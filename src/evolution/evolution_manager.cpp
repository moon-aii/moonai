#include "evolution/evolution_manager.hpp"
#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "core/types.hpp"
#include "evolution/crossover.hpp"
#include "evolution/mutation.hpp"

#ifdef MOONAI_ENABLE_CUDA
#include "evolution/backends/cuda/gpu_network_cache.hpp"
#include "simulation/backends/cuda/gpu_batch.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace moonai {

namespace {

void invalidate_gpu_cache(AgentRegistry &registry);

} // namespace

EvolutionManager::EvolutionManager(const SimulationConfig &config) : config_(config) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize_population(AgentRegistry &registry) const {
  registry.innovation_tracker = InnovationTracker();
  registry.innovation_tracker.set_counters(0, static_cast<std::uint32_t>(num_inputs_ + num_outputs_ + 1));
  registry.species.clear();
  registry.genomes.clear();
  registry.network_cache.clear();
  invalidate_gpu_cache(registry);
}

void EvolutionManager::initialize(AppState &state, int num_inputs, int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;

  Species::reset_id_counter();
  initialize_population(state.predator);
  initialize_population(state.prey);
}

using moonai::OUTPUT_COUNT;
using moonai::SENSOR_COUNT;

namespace {

void invalidate_gpu_cache(AgentRegistry &registry) {
#ifdef MOONAI_ENABLE_CUDA
  if (registry.gpu_network_cache) {
    registry.gpu_network_cache->invalidate();
  }
#else
  (void)registry;
#endif
}

class DenseReproductionGrid {
public:
  DenseReproductionGrid(float world_width, float world_height, float cell_size, std::size_t entity_count)
      : cell_size_(std::max(cell_size, 1.0f)),
        cols_(std::max(1, static_cast<int>(std::ceil(world_width / cell_size_)))),
        rows_(std::max(1, static_cast<int>(std::ceil(world_height / cell_size_)))),
        counts_(static_cast<std::size_t>(cols_ * rows_), 0), offsets_(static_cast<std::size_t>(cols_ * rows_) + 1, 0),
        write_offsets_(static_cast<std::size_t>(cols_ * rows_), 0), entries_(entity_count, INVALID_ENTITY) {}

  void build(const AgentRegistry &registry, std::size_t entity_count) {
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(offsets_.begin(), offsets_.end(), 0);

    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(registry.pos_x[idx], registry.pos_y[idx]);
      counts_[static_cast<std::size_t>(cell)] += 1;
    }

    for (std::size_t cell = 0; cell < counts_.size(); ++cell) {
      offsets_[cell + 1] = offsets_[cell] + counts_[cell];
    }

    std::copy(offsets_.begin(), offsets_.end() - 1, write_offsets_.begin());
    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(registry.pos_x[idx], registry.pos_y[idx]);
      const std::size_t slot = static_cast<std::size_t>(write_offsets_[static_cast<std::size_t>(cell)]++);
      entries_[slot] = idx;
    }
  }

  template <typename Callback> void for_each_candidate(Vec2 center, float radius, Callback &&callback) const {
    const int cells_to_check = std::max(1, static_cast<int>(std::ceil(radius / cell_size_)));
    const int base_x = cell_coord(center.x, cols_);
    const int base_y = cell_coord(center.y, rows_);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
      for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
        const int cell = flat_index(clamp_cell(base_x + dx, cols_), clamp_cell(base_y + dy, rows_));
        for (int slot = offsets_[static_cast<std::size_t>(cell)]; slot < offsets_[static_cast<std::size_t>(cell) + 1];
             ++slot) {
          callback(entries_[static_cast<std::size_t>(slot)]);
        }
      }
    }
  }

private:
  int cell_coord(float value, int limit) const {
    int coord = static_cast<int>(value / cell_size_);
    return clamp_cell(coord, limit);
  }

  int clamp_cell(int coord, int limit) const {
    if (coord < 0) {
      return 0;
    }
    if (coord >= limit) {
      return limit - 1;
    }
    return coord;
  }

  int flat_index(int x, int y) const {
    return y * cols_ + x;
  }

  int cell_index(float x, float y) const {
    return flat_index(cell_coord(x, cols_), cell_coord(y, rows_));
  }

  float cell_size_;
  int cols_;
  int rows_;
  std::vector<int> counts_;
  std::vector<int> offsets_;
  std::vector<int> write_offsets_;
  std::vector<uint32_t> entries_;
};

} // namespace

void EvolutionManager::compute_actions_for_population(AgentRegistry &registry, const std::vector<float> &sensors,
                                                      std::vector<float> &decisions_out) const {
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());

  std::vector<float> all_outputs;
  registry.network_cache.activate_batch(entity_count, sensors, all_outputs, SENSOR_COUNT, OUTPUT_COUNT);

  decisions_out.resize(entity_count * OUTPUT_COUNT);
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    decisions_out[idx * OUTPUT_COUNT] = all_outputs[idx * OUTPUT_COUNT];
    decisions_out[idx * OUTPUT_COUNT + 1] = all_outputs[idx * OUTPUT_COUNT + 1];
  }
}

bool EvolutionManager::run_inference(AppState &state) {
  MOONAI_PROFILE_SCOPE("evolution_run_inference");

#ifdef MOONAI_ENABLE_CUDA
  if (state.runtime.gpu_enabled) {
    if (!state.gpu_batch) {
      spdlog::error("GPU batch is not initialized for neural inference");
      enable_gpu(state, false);
      return false;
    }

    if (!launch_gpu_neural(state, *state.gpu_batch)) {
      enable_gpu(state, false);
      return false;
    }

    return true;
  }
#else
  if (state.runtime.gpu_enabled) {
    spdlog::warn("GPU path requested, but CUDA support is not compiled in; using CPU inference");
    state.runtime.gpu_enabled = false;
  }
#endif

  compute_actions_for_population(state.predator, state.step_buffers.predator_sensors,
                                 state.step_buffers.predator_decisions);
  compute_actions_for_population(state.prey, state.step_buffers.prey_sensors, state.step_buffers.prey_decisions);
  return true;
}

Genome EvolutionManager::create_initial_genome(AgentRegistry &registry, Random &rng) const {
  Genome genome(num_inputs_, num_outputs_);
  for (const auto &in_node : genome.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
      continue;
    }
    for (const auto &out_node : genome.nodes()) {
      if (out_node.type != NodeType::Output) {
        continue;
      }

      const std::uint32_t innovation = registry.innovation_tracker.get_innovation(in_node.id, out_node.id);
      genome.add_connection({in_node.id, out_node.id, rng.next_float(-1.0f, 1.0f), true, innovation});
    }
  }
  return genome;
}

Genome EvolutionManager::create_child_genome(AgentRegistry &registry, Random &rng, const Genome &parent_a,
                                             const Genome &parent_b) const {
  Genome child = Crossover::crossover(parent_a, parent_b, rng);
  Mutation::mutate(child, rng, config_, registry.innovation_tracker);
  return child;
}

void EvolutionManager::seed_initial_population(AppState &state) {
  state.predator.clear();
  state.prey.clear();
  state.runtime.next_agent_id = 1;
  initialize_population(state.predator);
  initialize_population(state.prey);

  const float grid_size = static_cast<float>(config_.grid_size);

  auto seed_predator = [&] {
    const uint32_t idx = state.predator.create();
    Genome genome = create_initial_genome(state.predator, state.runtime.rng);
    if (idx >= state.predator.genomes.size()) {
      state.predator.genomes.resize(idx + 1);
    }
    state.predator.genomes[idx] = genome;
    state.predator.network_cache.assign(idx, state.predator.genomes[idx]);

    state.predator.pos_x[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.predator.pos_y[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.predator.vel_x[idx] = 0.0f;
    state.predator.vel_y[idx] = 0.0f;
    state.predator.energy[idx] = config_.initial_energy;
    state.predator.age[idx] = 0;
    state.predator.alive[idx] = 1;
    state.predator.species_id[idx] = 0;
    state.predator.entity_id[idx] = state.runtime.next_agent_id++;
    state.predator.consumption[idx] = 0;
  };

  auto seed_prey = [&] {
    const uint32_t idx = state.prey.create();
    Genome genome = create_initial_genome(state.prey, state.runtime.rng);
    if (idx >= state.prey.genomes.size()) {
      state.prey.genomes.resize(idx + 1);
    }
    state.prey.genomes[idx] = genome;
    state.prey.network_cache.assign(idx, state.prey.genomes[idx]);

    state.prey.pos_x[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.prey.pos_y[idx] = state.runtime.rng.next_float(0.0f, grid_size);
    state.prey.vel_x[idx] = 0.0f;
    state.prey.vel_y[idx] = 0.0f;
    state.prey.energy[idx] = config_.initial_energy;
    state.prey.age[idx] = 0;
    state.prey.alive[idx] = 1;
    state.prey.species_id[idx] = 0;
    state.prey.entity_id[idx] = state.runtime.next_agent_id++;
    state.prey.consumption[idx] = 0;
  };

  for (int i = 0; i < config_.predator_count; ++i) {
    seed_predator();
  }
  for (int i = 0; i < config_.prey_count; ++i) {
    seed_prey();
  }

  invalidate_gpu_cache(state.predator);
  invalidate_gpu_cache(state.prey);
}

uint32_t EvolutionManager::create_offspring(AppState &state, AgentRegistry &registry, uint32_t parent_a,
                                            uint32_t parent_b, Vec2 spawn_position) {
  MOONAI_PROFILE_SCOPE("evolution_offspring_predator");
  if (!registry.valid(parent_a) || !registry.valid(parent_b) || parent_a >= registry.genomes.size() ||
      parent_b >= registry.genomes.size()) {
    return INVALID_ENTITY;
  }

  Genome child_genome =
      create_child_genome(registry, state.runtime.rng, registry.genomes[parent_a], registry.genomes[parent_b]);

  const uint32_t idx = registry.create();
  registry.pos_x[idx] = spawn_position.x;
  registry.pos_y[idx] = spawn_position.y;
  registry.vel_x[idx] = 0.0f;
  registry.vel_y[idx] = 0.0f;
  registry.energy[idx] = config_.offspring_initial_energy;
  registry.age[idx] = 0;
  registry.alive[idx] = 1;
  registry.species_id[idx] = registry.species_id[parent_a];
  registry.entity_id[idx] = state.runtime.next_agent_id++;
  registry.consumption[idx] = 0;

  if (idx >= registry.genomes.size()) {
    registry.genomes.resize(idx + 1);
  }
  registry.genomes[idx] = std::move(child_genome);
  registry.network_cache.assign(idx, registry.genomes[idx]);

  registry.energy[parent_a] -= config_.reproduction_energy_cost;
  registry.energy[parent_b] -= config_.reproduction_energy_cost;

  invalidate_gpu_cache(registry);

  return idx;
}

void EvolutionManager::refresh_population_species(AgentRegistry &registry) const {
  auto &species = registry.species;
  for (auto &entry : species) {
    entry.clear_members();
  }

  const uint32_t entity_count = static_cast<uint32_t>(registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (idx >= registry.genomes.size()) {
      continue;
    }

    Genome &genome = registry.genomes[idx];
    int assigned_species_id = -1;

    for (auto &entry : species) {
      if (entry.is_compatible(genome, config_.compatibility_threshold, config_.c1_excess, config_.c2_disjoint,
                              config_.c3_weight)) {
        entry.add_member(idx, genome);
        assigned_species_id = entry.id();
        break;
      }
    }

    if (assigned_species_id < 0) {
      Species entry(genome);
      entry.add_member(idx, genome);
      assigned_species_id = entry.id();
      species.push_back(std::move(entry));
    }

    registry.species_id[idx] = static_cast<uint32_t>(assigned_species_id);
  }

  for (auto &entry : species) {
    entry.refresh_summary();
  }

  species.erase(
      std::remove_if(species.begin(), species.end(), [](const Species &entry) { return entry.members().empty(); }),
      species.end());

  for (auto &entry : species) {
    const uint32_t representative_idx = entry.members().front().entity;
    if (representative_idx < registry.genomes.size()) {
      entry.set_representative(registry.genomes[representative_idx]);
    }
  }
}

void EvolutionManager::refresh_species(AppState &state) {
  MOONAI_PROFILE_SCOPE("refresh_species");
  refresh_population_species(state.predator);
  refresh_population_species(state.prey);
}

void EvolutionManager::reproduce_population(AppState &state, AgentRegistry &registry) {
  std::vector<uint8_t> used(registry.size(), 0);

  DenseReproductionGrid grid(static_cast<float>(config_.grid_size), static_cast<float>(config_.grid_size),
                             config_.mate_range, registry.size());
  grid.build(registry, registry.size());
  const float world_size = static_cast<float>(config_.grid_size);
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (registry.energy[idx] < config_.reproduction_energy_threshold || used[idx] != 0) {
      continue;
    }

    const Vec2 pos{registry.pos_x[idx], registry.pos_y[idx]};
    uint32_t best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    grid.for_each_candidate(pos, config_.mate_range, [&](uint32_t mate_id) {
      if (mate_id == idx || used[mate_id] != 0 || registry.energy[mate_id] < config_.reproduction_energy_threshold) {
        return;
      }

      const Vec2 mate_pos{registry.pos_x[mate_id], registry.pos_y[mate_id]};
      const Vec2 diff{mate_pos.x - pos.x, mate_pos.y - pos.y};
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    });

    if (best_mate == INVALID_ENTITY) {
      continue;
    }

    const Vec2 mate_pos{registry.pos_x[best_mate], registry.pos_y[best_mate]};
    const Vec2 mid_pos{(pos.x + mate_pos.x) * 0.5f, (pos.y + mate_pos.y) * 0.5f};
    const float clamped_x = std::clamp(mid_pos.x, 0.0f, world_size);
    const float clamped_y = std::clamp(mid_pos.y, 0.0f, world_size);
    const uint32_t child = create_offspring(state, registry, idx, best_mate, {clamped_x, clamped_y});
    if (child != INVALID_ENTITY) {
      ++state.metrics.step_delta.births;
    }
    used[idx] = 1;
    used[best_mate] = 1;
  }
}

void EvolutionManager::post_step(AppState &state) {
  MOONAI_PROFILE_SCOPE("evolution_post_step");

  reproduce_population(state, state.predator);
  reproduce_population(state, state.prey);

  if (config_.species_update_interval_steps > 0 && (state.runtime.step % config_.species_update_interval_steps) == 0) {
    refresh_species(state);
  }
}

#ifdef MOONAI_ENABLE_CUDA
namespace {

bool launch_population_gpu_neural(AgentRegistry &registry, gpu::GpuNetworkCache &gpu_cache,
                                  gpu::GpuPopulationBuffer &buffer, std::size_t count, cudaStream_t stream) {
  std::vector<std::pair<uint32_t, int>> network_entities_with_indices;
  network_entities_with_indices.reserve(count);

  const uint32_t entity_count = static_cast<uint32_t>(count);
  for (uint32_t entity = 0; entity < entity_count; ++entity) {
    if (registry.network_cache.has(entity)) {
      network_entities_with_indices.emplace_back(entity, static_cast<int>(entity));
    }
  }

  if (network_entities_with_indices.empty()) {
    return true;
  }

  if (gpu_cache.is_dirty() || gpu_cache.entity_mapping().size() != network_entities_with_indices.size() ||
      !std::equal(gpu_cache.entity_mapping().begin(), gpu_cache.entity_mapping().end(),
                  network_entities_with_indices.begin(),
                  [](uint32_t entity, const std::pair<uint32_t, int> &entity_with_index) {
                    return entity == entity_with_index.first;
                  })) {
    gpu_cache.build_from(registry.network_cache, network_entities_with_indices);
  }

  return gpu_cache.launch_inference_async(buffer.device_sensor_inputs(), buffer.device_brain_outputs(),
                                          network_entities_with_indices.size(), stream);
}

} // namespace
#endif

void EvolutionManager::enable_gpu(AppState &state, bool use_gpu) {
#ifdef MOONAI_ENABLE_CUDA
  if (use_gpu) {
    if (!state.predator.gpu_network_cache) {
      state.predator.gpu_network_cache = std::make_unique<gpu::GpuNetworkCache>();
      state.predator.gpu_network_cache->invalidate();
    }
    if (!state.prey.gpu_network_cache) {
      state.prey.gpu_network_cache = std::make_unique<gpu::GpuNetworkCache>();
      state.prey.gpu_network_cache->invalidate();
    }
    state.runtime.gpu_enabled = true;
    spdlog::info("GPU neural inference enabled");
  } else {
    state.predator.gpu_network_cache.reset();
    state.prey.gpu_network_cache.reset();
    state.gpu_batch.reset();
    state.runtime.gpu_enabled = false;
    spdlog::info("GPU neural inference disabled");
  }
#else
  state.runtime.gpu_enabled = false;
  if (use_gpu) {
    spdlog::warn("GPU requested, but this build was compiled without CUDA support");
  }
#endif
}

#ifdef MOONAI_ENABLE_CUDA
bool EvolutionManager::launch_gpu_neural(AppState &state, gpu::GpuBatch &gpu_batch) {
  MOONAI_PROFILE_SCOPE("gpu_neural", gpu_batch.stream());

  if (!state.predator.gpu_network_cache || !state.prey.gpu_network_cache) {
    spdlog::error("GPU neural caches not initialized");
    return false;
  }

  if (!launch_population_gpu_neural(state.predator, *state.predator.gpu_network_cache, gpu_batch.predator_buffer(),
                                    state.predator.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  if (!launch_population_gpu_neural(state.prey, *state.prey.gpu_network_cache, gpu_batch.prey_buffer(),
                                    state.prey.size(), gpu_batch.stream())) {
    gpu_batch.mark_error();
    return false;
  }

  return true;
}
#endif

} // namespace moonai
