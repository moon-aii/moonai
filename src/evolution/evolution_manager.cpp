#include "evolution/evolution_manager.hpp"

#include "core/profiler_types.hpp"
#include "evolution/crossover.hpp"
#include "simulation/predator.hpp"
#include "simulation/prey.hpp"

#include <algorithm>
#include <limits>

#ifdef MOONAI_ENABLE_CUDA
#include <spdlog/spdlog.h>
#endif

namespace moonai {

#ifdef MOONAI_ENABLE_CUDA
namespace {

gpu::GpuNetworkData build_gpu_network_data(const SimulationManager &sim,
                                           const std::string &activation_fn) {
  gpu::GpuNetworkData data;

  if (activation_fn == "tanh") {
    data.activation_fn_id = 1;
  } else if (activation_fn == "relu") {
    data.activation_fn_id = 2;
  } else {
    data.activation_fn_id = 0;
  }

  const int agent_count = static_cast<int>(sim.agents().size());
  data.descs.resize(static_cast<std::size_t>(agent_count));

  int total_nodes = 0;
  int total_eval = 0;
  int total_conn = 0;
  int total_out = 0;

  for (int i = 0; i < agent_count; ++i) {
    const NeuralNetwork *network =
        sim.agents()[static_cast<std::size_t>(i)]->network();
    if (network == nullptr) {
      continue;
    }

    const auto &nodes = network->raw_nodes();
    const auto &eval_order = network->eval_order();
    const auto &incoming = network->incoming();
    const auto &node_index = network->node_index_map();

    int num_conn = 0;
    for (auto node_id : eval_order) {
      num_conn += static_cast<int>(incoming[node_index.at(node_id)].size());
    }
    const int num_out = static_cast<int>(
        std::count_if(nodes.begin(), nodes.end(), [](const auto &node) {
          return node.type == NodeType::Output;
        }));

    data.descs[static_cast<std::size_t>(i)] = {
        static_cast<int>(nodes.size()),
        static_cast<int>(eval_order.size()),
        network->num_inputs(),
        network->num_outputs(),
        total_nodes,
        total_eval,
        total_conn,
        total_out};

    total_nodes += static_cast<int>(nodes.size());
    total_eval += static_cast<int>(eval_order.size());
    total_conn += num_conn;
    total_out += num_out;
  }

  data.node_types.resize(static_cast<std::size_t>(total_nodes));
  data.eval_order.resize(static_cast<std::size_t>(total_eval));
  data.conn_ptr.resize(static_cast<std::size_t>(total_eval));
  data.in_count.resize(static_cast<std::size_t>(total_eval));
  data.conn_from.resize(static_cast<std::size_t>(total_conn));
  data.conn_w.resize(static_cast<std::size_t>(total_conn));
  data.out_indices.resize(static_cast<std::size_t>(total_out));

  for (int i = 0; i < agent_count; ++i) {
    const NeuralNetwork *network =
        sim.agents()[static_cast<std::size_t>(i)]->network();
    if (network == nullptr) {
      continue;
    }

    const auto &nodes = network->raw_nodes();
    const auto &eval_order = network->eval_order();
    const auto &incoming = network->incoming();
    const auto &node_index = network->node_index_map();
    const auto &desc = data.descs[static_cast<std::size_t>(i)];

    for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
      uint8_t type = 2;
      switch (nodes[static_cast<std::size_t>(j)].type) {
        case NodeType::Input:
          type = 0;
          break;
        case NodeType::Bias:
          type = 1;
          break;
        case NodeType::Hidden:
          type = 2;
          break;
        case NodeType::Output:
          type = 3;
          break;
      }
      data.node_types[static_cast<std::size_t>(desc.node_off + j)] = type;
    }

    int conn_running = 0;
    for (int j = 0; j < static_cast<int>(eval_order.size()); ++j) {
      const std::uint32_t node_id = eval_order[static_cast<std::size_t>(j)];
      const int node_slot = node_index.at(node_id);

      data.eval_order[static_cast<std::size_t>(desc.eval_off + j)] = node_slot;
      data.conn_ptr[static_cast<std::size_t>(desc.eval_off + j)] = conn_running;
      data.in_count[static_cast<std::size_t>(desc.eval_off + j)] =
          static_cast<int>(
              incoming[static_cast<std::size_t>(node_slot)].size());

      for (const auto &[from_idx, weight] :
           incoming[static_cast<std::size_t>(node_slot)]) {
        data.conn_from[static_cast<std::size_t>(desc.conn_off + conn_running)] =
            from_idx;
        data.conn_w[static_cast<std::size_t>(desc.conn_off + conn_running)] =
            weight;
        ++conn_running;
      }
    }

    int out_running = 0;
    for (int j = 0; j < static_cast<int>(nodes.size()); ++j) {
      if (nodes[static_cast<std::size_t>(j)].type == NodeType::Output) {
        data.out_indices[static_cast<std::size_t>(desc.out_off + out_running)] =
            j;
        ++out_running;
      }
    }
  }

  return data;
}

} // namespace
#endif

EvolutionManager::EvolutionManager(const SimulationConfig &config, Random &rng)
    : config_(config), rng_(rng) {}

EvolutionManager::~EvolutionManager() = default;

void EvolutionManager::initialize(int num_inputs, int num_outputs) {
  num_inputs_ = num_inputs;
  num_outputs_ = num_outputs;
  species_.clear();
  species_refresh_step_ = -1;
  gpu_runtime_ready_ = false;
  gpu_warning_emitted_ = false;
  gpu_layout_agent_ids_.clear();
#ifdef MOONAI_ENABLE_CUDA
  gpu_batch_.reset();
  gpu_activation_function_.clear();
#endif
}

Genome EvolutionManager::create_initial_genome() const {
  Genome genome(num_inputs_, num_outputs_);
  for (const auto &in_node : genome.nodes()) {
    if (in_node.type != NodeType::Input && in_node.type != NodeType::Bias) {
      continue;
    }
    for (const auto &out_node : genome.nodes()) {
      if (out_node.type != NodeType::Output) {
        continue;
      }
      const std::uint32_t innov =
          const_cast<InnovationTracker &>(tracker_).get_innovation(in_node.id,
                                                                   out_node.id);
      genome.add_connection({in_node.id, out_node.id,
                             const_cast<Random &>(rng_).next_float(-1.0f, 1.0f),
                             true, innov});
    }
  }
  return genome;
}

Genome EvolutionManager::create_child_genome(const Genome &parent_a,
                                             const Genome &parent_b) const {
  Genome child = Crossover::crossover(parent_a, parent_b, rng_);
  Mutation::mutate(child, rng_, config_,
                   const_cast<InnovationTracker &>(tracker_));
  return child;
}

std::unique_ptr<Agent> EvolutionManager::create_agent(AgentId id,
                                                      AgentType type,
                                                      Vec2 position,
                                                      Genome genome) const {
  std::unique_ptr<Agent> agent;
  if (type == AgentType::Predator) {
    agent = std::make_unique<Predator>(
        id, position, config_.predator_speed, config_.vision_range,
        config_.initial_energy, config_.attack_range);
  } else {
    agent =
        std::make_unique<Prey>(id, position, config_.prey_speed,
                               config_.vision_range, config_.initial_energy);
  }
  agent->set_genome(std::move(genome), config_.activation_function);
  return agent;
}

void EvolutionManager::seed_initial_population(SimulationManager &sim) {
  for (int i = 0; i < config_.predator_count; ++i) {
    const AgentId id = static_cast<AgentId>(i);
    Vec2 pos{rng_.next_float(0.0f, static_cast<float>(config_.grid_size)),
             rng_.next_float(0.0f, static_cast<float>(config_.grid_size))};
    sim.spawn_agent(
        create_agent(id, AgentType::Predator, pos, create_initial_genome()));
  }

  const AgentId prey_base = static_cast<AgentId>(config_.predator_count);
  for (int i = 0; i < config_.prey_count; ++i) {
    const AgentId id = static_cast<AgentId>(prey_base + i);
    Vec2 pos{rng_.next_float(0.0f, static_cast<float>(config_.grid_size)),
             rng_.next_float(0.0f, static_cast<float>(config_.grid_size))};
    sim.spawn_agent(
        create_agent(id, AgentType::Prey, pos, create_initial_genome()));
  }

  refresh_species(sim);
  refresh_fitness(sim);
}

void EvolutionManager::compute_actions(const SimulationManager &sim,
                                       std::vector<Vec2> &actions) {
  if (actions.size() < sim.agents().size()) {
    actions.resize(sim.agents().size(), {0.0f, 0.0f});
  }
  std::fill(actions.begin(), actions.end(), Vec2{0.0f, 0.0f});

  if (sim.agents().empty()) {
    return;
  }

  // CPU-only path: build sensors and run inference on CPU
  MOONAI_PROFILE_MARK_CPU_USED(true);
  MOONAI_PROFILE_SCOPE(ProfileEvent::CpuEvalTotal);
  for (std::size_t slot : sim.alive_agent_indices()) {
    const Agent *agent = sim.agents()[slot].get();
    if (agent->network() == nullptr) {
      continue;
    }
    std::vector<float> sensors;
    {
      MOONAI_PROFILE_SCOPE(ProfileEvent::CpuSensorBuild);
      sensors = sim.get_sensors(slot).to_vector();
    }
    std::vector<float> output;
    {
      MOONAI_PROFILE_SCOPE(ProfileEvent::CpuNnActivate);
      output = agent->network()->activate(sensors);
    }
    if (output.size() >= 2) {
      actions[slot] = {output[0] * 2.0f - 1.0f, output[1] * 2.0f - 1.0f};
    }
  }
}

AgentId EvolutionManager::create_offspring(SimulationManager &sim,
                                           AgentId parent_a_id,
                                           AgentId parent_b_id,
                                           Vec2 spawn_position) {
  Agent *parent_a = sim.agent_by_id(parent_a_id);
  Agent *parent_b = sim.agent_by_id(parent_b_id);
  if (parent_a == nullptr || parent_b == nullptr) {
    return std::numeric_limits<AgentId>::max();
  }

  Genome child = create_child_genome(parent_a->genome(), parent_b->genome());
  const AgentType role = parent_a->type();
  const AgentId child_id = sim.next_available_agent_id();

  auto child_agent =
      create_agent(child_id, role, spawn_position, std::move(child));
  child_agent->set_energy(config_.offspring_initial_energy);
  child_agent->set_species_id(parent_a->species_id());
  sim.spawn_agent(std::move(child_agent));
  sim.record_event(SimEvent{SimEvent::Birth, child_id, child_id, parent_a_id,
                            parent_b_id, spawn_position});

  for (Agent *parent : {parent_a, parent_b}) {
    parent->drain_energy(config_.reproduction_energy_cost);
    parent->set_reproduction_cooldown(config_.reproduction_cooldown_steps);
    parent->add_offspring();
  }

  species_refresh_step_ = -1;
  gpu_runtime_ready_ = false;
  gpu_layout_agent_ids_.clear();
  if (Agent *child_agent_ptr = sim.agent_by_id(child_id);
      child_agent_ptr != nullptr && child_agent_ptr->species_id() < 0) {
    child_agent_ptr->set_species_id(parent_a->species_id());
  }
  return child_id;
}

float EvolutionManager::default_fitness(const Agent &agent) const {
  const float energy_ratio =
      std::max(0.0f, agent.energy()) / std::max(config_.initial_energy, 1.0f);
  const float survival_score =
      static_cast<float>(agent.age()) /
      static_cast<float>(std::max(config_.report_interval_steps, 1));
  const float forage_score = (agent.type() == AgentType::Predator)
                                 ? static_cast<float>(agent.kills())
                                 : static_cast<float>(agent.food_eaten());
  const float offspring_score =
      static_cast<float>(agent.offspring_count()) * 5.0f;
  const float complexity = static_cast<float>(agent.genome().complexity());
  return std::max(0.0f, config_.fitness_survival_weight * survival_score +
                            config_.fitness_kill_weight * forage_score +
                            config_.fitness_energy_weight * energy_ratio +
                            offspring_score -
                            config_.complexity_penalty_weight * complexity);
}

void EvolutionManager::refresh_fitness(const SimulationManager &sim) {
  for (const auto &agent : sim.agents()) {
    agent->genome().set_fitness(default_fitness(*agent));
  }
}

void EvolutionManager::refresh_species(SimulationManager &sim) {
  if (species_refresh_step_ == sim.current_step()) {
    return;
  }

  refresh_fitness(sim);
  species_.clear();
  for (std::size_t slot : sim.alive_agent_indices()) {
    auto &agent = sim.agents()[slot];
    Genome &genome = agent->genome();
    auto it = std::find_if(
        species_.begin(), species_.end(), [&](const Species &species) {
          return species.is_compatible(genome, config_.compatibility_threshold,
                                       config_.c1_excess, config_.c2_disjoint,
                                       config_.c3_weight);
        });
    if (it == species_.end()) {
      species_.emplace_back(genome);
      it = std::prev(species_.end());
    }
    it->add_member(agent->id(), genome);
    agent->set_species_id(it->id());
  }

  for (auto &species : species_) {
    species.refresh_summary();
    if (!species.members().empty()) {
      const Agent *representative =
          sim.agent_by_id(species.members().front().agent_id);
      if (representative != nullptr) {
        species.update_representative(representative->genome());
      }
    }
  }

  species_refresh_step_ = sim.current_step();
}

const Genome *EvolutionManager::genome_at(const SimulationManager &sim,
                                          int idx) const {
  if (idx < 0 || idx >= static_cast<int>(sim.agents().size())) {
    return nullptr;
  }
  return &sim.agents()[static_cast<std::size_t>(idx)]->genome();
}

NeuralNetwork *EvolutionManager::network_at(const SimulationManager &sim,
                                            int idx) const {
  if (idx < 0 || idx >= static_cast<int>(sim.agents().size())) {
    return nullptr;
  }
  return sim.agents()[static_cast<std::size_t>(idx)]->network();
}

void EvolutionManager::get_fitness_by_type(const SimulationManager &sim,
                                           float &best_predator,
                                           float &avg_predator,
                                           float &best_prey,
                                           float &avg_prey) const {
  best_predator = 0.0f;
  avg_predator = 0.0f;
  best_prey = 0.0f;
  avg_prey = 0.0f;

  float predator_sum = 0.0f;
  float prey_sum = 0.0f;
  int predator_count = 0;
  int prey_count = 0;
  for (const auto &agent : sim.agents()) {
    const float fitness = agent->genome().fitness();
    if (agent->type() == AgentType::Predator) {
      best_predator = std::max(best_predator, fitness);
      predator_sum += fitness;
      ++predator_count;
    } else {
      best_prey = std::max(best_prey, fitness);
      prey_sum += fitness;
      ++prey_count;
    }
  }

  if (predator_count > 0) {
    avg_predator = predator_sum / static_cast<float>(predator_count);
  }
  if (prey_count > 0) {
    avg_prey = prey_sum / static_cast<float>(prey_count);
  }
}

bool EvolutionManager::current_gpu_layout_matches(
    const SimulationManager &sim) const {
  if (!gpu_runtime_ready_ ||
      gpu_layout_agent_ids_.size() != sim.agents().size() ||
      gpu_activation_function_ != config_.activation_function) {
    return false;
  }

  for (std::size_t i = 0; i < sim.agents().size(); ++i) {
    if (gpu_layout_agent_ids_[i] != sim.agents()[i]->id()) {
      return false;
    }
  }
  return true;
}

bool EvolutionManager::rebuild_gpu_runtime(const SimulationManager &sim) {
#ifndef MOONAI_ENABLE_CUDA
  (void)sim;
  return false;
#else
  if (!use_gpu_) {
    return false;
  }

  if (gpu_batch_ == nullptr) {
    if (!gpu::init_cuda()) {
      return false;
    }
    gpu::print_device_info();
  }

  gpu_batch_ = std::make_unique<gpu::GpuBatch>(
      static_cast<int>(sim.agents().size()), num_inputs_, num_outputs_,
      static_cast<float>(config_.grid_size),
      static_cast<float>(config_.grid_size));
  if (!gpu_batch_->ok()) {
    gpu_batch_.reset();
    return false;
  }

  MOONAI_PROFILE_SCOPE(ProfileEvent::PrepareGpuWindow);
  gpu_batch_->upload_network_data(
      build_gpu_network_data(sim, config_.activation_function));
  if (!gpu_batch_->ok()) {
    gpu_batch_.reset();
    return false;
  }

  gpu_activation_function_ = config_.activation_function;
  gpu_layout_agent_ids_.resize(sim.agents().size());
  for (std::size_t i = 0; i < sim.agents().size(); ++i) {
    gpu_layout_agent_ids_[i] = sim.agents()[i]->id();
  }
  gpu_runtime_ready_ = true;
  gpu_warning_emitted_ = false;
  return true;
#endif
}

bool EvolutionManager::step_gpu(SimulationManager &sim, int step_index) {
#ifndef MOONAI_ENABLE_CUDA
  (void)sim;
  (void)step_index;
  return false;
#else
  if (!use_gpu_) {
    return false;
  }

  // Check if we need to rebuild GPU runtime (births/deaths changed agent count)
  if (!current_gpu_layout_matches(sim) && !rebuild_gpu_runtime(sim)) {
    gpu_runtime_ready_ = false;
    return false;
  }

  if (gpu_batch_ == nullptr) {
    gpu_runtime_ready_ = false;
    return false;
  }

  auto &batch = *gpu_batch_;
  MOONAI_PROFILE_MARK_GPU_USED(true);

  // 1. Prepare agent states for upload
  std::vector<gpu::GpuAgentState> agent_states;
  agent_states.reserve(sim.agents().size());

  for (const auto &agent : sim.agents()) {
    gpu::GpuAgentState state;
    state.pos_x = agent->position().x;
    state.pos_y = agent->position().y;
    state.vel_x = agent->velocity().x;
    state.vel_y = agent->velocity().y;
    state.speed = agent->speed();
    state.vision_range = agent->vision_range();
    state.energy = agent->energy();
    state.distance_traveled = agent->distance_traveled();
    state.age = agent->age();
    state.kills = agent->kills();
    state.food_eaten = agent->food_eaten();
    state.id = static_cast<unsigned int>(agent->id());
    state.type = (agent->type() == AgentType::Predator) ? 0U : 1U;
    state.alive = agent->alive() ? 1U : 0U;
    agent_states.push_back(state);
  }

  // 2. Upload agent states to GPU
  {
    MOONAI_PROFILE_SCOPE(ProfileEvent::GpuPackInputs);
    batch.upload_agent_states_async(agent_states.data(),
                                    static_cast<int>(agent_states.size()));
  }

  // 3. Prepare and upload food states
  auto &food_vec = sim.environment().mutable_food();
  std::vector<gpu::GpuFoodState> food_states;
  food_states.reserve(food_vec.size());
  for (const auto &food : food_vec) {
    gpu::GpuFoodState state;
    state.pos_x = food.position.x;
    state.pos_y = food.position.y;
    state.active = food.active ? 1U : 0U;
    food_states.push_back(state);
  }
  if (!food_states.empty()) {
    batch.upload_food_states_async(food_states.data(),
                                   static_cast<int>(food_states.size()));
  }

  // 4. Launch full ecology step on GPU
  gpu::EcologyStepParams params;
  params.dt = 1.0f / static_cast<float>(config_.target_fps);
  params.world_width = static_cast<float>(config_.grid_size);
  params.world_height = static_cast<float>(config_.grid_size);
  params.has_walls = (config_.boundary_mode == BoundaryMode::Clamp);
  params.energy_drain_per_step = config_.energy_drain_per_step;
  params.target_fps = config_.target_fps;
  params.food_pickup_range = config_.food_pickup_range;
  params.attack_range = config_.attack_range;
  params.max_energy = config_.initial_energy;
  params.energy_gain_from_food = config_.energy_gain_from_food;
  params.energy_gain_from_kill = config_.energy_gain_from_kill;
  params.food_respawn_rate = config_.food_respawn_rate;
  params.seed = config_.seed;
  params.step_index = step_index;

  {
    MOONAI_PROFILE_SCOPE(ProfileEvent::GpuLaunch);
    batch.launch_ecology_step_async(params);
  }

  // 5. Download results back to CPU
  std::vector<float> new_pos_x(sim.agents().size());
  std::vector<float> new_pos_y(sim.agents().size());
  std::vector<float> new_energy(sim.agents().size());
  std::vector<unsigned int> new_alive(sim.agents().size());
  std::vector<int> new_kills(sim.agents().size());
  std::vector<int> new_food_eaten(sim.agents().size());
  std::vector<int> new_age(sim.agents().size());
  std::vector<float> new_distance(sim.agents().size());

  {
    MOONAI_PROFILE_SCOPE(ProfileEvent::GpuOutputConvert);
    batch.download_agent_changes_async(new_pos_x.data(), new_pos_y.data(),
                                       new_energy.data(), new_alive.data(),
                                       static_cast<int>(sim.agents().size()));
    batch.download_agent_counters_async(new_kills.data(), new_food_eaten.data(),
                                        new_age.data(), new_distance.data(),
                                        static_cast<int>(sim.agents().size()));
  }

  // 6. Download food states from GPU
  std::vector<gpu::GpuFoodState> new_food_states;
  if (!food_states.empty()) {
    new_food_states.resize(food_states.size());
    batch.download_food_states_async(new_food_states.data(),
                                     static_cast<int>(new_food_states.size()));
  }

  batch.synchronize();

  if (!batch.ok()) {
    gpu_runtime_ready_ = false;
    gpu_batch_.reset();
    return false;
  }

  // 7. Update simulation state with GPU results
  int total_food_eaten_this_step = 0;
  int total_kills_this_step = 0;
  for (std::size_t i = 0; i < sim.agents().size(); ++i) {
    auto &agent = sim.agents()[i];
    agent->set_position({new_pos_x[i], new_pos_y[i]});
    agent->set_energy(new_energy[i]);
    agent->set_age(new_age[i]);
    agent->set_distance_traveled(new_distance[i]);

    // Only process agents that were alive at the start of this step
    if (agent->alive()) {
      // Record kill events (counter increased on GPU)
      int kill_delta = new_kills[i] - agent->kills();
      total_kills_this_step += kill_delta;
      for (int k = 0; k < kill_delta; ++k) {
        sim.record_event(SimEvent{SimEvent::Kill, agent->id(), agent->id(), 0,
                                  0, agent->position()});
      }
      agent->set_kills(new_kills[i]);

      // Record food events (counter increased on GPU)
      int food_delta = new_food_eaten[i] - agent->food_eaten();
      total_food_eaten_this_step += food_delta;
      for (int f = 0; f < food_delta; ++f) {
        sim.record_event(SimEvent{SimEvent::Food, agent->id(),
                                  static_cast<AgentId>(f), 0, 0,
                                  agent->position()});
      }
      agent->set_food_eaten(new_food_eaten[i]);

      // Check if agent died this step
      if (new_alive[i] == 0U) {
        agent->set_alive(false);
        // Record death event
        sim.record_event(SimEvent{SimEvent::Death, agent->id(), agent->id(),
                                  agent->id(), agent->id(), agent->position()});
      }
    }
  }

  // Debug: Print food/kill counts every 60 steps
  if (step_index % 60 == 0 && step_index > 0) {
    spdlog::info("GPU Step {}: food_eaten={}, kills={}", step_index,
                 total_food_eaten_this_step, total_kills_this_step);
  }

  // 8. Update food states
  for (std::size_t i = 0; i < new_food_states.size(); ++i) {
    auto &food = food_vec[i];
    food.position.x = new_food_states[i].pos_x;
    food.position.y = new_food_states[i].pos_y;
    food.active = (new_food_states[i].active != 0U);
  }

  // 9. Refresh simulation state (rebuild spatial grids, alive indices)
  sim.refresh_state();

  // 10. Increment step counter so GUI and species refresh work correctly
  sim.increment_step();

  return true;
#endif
}

} // namespace moonai
