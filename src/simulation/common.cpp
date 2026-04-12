#include "simulation/common.hpp"

#include "core/metrics.hpp"
#include "evolution/evolution_manager.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace moonai::common {

namespace {

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
    if (coord < 0) {
      return 0;
    }
    if (coord >= limit) {
      return limit - 1;
    }
    return coord;
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

void reproduction(AppState &state, EvolutionManager &evolution, AgentRegistry &registry,
                  const SimulationConfig &config) {
  std::vector<uint8_t> used(registry.size(), 0);

  DenseReproductionGrid grid(static_cast<float>(config.grid_size), static_cast<float>(config.grid_size),
                             config.mate_range, registry.size());
  grid.build(registry, registry.size());
  const float world_size = static_cast<float>(config.grid_size);
  const uint32_t entity_count = static_cast<uint32_t>(registry.size());

  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    if (registry.energy[idx] < config.reproduction_energy_threshold || used[idx] != 0) {
      continue;
    }

    const Vec2 pos{registry.pos_x[idx], registry.pos_y[idx]};
    uint32_t best_mate = INVALID_ENTITY;
    float best_dist_sq = config.mate_range * config.mate_range;

    grid.for_each_candidate(pos, config.mate_range, [&](uint32_t mate_id) {
      if (mate_id == idx || used[mate_id] != 0 || registry.energy[mate_id] < config.reproduction_energy_threshold) {
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

    if (best_mate != INVALID_ENTITY) {
      const Vec2 mate_pos{registry.pos_x[best_mate], registry.pos_y[best_mate]};
      const Vec2 mid_pos{(pos.x + mate_pos.x) * 0.5f, (pos.y + mate_pos.y) * 0.5f};
      const float clamped_x = std::clamp(mid_pos.x, 0.0f, world_size);
      const float clamped_y = std::clamp(mid_pos.y, 0.0f, world_size);
      const uint32_t child = evolution.create_offspring(state, registry, idx, best_mate, {clamped_x, clamped_y});
      if (child != INVALID_ENTITY) {
        ++state.metrics.step_delta.births;
      }
      used[idx] = 1;
      used[best_mate] = 1;
    }
  }
}

} // namespace

void run(AppState &state, EvolutionManager &evolution, const SimulationConfig &config) {
  // Compact registries to remove dead entities
  state.predator.compact();
  state.prey.compact();

  // Respawn food
  state.food.respawn_step(config, state.runtime.step, state.runtime.rng.seed());

  // Reproduction for both species
  reproduction(state, evolution, state.predator, config);
  reproduction(state, evolution, state.prey, config);

  // Refresh species if interval reached
  if (config.species_update_interval_steps > 0 && (state.runtime.step % config.species_update_interval_steps) == 0) {
    evolution.refresh_species(state);
  }
}

} // namespace moonai::common