#include "simulation/simulation_manager.hpp"

#include "core/app_state.hpp"
#include "core/profiler_macros.hpp"
#include "evolution/evolution_manager.hpp"
#include "simulation/simulation_step_systems.hpp"

#include <algorithm>
#include <cmath>

namespace moonai {

namespace {

Vec2 wrap_diff(Vec2 diff, float world_width, float world_height) {
  if (std::abs(diff.x) > world_width * 0.5f) {
    diff.x = diff.x > 0.0f ? diff.x - world_width : diff.x + world_width;
  }
  if (std::abs(diff.y) > world_height * 0.5f) {
    diff.y = diff.y > 0.0f ? diff.y - world_height : diff.y + world_height;
  }
  return diff;
}

float wrap_coord(float value, float limit) {
  while (value < 0.0f) {
    value += limit;
  }
  while (value >= limit) {
    value -= limit;
  }
  return value;
}

class DenseReproductionGrid {
public:
  DenseReproductionGrid(float world_width, float world_height, float cell_size,
                        std::size_t entity_count)
      : cell_size_(std::max(cell_size, 1.0f)),
        cols_(
            std::max(1, static_cast<int>(std::ceil(world_width / cell_size_)))),
        rows_(std::max(1,
                       static_cast<int>(std::ceil(world_height / cell_size_)))),
        counts_(static_cast<std::size_t>(cols_ * rows_), 0),
        offsets_(static_cast<std::size_t>(cols_ * rows_) + 1, 0),
        write_offsets_(static_cast<std::size_t>(cols_ * rows_), 0),
        entries_(entity_count, INVALID_ENTITY) {}

  void build(const PositionSoA &positions, std::size_t entity_count) {
    std::fill(counts_.begin(), counts_.end(), 0);
    std::fill(offsets_.begin(), offsets_.end(), 0);

    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(positions.x[idx], positions.y[idx]);
      counts_[static_cast<std::size_t>(cell)] += 1;
    }

    for (std::size_t cell = 0; cell < counts_.size(); ++cell) {
      offsets_[cell + 1] = offsets_[cell] + counts_[cell];
    }

    std::copy(offsets_.begin(), offsets_.end() - 1, write_offsets_.begin());
    for (uint32_t idx = 0; idx < entity_count; ++idx) {
      const int cell = cell_index(positions.x[idx], positions.y[idx]);
      const std::size_t slot = static_cast<std::size_t>(
          write_offsets_[static_cast<std::size_t>(cell)]++);
      entries_[slot] = idx;
    }
  }

  template <typename Callback>
  void for_each_candidate(Vec2 center, float radius,
                          Callback &&callback) const {
    const int cells_to_check =
        std::max(1, static_cast<int>(std::ceil(radius / cell_size_)));
    const int base_x = cell_coord(center.x, cols_);
    const int base_y = cell_coord(center.y, rows_);

    for (int dy = -cells_to_check; dy <= cells_to_check; ++dy) {
      for (int dx = -cells_to_check; dx <= cells_to_check; ++dx) {
        const int cell = flat_index(wrap_cell(base_x + dx, cols_),
                                    wrap_cell(base_y + dy, rows_));
        for (int slot = offsets_[static_cast<std::size_t>(cell)];
             slot < offsets_[static_cast<std::size_t>(cell) + 1]; ++slot) {
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

  int wrap_cell(int coord, int limit) const {
    coord %= limit;
    if (coord < 0) {
      coord += limit;
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

void SimulationManager::compact_registry(AppState &state,
                                         EvolutionManager &evolution) {
  const auto result = state.registry.compact_dead();
  for (const auto &[from, to] : result.moved) {
    evolution.on_entity_moved(state, from, to);
  }
  for (uint32_t removed : result.removed) {
    evolution.on_entity_destroyed(state, removed);
  }
}

void SimulationManager::refresh_world_state_after_step(AppState &state) {
  MOONAI_PROFILE_SCOPE("food_respawn");
  state.food_store.respawn_step(config_, state.runtime.step,
                                state.runtime.rng.seed());
}

void SimulationManager::step(AppState &state, EvolutionManager &evolution) {
  MOONAI_PROFILE_SCOPE("simulation_step_cpu");

  state.runtime.last_step_events.clear();
  state.runtime.pending_offspring.clear();
  state.runtime.step_events.clear();

  const std::vector<uint8_t> was_alive = state.registry.vitals.alive;
  const std::vector<uint8_t> was_food_active = state.food_store.active;
  std::vector<int> food_consumed_by(state.food_store.size(), -1);
  std::vector<int> killed_by(state.registry.size(), -1);
  std::vector<uint32_t> kill_counts(state.registry.size(), 0U);

  simulation_detail::build_sensors(state.registry, state.food_store, config_);
  evolution.compute_actions(state);
  simulation_detail::update_vitals(state.registry, config_);
  simulation_detail::process_food(state.registry, state.food_store, config_,
                                  food_consumed_by);
  simulation_detail::process_combat(state.registry, config_, killed_by,
                                    kill_counts);
  simulation_detail::apply_movement(state.registry, config_);

  simulation_detail::collect_cpu_step_events(
      state.registry, state.food_store, was_alive, was_food_active,
      food_consumed_by, killed_by, kill_counts, state.runtime.last_step_events);
  compact_registry(state, evolution);
  refresh_world_state_after_step(state);
  state.runtime.pending_offspring = find_reproduction_pairs(state);
  simulation_detail::accumulate_events(state.runtime.step_events,
                                       state.runtime.last_step_events);
}

std::vector<PendingOffspring>
SimulationManager::find_reproduction_pairs(const AppState &state) const {
  MOONAI_PROFILE_SCOPE("find_reproduction_pairs");
  std::vector<PendingOffspring> pairs;
  std::vector<uint8_t> used(state.registry.size(), 0);

  const auto &positions = state.registry.positions;
  const auto &vitals = state.registry.vitals;
  const auto &identity = state.registry.identity;
  DenseReproductionGrid grid(static_cast<float>(config_.grid_size),
                             static_cast<float>(config_.grid_size),
                             config_.mate_range, state.registry.size());
  grid.build(positions, state.registry.size());
  const float world_size = static_cast<float>(config_.grid_size);

  const uint32_t entity_count = static_cast<uint32_t>(state.registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    const uint32_t entity = idx;
    if (vitals.energy[idx] < config_.reproduction_energy_threshold ||
        used[idx] != 0) {
      continue;
    }

    const Vec2 pos{positions.x[idx], positions.y[idx]};
    uint32_t best_mate = INVALID_ENTITY;
    float best_dist_sq = config_.mate_range * config_.mate_range;

    grid.for_each_candidate(pos, config_.mate_range, [&](uint32_t mate_id) {
      if (mate_id == entity || used[static_cast<std::size_t>(mate_id)] != 0) {
        return;
      }

      const uint32_t mate_idx = mate_id;
      if (identity.type[mate_idx] != identity.type[idx] ||
          vitals.energy[mate_idx] < config_.reproduction_energy_threshold) {
        return;
      }

      const Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      const Vec2 diff = wrap_diff(mate_pos - pos, world_size, world_size);
      const float dist_sq = diff.x * diff.x + diff.y * diff.y;
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_mate = mate_id;
      }
    });

    if (best_mate != INVALID_ENTITY) {
      const uint32_t mate_idx = best_mate;
      const Vec2 mate_pos{positions.x[mate_idx], positions.y[mate_idx]};
      const Vec2 diff = wrap_diff(mate_pos - pos, world_size, world_size);

      PendingOffspring pair;
      pair.parent_a = entity;
      pair.parent_b = best_mate;
      pair.spawn_position = {wrap_coord(pos.x + diff.x * 0.5f, world_size),
                             wrap_coord(pos.y + diff.y * 0.5f, world_size)};
      pairs.push_back(pair);
      used[idx] = 1;
      used[mate_idx] = 1;
    }
  }

  return pairs;
}

} // namespace moonai
