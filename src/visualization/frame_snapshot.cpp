#include "visualization/frame_snapshot.hpp"

#include "evolution/neural_network.hpp"
#include "simulation/components.hpp"
#include "visualization/constants.hpp"

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

} // namespace

void update_selected_activations(AppState &state) {
  state.ui.selected_node_activations.clear();

  const uint32_t selected =
      state.registry.find_by_agent_id(state.ui.selected_agent_id);
  if (selected == INVALID_ENTITY || !state.registry.valid(selected)) {
    return;
  }

  const auto *genome = genome_for(state, selected);
  if (!genome) {
    return;
  }

  const uint32_t idx = selected;
  const float *sensors = state.registry.sensors.input_ptr(idx);
  std::vector<float> sensor_values(sensors, sensors + SensorSoA::INPUT_COUNT);

  NeuralNetwork *network = state.evolution.network_cache.get_network(selected);
  if (!network) {
    return;
  }

  network->activate(sensor_values);
  for (const auto &[node_id, node_index] : network->node_index_map()) {
    if (node_index >= 0 &&
        node_index < static_cast<int>(network->last_activations().size())) {
      state.ui.selected_node_activations[node_id] =
          network->last_activations()[node_index];
    }
  }
}

FrameSnapshot build_frame_snapshot(const AppState &state,
                                   const AppConfig &config) {
  FrameSnapshot frame;
  frame.world_width = config.sim_config.grid_size;
  frame.world_height = config.sim_config.grid_size;

  float pred_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  const auto &positions = state.registry.positions;
  const auto &motion = state.registry.motion;
  const auto &identity = state.registry.identity;
  const auto &vitals = state.registry.vitals;
  const auto &stats = state.registry.stats;

  frame.foods.reserve(static_cast<std::size_t>(state.metrics.live.active_food));
  for (std::size_t i = 0; i < state.food_store.size(); ++i) {
    if (!state.food_store.active[i]) {
      continue;
    }
    frame.foods.push_back(RenderFood{
        Vec2{state.food_store.pos_x[i], state.food_store.pos_y[i]}});
  }

  const uint32_t entity_count = static_cast<uint32_t>(state.registry.size());
  for (uint32_t idx = 0; idx < entity_count; ++idx) {
    float energy_ratio = vitals.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);

    if (identity.type[idx] == IdentitySoA::TYPE_PREDATOR) {
      pred_dist[bucket] += 1.0f;
    } else {
      prey_dist[bucket] += 1.0f;
    }

    frame.agents.push_back(RenderAgent{
        idx, identity.entity_id[idx], Vec2{positions.x[idx], positions.y[idx]},
        Vec2{motion.vel_x[idx], motion.vel_y[idx]}, identity.type[idx]});
  }

  if (state.metrics.live.alive_predators > 0) {
    for (float &value : pred_dist) {
      value /= state.metrics.live.alive_predators;
    }
  }
  if (state.metrics.live.alive_prey > 0) {
    for (float &value : prey_dist) {
      value /= state.metrics.live.alive_prey;
    }
  }

  frame.overlay_stats.step = state.runtime.step;
  frame.overlay_stats.max_steps = config.sim_config.max_steps;
  frame.overlay_stats.alive_predators = state.metrics.live.alive_predators;
  frame.overlay_stats.alive_prey = state.metrics.live.alive_prey;
  frame.overlay_stats.active_food = state.metrics.live.active_food;
  frame.overlay_stats.num_species = state.metrics.live.num_species;
  frame.overlay_stats.speed_multiplier = state.ui.speed_multiplier;
  frame.overlay_stats.paused = state.ui.paused;
  frame.overlay_stats.experiment_name = config.experiment_name;
  frame.overlay_stats.total_kills = state.runtime.total_events.kills;
  frame.overlay_stats.total_food_eaten = state.runtime.total_events.food_eaten;
  frame.overlay_stats.total_births = state.runtime.total_events.births;
  frame.overlay_stats.total_deaths = state.runtime.total_events.deaths;
  for (int i = 0; i < 5; ++i) {
    frame.overlay_stats.predator_energy_dist[i] = pred_dist[i];
    frame.overlay_stats.prey_energy_dist[i] = prey_dist[i];
  }

  const uint32_t selected =
      state.registry.find_by_agent_id(state.ui.selected_agent_id);
  if (selected != INVALID_ENTITY && state.registry.valid(selected)) {
    const uint32_t idx = selected;
    const Genome *genome = genome_for(state, selected);
    if (genome) {
      frame.overlay_stats.selected_agent =
          static_cast<int>(identity.entity_id[idx]);
      frame.overlay_stats.selected_energy = vitals.energy[idx];
      frame.overlay_stats.selected_age = vitals.age[idx];
      frame.overlay_stats.selected_kills = stats.kills[idx];
      frame.overlay_stats.selected_food_eaten = stats.food_eaten[idx];
      frame.overlay_stats.selected_genome_complexity = static_cast<int>(
          genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = identity.entity_id[idx];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{positions.x[idx], positions.y[idx]};
      frame.selected_vision_range = config.sim_config.vision_range;
      frame.selected_node_activations = state.ui.selected_node_activations;

      const Vec2 selected_pos{positions.x[idx], positions.y[idx]};
      for (uint32_t other_idx = 0; other_idx < entity_count; ++other_idx) {
        if (other_idx == idx) {
          continue;
        }

        const Vec2 other_pos{positions.x[other_idx], positions.y[other_idx]};
        Vec2 diff = other_pos - selected_pos;
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }

        const sf::Color color =
            identity.type[other_idx] == IdentitySoA::TYPE_PREDATOR
                ? sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                            chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)
                : sf::Color(chart_colors::PREY_R, chart_colors::PREY_G,
                            chart_colors::PREY_B, visual::SENSOR_ALPHA);
        frame.sensor_lines.push_back(
            RenderLine{selected_pos, other_pos, color});
      }

      for (const auto &food : frame.foods) {
        const Vec2 diff =
            wrap_diff(food.position - selected_pos,
                      static_cast<float>(config.sim_config.grid_size),
                      static_cast<float>(config.sim_config.grid_size));
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }

        frame.sensor_lines.push_back(RenderLine{
            selected_pos, food.position,
            sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G,
                      chart_colors::FOOD_B, visual::FOOD_SENSOR_ALPHA)});
      }
    }
  }

  return frame;
}

} // namespace moonai
