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

  if (state.ui.selected_agent_id == 0) {
    return;
  }

  auto populate_activations = [&](const auto &registry,
                                  const auto &network_cache,
                                  const Genome *genome, uint32_t selected) {
    if (selected == INVALID_ENTITY || !registry.valid(selected)) {
      return false;
    }
    if (!genome) {
      return false;
    }

    const float *sensors = registry.agents.input_ptr(selected);
    std::vector<float> sensor_values(sensors, sensors + AgentSoA::INPUT_COUNT);

    NeuralNetwork *network = network_cache.get_network(selected);
    if (!network) {
      return false;
    }

    network->activate(sensor_values);
    for (const auto &[node_id, node_index] : network->node_index_map()) {
      if (node_index >= 0 &&
          node_index < static_cast<int>(network->last_activations().size())) {
        state.ui.selected_node_activations[node_id] =
            network->last_activations()[node_index];
      }
    }
    return true;
  };

  const uint32_t predator_selected =
      state.predators.find_by_agent_id(state.ui.selected_agent_id);
  if (populate_activations(
          state.predators, state.evolution.predators.network_cache,
          predator_genome_for(state, predator_selected), predator_selected)) {
    return;
  }

  const uint32_t prey_selected =
      state.prey.find_by_agent_id(state.ui.selected_agent_id);
  populate_activations(state.prey, state.evolution.prey.network_cache,
                       prey_genome_for(state, prey_selected), prey_selected);
}

FrameSnapshot build_frame_snapshot(const AppState &state,
                                   const AppConfig &config) {
  FrameSnapshot frame;
  frame.world_width = config.sim_config.grid_size;
  frame.world_height = config.sim_config.grid_size;

  float pred_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  frame.foods.reserve(static_cast<std::size_t>(state.metrics.live.active_food));
  for (std::size_t i = 0; i < state.food_store.size(); ++i) {
    if (!state.food_store.active[i]) {
      continue;
    }
    frame.foods.push_back(RenderFood{Vec2{state.food_store.positions.x[i],
                                          state.food_store.positions.y[i]}});
  }

  frame.predators.reserve(state.predators.size());
  for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
    float energy_ratio =
        state.predators.agents.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    pred_dist[bucket] += 1.0f;

    frame.predators.push_back(
        RenderAgent{idx, state.predators.agents.entity_id[idx],
                    Vec2{state.predators.positions.x[idx],
                         state.predators.positions.y[idx]},
                    Vec2{state.predators.agents.vel_x[idx],
                         state.predators.agents.vel_y[idx]}});
  }

  frame.prey.reserve(state.prey.size());
  for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
    float energy_ratio =
        state.prey.agents.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    prey_dist[bucket] += 1.0f;

    frame.prey.push_back(RenderAgent{
        idx, state.prey.agents.entity_id[idx],
        Vec2{state.prey.positions.x[idx], state.prey.positions.y[idx]},
        Vec2{state.prey.agents.vel_x[idx], state.prey.agents.vel_y[idx]}});
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
  frame.overlay_stats.predator_species = state.metrics.live.predator_species;
  frame.overlay_stats.prey_species = state.metrics.live.prey_species;
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

  const uint32_t predator_selected =
      state.predators.find_by_agent_id(state.ui.selected_agent_id);
  if (predator_selected != INVALID_ENTITY &&
      state.predators.valid(predator_selected)) {
    const Genome *genome = predator_genome_for(state, predator_selected);
    if (genome) {
      frame.overlay_stats.selected_agent =
          static_cast<int>(state.predators.agents.entity_id[predator_selected]);
      frame.overlay_stats.selected_energy =
          state.predators.agents.energy[predator_selected];
      frame.overlay_stats.selected_age =
          state.predators.agents.age[predator_selected];
      frame.overlay_stats.selected_kills =
          state.predators.predator.kills[predator_selected];
      frame.overlay_stats.selected_food_eaten = 0;
      frame.overlay_stats.selected_genome_complexity = static_cast<int>(
          genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id =
          state.predators.agents.entity_id[predator_selected];
      frame.has_selected_vision = true;
      frame.selected_position =
          Vec2{state.predators.positions.x[predator_selected],
               state.predators.positions.y[predator_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      frame.selected_node_activations = state.ui.selected_node_activations;

      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
        if (idx == predator_selected) {
          continue;
        }
        const Vec2 other_pos{state.predators.positions.x[idx],
                             state.predators.positions.y[idx]};
        const Vec2 diff =
            wrap_diff(other_pos - selected_pos,
                      static_cast<float>(config.sim_config.grid_size),
                      static_cast<float>(config.sim_config.grid_size));
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos,
            Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                      chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)});
      }
      for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
        const Vec2 other_pos{state.prey.positions.x[idx],
                             state.prey.positions.y[idx]};
        const Vec2 diff =
            wrap_diff(other_pos - selected_pos,
                      static_cast<float>(config.sim_config.grid_size),
                      static_cast<float>(config.sim_config.grid_size));
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(
            RenderLine{selected_pos,
                       Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
                       sf::Color(chart_colors::PREY_R, chart_colors::PREY_G,
                                 chart_colors::PREY_B, visual::SENSOR_ALPHA)});
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

    return frame;
  }

  const uint32_t prey_selected =
      state.prey.find_by_agent_id(state.ui.selected_agent_id);
  if (prey_selected != INVALID_ENTITY && state.prey.valid(prey_selected)) {
    const Genome *genome = prey_genome_for(state, prey_selected);
    if (genome) {
      frame.overlay_stats.selected_agent =
          static_cast<int>(state.prey.agents.entity_id[prey_selected]);
      frame.overlay_stats.selected_energy =
          state.prey.agents.energy[prey_selected];
      frame.overlay_stats.selected_age = state.prey.agents.age[prey_selected];
      frame.overlay_stats.selected_kills = 0;
      frame.overlay_stats.selected_food_eaten =
          state.prey.prey.food_eaten[prey_selected];
      frame.overlay_stats.selected_genome_complexity = static_cast<int>(
          genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.prey.agents.entity_id[prey_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.prey.positions.x[prey_selected],
                                     state.prey.positions.y[prey_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      frame.selected_node_activations = state.ui.selected_node_activations;

      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
        const Vec2 other_pos{state.predators.positions.x[idx],
                             state.predators.positions.y[idx]};
        const Vec2 diff =
            wrap_diff(other_pos - selected_pos,
                      static_cast<float>(config.sim_config.grid_size),
                      static_cast<float>(config.sim_config.grid_size));
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos,
            Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
            sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                      chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)});
      }
      for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
        if (idx == prey_selected) {
          continue;
        }
        const Vec2 other_pos{state.prey.positions.x[idx],
                             state.prey.positions.y[idx]};
        const Vec2 diff =
            wrap_diff(other_pos - selected_pos,
                      static_cast<float>(config.sim_config.grid_size),
                      static_cast<float>(config.sim_config.grid_size));
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(
            RenderLine{selected_pos,
                       Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
                       sf::Color(chart_colors::PREY_R, chart_colors::PREY_G,
                                 chart_colors::PREY_B, visual::SENSOR_ALPHA)});
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
