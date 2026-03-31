#include "visualization/frame_snapshot.hpp"

#include "visualization/constants.hpp"

#include <algorithm>
#include <cmath>

namespace moonai {

void update_selected_activations(AppState &state) {
  // Structure-only visualization: no activation computation needed
  // Activations map remains empty, overlay will use type-based colors
  state.ui.selected_node_activations.clear();
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
    frame.foods.push_back(
        RenderFood{Vec2{state.food_store.pos_x[i], state.food_store.pos_y[i]}});
  }

  frame.predators.reserve(state.predators.size());
  for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
    float energy_ratio =
        state.predators.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    pred_dist[bucket] += 1.0f;

    frame.predators.push_back(RenderAgent{
        idx, state.predators.entity_id[idx],
        Vec2{state.predators.pos_x[idx], state.predators.pos_y[idx]},
        Vec2{state.predators.vel_x[idx], state.predators.vel_y[idx]}});
  }

  frame.prey.reserve(state.prey.size());
  for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
    float energy_ratio =
        state.prey.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    prey_dist[bucket] += 1.0f;

    frame.prey.push_back(
        RenderAgent{idx, state.prey.entity_id[idx],
                    Vec2{state.prey.pos_x[idx], state.prey.pos_y[idx]},
                    Vec2{state.prey.vel_x[idx], state.prey.vel_y[idx]}});
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
          static_cast<int>(state.predators.entity_id[predator_selected]);
      frame.overlay_stats.selected_energy =
          state.predators.energy[predator_selected];
      frame.overlay_stats.selected_age = state.predators.age[predator_selected];
      frame.overlay_stats.selected_kills =
          state.predators.consumption[predator_selected];
      frame.overlay_stats.selected_food_eaten = 0;
      frame.overlay_stats.selected_genome_complexity = static_cast<int>(
          genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.predators.entity_id[predator_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.predators.pos_x[predator_selected],
                                     state.predators.pos_y[predator_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
        if (idx == predator_selected) {
          continue;
        }
        const Vec2 other_pos{state.predators.pos_x[idx],
                             state.predators.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x,
                        other_pos.y - selected_pos.y};
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
        const Vec2 other_pos{state.prey.pos_x[idx], state.prey.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x,
                        other_pos.y - selected_pos.y};
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
        const Vec2 diff{food.position.x - selected_pos.x,
                        food.position.y - selected_pos.y};
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
          static_cast<int>(state.prey.entity_id[prey_selected]);
      frame.overlay_stats.selected_energy = state.prey.energy[prey_selected];
      frame.overlay_stats.selected_age = state.prey.age[prey_selected];
      frame.overlay_stats.selected_kills = 0;
      frame.overlay_stats.selected_food_eaten =
          state.prey.consumption[prey_selected];
      frame.overlay_stats.selected_genome_complexity = static_cast<int>(
          genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.prey.entity_id[prey_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.prey.pos_x[prey_selected],
                                     state.prey.pos_y[prey_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;

      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predators.size(); ++idx) {
        const Vec2 other_pos{state.predators.pos_x[idx],
                             state.predators.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x,
                        other_pos.y - selected_pos.y};
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
        const Vec2 other_pos{state.prey.pos_x[idx], state.prey.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x,
                        other_pos.y - selected_pos.y};
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
        const Vec2 diff{food.position.x - selected_pos.x,
                        food.position.y - selected_pos.y};
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
