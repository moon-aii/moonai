#include "visualization/frame_snapshot.hpp"
#include "evolution/genome.hpp"
#include "visualization/constants.hpp"

#include <algorithm>

namespace moonai {

static inline const Genome *predator_genome_for(const AppState &state, uint32_t entity) {
  if (entity == INVALID_ENTITY || entity >= state.predator.genomes.size()) {
    return nullptr;
  }
  return &state.predator.genomes[entity];
}

static inline const Genome *prey_genome_for(const AppState &state, uint32_t entity) {
  if (entity == INVALID_ENTITY || entity >= state.prey.genomes.size()) {
    return nullptr;
  }
  return &state.prey.genomes[entity];
}

FrameSnapshot build_frame_snapshot(const AppState &state, const AppConfig &config) {
  FrameSnapshot frame;
  frame.world_width = config.sim_config.grid_size;
  frame.world_height = config.sim_config.grid_size;

  float pred_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float prey_dist[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  frame.foods.reserve(static_cast<std::size_t>(state.metrics.live.active_food));
  for (std::size_t i = 0; i < state.food.size(); ++i) {
    if (!state.food.active[i]) {
      continue;
    }
    frame.foods.push_back(RenderFood{Vec2{state.food.pos_x[i], state.food.pos_y[i]}});
  }

  frame.predators.reserve(state.predator.size());
  for (uint32_t idx = 0; idx < state.predator.size(); ++idx) {
    float energy_ratio = state.predator.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    pred_dist[bucket] += 1.0f;

    frame.predators.push_back(RenderAgent{idx, state.predator.entity_id[idx],
                                          Vec2{state.predator.pos_x[idx], state.predator.pos_y[idx]},
                                          Vec2{state.predator.vel_x[idx], state.predator.vel_y[idx]}});
  }

  frame.prey.reserve(state.prey.size());
  for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
    float energy_ratio = state.prey.energy[idx] / config.sim_config.initial_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    prey_dist[bucket] += 1.0f;

    frame.prey.push_back(RenderAgent{idx, state.prey.entity_id[idx], Vec2{state.prey.pos_x[idx], state.prey.pos_y[idx]},
                                     Vec2{state.prey.vel_x[idx], state.prey.vel_y[idx]}});
  }

  if (state.metrics.live.predator_count > 0) {
    for (float &value : pred_dist) {
      value /= state.metrics.live.predator_count;
    }
  }
  if (state.metrics.live.prey_count > 0) {
    for (float &value : prey_dist) {
      value /= state.metrics.live.prey_count;
    }
  }

  frame.overlay_stats.step = state.metrics.live.step;
  frame.overlay_stats.max_steps = config.sim_config.max_steps;
  frame.overlay_stats.alive_predator = state.metrics.live.predator_count;
  frame.overlay_stats.alive_prey = state.metrics.live.prey_count;
  frame.overlay_stats.active_food = state.metrics.live.active_food;
  frame.overlay_stats.predator_species = state.metrics.live.predator_species;
  frame.overlay_stats.prey_species = state.metrics.live.prey_species;
  frame.overlay_stats.speed_multiplier = state.ui.speed_multiplier;
  frame.overlay_stats.paused = state.ui.paused;
  frame.overlay_stats.experiment_name = config.experiment_name;
  frame.overlay_stats.total_kills = state.metrics.totals.kills;
  frame.overlay_stats.total_food_eaten = state.metrics.totals.food_eaten;
  frame.overlay_stats.total_births = state.metrics.totals.births;
  frame.overlay_stats.total_deaths = state.metrics.totals.deaths;
  for (int i = 0; i < 5; ++i) {
    frame.overlay_stats.predator_energy_dist[i] = pred_dist[i];
    frame.overlay_stats.prey_energy_dist[i] = prey_dist[i];
  }

  const uint32_t predator_selected = state.predator.find_by_agent_id(state.ui.selected_agent_id);
  if (predator_selected != INVALID_ENTITY && state.predator.valid(predator_selected)) {
    const Genome *genome = predator_genome_for(state, predator_selected);
    if (genome) {
      frame.overlay_stats.selected_agent = static_cast<int>(state.predator.entity_id[predator_selected]);
      frame.overlay_stats.selected_energy = state.predator.energy[predator_selected];
      frame.overlay_stats.selected_age = state.predator.age[predator_selected];
      frame.overlay_stats.selected_kills = state.predator.consumption[predator_selected];
      frame.overlay_stats.selected_food_eaten = 0;
      frame.overlay_stats.selected_genome_complexity =
          static_cast<int>(genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.predator.entity_id[predator_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.predator.pos_x[predator_selected], state.predator.pos_y[predator_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predator.size(); ++idx) {
        if (idx == predator_selected) {
          continue;
        }
        const Vec2 other_pos{state.predator.pos_x[idx], state.predator.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x, other_pos.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{selected_pos, Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
                                                sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                                                          chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)});
      }
      for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
        const Vec2 other_pos{state.prey.pos_x[idx], state.prey.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x, other_pos.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos, Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B, visual::SENSOR_ALPHA)});
      }
      for (const auto &food : frame.foods) {
        const Vec2 diff{food.position.x - selected_pos.x, food.position.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos, food.position,
            sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B, visual::FOOD_SENSOR_ALPHA)});
      }
    }

    return frame;
  }

  const uint32_t prey_selected = state.prey.find_by_agent_id(state.ui.selected_agent_id);
  if (prey_selected != INVALID_ENTITY && state.prey.valid(prey_selected)) {
    const Genome *genome = prey_genome_for(state, prey_selected);
    if (genome) {
      frame.overlay_stats.selected_agent = static_cast<int>(state.prey.entity_id[prey_selected]);
      frame.overlay_stats.selected_energy = state.prey.energy[prey_selected];
      frame.overlay_stats.selected_age = state.prey.age[prey_selected];
      frame.overlay_stats.selected_kills = 0;
      frame.overlay_stats.selected_food_eaten = state.prey.consumption[prey_selected];
      frame.overlay_stats.selected_genome_complexity =
          static_cast<int>(genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.prey.entity_id[prey_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.prey.pos_x[prey_selected], state.prey.pos_y[prey_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;

      const Vec2 selected_pos = frame.selected_position;
      for (uint32_t idx = 0; idx < state.predator.size(); ++idx) {
        const Vec2 other_pos{state.predator.pos_x[idx], state.predator.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x, other_pos.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{selected_pos, Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
                                                sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G,
                                                          chart_colors::PREDATOR_B, visual::SENSOR_ALPHA)});
      }
      for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
        if (idx == prey_selected) {
          continue;
        }
        const Vec2 other_pos{state.prey.pos_x[idx], state.prey.pos_y[idx]};
        const Vec2 diff{other_pos.x - selected_pos.x, other_pos.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos, Vec2{selected_pos.x + diff.x, selected_pos.y + diff.y},
            sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B, visual::SENSOR_ALPHA)});
      }
      for (const auto &food : frame.foods) {
        const Vec2 diff{food.position.x - selected_pos.x, food.position.y - selected_pos.y};
        if (diff.length() > config.sim_config.vision_range) {
          continue;
        }
        frame.sensor_lines.push_back(RenderLine{
            selected_pos, food.position,
            sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B, visual::FOOD_SENSOR_ALPHA)});
      }
    }
  }

  return frame;
}

} // namespace moonai
