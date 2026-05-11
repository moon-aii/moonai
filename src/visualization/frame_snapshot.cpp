#include "visualization/frame_snapshot.hpp"
#include "evolution/genome.hpp"
#include "visualization/constants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace moonai {

namespace {

constexpr std::size_t kNearestSensorTargets = 5;

struct NearestSensorTarget {
  Vec2 position{};
  float dist_sq = std::numeric_limits<float>::infinity();
};

using NearestSensorTargetList = std::array<NearestSensorTarget, kNearestSensorTargets>;

void insert_nearest_target(NearestSensorTargetList &targets, Vec2 position, float dist_sq) {
  if (dist_sq >= targets.back().dist_sq) {
    return;
  }

  std::size_t insert_at = targets.size() - 1;
  while (insert_at > 0 && dist_sq < targets[insert_at - 1].dist_sq) {
    targets[insert_at] = targets[insert_at - 1];
    --insert_at;
  }

  targets[insert_at] = NearestSensorTarget{position, dist_sq};
}

void collect_nearest_agents(const AgentRegistry &registry, Vec2 selected_pos, float vision_sq, uint32_t excluded_entity,
                            NearestSensorTargetList &targets) {
  for (uint32_t idx = 0; idx < registry.size(); ++idx) {
    if (idx == excluded_entity) {
      continue;
    }

    const Vec2 other_pos{registry.pos_x[idx], registry.pos_y[idx]};
    const float dx = other_pos.x - selected_pos.x;
    const float dy = other_pos.y - selected_pos.y;
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq <= 0.0f || dist_sq > vision_sq) {
      continue;
    }

    insert_nearest_target(targets, other_pos, dist_sq);
  }
}

void collect_nearest_food(const Food &food, Vec2 selected_pos, float vision_sq, NearestSensorTargetList &targets) {
  for (std::size_t idx = 0; idx < food.size(); ++idx) {
    if (!food.active[idx]) {
      continue;
    }

    const Vec2 other_pos{food.pos_x[idx], food.pos_y[idx]};
    const float dx = other_pos.x - selected_pos.x;
    const float dy = other_pos.y - selected_pos.y;
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq <= 0.0f || dist_sq > vision_sq) {
      continue;
    }

    insert_nearest_target(targets, other_pos, dist_sq);
  }
}

void append_sensor_lines(std::vector<RenderLine> &lines, Vec2 selected_pos, const NearestSensorTargetList &targets,
                         sf::Color color, const std::array<std::uint8_t, kNearestSensorTargets> &alphas) {
  for (std::size_t idx = 0; idx < targets.size(); ++idx) {
    if (!std::isfinite(targets[idx].dist_sq)) {
      continue;
    }

    color.a = alphas[idx];
    lines.push_back(RenderLine{selected_pos, targets[idx].position, color});
  }
}

void collect_selected_sensor_lines(const AppState &state, Vec2 selected_pos, float vision_range,
                                   const AgentRegistry &same_species, uint32_t selected_index,
                                   sf::Color same_species_color, const AgentRegistry &other_species,
                                   sf::Color other_species_color, FrameSnapshot &frame) {
  const float vision_sq = vision_range * vision_range;
  NearestSensorTargetList same_species_targets{};
  NearestSensorTargetList other_species_targets{};
  NearestSensorTargetList food_targets{};

  collect_nearest_agents(same_species, selected_pos, vision_sq, selected_index, same_species_targets);
  collect_nearest_agents(other_species, selected_pos, vision_sq, INVALID_ENTITY, other_species_targets);
  collect_nearest_food(state.food, selected_pos, vision_sq, food_targets);

  append_sensor_lines(frame.sensor_lines, selected_pos, same_species_targets, same_species_color,
                      {140, 120, 100, 84, 68});
  append_sensor_lines(frame.sensor_lines, selected_pos, other_species_targets, other_species_color,
                      {140, 120, 100, 84, 68});
  append_sensor_lines(frame.sensor_lines, selected_pos, food_targets,
                      sf::Color(chart_colors::FOOD_R, chart_colors::FOOD_G, chart_colors::FOOD_B),
                      {120, 102, 84, 68, 54});
}

} // namespace

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

  frame.foods.reserve(static_cast<std::size_t>(state.metrics.active_food));
  for (std::size_t i = 0; i < state.food.size(); ++i) {
    if (!state.food.active[i]) {
      continue;
    }
    frame.foods.push_back(RenderFood{Vec2{state.food.pos_x[i], state.food.pos_y[i]}});
  }

  frame.predators.reserve(state.predator.size());
  for (uint32_t idx = 0; idx < state.predator.size(); ++idx) {
    float energy_ratio = state.predator.energy[idx] / config.sim_config.max_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    pred_dist[bucket] += 1.0f;

    frame.predators.push_back(RenderAgent{idx, state.predator.entity_id[idx],
                                          Vec2{state.predator.pos_x[idx], state.predator.pos_y[idx]},
                                          Vec2{state.predator.vel_x[idx], state.predator.vel_y[idx]}});
  }

  frame.prey.reserve(state.prey.size());
  for (uint32_t idx = 0; idx < state.prey.size(); ++idx) {
    float energy_ratio = state.prey.energy[idx] / config.sim_config.max_energy;
    energy_ratio = std::clamp(energy_ratio, 0.0f, 1.0f);
    const int bucket = std::min(static_cast<int>(energy_ratio * 5.0f), 4);
    prey_dist[bucket] += 1.0f;

    frame.prey.push_back(RenderAgent{idx, state.prey.entity_id[idx], Vec2{state.prey.pos_x[idx], state.prey.pos_y[idx]},
                                     Vec2{state.prey.vel_x[idx], state.prey.vel_y[idx]}});
  }

  if (state.metrics.predator_count > 0) {
    for (float &value : pred_dist) {
      value /= state.metrics.predator_count;
    }
  }
  if (state.metrics.prey_count > 0) {
    for (float &value : prey_dist) {
      value /= state.metrics.prey_count;
    }
  }

  frame.overlay_stats.step = state.metrics.step;
  frame.overlay_stats.max_steps = config.sim_config.max_steps;
  frame.overlay_stats.alive_predator = state.metrics.predator_count;
  frame.overlay_stats.alive_prey = state.metrics.prey_count;
  frame.overlay_stats.active_food = state.metrics.active_food;
  frame.overlay_stats.predator_species = state.metrics.predator_species;
  frame.overlay_stats.prey_species = state.metrics.prey_species;
  frame.overlay_stats.avg_predator_complexity = state.metrics.avg_predator_complexity;
  frame.overlay_stats.avg_prey_complexity = state.metrics.avg_prey_complexity;
  frame.overlay_stats.avg_predator_energy = state.metrics.avg_predator_energy;
  frame.overlay_stats.avg_prey_energy = state.metrics.avg_prey_energy;
  frame.overlay_stats.speed_multiplier = state.ui.speed_multiplier;
  frame.overlay_stats.paused = state.ui.paused;
  frame.overlay_stats.experiment_name = config.experiment_name;
  frame.overlay_stats.total_kills = state.metrics.kills;
  frame.overlay_stats.total_food_eaten = state.metrics.food_eaten;
  frame.overlay_stats.total_predator_births = state.metrics.predator_births;
  frame.overlay_stats.total_prey_births = state.metrics.prey_births;
  frame.overlay_stats.total_predator_deaths = state.metrics.predator_deaths;
  frame.overlay_stats.total_prey_deaths = state.metrics.prey_deaths;
  frame.overlay_stats.max_predator_generation = state.metrics.max_predator_generation;
  frame.overlay_stats.avg_predator_generation = state.metrics.avg_predator_generation;
  frame.overlay_stats.max_prey_generation = state.metrics.max_prey_generation;
  frame.overlay_stats.avg_prey_generation = state.metrics.avg_prey_generation;
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
      frame.overlay_stats.selected_generation = state.predator.generation[predator_selected];
      frame.overlay_stats.selected_genome_complexity =
          static_cast<int>(genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.predator.entity_id[predator_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.predator.pos_x[predator_selected], state.predator.pos_y[predator_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      collect_selected_sensor_lines(
          state, frame.selected_position, config.sim_config.vision_range, state.predator, predator_selected,
          sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B), state.prey,
          sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B), frame);
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
      frame.overlay_stats.selected_generation = state.prey.generation[prey_selected];
      frame.overlay_stats.selected_genome_complexity =
          static_cast<int>(genome->nodes().size() + genome->connections().size());
      frame.selected_genome = genome;
      frame.selected_agent_id = state.prey.entity_id[prey_selected];
      frame.has_selected_vision = true;
      frame.selected_position = Vec2{state.prey.pos_x[prey_selected], state.prey.pos_y[prey_selected]};
      frame.selected_vision_range = config.sim_config.vision_range;
      collect_selected_sensor_lines(
          state, frame.selected_position, config.sim_config.vision_range, state.prey, prey_selected,
          sf::Color(chart_colors::PREY_R, chart_colors::PREY_G, chart_colors::PREY_B), state.predator,
          sf::Color(chart_colors::PREDATOR_R, chart_colors::PREDATOR_G, chart_colors::PREDATOR_B), frame);
    }
  }

  return frame;
}

} // namespace moonai
