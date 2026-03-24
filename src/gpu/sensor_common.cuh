#pragma once

#include <cuda_runtime.h>

namespace moonai::gpu {

struct SensorBuildView {
    const float* agent_pos_x;
    const float* agent_pos_y;
    const float* agent_vel_x;
    const float* agent_vel_y;
    const float* agent_speed;
    const float* agent_vision;
    const float* agent_energy;
    const unsigned int* agent_ids;
    const unsigned int* agent_types;
    const unsigned int* agent_alive;
    const float* food_pos_x;
    const float* food_pos_y;
    const unsigned int* food_active;
    const int* agent_cell_offsets;
    const unsigned int* agent_cell_ids;
    const int* food_cell_offsets;
    const unsigned int* food_cell_ids;
    float* inputs;
    int num_agents;
    int food_count;
    int agent_cols;
    int agent_rows;
    float agent_cell_size;
    int food_cols;
    int food_rows;
    float food_cell_size;
    int num_inputs;
    float world_width;
    float world_height;
    float max_energy;
    bool has_walls;
};

__device__ __forceinline__ float sensor_clampf(float value, float min_value, float max_value) {
    return fminf(fmaxf(value, min_value), max_value);
}

template<bool HasWalls>
__device__ __forceinline__ void sensor_apply_wrap(float& dx, float& dy, float world_width,
                                                  float world_height) {
    if constexpr (HasWalls) {
        (void)world_width;
        (void)world_height;
        return;
    }

    const float half_width = world_width * 0.5f;
    const float half_height = world_height * 0.5f;
    if (fabsf(dx) > half_width) {
        dx = dx > 0.0f ? dx - world_width : dx + world_width;
    }
    if (fabsf(dy) > half_height) {
        dy = dy > 0.0f ? dy - world_height : dy + world_height;
    }
}

__device__ __forceinline__ void sensor_apply_wrap(float& dx, float& dy, float world_width,
                                                  float world_height, bool has_walls) {
    if (has_walls) {
        sensor_apply_wrap<true>(dx, dy, world_width, world_height);
    } else {
        sensor_apply_wrap<false>(dx, dy, world_width, world_height);
    }
}

__device__ __forceinline__ int sensor_clamp_index(int value, int min_value, int max_value) {
    return max(min_value, min(value, max_value));
}

template<bool HasWalls>
__device__ __forceinline__ bool cell_may_intersect_radius(
    int cx,
    int cy,
    float cell_size,
    float origin_x,
    float origin_y,
    float radius,
    float world_width,
    float world_height) {
    const float center_x = (static_cast<float>(cx) + 0.5f) * cell_size;
    const float center_y = (static_cast<float>(cy) + 0.5f) * cell_size;
    float dx = center_x - origin_x;
    float dy = center_y - origin_y;
    sensor_apply_wrap<HasWalls>(dx, dy, world_width, world_height);

    const float half_size = cell_size * 0.5f;
    const float nearest_x = fmaxf(fabsf(dx) - half_size, 0.0f);
    const float nearest_y = fmaxf(fabsf(dy) - half_size, 0.0f);
    return nearest_x * nearest_x + nearest_y * nearest_y <= radius * radius;
}

template<bool HasWalls>
__device__ __forceinline__ void build_sensor_inputs_for_agent(const SensorBuildView& view,
                                                              int agent_idx) {
    float* out = view.inputs + static_cast<size_t>(agent_idx) * view.num_inputs;
    out[0] = -1.0f; out[1] = 0.0f;
    out[2] = -1.0f; out[3] = 0.0f;
    out[4] = -1.0f; out[5] = 0.0f;
    out[6] = 1.0f; out[7] = 0.0f; out[8] = 0.0f;
    out[9] = 0.0f; out[10] = 0.0f;
    out[11] = 1.0f; out[12] = 1.0f; out[13] = 1.0f; out[14] = 1.0f;

    if (view.agent_alive[agent_idx] == 0U) {
        return;
    }

    const float self_x = view.agent_pos_x[agent_idx];
    const float self_y = view.agent_pos_y[agent_idx];
    const float self_vision = view.agent_vision[agent_idx];
    const float vision_sq = self_vision * self_vision;
    const float inv_vision = self_vision > 0.0f ? 1.0f / self_vision : 0.0f;
    const int agent_cells_to_check = static_cast<int>(self_vision / view.agent_cell_size) + 1;
    const int food_cells_to_check = static_cast<int>(self_vision / view.food_cell_size) + 1;
    const int agent_cx = sensor_clamp_index(static_cast<int>(self_x / view.agent_cell_size), 0, view.agent_cols - 1);
    const int agent_cy = sensor_clamp_index(static_cast<int>(self_y / view.agent_cell_size), 0, view.agent_rows - 1);
    const int food_cx = sensor_clamp_index(static_cast<int>(self_x / view.food_cell_size), 0, view.food_cols - 1);
    const int food_cy = sensor_clamp_index(static_cast<int>(self_y / view.food_cell_size), 0, view.food_rows - 1);

    float nearest_pred_dist_sq = INFINITY;
    float nearest_prey_dist_sq = INFINITY;
    float nearest_food_dist_sq = INFINITY;
    float pred_dx = 0.0f;
    float pred_dy = 0.0f;
    float prey_dx = 0.0f;
    float prey_dy = 0.0f;
    float food_dx = 0.0f;
    float food_dy = 0.0f;
    int local_predators = 0;
    int local_prey = 0;

    for (int dy_cell = -agent_cells_to_check; dy_cell <= agent_cells_to_check; ++dy_cell) {
        for (int dx_cell = -agent_cells_to_check; dx_cell <= agent_cells_to_check; ++dx_cell) {
            const int nx = agent_cx + dx_cell;
            const int ny = agent_cy + dy_cell;
            if (nx < 0 || nx >= view.agent_cols || ny < 0 || ny >= view.agent_rows) {
                continue;
            }
            if (!cell_may_intersect_radius<HasWalls>(nx, ny, view.agent_cell_size, self_x, self_y,
                                                     self_vision, view.world_width, view.world_height)) {
                continue;
            }
            const int cell = ny * view.agent_cols + nx;
            const int start = view.agent_cell_offsets[cell];
            const int end = view.agent_cell_offsets[cell + 1];
            for (int slot = start; slot < end; ++slot) {
                const unsigned int other_idx = view.agent_cell_ids[slot];
                if (other_idx == static_cast<unsigned int>(agent_idx) || view.agent_alive[other_idx] == 0U) {
                    continue;
                }
                float dx = view.agent_pos_x[other_idx] - self_x;
                float dy = view.agent_pos_y[other_idx] - self_y;
                sensor_apply_wrap<HasWalls>(dx, dy, view.world_width, view.world_height);
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq > vision_sq) {
                    continue;
                }
                if (view.agent_types[other_idx] == 0U) {
                    ++local_predators;
                    if (dist_sq < nearest_pred_dist_sq) {
                        nearest_pred_dist_sq = dist_sq;
                        pred_dx = dx;
                        pred_dy = dy;
                    }
                } else {
                    ++local_prey;
                    if (dist_sq < nearest_prey_dist_sq) {
                        nearest_prey_dist_sq = dist_sq;
                        prey_dx = dx;
                        prey_dy = dy;
                    }
                }
            }
        }
    }

    for (int dy_cell = -food_cells_to_check; dy_cell <= food_cells_to_check; ++dy_cell) {
        for (int dx_cell = -food_cells_to_check; dx_cell <= food_cells_to_check; ++dx_cell) {
            const int nx = food_cx + dx_cell;
            const int ny = food_cy + dy_cell;
            if (nx < 0 || nx >= view.food_cols || ny < 0 || ny >= view.food_rows) {
                continue;
            }
            if (!cell_may_intersect_radius<HasWalls>(nx, ny, view.food_cell_size, self_x, self_y,
                                                     self_vision, view.world_width, view.world_height)) {
                continue;
            }
            const int cell = ny * view.food_cols + nx;
            const int start = view.food_cell_offsets[cell];
            const int end = view.food_cell_offsets[cell + 1];
            for (int slot = start; slot < end; ++slot) {
                const unsigned int food_idx = view.food_cell_ids[slot];
                if (food_idx >= static_cast<unsigned int>(view.food_count) || view.food_active[food_idx] == 0U) {
                    continue;
                }
                float dx = view.food_pos_x[food_idx] - self_x;
                float dy = view.food_pos_y[food_idx] - self_y;
                sensor_apply_wrap<HasWalls>(dx, dy, view.world_width, view.world_height);
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq > vision_sq) {
                    continue;
                }
                if (dist_sq < nearest_food_dist_sq) {
                    nearest_food_dist_sq = dist_sq;
                    food_dx = dx;
                    food_dy = dy;
                }
            }
        }
    }

    if (nearest_pred_dist_sq < INFINITY) {
        out[0] = sqrtf(nearest_pred_dist_sq) * inv_vision;
        out[1] = atan2f(pred_dy, pred_dx) * 0.31830988618f;
    }
    if (nearest_prey_dist_sq < INFINITY) {
        out[2] = sqrtf(nearest_prey_dist_sq) * inv_vision;
        out[3] = atan2f(prey_dy, prey_dx) * 0.31830988618f;
    }
    if (nearest_food_dist_sq < INFINITY) {
        out[4] = sqrtf(nearest_food_dist_sq) * inv_vision;
        out[5] = atan2f(food_dy, food_dx) * 0.31830988618f;
    }
    out[6] = sensor_clampf(view.agent_energy[agent_idx] / (view.max_energy * 2.0f), 0.0f, 1.0f);
    if (view.agent_speed[agent_idx] > 0.0f) {
        const float inv_speed = 1.0f / view.agent_speed[agent_idx];
        out[7] = sensor_clampf(view.agent_vel_x[agent_idx] * inv_speed, -1.0f, 1.0f);
        out[8] = sensor_clampf(view.agent_vel_y[agent_idx] * inv_speed, -1.0f, 1.0f);
    }
    out[9] = sensor_clampf(static_cast<float>(local_predators) * 0.1f, 0.0f, 1.0f);
    out[10] = sensor_clampf(static_cast<float>(local_prey) * 0.1f, 0.0f, 1.0f);
    if constexpr (HasWalls) {
        out[11] = sensor_clampf(self_x / self_vision, 0.0f, 1.0f);
        out[12] = sensor_clampf((view.world_width - self_x) / self_vision, 0.0f, 1.0f);
        out[13] = sensor_clampf(self_y / self_vision, 0.0f, 1.0f);
        out[14] = sensor_clampf((view.world_height - self_y) / self_vision, 0.0f, 1.0f);
    }
}

} // namespace moonai::gpu
