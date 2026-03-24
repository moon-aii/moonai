#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

#include "core/deterministic_respawn.hpp"

#include <cmath>

namespace moonai::gpu {

namespace {

__device__ __forceinline__ float clampf(float value, float min_value, float max_value) {
    return fminf(fmaxf(value, min_value), max_value);
}

__device__ __forceinline__ float length2(float x, float y) {
    return sqrtf(x * x + y * y);
}

__device__ __forceinline__ void normalize_inplace(float& x, float& y) {
    const float len = length2(x, y);
    if (len < 1e-6f) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    x /= len;
    y /= len;
}

__device__ __forceinline__ float angle_norm(float dx, float dy) {
    return atan2f(dy, dx) / 3.14159265f;
}

__device__ __forceinline__ void apply_wrap(float& dx, float& dy, float world_width,
                                           float world_height, bool has_walls) {
    if (has_walls) {
        return;
    }
    if (fabsf(dx) > world_width * 0.5f) {
        dx = dx > 0.0f ? dx - world_width : dx + world_width;
    }
    if (fabsf(dy) > world_height * 0.5f) {
        dy = dy > 0.0f ? dy - world_height : dy + world_height;
    }
}

__device__ __forceinline__ int clamp_index(int value, int min_value, int max_value) {
    return max(min_value, min(value, max_value));
}

__global__ void bin_agents_kernel(const GpuAgentState* agents, int num_agents,
                                  int cols, int rows, float cell_size,
                                  int* cell_counts, unsigned int* cell_ids,
                                  int bin_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) {
        return;
    }
    const auto& agent = agents[idx];
    if (!agent.alive) {
        return;
    }
    const int cx = clamp_index(static_cast<int>(agent.pos_x / cell_size), 0, cols - 1);
    const int cy = clamp_index(static_cast<int>(agent.pos_y / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    const int slot = atomicAdd(&cell_counts[cell], 1);
    if (slot < bin_stride) {
        cell_ids[cell * bin_stride + slot] = static_cast<unsigned int>(idx);
    }
}

__global__ void bin_food_kernel(const GpuFoodState* food, int food_count,
                                int cols, int rows, float cell_size,
                                int* cell_counts, unsigned int* cell_ids,
                                int bin_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count) {
        return;
    }
    const auto& item = food[idx];
    if (!item.active) {
        return;
    }
    const int cx = clamp_index(static_cast<int>(item.pos_x / cell_size), 0, cols - 1);
    const int cy = clamp_index(static_cast<int>(item.pos_y / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    const int slot = atomicAdd(&cell_counts[cell], 1);
    if (slot < bin_stride) {
        cell_ids[cell * bin_stride + slot] = static_cast<unsigned int>(idx);
    }
}

__global__ void sensor_build_resident_kernel(const GpuAgentState* agents,
                                             const GpuFoodState* food,
                                             const int* agent_cell_counts,
                                             const unsigned int* agent_cell_ids,
                                             const int* food_cell_counts,
                                             const unsigned int* food_cell_ids,
                                             float* inputs,
                                             int num_agents,
                                             int food_count,
                                             int agent_cols,
                                             int agent_rows,
                                             float agent_cell_size,
                                             int food_cols,
                                             int food_rows,
                                             float food_cell_size,
                                             int agent_bin_stride,
                                             int food_bin_stride,
                                             int num_inputs,
                                             float world_width,
                                             float world_height,
                                             float max_energy,
                                             bool has_walls) {
    const int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) {
        return;
    }

    float* out = inputs + static_cast<size_t>(agent_idx) * num_inputs;
    out[0] = -1.0f; out[1] = 0.0f;
    out[2] = -1.0f; out[3] = 0.0f;
    out[4] = -1.0f; out[5] = 0.0f;
    out[6] = 1.0f; out[7] = 0.0f; out[8] = 0.0f;
    out[9] = 0.0f; out[10] = 0.0f;
    out[11] = 1.0f; out[12] = 1.0f; out[13] = 1.0f; out[14] = 1.0f;

    const GpuAgentState self = agents[agent_idx];
    if (!self.alive) {
        return;
    }

    const float vision = self.vision_range;
    const float vision_sq = vision * vision;
    float nearest_pred_dist_sq = INFINITY;
    float nearest_prey_dist_sq = INFINITY;
    float nearest_food_dist_sq = INFINITY;
    float pred_dx = 0.0f, pred_dy = 0.0f;
    float prey_dx = 0.0f, prey_dy = 0.0f;
    float food_dx = 0.0f, food_dy = 0.0f;
    int local_predators = 0;
    int local_prey = 0;

    const int agent_cells_to_check = static_cast<int>(ceilf(vision / agent_cell_size));
    const int agent_cx = clamp_index(static_cast<int>(self.pos_x / agent_cell_size), 0, agent_cols - 1);
    const int agent_cy = clamp_index(static_cast<int>(self.pos_y / agent_cell_size), 0, agent_rows - 1);
    for (int dy_cell = -agent_cells_to_check; dy_cell <= agent_cells_to_check; ++dy_cell) {
        for (int dx_cell = -agent_cells_to_check; dx_cell <= agent_cells_to_check; ++dx_cell) {
            const int nx = agent_cx + dx_cell;
            const int ny = agent_cy + dy_cell;
            if (nx < 0 || nx >= agent_cols || ny < 0 || ny >= agent_rows) {
                continue;
            }
            const int cell = ny * agent_cols + nx;
            const int count = min(agent_cell_counts[cell], agent_bin_stride);
            for (int slot = 0; slot < count; ++slot) {
                const unsigned int other_idx = agent_cell_ids[cell * agent_bin_stride + slot];
                const GpuAgentState other = agents[other_idx];
                if (!other.alive || other.id == self.id) {
                    continue;
                }
                float dx = other.pos_x - self.pos_x;
                float dy = other.pos_y - self.pos_y;
                apply_wrap(dx, dy, world_width, world_height, has_walls);
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq > vision_sq) {
                    continue;
                }
                if (other.type == 0U) {
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

    const int food_cells_to_check = static_cast<int>(ceilf(vision / food_cell_size));
    const int food_cx = clamp_index(static_cast<int>(self.pos_x / food_cell_size), 0, food_cols - 1);
    const int food_cy = clamp_index(static_cast<int>(self.pos_y / food_cell_size), 0, food_rows - 1);
    for (int dy_cell = -food_cells_to_check; dy_cell <= food_cells_to_check; ++dy_cell) {
        for (int dx_cell = -food_cells_to_check; dx_cell <= food_cells_to_check; ++dx_cell) {
            const int nx = food_cx + dx_cell;
            const int ny = food_cy + dy_cell;
            if (nx < 0 || nx >= food_cols || ny < 0 || ny >= food_rows) {
                continue;
            }
            const int cell = ny * food_cols + nx;
            const int count = min(food_cell_counts[cell], food_bin_stride);
            for (int slot = 0; slot < count; ++slot) {
                const unsigned int food_idx = food_cell_ids[cell * food_bin_stride + slot];
                if (food_idx >= static_cast<unsigned int>(food_count) || !food[food_idx].active) {
                    continue;
                }
                float dx = food[food_idx].pos_x - self.pos_x;
                float dy = food[food_idx].pos_y - self.pos_y;
                apply_wrap(dx, dy, world_width, world_height, has_walls);
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
        out[0] = sqrtf(nearest_pred_dist_sq) / vision;
        out[1] = angle_norm(pred_dx, pred_dy);
    }
    if (nearest_prey_dist_sq < INFINITY) {
        out[2] = sqrtf(nearest_prey_dist_sq) / vision;
        out[3] = angle_norm(prey_dx, prey_dy);
    }
    if (nearest_food_dist_sq < INFINITY) {
        out[4] = sqrtf(nearest_food_dist_sq) / vision;
        out[5] = angle_norm(food_dx, food_dy);
    }
    out[6] = clampf(self.energy / (max_energy * 2.0f), 0.0f, 1.0f);
    if (self.speed > 0.0f) {
        out[7] = clampf(self.vel_x / self.speed, -1.0f, 1.0f);
        out[8] = clampf(self.vel_y / self.speed, -1.0f, 1.0f);
    }
    out[9] = clampf(static_cast<float>(local_predators) / 10.0f, 0.0f, 1.0f);
    out[10] = clampf(static_cast<float>(local_prey) / 10.0f, 0.0f, 1.0f);
    if (has_walls) {
        out[11] = clampf(self.pos_x / vision, 0.0f, 1.0f);
        out[12] = clampf((world_width - self.pos_x) / vision, 0.0f, 1.0f);
        out[13] = clampf(self.pos_y / vision, 0.0f, 1.0f);
        out[14] = clampf((world_height - self.pos_y) / vision, 0.0f, 1.0f);
    }
}

__global__ void movement_kernel(GpuAgentState* agents, const float* outputs, int num_agents,
                                int num_outputs, float dt, float world_width,
                                float world_height, bool has_walls,
                                float energy_drain_per_tick, int target_fps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) {
        return;
    }
    auto& agent = agents[idx];
    if (!agent.alive) {
        return;
    }
    float dir_x = 0.0f;
    float dir_y = 0.0f;
    if (num_outputs >= 2) {
        dir_x = outputs[idx * num_outputs + 0] * 2.0f - 1.0f;
        dir_y = outputs[idx * num_outputs + 1] * 2.0f - 1.0f;
    }
    normalize_inplace(dir_x, dir_y);
    agent.vel_x = dir_x * agent.speed;
    agent.vel_y = dir_y * agent.speed;
    agent.pos_x += agent.vel_x * dt;
    agent.pos_y += agent.vel_y * dt;
    agent.distance_traveled += agent.speed * dt;
    agent.age += 1;
    agent.energy -= energy_drain_per_tick * dt * static_cast<float>(target_fps);

    if (has_walls) {
        agent.pos_x = clampf(agent.pos_x, 0.0f, world_width);
        agent.pos_y = clampf(agent.pos_y, 0.0f, world_height);
    } else {
        if (agent.pos_x < 0.0f) agent.pos_x += world_width;
        if (agent.pos_x >= world_width) agent.pos_x -= world_width;
        if (agent.pos_y < 0.0f) agent.pos_y += world_height;
        if (agent.pos_y >= world_height) agent.pos_y -= world_height;
    }

    if (agent.energy <= 0.0f) {
        agent.alive = 0U;
    }
}

__global__ void prey_food_kernel(GpuAgentState* agents, GpuFoodState* food, int num_agents,
                                 const int* food_cell_counts, const unsigned int* food_cell_ids,
                                 int food_cols, int food_rows, float food_cell_size,
                                 int food_bin_stride, int food_count,
                                 float food_pickup_range, float energy_gain_from_food,
                                 float world_width, float world_height, bool has_walls) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) {
        return;
    }
    auto& agent = agents[idx];
    if (!agent.alive || agent.type != 1U) {
        return;
    }

    const float range_sq = food_pickup_range * food_pickup_range;
    const int cells_to_check = static_cast<int>(ceilf(food_pickup_range / food_cell_size));
    const int cx = clamp_index(static_cast<int>(agent.pos_x / food_cell_size), 0, food_cols - 1);
    const int cy = clamp_index(static_cast<int>(agent.pos_y / food_cell_size), 0, food_rows - 1);
    for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
        for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
            const int nx = cx + dx_cell;
            const int ny = cy + dy_cell;
            if (nx < 0 || nx >= food_cols || ny < 0 || ny >= food_rows) {
                continue;
            }
            const int cell = ny * food_cols + nx;
            const int count = min(food_cell_counts[cell], food_bin_stride);
            for (int slot = 0; slot < count; ++slot) {
                const unsigned int food_idx = food_cell_ids[cell * food_bin_stride + slot];
                if (food_idx >= static_cast<unsigned int>(food_count) || !food[food_idx].active) {
                    continue;
                }
                float dx = food[food_idx].pos_x - agent.pos_x;
                float dy = food[food_idx].pos_y - agent.pos_y;
                apply_wrap(dx, dy, world_width, world_height, has_walls);
                if (dx * dx + dy * dy > range_sq) {
                    continue;
                }
                if (atomicCAS(&food[food_idx].active, 1U, 0U) == 1U) {
                    agent.energy += energy_gain_from_food;
                    agent.food_eaten += 1;
                    return;
                }
            }
        }
    }
}

__global__ void predator_attack_kernel(GpuAgentState* agents, int num_agents,
                                       const int* agent_cell_counts,
                                       const unsigned int* agent_cell_ids,
                                       int agent_cols, int agent_rows, float agent_cell_size,
                                       int agent_bin_stride, float attack_range,
                                       float energy_gain_from_kill, float world_width,
                                       float world_height, bool has_walls) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents) {
        return;
    }
    auto& predator = agents[idx];
    if (!predator.alive || predator.type != 0U) {
        return;
    }

    const float range_sq = attack_range * attack_range;
    const int cells_to_check = static_cast<int>(ceilf(attack_range / agent_cell_size));
    const int cx = clamp_index(static_cast<int>(predator.pos_x / agent_cell_size), 0, agent_cols - 1);
    const int cy = clamp_index(static_cast<int>(predator.pos_y / agent_cell_size), 0, agent_rows - 1);
    for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
        for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
            const int nx = cx + dx_cell;
            const int ny = cy + dy_cell;
            if (nx < 0 || nx >= agent_cols || ny < 0 || ny >= agent_rows) {
                continue;
            }
            const int cell = ny * agent_cols + nx;
            const int count = min(agent_cell_counts[cell], agent_bin_stride);
            for (int slot = 0; slot < count; ++slot) {
                const unsigned int prey_idx = agent_cell_ids[cell * agent_bin_stride + slot];
                if (prey_idx == static_cast<unsigned int>(idx)) {
                    continue;
                }
                auto& prey = agents[prey_idx];
                if (!prey.alive || prey.type != 1U) {
                    continue;
                }
                float dx = prey.pos_x - predator.pos_x;
                float dy = prey.pos_y - predator.pos_y;
                apply_wrap(dx, dy, world_width, world_height, has_walls);
                if (dx * dx + dy * dy > range_sq) {
                    continue;
                }
                if (atomicCAS(&prey.alive, 1U, 0U) == 1U) {
                    predator.kills += 1;
                    predator.energy += energy_gain_from_kill;
                    return;
                }
            }
        }
    }
}

__global__ void respawn_food_kernel(GpuFoodState* food, int food_count, float respawn_rate,
                                    float world_width, float world_height,
                                    std::uint64_t seed, int tick_index) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count) {
        return;
    }
    auto& item = food[idx];
    if (item.active) {
        return;
    }
    if (!respawn::should_respawn(seed, tick_index, static_cast<std::uint32_t>(idx), respawn_rate)) {
        return;
    }
    item.pos_x = respawn::respawn_x(seed, tick_index, static_cast<std::uint32_t>(idx), world_width);
    item.pos_y = respawn::respawn_y(seed, tick_index, static_cast<std::uint32_t>(idx), world_height);
    item.active = 1U;
}

void rebuild_bins(GpuBatch& batch, cudaStream_t stream) {
    const size_t agent_cell_bytes = static_cast<size_t>(batch.agent_cols() * batch.agent_rows()) * sizeof(int);
    const size_t food_cell_bytes = static_cast<size_t>(batch.food_cols() * batch.food_rows()) * sizeof(int);
    CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_counts(), 0, agent_cell_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(batch.d_food_cell_counts(), 0, food_cell_bytes, stream));

    const int agent_block = 256;
    const int food_block = 256;
    const int agent_grid = (batch.num_agents() + agent_block - 1) / agent_block;
    const int food_grid = (batch.food_count() + food_block - 1) / food_block;

    bin_agents_kernel<<<agent_grid, agent_block, 0, stream>>>(
        batch.d_agent_states(), batch.num_agents(), batch.agent_cols(), batch.agent_rows(),
        batch.agent_cell_size(), batch.d_agent_cell_counts(), batch.d_agent_cell_ids(),
        batch.agent_bin_stride());
    bin_food_kernel<<<food_grid, food_block, 0, stream>>>(
        batch.d_food_states(), batch.food_count(), batch.food_cols(), batch.food_rows(),
        batch.food_cell_size(), batch.d_food_cell_counts(), batch.d_food_cell_ids(),
        batch.food_bin_stride());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void batch_build_sensors_resident(GpuBatch& batch, float world_width, float world_height,
                                  float max_energy, bool has_walls) {
    const int n = batch.num_agents();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
    sensor_build_resident_kernel<<<grid_size, block_size, 0, stream>>>(
        batch.d_agent_states(), batch.d_food_states(),
        batch.d_agent_cell_counts(), batch.d_agent_cell_ids(),
        batch.d_food_cell_counts(), batch.d_food_cell_ids(),
        batch.d_inputs(), n, batch.food_count(),
        batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
        batch.food_cols(), batch.food_rows(), batch.food_cell_size(),
        batch.agent_bin_stride(), batch.food_bin_stride(),
        batch.num_inputs(), world_width, world_height, max_energy, has_walls);
    CUDA_CHECK(cudaGetLastError());
}

void batch_simulate_tick_resident(GpuBatch& batch, float dt, float world_width,
                                  float world_height, bool has_walls,
                                  float energy_drain_per_tick, int target_fps,
                                  float food_pickup_range, float attack_range,
                                  float energy_gain_from_food, float energy_gain_from_kill,
                                  float food_respawn_rate, std::uint64_t seed,
                                  int tick_index) {
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
    const int agent_block = 256;
    const int food_block = 256;
    const int agent_grid = (batch.num_agents() + agent_block - 1) / agent_block;
    const int food_grid = (batch.food_count() + food_block - 1) / food_block;

    rebuild_bins(batch, stream);

    movement_kernel<<<agent_grid, agent_block, 0, stream>>>(
        batch.d_agent_states(), batch.d_outputs(), batch.num_agents(), batch.num_outputs(),
        dt, world_width, world_height, has_walls, energy_drain_per_tick, target_fps);

    rebuild_bins(batch, stream);

    prey_food_kernel<<<agent_grid, agent_block, 0, stream>>>(
        batch.d_agent_states(), batch.d_food_states(), batch.num_agents(),
        batch.d_food_cell_counts(), batch.d_food_cell_ids(),
        batch.food_cols(), batch.food_rows(), batch.food_cell_size(), batch.food_bin_stride(),
        batch.food_count(), food_pickup_range, energy_gain_from_food,
        world_width, world_height, has_walls);
    predator_attack_kernel<<<agent_grid, agent_block, 0, stream>>>(
        batch.d_agent_states(), batch.num_agents(),
        batch.d_agent_cell_counts(), batch.d_agent_cell_ids(),
        batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
        batch.agent_bin_stride(), attack_range, energy_gain_from_kill,
        world_width, world_height, has_walls);
    respawn_food_kernel<<<food_grid, food_block, 0, stream>>>(
        batch.d_food_states(), batch.food_count(), food_respawn_rate, world_width,
        world_height, seed, tick_index);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace moonai::gpu
