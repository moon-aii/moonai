#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

#include <cmath>

namespace moonai::gpu {

namespace {

__device__ __forceinline__ float clampf(float value, float min_value, float max_value) {
    return fminf(fmaxf(value, min_value), max_value);
}

__device__ __forceinline__ float normalize_angle(float dx, float dy) {
    return atan2f(dy, dx) / 3.14159265f;
}

__device__ __forceinline__ void apply_wrap_diff(float& dx, float& dy, float world_width,
                                                float world_height, bool has_walls) {
    if (has_walls) {
        return;
    }
    if (fabsf(dx) > world_width * 0.5f) {
        dx = (dx > 0.0f) ? dx - world_width : dx + world_width;
    }
    if (fabsf(dy) > world_height * 0.5f) {
        dy = (dy > 0.0f) ? dy - world_height : dy + world_height;
    }
}

__device__ __forceinline__ int clamp_index(int value, int min_value, int max_value) {
    return max(min_value, min(value, max_value));
}

__global__ void sensor_build_kernel(const GpuAgentState* __restrict__ agents,
                                    const int* __restrict__ agent_cell_offsets,
                                    const GpuGridEntry* __restrict__ agent_entries,
                                    const int* __restrict__ food_cell_offsets,
                                    const GpuGridEntry* __restrict__ food_entries,
                                    float* __restrict__ inputs,
                                    int num_agents,
                                    int agent_cols,
                                    int agent_rows,
                                    float agent_cell_size,
                                    int food_cols,
                                    int food_rows,
                                    float food_cell_size,
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
    out[0] = -1.0f;
    out[1] = 0.0f;
    out[2] = -1.0f;
    out[3] = 0.0f;
    out[4] = -1.0f;
    out[5] = 0.0f;
    out[6] = 1.0f;
    out[7] = 0.0f;
    out[8] = 0.0f;
    out[9] = 0.0f;
    out[10] = 0.0f;
    out[11] = 1.0f;
    out[12] = 1.0f;
    out[13] = 1.0f;
    out[14] = 1.0f;

    const GpuAgentState self = agents[agent_idx];
    if (!self.alive) {
        return;
    }

    const float vision = self.vision_range;
    const float vision_sq = vision * vision;
    float nearest_pred_dist_sq = INFINITY;
    float nearest_prey_dist_sq = INFINITY;
    float nearest_food_dist_sq = INFINITY;
    float nearest_pred_dx = 0.0f;
    float nearest_pred_dy = 0.0f;
    float nearest_prey_dx = 0.0f;
    float nearest_prey_dy = 0.0f;
    float nearest_food_dx = 0.0f;
    float nearest_food_dy = 0.0f;
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
            const int cell_index = ny * agent_cols + nx;
            const int start = agent_cell_offsets[cell_index];
            const int end = agent_cell_offsets[cell_index + 1];
            for (int i = start; i < end; ++i) {
                const GpuGridEntry entry = agent_entries[i];
                const GpuAgentState other = agents[entry.id];
                if (!other.alive || other.id == self.id) {
                    continue;
                }

                float dx = entry.pos_x - self.pos_x;
                float dy = entry.pos_y - self.pos_y;
                apply_wrap_diff(dx, dy, world_width, world_height, has_walls);
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq > vision_sq) {
                    continue;
                }

                if (other.type == 0) {
                    ++local_predators;
                    if (dist_sq < nearest_pred_dist_sq) {
                        nearest_pred_dist_sq = dist_sq;
                        nearest_pred_dx = dx;
                        nearest_pred_dy = dy;
                    }
                } else {
                    ++local_prey;
                    if (dist_sq < nearest_prey_dist_sq) {
                        nearest_prey_dist_sq = dist_sq;
                        nearest_prey_dx = dx;
                        nearest_prey_dy = dy;
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
            const int cell_index = ny * food_cols + nx;
            const int start = food_cell_offsets[cell_index];
            const int end = food_cell_offsets[cell_index + 1];
            for (int i = start; i < end; ++i) {
                const GpuGridEntry entry = food_entries[i];
                float dx = entry.pos_x - self.pos_x;
                float dy = entry.pos_y - self.pos_y;
                apply_wrap_diff(dx, dy, world_width, world_height, has_walls);
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq > vision_sq) {
                    continue;
                }
                if (dist_sq < nearest_food_dist_sq) {
                    nearest_food_dist_sq = dist_sq;
                    nearest_food_dx = dx;
                    nearest_food_dy = dy;
                }
            }
        }
    }

    if (nearest_pred_dist_sq < INFINITY) {
        out[0] = sqrtf(nearest_pred_dist_sq) / vision;
        out[1] = normalize_angle(nearest_pred_dx, nearest_pred_dy);
    }
    if (nearest_prey_dist_sq < INFINITY) {
        out[2] = sqrtf(nearest_prey_dist_sq) / vision;
        out[3] = normalize_angle(nearest_prey_dx, nearest_prey_dy);
    }
    if (nearest_food_dist_sq < INFINITY) {
        out[4] = sqrtf(nearest_food_dist_sq) / vision;
        out[5] = normalize_angle(nearest_food_dx, nearest_food_dy);
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

} // namespace

void batch_build_sensors(GpuBatch& batch, float world_width, float world_height,
                         float max_energy, bool has_walls) {
    const int n = batch.num_agents();
    int min_grid_size = 0;
    int block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        sensor_build_kernel,
        0,
        0));
    if (block_size <= 0) {
        block_size = 256;
    }
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());

    sensor_build_kernel<<<grid_size, block_size, 0, stream>>>(
        batch.d_agent_states(),
        batch.d_agent_cell_offsets(),
        batch.d_agent_grid_entries(),
        batch.d_food_cell_offsets(),
        batch.d_food_grid_entries(),
        batch.d_inputs(),
        n,
        batch.agent_cols(),
        batch.agent_rows(),
        batch.agent_cell_size(),
        batch.food_cols(),
        batch.food_rows(),
        batch.food_cell_size(),
        batch.num_inputs(),
        world_width,
        world_height,
        max_energy,
        has_walls);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace moonai::gpu
