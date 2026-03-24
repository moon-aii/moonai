#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"
#include "gpu/sensor_common.cuh"

#include "core/deterministic_respawn.hpp"
#include "core/profiler.hpp"

#include <cub/cub.cuh>

#include <vector>

namespace moonai::gpu {

namespace {

constexpr int kBlockSize = 256;

enum class AgentBinMode {
    AllAgents,
    PreyOnly,
};

#ifdef MOONAI_BUILD_PROFILER
__device__ __forceinline__ unsigned long long read_globaltimer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void resident_stage_begin_kernel(unsigned long long* last_timestamp_ns) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned long long now = read_globaltimer_ns();
        *last_timestamp_ns = now == 0ULL ? 1ULL : now;
    }
}

__global__ void resident_stage_end_kernel(unsigned long long* stage_accum_ns,
                                          unsigned long long* last_timestamp_ns,
                                          int stage_index) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const unsigned long long now = read_globaltimer_ns();
        const unsigned long long last = *last_timestamp_ns;
        if (last != 0ULL) {
            stage_accum_ns[stage_index] += now - last;
        }
        *last_timestamp_ns = now;
    }
}

__host__ __forceinline__ void record_stage_begin(GpuBatch& batch, cudaStream_t stream) {
    resident_stage_begin_kernel<<<1, 1, 0, stream>>>(batch.d_resident_stage_last_timestamp_ns());
}

__host__ __forceinline__ void record_stage_end(GpuBatch& batch, cudaStream_t stream, GpuStageTiming stage) {
    resident_stage_end_kernel<<<1, 1, 0, stream>>>(
        batch.d_resident_stage_accum_ns(), batch.d_resident_stage_last_timestamp_ns(),
        static_cast<int>(stage));
}
#else
__host__ __forceinline__ void record_stage_begin(GpuBatch&, cudaStream_t) {}
template<typename Stage>
__host__ __forceinline__ void record_stage_end(GpuBatch&, cudaStream_t, Stage) {}
#endif

__device__ __forceinline__ void normalize_inplace(float& x, float& y) {
    const float len_sq = x * x + y * y;
    if (len_sq < 1e-12f) {
        x = 0.0f;
        y = 0.0f;
        return;
    }
    const float inv_len = rsqrtf(len_sq);
    x *= inv_len;
    y *= inv_len;
}

__global__ void bin_agents_kernel(const float* pos_x, const float* pos_y,
                                  const unsigned int* alive, int num_agents,
                                  int cols, int rows, float cell_size,
                                  int* cell_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || alive[idx] == 0U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    atomicAdd(&cell_counts[cell], 1);
}

__global__ void bin_prey_kernel(const float* pos_x, const float* pos_y,
                                const unsigned int* types, const unsigned int* alive,
                                int num_agents, int cols, int rows, float cell_size,
                                int* cell_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || alive[idx] == 0U || types[idx] != 1U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    atomicAdd(&cell_counts[cell], 1);
}

__global__ void bin_food_kernel(const float* pos_x, const float* pos_y,
                                const unsigned int* active, int food_count,
                                int cols, int rows, float cell_size,
                                int* cell_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count || active[idx] == 0U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    atomicAdd(&cell_counts[cell], 1);
}

__global__ void scatter_agents_kernel(const float* pos_x, const float* pos_y,
                                      const unsigned int* alive, int num_agents,
                                      int cols, int rows, float cell_size,
                                      const int* cell_offsets, int* cell_write_counts,
                                      unsigned int* cell_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || alive[idx] == 0U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    const int slot = atomicAdd(&cell_write_counts[cell], 1);
    cell_ids[cell_offsets[cell] + slot] = static_cast<unsigned int>(idx);
}

__global__ void scatter_prey_kernel(const float* pos_x, const float* pos_y,
                                    const unsigned int* types, const unsigned int* alive,
                                    int num_agents, int cols, int rows, float cell_size,
                                    const int* cell_offsets, int* cell_write_counts,
                                    unsigned int* cell_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || alive[idx] == 0U || types[idx] != 1U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    const int slot = atomicAdd(&cell_write_counts[cell], 1);
    cell_ids[cell_offsets[cell] + slot] = static_cast<unsigned int>(idx);
}

__global__ void scatter_food_kernel(const float* pos_x, const float* pos_y,
                                    const unsigned int* active, int food_count,
                                    int cols, int rows, float cell_size,
                                    const int* cell_offsets, int* cell_write_counts,
                                    unsigned int* cell_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count || active[idx] == 0U) {
        return;
    }
    const int cx = sensor_clamp_index(static_cast<int>(pos_x[idx] / cell_size), 0, cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(pos_y[idx] / cell_size), 0, rows - 1);
    const int cell = cy * cols + cx;
    const int slot = atomicAdd(&cell_write_counts[cell], 1);
    cell_ids[cell_offsets[cell] + slot] = static_cast<unsigned int>(idx);
}

template<bool HasWalls>
__global__ void sensor_build_kernel(SensorBuildView view) {
    const int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= view.num_agents) {
        return;
    }
    build_sensor_inputs_for_agent<HasWalls>(view, agent_idx);
}

__global__ void movement_kernel(float* pos_x, float* pos_y,
                                float* vel_x, float* vel_y,
                                const float* speed, float* energy,
                                float* distance_traveled, int* age,
                                unsigned int* alive, const float* outputs,
                                int num_agents, int num_outputs, float dt,
                                float world_width, float world_height,
                                bool has_walls, float energy_drain_per_tick,
                                int target_fps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || alive[idx] == 0U) {
        return;
    }
    float dir_x = 0.0f;
    float dir_y = 0.0f;
    if (num_outputs >= 2) {
        dir_x = outputs[idx * num_outputs + 0] * 2.0f - 1.0f;
        dir_y = outputs[idx * num_outputs + 1] * 2.0f - 1.0f;
    }
    normalize_inplace(dir_x, dir_y);
    vel_x[idx] = dir_x * speed[idx];
    vel_y[idx] = dir_y * speed[idx];
    pos_x[idx] += vel_x[idx] * dt;
    pos_y[idx] += vel_y[idx] * dt;
    distance_traveled[idx] += speed[idx] * dt;
    age[idx] += 1;
    energy[idx] -= energy_drain_per_tick * dt * static_cast<float>(target_fps);

    if (has_walls) {
        pos_x[idx] = sensor_clampf(pos_x[idx], 0.0f, world_width);
        pos_y[idx] = sensor_clampf(pos_y[idx], 0.0f, world_height);
    } else {
        if (pos_x[idx] < 0.0f) pos_x[idx] += world_width;
        if (pos_x[idx] >= world_width) pos_x[idx] -= world_width;
        if (pos_y[idx] < 0.0f) pos_y[idx] += world_height;
        if (pos_y[idx] >= world_height) pos_y[idx] -= world_height;
    }

    if (energy[idx] <= 0.0f) {
        alive[idx] = 0U;
    }
}

__global__ void prey_food_kernel(
    unsigned int* agent_alive,
    const unsigned int* agent_types,
    const float* agent_pos_x,
    const float* agent_pos_y,
    float* agent_energy,
    int* agent_food_eaten,
    const float* food_pos_x,
    const float* food_pos_y,
    unsigned int* food_active,
    int num_agents,
    const int* food_cell_offsets,
    const unsigned int* food_cell_ids,
    int food_cols,
    int food_rows,
    float food_cell_size,
    int food_count,
    float food_pickup_range,
    float energy_gain_from_food,
    float world_width,
    float world_height,
    bool has_walls) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || agent_alive[idx] == 0U || agent_types[idx] != 1U) {
        return;
    }

    const float range_sq = food_pickup_range * food_pickup_range;
    const int cells_to_check = static_cast<int>(food_pickup_range / food_cell_size) + 1;
    const int cx = sensor_clamp_index(static_cast<int>(agent_pos_x[idx] / food_cell_size), 0, food_cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(agent_pos_y[idx] / food_cell_size), 0, food_rows - 1);
    for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
        for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
            const int nx = cx + dx_cell;
            const int ny = cy + dy_cell;
            if (nx < 0 || nx >= food_cols || ny < 0 || ny >= food_rows) {
                continue;
            }
            const int cell = ny * food_cols + nx;
            const int start = food_cell_offsets[cell];
            const int end = food_cell_offsets[cell + 1];
            for (int slot = start; slot < end; ++slot) {
                const unsigned int food_idx = food_cell_ids[slot];
                if (food_idx >= static_cast<unsigned int>(food_count) || food_active[food_idx] == 0U) {
                    continue;
                }
                float dx = food_pos_x[food_idx] - agent_pos_x[idx];
                float dy = food_pos_y[food_idx] - agent_pos_y[idx];
                sensor_apply_wrap(dx, dy, world_width, world_height, has_walls);
                if (dx * dx + dy * dy > range_sq) {
                    continue;
                }
                if (atomicCAS(&food_active[food_idx], 1U, 0U) == 1U) {
                    agent_energy[idx] += energy_gain_from_food;
                    agent_food_eaten[idx] += 1;
                    return;
                }
            }
        }
    }
}

__global__ void predator_attack_kernel(
    unsigned int* agent_alive,
    const unsigned int* agent_types,
    const float* agent_pos_x,
    const float* agent_pos_y,
    float* agent_energy,
    int* agent_kills,
    int num_agents,
    const int* agent_cell_offsets,
    const unsigned int* agent_cell_ids,
    int agent_cols,
    int agent_rows,
    float agent_cell_size,
    float attack_range,
    float energy_gain_from_kill,
    float world_width,
    float world_height,
    bool has_walls) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_agents || agent_alive[idx] == 0U || agent_types[idx] != 0U) {
        return;
    }

    const float range_sq = attack_range * attack_range;
    const int cells_to_check = static_cast<int>(attack_range / agent_cell_size) + 1;
    const int cx = sensor_clamp_index(static_cast<int>(agent_pos_x[idx] / agent_cell_size), 0, agent_cols - 1);
    const int cy = sensor_clamp_index(static_cast<int>(agent_pos_y[idx] / agent_cell_size), 0, agent_rows - 1);
    for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
        for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
            const int nx = cx + dx_cell;
            const int ny = cy + dy_cell;
            if (nx < 0 || nx >= agent_cols || ny < 0 || ny >= agent_rows) {
                continue;
            }
            const int cell = ny * agent_cols + nx;
            const int start = agent_cell_offsets[cell];
            const int end = agent_cell_offsets[cell + 1];
            for (int slot = start; slot < end; ++slot) {
                const unsigned int prey_idx = agent_cell_ids[slot];
                if (prey_idx == static_cast<unsigned int>(idx)
                    || agent_alive[prey_idx] == 0U
                    || agent_types[prey_idx] != 1U) {
                    continue;
                }
                float dx = agent_pos_x[prey_idx] - agent_pos_x[idx];
                float dy = agent_pos_y[prey_idx] - agent_pos_y[idx];
                sensor_apply_wrap(dx, dy, world_width, world_height, has_walls);
                if (dx * dx + dy * dy > range_sq) {
                    continue;
                }
                if (atomicCAS(&agent_alive[prey_idx], 1U, 0U) == 1U) {
                    agent_kills[idx] += 1;
                    agent_energy[idx] += energy_gain_from_kill;
                    return;
                }
            }
        }
    }
}

__global__ void respawn_food_kernel(float* food_pos_x, float* food_pos_y,
                                    unsigned int* food_active, int food_count,
                                    float respawn_rate, float world_width,
                                    float world_height, std::uint64_t seed,
                                    const int* tick_index_ptr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= food_count || food_active[idx] != 0U) {
        return;
    }
    const int tick_index = *tick_index_ptr;
    if (!respawn::should_respawn(seed, tick_index, static_cast<std::uint32_t>(idx), respawn_rate)) {
        return;
    }
    food_pos_x[idx] = respawn::respawn_x(seed, tick_index, static_cast<std::uint32_t>(idx), world_width);
    food_pos_y[idx] = respawn::respawn_y(seed, tick_index, static_cast<std::uint32_t>(idx), world_height);
    food_active[idx] = 1U;
}

void rebuild_agent_bins(GpuBatch& batch, cudaStream_t stream, AgentBinMode mode) {
    const size_t agent_cell_bytes = static_cast<size_t>(batch.agent_cols() * batch.agent_rows()) * sizeof(int);
    CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_counts(), 0, agent_cell_bytes, stream));

    const int agent_grid = (batch.num_agents() + kBlockSize - 1) / kBlockSize;

    if (mode == AgentBinMode::AllAgents) {
        bin_agents_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
            batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_alive(),
            batch.num_agents(), batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
            batch.d_agent_cell_counts());
    } else {
        bin_prey_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
            batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_types(), batch.d_agent_alive(),
            batch.num_agents(), batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
            batch.d_agent_cell_counts());
    }

    CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_offsets(), 0, sizeof(int), stream));

    size_t agent_scan_bytes = batch.agent_scan_temp_bytes();
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(batch.d_scan_temp_storage(), agent_scan_bytes,
        batch.d_agent_cell_counts(), batch.d_agent_cell_offsets() + 1, batch.agent_cell_count(), stream));

    CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_write_counts(), 0, agent_cell_bytes, stream));

    if (mode == AgentBinMode::AllAgents) {
        scatter_agents_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
            batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_alive(),
            batch.num_agents(), batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
            batch.d_agent_cell_offsets(), batch.d_agent_cell_write_counts(), batch.d_agent_cell_ids());
    } else {
        scatter_prey_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
            batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_types(), batch.d_agent_alive(),
            batch.num_agents(), batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
            batch.d_agent_cell_offsets(), batch.d_agent_cell_write_counts(), batch.d_agent_cell_ids());
    }
    CUDA_CHECK(cudaGetLastError());
}

void rebuild_food_bins(GpuBatch& batch, cudaStream_t stream) {
    const size_t food_cell_bytes = static_cast<size_t>(batch.food_cols() * batch.food_rows()) * sizeof(int);
    CUDA_CHECK(cudaMemsetAsync(batch.d_food_cell_counts(), 0, food_cell_bytes, stream));

    const int food_grid = (batch.food_count() + kBlockSize - 1) / kBlockSize;
    bin_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
        batch.d_food_pos_x(), batch.d_food_pos_y(), batch.d_food_active(),
        batch.food_count(), batch.food_cols(), batch.food_rows(), batch.food_cell_size(),
        batch.d_food_cell_counts());

    CUDA_CHECK(cudaMemsetAsync(batch.d_food_cell_offsets(), 0, sizeof(int), stream));

    size_t food_scan_bytes = batch.food_scan_temp_bytes();
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(batch.d_scan_temp_storage(), food_scan_bytes,
        batch.d_food_cell_counts(), batch.d_food_cell_offsets() + 1, batch.food_cell_count(), stream));

    CUDA_CHECK(cudaMemsetAsync(batch.d_food_cell_write_counts(), 0, food_cell_bytes, stream));
    scatter_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
        batch.d_food_pos_x(), batch.d_food_pos_y(), batch.d_food_active(),
        batch.food_count(), batch.food_cols(), batch.food_rows(), batch.food_cell_size(),
        batch.d_food_cell_offsets(), batch.d_food_cell_write_counts(), batch.d_food_cell_ids());
    CUDA_CHECK(cudaGetLastError());
}

void rebuild_bins(GpuBatch& batch, cudaStream_t stream) {
    record_stage_begin(batch, stream);
    rebuild_agent_bins(batch, stream, AgentBinMode::AllAgents);
    rebuild_food_bins(batch, stream);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickBinRebuildPre);
    CUDA_CHECK(cudaGetLastError());
}

void launch_resident_tick_sequence(GpuBatch& batch, const ResidentTickParams& params,
                                   cudaStream_t stream) {
    const int agent_grid = (batch.num_agents() + kBlockSize - 1) / kBlockSize;
    const int food_grid = (batch.food_count() + kBlockSize - 1) / kBlockSize;

    CUDA_CHECK(cudaMemcpyAsync(batch.d_tick_index(), batch.host_tick_index(), sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    rebuild_agent_bins(batch, stream, AgentBinMode::AllAgents);
    rebuild_food_bins(batch, stream);

    SensorBuildView view{
        batch.d_agent_pos_x(),
        batch.d_agent_pos_y(),
        batch.d_agent_vel_x(),
        batch.d_agent_vel_y(),
        batch.d_agent_speed(),
        batch.d_agent_vision(),
        batch.d_agent_energy(),
        batch.d_agent_ids(),
        batch.d_agent_types(),
        batch.d_agent_alive(),
        batch.d_food_pos_x(),
        batch.d_food_pos_y(),
        batch.d_food_active(),
        batch.d_agent_cell_offsets(),
        batch.d_agent_cell_ids(),
        batch.d_food_cell_offsets(),
        batch.d_food_cell_ids(),
        batch.d_inputs(),
        batch.num_agents(),
        batch.food_count(),
        batch.agent_cols(),
        batch.agent_rows(),
        batch.agent_cell_size(),
        batch.food_cols(),
        batch.food_rows(),
        batch.food_cell_size(),
        batch.num_inputs(),
        params.world_width,
        params.world_height,
        params.max_energy,
        params.has_walls,
    };

    if (params.has_walls) {
        sensor_build_kernel<true><<<agent_grid, kBlockSize, 0, stream>>>(view);
    } else {
        sensor_build_kernel<false><<<agent_grid, kBlockSize, 0, stream>>>(view);
    }
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickSensorBuild);

    batch_neural_inference(batch);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickInference);

    movement_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_vel_x(), batch.d_agent_vel_y(),
        batch.d_agent_speed(), batch.d_agent_energy(), batch.d_agent_distance_traveled(),
        batch.d_agent_age(), batch.d_agent_alive(), batch.d_outputs(), batch.num_agents(),
        batch.num_outputs(), params.dt, params.world_width, params.world_height,
        params.has_walls, params.energy_drain_per_tick, params.target_fps);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickMovement);

    rebuild_agent_bins(batch, stream, AgentBinMode::PreyOnly);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickBinRebuildPost);

    prey_food_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_alive(), batch.d_agent_types(), batch.d_agent_pos_x(), batch.d_agent_pos_y(),
        batch.d_agent_energy(), batch.d_agent_food_eaten(), batch.d_food_pos_x(), batch.d_food_pos_y(),
        batch.d_food_active(), batch.num_agents(), batch.d_food_cell_offsets(), batch.d_food_cell_ids(),
        batch.food_cols(), batch.food_rows(), batch.food_cell_size(),
        batch.food_count(), params.food_pickup_range, params.energy_gain_from_food,
        params.world_width, params.world_height, params.has_walls);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickPreyFood);
    predator_attack_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_alive(), batch.d_agent_types(), batch.d_agent_pos_x(), batch.d_agent_pos_y(),
        batch.d_agent_energy(), batch.d_agent_kills(), batch.num_agents(), batch.d_agent_cell_offsets(),
        batch.d_agent_cell_ids(), batch.agent_cols(), batch.agent_rows(), batch.agent_cell_size(),
        params.attack_range, params.energy_gain_from_kill,
        params.world_width, params.world_height, params.has_walls);
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickPredatorAttack);
    respawn_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
        batch.d_food_pos_x(), batch.d_food_pos_y(), batch.d_food_active(), batch.food_count(),
        params.food_respawn_rate, params.world_width, params.world_height, params.seed,
        batch.d_tick_index());
    record_stage_end(batch, stream, GpuStageTiming::ResidentTickRespawn);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void batch_build_sensors_resident(GpuBatch& batch, float world_width, float world_height,
                                  float max_energy, bool has_walls) {
    batch_build_sensors(batch, world_width, world_height, max_energy, has_walls);
}

void batch_rebuild_compact_bins(GpuBatch& batch) {
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
    rebuild_bins(batch, stream);
}

void batch_prepare_resident_tick_graph(GpuBatch& batch, const ResidentTickParams& params) {
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_resident_tick_sequence(batch, params, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    if (graph == nullptr) {
        return;
    }

    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    if (exec == nullptr) {
        cudaGraphDestroy(graph);
        return;
    }

    batch.resident_tick_graph_ = static_cast<void*>(graph);
    batch.resident_tick_graph_exec_ = static_cast<void*>(exec);
    batch.resident_graph_valid_ = true;
}

void batch_simulate_tick_resident(GpuBatch& batch, float dt, float world_width,
                                  float world_height, bool has_walls,
                                  float energy_drain_per_tick, int target_fps,
                                  float food_pickup_range, float attack_range,
                                  float energy_gain_from_food, float energy_gain_from_kill,
                                  float food_respawn_rate, std::uint64_t seed,
                                  int tick_index) {
    cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
    *batch.host_tick_index() = tick_index;
    if (batch.resident_graph_valid_ && batch.resident_tick_graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(batch.resident_tick_graph_exec_), stream));
    } else {
        const ResidentTickParams params{
            dt,
            world_width,
            world_height,
            has_walls,
            energy_drain_per_tick,
            target_fps,
            food_pickup_range,
            attack_range,
            batch.resident_tick_params_.max_energy,
            energy_gain_from_food,
            energy_gain_from_kill,
            food_respawn_rate,
            seed,
        };
        launch_resident_tick_sequence(batch, params, stream);
    }
}

} // namespace moonai::gpu
