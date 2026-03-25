#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch.hpp"
#include "gpu/sensor_common.cuh"

#include <cub/cub.cuh>

#include <vector>

namespace moonai::gpu {

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ void normalize_inplace(float &x, float &y) {
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

__global__ void bin_agents_kernel(const float *pos_x, const float *pos_y,
                                  const unsigned int *alive, int num_agents,
                                  int cols, int rows, float cell_size,
                                  int *cell_counts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || alive[idx] == 0U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void bin_prey_kernel(const float *pos_x, const float *pos_y,
                                const unsigned int *types,
                                const unsigned int *alive, int num_agents,
                                int cols, int rows, float cell_size,
                                int *cell_counts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || alive[idx] == 0U || types[idx] != 1U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void bin_food_kernel(const float *pos_x, const float *pos_y,
                                const unsigned int *active, int food_count,
                                int cols, int rows, float cell_size,
                                int *cell_counts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || active[idx] == 0U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void scatter_agents_kernel(
    const float *pos_x, const float *pos_y, const unsigned int *types,
    const unsigned int *alive, int num_agents, int cols, int rows,
    float cell_size, const int *cell_offsets, int *cell_write_counts,
    unsigned int *cell_ids, GpuSensorAgentEntry *sensor_entries) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || alive[idx] == 0U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  const int slot = atomicAdd(&cell_write_counts[cell], 1);
  const int write_index = cell_offsets[cell] + slot;
  cell_ids[write_index] = static_cast<unsigned int>(idx);
  sensor_entries[write_index] = GpuSensorAgentEntry{
      static_cast<unsigned int>(idx),
      types[idx],
      pos_x[idx],
      pos_y[idx],
  };
}

__global__ void scatter_prey_kernel(const float *pos_x, const float *pos_y,
                                    const unsigned int *types,
                                    const unsigned int *alive, int num_agents,
                                    int cols, int rows, float cell_size,
                                    const int *cell_offsets,
                                    int *cell_write_counts,
                                    unsigned int *cell_ids) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || alive[idx] == 0U || types[idx] != 1U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  const int slot = atomicAdd(&cell_write_counts[cell], 1);
  cell_ids[cell_offsets[cell] + slot] = static_cast<unsigned int>(idx);
}

__global__ void scatter_food_kernel(const float *pos_x, const float *pos_y,
                                    const unsigned int *active, int food_count,
                                    int cols, int rows, float cell_size,
                                    const int *cell_offsets,
                                    int *cell_write_counts,
                                    unsigned int *cell_ids,
                                    GpuSensorFoodEntry *sensor_entries) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || active[idx] == 0U) {
    return;
  }
  const float inv_cell_size = 1.0f / cell_size;
  const int cx = sensor_clamp_index(
      static_cast<int>(pos_x[idx] * inv_cell_size), 0, cols - 1);
  const int cy = sensor_clamp_index(
      static_cast<int>(pos_y[idx] * inv_cell_size), 0, rows - 1);
  const int cell = cy * cols + cx;
  const int slot = atomicAdd(&cell_write_counts[cell], 1);
  const int write_index = cell_offsets[cell] + slot;
  cell_ids[write_index] = static_cast<unsigned int>(idx);
  sensor_entries[write_index] = GpuSensorFoodEntry{pos_x[idx], pos_y[idx]};
}

__global__ void movement_kernel(float *pos_x, float *pos_y, float *vel_x,
                                float *vel_y, const float *speed, float *energy,
                                float *distance_traveled, int *age,
                                unsigned int *alive, const float *outputs,
                                int num_agents, int num_outputs, float dt,
                                float world_width, float world_height,
                                bool has_walls, float energy_drain_per_step,
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
  energy[idx] -= energy_drain_per_step * dt * static_cast<float>(target_fps);

  if (has_walls) {
    pos_x[idx] = fminf(fmaxf(pos_x[idx], 0.0f), world_width);
    pos_y[idx] = fminf(fmaxf(pos_y[idx], 0.0f), world_height);
  } else {
    if (pos_x[idx] < 0.0f)
      pos_x[idx] += world_width;
    if (pos_x[idx] >= world_width)
      pos_x[idx] -= world_width;
    if (pos_y[idx] < 0.0f)
      pos_y[idx] += world_height;
    if (pos_y[idx] >= world_height)
      pos_y[idx] -= world_height;
  }

  if (energy[idx] <= 0.0f) {
    alive[idx] = 0U;
  }
}

template <bool HasWalls>
__global__ void prey_food_kernel(
    unsigned int *agent_alive, const unsigned int *agent_types,
    const float *agent_pos_x, const float *agent_pos_y, float *agent_energy,
    int *agent_food_eaten, const float *food_pos_x, const float *food_pos_y,
    unsigned int *food_active, int num_agents, const int *food_cell_offsets,
    const unsigned int *food_cell_ids, int food_cols, int food_rows,
    float food_cell_size, float food_pickup_range, float energy_gain_from_food,
    float world_width, float world_height, bool has_walls) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || agent_alive[idx] == 0U || agent_types[idx] != 1U) {
    return;
  }

  (void)has_walls;

  const float self_x = agent_pos_x[idx];
  const float self_y = agent_pos_y[idx];
  const float range_sq = food_pickup_range * food_pickup_range;
  const int cells_to_check =
      static_cast<int>(food_pickup_range / food_cell_size) + 1;
  const int cx = sensor_clamp_index(static_cast<int>(self_x / food_cell_size),
                                    0, food_cols - 1);
  const int cy = sensor_clamp_index(static_cast<int>(self_y / food_cell_size),
                                    0, food_rows - 1);
  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const int ny = cy + dy_cell;
    if (ny < 0 || ny >= food_rows) {
      continue;
    }
    const int row_base = ny * food_cols;
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int nx = cx + dx_cell;
      if (nx < 0 || nx >= food_cols) {
        continue;
      }
      if (!cell_may_intersect_radius<HasWalls>(nx, ny, food_cell_size, self_x,
                                               self_y, food_pickup_range,
                                               world_width, world_height)) {
        continue;
      }
      const int cell = row_base + nx;
      const int start = food_cell_offsets[cell];
      const int end = food_cell_offsets[cell + 1];
      for (int slot = start; slot < end; ++slot) {
        const unsigned int food_idx = food_cell_ids[slot];
        float dx = food_pos_x[food_idx] - self_x;
        float dy = food_pos_y[food_idx] - self_y;
        sensor_apply_wrap<HasWalls>(dx, dy, world_width, world_height);
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

template <bool HasWalls>
__global__ void predator_attack_kernel(
    unsigned int *agent_alive, const unsigned int *agent_types,
    const float *agent_pos_x, const float *agent_pos_y, float *agent_energy,
    int *agent_kills, int num_agents, const int *agent_cell_offsets,
    const unsigned int *agent_cell_ids, int agent_cols, int agent_rows,
    float agent_cell_size, float attack_range, float energy_gain_from_kill,
    float world_width, float world_height, bool has_walls) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_agents || agent_alive[idx] == 0U || agent_types[idx] != 0U) {
    return;
  }

  (void)has_walls;

  const float self_x = agent_pos_x[idx];
  const float self_y = agent_pos_y[idx];
  const float range_sq = attack_range * attack_range;
  const int cells_to_check =
      static_cast<int>(attack_range / agent_cell_size) + 1;
  const int cx = sensor_clamp_index(static_cast<int>(self_x / agent_cell_size),
                                    0, agent_cols - 1);
  const int cy = sensor_clamp_index(static_cast<int>(self_y / agent_cell_size),
                                    0, agent_rows - 1);
  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const int ny = cy + dy_cell;
    if (ny < 0 || ny >= agent_rows) {
      continue;
    }
    const int row_base = ny * agent_cols;
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int nx = cx + dx_cell;
      if (nx < 0 || nx >= agent_cols) {
        continue;
      }
      if (!cell_may_intersect_radius<HasWalls>(nx, ny, agent_cell_size, self_x,
                                               self_y, attack_range,
                                               world_width, world_height)) {
        continue;
      }
      const int cell = row_base + nx;
      const int start = agent_cell_offsets[cell];
      const int end = agent_cell_offsets[cell + 1];
      for (int slot = start; slot < end; ++slot) {
        const unsigned int prey_idx = agent_cell_ids[slot];
        if (prey_idx == static_cast<unsigned int>(idx) ||
            agent_alive[prey_idx] == 0U || agent_types[prey_idx] != 1U) {
          continue;
        }
        float dx = agent_pos_x[prey_idx] - self_x;
        float dy = agent_pos_y[prey_idx] - self_y;
        sensor_apply_wrap<HasWalls>(dx, dy, world_width, world_height);
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

__device__ __forceinline__ unsigned long long rotl(unsigned long long x,
                                                     int k) {
  return (x << k) | (x >> (64 - k));
}

__device__ unsigned long long xoshiro256ss(unsigned long long s[4]) {
  unsigned long long result = rotl(s[1] * 5, 7) * 9;
  unsigned long long t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

__device__ void init_rng(unsigned long long s[4], std::uint64_t seed,
                         int tick_index, int idx) {
  s[0] = seed + static_cast<unsigned long long>(tick_index) * 0x9e3779b97f4a7c15ULL +
         static_cast<unsigned long long>(idx) * 0xbf58476d1ce4e5b9ULL;
  s[1] = seed + static_cast<unsigned long long>(tick_index) * 0xbf58476d1ce4e5b9ULL +
         static_cast<unsigned long long>(idx) * 0x94d049bb133111ebULL;
  s[2] = seed + static_cast<unsigned long long>(tick_index) * 0x94d049bb133111ebULL +
         static_cast<unsigned long long>(idx) * 0x9e3779b97f4a7c15ULL;
  s[3] = seed + static_cast<unsigned long long>(tick_index) * 0x9e3779b97f4a7c15ULL +
         static_cast<unsigned long long>(idx) * 0xbf58476d1ce4e5b9ULL;
}

__device__ float rng_float(unsigned long long s[4]) {
  return static_cast<float>(xoshiro256ss(s) & 0xFFFFFFFFULL) /
         static_cast<float>(0xFFFFFFFFULL);
}

__global__ void respawn_food_kernel(float *food_pos_x, float *food_pos_y,
                                    unsigned int *food_active, int food_count,
                                    float respawn_rate, float world_width,
                                    float world_height, std::uint64_t seed,
                                    int step_index) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] != 0U) {
    return;
  }

  unsigned long long rng_state[4];
  init_rng(rng_state, seed, step_index, idx);

  // Check if this food should respawn
  if (rng_float(rng_state) >= respawn_rate) {
    return;
  }

  // Generate new position
  food_pos_x[idx] = rng_float(rng_state) * world_width;
  food_pos_y[idx] = rng_float(rng_state) * world_height;
  food_active[idx] = 1U;
}

void rebuild_agent_bins(GpuBatch &batch, cudaStream_t stream,
                        bool prey_only) {
  const size_t agent_cell_bytes =
      static_cast<size_t>(batch.agent_cols() * batch.agent_rows()) *
      sizeof(int);
  CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_counts(), 0, agent_cell_bytes,
                             stream));

  const int agent_grid = (batch.num_agents() + kBlockSize - 1) / kBlockSize;

  if (!prey_only) {
    bin_agents_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_alive(),
        batch.num_agents(), batch.agent_cols(), batch.agent_rows(),
        batch.agent_cell_size(), batch.d_agent_cell_counts());
  } else {
    bin_prey_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_types(),
        batch.d_agent_alive(), batch.num_agents(), batch.agent_cols(),
        batch.agent_rows(), batch.agent_cell_size(),
        batch.d_agent_cell_counts());
  }

  size_t agent_scan_bytes = batch.agent_scan_temp_bytes();
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      batch.d_scan_temp_storage(), agent_scan_bytes,
      batch.d_agent_cell_counts(), batch.d_agent_cell_offsets(),
      batch.agent_cell_count(), stream));

  CUDA_CHECK(cudaMemsetAsync(batch.d_agent_cell_write_counts(), 0,
                             agent_cell_bytes, stream));

  if (!prey_only) {
    scatter_agents_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_types(),
        batch.d_agent_alive(), batch.num_agents(), batch.agent_cols(),
        batch.agent_rows(), batch.agent_cell_size(),
        batch.d_agent_cell_offsets(), batch.d_agent_cell_write_counts(),
        batch.d_agent_cell_ids(), batch.d_sensor_agent_entries());
  } else {
    scatter_prey_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
        batch.d_agent_pos_x(), batch.d_agent_pos_y(), batch.d_agent_types(),
        batch.d_agent_alive(), batch.num_agents(), batch.agent_cols(),
        batch.agent_rows(), batch.agent_cell_size(),
        batch.d_agent_cell_offsets(), batch.d_agent_cell_write_counts(),
        batch.d_agent_cell_ids());
  }
  CUDA_CHECK(cudaGetLastError());
}

void rebuild_food_bins(GpuBatch &batch, cudaStream_t stream) {
  const size_t food_cell_bytes =
      static_cast<size_t>(batch.food_cols() * batch.food_rows()) * sizeof(int);
  CUDA_CHECK(
      cudaMemsetAsync(batch.d_food_cell_counts(), 0, food_cell_bytes, stream));

  const int food_grid = (batch.food_count() + kBlockSize - 1) / kBlockSize;
  bin_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
      batch.d_food_pos_x(), batch.d_food_pos_y(), batch.d_food_active(),
      batch.food_count(), batch.food_cols(), batch.food_rows(),
      batch.food_cell_size(), batch.d_food_cell_counts());

  size_t food_scan_bytes = batch.food_scan_temp_bytes();
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      batch.d_scan_temp_storage(), food_scan_bytes, batch.d_food_cell_counts(),
      batch.d_food_cell_offsets(), batch.food_cell_count(), stream));

  CUDA_CHECK(cudaMemsetAsync(batch.d_food_cell_write_counts(), 0,
                             food_cell_bytes, stream));
  scatter_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
      batch.d_food_pos_x(), batch.d_food_pos_y(), batch.d_food_active(),
      batch.food_count(), batch.food_cols(), batch.food_rows(),
      batch.food_cell_size(), batch.d_food_cell_offsets(),
      batch.d_food_cell_write_counts(), batch.d_food_cell_ids(),
      batch.d_sensor_food_entries());
  CUDA_CHECK(cudaGetLastError());
}

void rebuild_bins(GpuBatch &batch, cudaStream_t stream) {
  rebuild_agent_bins(batch, stream, false);
  rebuild_food_bins(batch, stream);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ── Public API Implementation ───────────────────────────────────────────────

void batch_rebuild_compact_bins(GpuBatch &batch) {
  cudaStream_t stream = static_cast<cudaStream_t>(batch.stream_handle());
  rebuild_bins(batch, stream);
}

void GpuBatch::rebuild_bins_async(float world_width, float world_height) {
  if (had_error_) {
    return;
  }

  // Ensure bin arrays are allocated with appropriate sizes
  const float cell_size = 100.0f;
  int cols = static_cast<int>(world_width / cell_size) + 1;
  int rows = static_cast<int>(world_height / cell_size) + 1;
  int cell_bins = cols * rows;

  // Allocate agent bin arrays if needed
  if (d_agent_cell_offsets_ == nullptr || agent_cell_capacity_ < cell_bins) {
    if (d_agent_cell_offsets_)
      cudaFree(d_agent_cell_offsets_);
    if (d_agent_cell_counts_)
      cudaFree(d_agent_cell_counts_);
    if (d_agent_cell_write_counts_)
      cudaFree(d_agent_cell_write_counts_);
    if (d_agent_cell_ids_)
      cudaFree(d_agent_cell_ids_);
    if (d_sensor_agent_entries_)
      cudaFree(d_sensor_agent_entries_);

    CUDA_CHECK(cudaMalloc(&d_agent_cell_offsets_, (cell_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_agent_cell_counts_, cell_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_agent_cell_write_counts_, cell_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_agent_cell_ids_, num_agents_ * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_sensor_agent_entries_, num_agents_ * sizeof(GpuSensorAgentEntry)));

    agent_cols_ = cols;
    agent_rows_ = rows;
    agent_cell_size_ = cell_size;
    agent_cell_capacity_ = cell_bins;
  }

  // Allocate food bin arrays if needed
  if (d_food_cell_offsets_ == nullptr || food_cell_capacity_ < cell_bins) {
    if (d_food_cell_offsets_)
      cudaFree(d_food_cell_offsets_);
    if (d_food_cell_counts_)
      cudaFree(d_food_cell_counts_);
    if (d_food_cell_write_counts_)
      cudaFree(d_food_cell_write_counts_);
    if (d_food_cell_ids_)
      cudaFree(d_food_cell_ids_);
    if (d_sensor_food_entries_)
      cudaFree(d_sensor_food_entries_);

    int food_capacity = std::max(food_count_, 1);
    CUDA_CHECK(cudaMalloc(&d_food_cell_offsets_, (cell_bins + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_food_cell_counts_, cell_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_food_cell_write_counts_, cell_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_food_cell_ids_, food_capacity * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_sensor_food_entries_, food_capacity * sizeof(GpuSensorFoodEntry)));

    food_cols_ = cols;
    food_rows_ = rows;
    food_cell_size_ = cell_size;
    food_cell_capacity_ = cell_bins;
  }

  batch_rebuild_compact_bins(*this);
}

void GpuBatch::rebuild_food_bins_async(float world_width, float world_height) {
  (void)world_width;
  (void)world_height;
  if (had_error_) {
    return;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());
  rebuild_food_bins(*this, stream);
}

void GpuBatch::rebuild_prey_bins_async(float world_width, float world_height) {
  (void)world_width;
  (void)world_height;
  if (had_error_) {
    return;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());
  rebuild_agent_bins(*this, stream, true);
}

void GpuBatch::build_sensors_async(float world_width, float world_height,
                                   float max_energy, bool has_walls) {
  if (had_error_) {
    return;
  }
  batch_build_sensors(*this, world_width, world_height, max_energy, has_walls);
}

void GpuBatch::apply_movement_async(const EcologyStepParams &params) {
  if (had_error_) {
    return;
  }
  const int agent_grid = (num_agents_ + kBlockSize - 1) / kBlockSize;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());

  movement_kernel<<<agent_grid, kBlockSize, 0, stream>>>(
      d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_, d_agent_vel_y_,
      d_agent_speed_, d_agent_energy_, d_agent_distance_traveled_, d_agent_age_,
      d_agent_alive_, d_outputs_, num_agents_, num_outputs_, params.dt,
      params.world_width, params.world_height, params.has_walls,
      params.energy_drain_per_step, params.target_fps);

  CUDA_CHECK(cudaGetLastError());
}

void GpuBatch::process_prey_food_async(float world_width, float world_height,
                                       bool has_walls, float food_pickup_range,
                                       float energy_gain_from_food) {
  if (had_error_) {
    return;
  }
  const int agent_grid = (num_agents_ + kBlockSize - 1) / kBlockSize;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());

  if (has_walls) {
    prey_food_kernel<true><<<agent_grid, kBlockSize, 0, stream>>>(
        d_agent_alive_, d_agent_types_, d_agent_pos_x_, d_agent_pos_y_,
        d_agent_energy_, d_agent_food_eaten_, d_food_pos_x_, d_food_pos_y_,
        d_food_active_, num_agents_, d_food_cell_offsets_, d_food_cell_ids_,
        food_cols_, food_rows_, food_cell_size_, food_pickup_range,
        energy_gain_from_food, world_width, world_height, has_walls);
  } else {
    prey_food_kernel<false><<<agent_grid, kBlockSize, 0, stream>>>(
        d_agent_alive_, d_agent_types_, d_agent_pos_x_, d_agent_pos_y_,
        d_agent_energy_, d_agent_food_eaten_, d_food_pos_x_, d_food_pos_y_,
        d_food_active_, num_agents_, d_food_cell_offsets_, d_food_cell_ids_,
        food_cols_, food_rows_, food_cell_size_, food_pickup_range,
        energy_gain_from_food, world_width, world_height, has_walls);
  }

  CUDA_CHECK(cudaGetLastError());
}

void GpuBatch::process_predator_attacks_async(float world_width,
                                              float world_height, bool has_walls,
                                              float attack_range,
                                              float energy_gain_from_kill) {
  if (had_error_) {
    return;
  }
  const int agent_grid = (num_agents_ + kBlockSize - 1) / kBlockSize;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());

  if (has_walls) {
    predator_attack_kernel<true><<<agent_grid, kBlockSize, 0, stream>>>(
        d_agent_alive_, d_agent_types_, d_agent_pos_x_, d_agent_pos_y_,
        d_agent_energy_, d_agent_kills_, num_agents_, d_agent_cell_offsets_,
        d_agent_cell_ids_, agent_cols_, agent_rows_, agent_cell_size_,
        attack_range, energy_gain_from_kill, world_width, world_height,
        has_walls);
  } else {
    predator_attack_kernel<false><<<agent_grid, kBlockSize, 0, stream>>>(
        d_agent_alive_, d_agent_types_, d_agent_pos_x_, d_agent_pos_y_,
        d_agent_energy_, d_agent_kills_, num_agents_, d_agent_cell_offsets_,
        d_agent_cell_ids_, agent_cols_, agent_rows_, agent_cell_size_,
        attack_range, energy_gain_from_kill, world_width, world_height,
        has_walls);
  }

  CUDA_CHECK(cudaGetLastError());
}

void GpuBatch::respawn_food_async(float world_width, float world_height,
                                  float respawn_rate, std::uint64_t seed,
                                  int step_index) {
  if (had_error_) {
    return;
  }
  if (food_count_ <= 0) {
    return;
  }

  const int food_grid = (food_count_ + kBlockSize - 1) / kBlockSize;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_handle());

  respawn_food_kernel<<<food_grid, kBlockSize, 0, stream>>>(
      d_food_pos_x_, d_food_pos_y_, d_food_active_, food_count_, respawn_rate,
      world_width, world_height, seed, step_index);

  CUDA_CHECK(cudaGetLastError());
}

void GpuBatch::launch_ecology_step_async(const EcologyStepParams &params) {
  if (had_error_) {
    return;
  }

  // Launch the full sequence of kernels
  // Phase 1: Sensing and Decision Making (using current positions)
  // 1. Rebuild spatial bins for sensing
  rebuild_bins_async(params.world_width, params.world_height);

  // 2. Build sensors
  build_sensors_async(params.world_width, params.world_height, params.max_energy,
                      params.has_walls);

  // 3. Neural inference
  launch_inference_async();

  // Phase 2: Movement (agents move to new positions)
  // 4. Apply movement
  apply_movement_async(params);

  // Phase 3: Interactions (using new positions)
  // 5. Rebuild food bins (food positions are static, but agents moved)
  rebuild_food_bins_async(params.world_width, params.world_height);

  // 6. Rebuild prey bins for attack queries (agents are at new positions)
  rebuild_prey_bins_async(params.world_width, params.world_height);

  // 7. Process prey food (prey eat food at their new positions)
  process_prey_food_async(params.world_width, params.world_height,
                          params.has_walls, params.food_pickup_range,
                          params.energy_gain_from_food);

  // 8. Process predator attacks (predators attack prey at new positions)
  process_predator_attacks_async(params.world_width, params.world_height,
                                 params.has_walls, params.attack_range,
                                 params.energy_gain_from_kill);

  // 9. Respawn food
  respawn_food_async(params.world_width, params.world_height,
                     params.food_respawn_rate, params.seed, params.step_index);
}

} // namespace moonai::gpu
