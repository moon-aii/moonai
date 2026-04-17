#include "core/profiler_macros.hpp"
#include "simulation/batch.hpp"
#include "simulation/checks.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace moonai::simulation {

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kNearestTargetsPerType = 5;
constexpr int kTargetCoordinateCount = 2;
constexpr int kPerTypeSensorCount = kNearestTargetsPerType * kTargetCoordinateCount;
constexpr int kTargetTypeCount = 3;
constexpr int kTargetSensorCount = kPerTypeSensorCount * kTargetTypeCount;
constexpr int kSelfStateSensorCount = 3;
constexpr int kWallSensorCount = 2;
constexpr int kSelfStateOffset = kTargetSensorCount;
constexpr int kWallSensorOffset = kSelfStateOffset + kSelfStateSensorCount;
constexpr int kUnclaimed = 0x7f7f7f7f;

static_assert(SENSOR_COUNT == kWallSensorOffset + kWallSensorCount, "Sensor layout constants must match SENSOR_COUNT.");

std::size_t next_power_of_2(std::size_t n) {
  if (n == 0) {
    return 1;
  }
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

__device__ float clampf(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

template <int N>
__device__ void insert_nearest_candidate(float dx, float dy, float dist_sq, float (&best_dx)[N], float (&best_dy)[N],
                                         float (&best_dist_sq)[N]) {
  if (dist_sq >= best_dist_sq[N - 1]) {
    return;
  }

  int insert_at = N - 1;
  while (insert_at > 0 && dist_sq < best_dist_sq[insert_at - 1]) {
    best_dist_sq[insert_at] = best_dist_sq[insert_at - 1];
    best_dx[insert_at] = best_dx[insert_at - 1];
    best_dy[insert_at] = best_dy[insert_at - 1];
    --insert_at;
  }

  best_dist_sq[insert_at] = dist_sq;
  best_dx[insert_at] = dx;
  best_dy[insert_at] = dy;
}

template <int N>
__device__ void encode_nearest_targets(const float (&best_dx)[N], const float (&best_dy)[N],
                                       const float (&best_dist_sq)[N], float vision_range, float *out) {
  for (int i = 0; i < N; ++i) {
    if (best_dist_sq[i] == INFINITY) {
      out[i * 2] = 0.0f;
      out[i * 2 + 1] = 0.0f;
      continue;
    }

    const float dist = sqrtf(best_dist_sq[i]);
    if (dist <= 1e-6f) {
      out[i * 2] = 0.0f;
      out[i * 2 + 1] = 0.0f;
      continue;
    }

    const float proximity = clampf(1.0f - (dist / vision_range), 0.0f, 1.0f);
    const float inv_dist = 1.0f / dist;
    out[i * 2] = clampf(best_dx[i] * inv_dist * proximity, -1.0f, 1.0f);
    out[i * 2 + 1] = clampf(best_dy[i] * inv_dist * proximity, -1.0f, 1.0f);
  }
}

__device__ float encode_axis_wall_sensor(float negative_side_dist, float positive_side_dist, float vision_range) {
  const bool negative_in_range = negative_side_dist < vision_range;
  const bool positive_in_range = positive_side_dist < vision_range;

  if (!negative_in_range && !positive_in_range) {
    return 0.0f;
  }

  if (negative_in_range && (!positive_in_range || negative_side_dist <= positive_side_dist)) {
    return -(1.0f - (negative_side_dist / vision_range));
  }

  return 1.0f - (positive_side_dist / vision_range);
}

__device__ int cell_coord(float pos, float cell_size, int limit) {
  int coord = static_cast<int>(pos / cell_size);
  if (coord < 0) {
    return 0;
  }
  if (coord >= limit) {
    return limit - 1;
  }
  return coord;
}

__device__ bool cell_may_intersect_radius(int cx, int cy, float cell_size, float origin_x, float origin_y,
                                          float radius) {
  const float center_x = (static_cast<float>(cx) + 0.5f) * cell_size;
  const float center_y = (static_cast<float>(cy) + 0.5f) * cell_size;
  float dx = center_x - origin_x;
  float dy = center_y - origin_y;

  const float half_size = cell_size * 0.5f;
  const float nearest_x = fmaxf(fabsf(dx) - half_size, 0.0f);
  const float nearest_y = fmaxf(fabsf(dy) - half_size, 0.0f);
  return nearest_x * nearest_x + nearest_y * nearest_y <= radius * radius;
}

__global__ void kernel_count_population_cells_from_positions(const float *__restrict__ pos_x,
                                                             const float *__restrict__ pos_y,
                                                             const uint32_t *__restrict__ alive,
                                                             int *__restrict__ cell_counts, int population_count,
                                                             int grid_cols, int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= population_count || alive[idx] == 0) {
    return;
  }

  const int cx = cell_coord(pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[cy * grid_cols + cx], 1);
}

__global__ void
kernel_scatter_population_cells_from_positions(const float *__restrict__ pos_x, const float *__restrict__ pos_y,
                                               const uint32_t *__restrict__ alive, int *__restrict__ cell_offsets,
                                               PopulationEntry *__restrict__ entries, int population_count,
                                               int grid_cols, int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= population_count || alive[idx] == 0) {
    return;
  }

  const int cx = cell_coord(pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(pos_y[idx], cell_size, grid_rows);
  const int slot = atomicAdd(&cell_offsets[cy * grid_cols + cx], 1);
  entries[slot] = PopulationEntry{static_cast<unsigned int>(idx), pos_x[idx], pos_y[idx], 0.0f};
}

__global__ void kernel_count_food_cells_from_positions(const float *__restrict__ food_pos_x,
                                                       const float *__restrict__ food_pos_y,
                                                       const uint32_t *__restrict__ food_active,
                                                       int *__restrict__ cell_counts, int food_count, int grid_cols,
                                                       int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int cx = cell_coord(food_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(food_pos_y[idx], cell_size, grid_rows);
  atomicAdd(&cell_counts[cy * grid_cols + cx], 1);
}

__global__ void kernel_scatter_food_cells_from_positions(const float *__restrict__ food_pos_x,
                                                         const float *__restrict__ food_pos_y,
                                                         const uint32_t *__restrict__ food_active,
                                                         int *__restrict__ cell_offsets,
                                                         FoodEntry *__restrict__ entries, int food_count, int grid_cols,
                                                         int grid_rows, float cell_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int cx = cell_coord(food_pos_x[idx], cell_size, grid_cols);
  const int cy = cell_coord(food_pos_y[idx], cell_size, grid_rows);
  const int slot = atomicAdd(&cell_offsets[cy * grid_cols + cx], 1);
  entries[slot] = FoodEntry{static_cast<unsigned int>(idx), food_pos_x[idx], food_pos_y[idx], 0.0f};
}

template <bool SelfIsPredator>
__global__ void
kernel_build_sensors(const float *__restrict__ self_pos_x, const float *__restrict__ self_pos_y,
                     const float *__restrict__ self_vel_x, const float *__restrict__ self_vel_y,
                     const uint32_t *__restrict__ self_alive, const float *__restrict__ self_energy,
                     const int *__restrict__ predator_cell_offsets,
                     const PopulationEntry *__restrict__ predator_entries, const int *__restrict__ prey_cell_offsets,
                     const PopulationEntry *__restrict__ prey_entries, const int *__restrict__ food_cell_offsets,
                     const FoodEntry *__restrict__ food_entries, float *__restrict__ sensor_inputs, int self_count,
                     int grid_cols, int grid_rows, float grid_cell_size, float world_width, float world_height,
                     float vision_range, float max_energy, float agent_speed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= self_count) {
    return;
  }

  float *out = sensor_inputs + idx * SENSOR_COUNT;
  for (int sensor_idx = 0; sensor_idx < SENSOR_COUNT; ++sensor_idx) {
    out[sensor_idx] = 0.0f;
  }

  if (self_alive[idx] == 0) {
    return;
  }

  const float px = self_pos_x[idx];
  const float py = self_pos_y[idx];
  const float vision_sq = vision_range * vision_range;
  const int cells_to_check = static_cast<int>(vision_range / grid_cell_size) + 1;
  const int base_cx = cell_coord(px, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(py, grid_cell_size, grid_rows);

  float predator_dx[kNearestTargetsPerType];
  float predator_dy[kNearestTargetsPerType];
  float predator_dist_sq[kNearestTargetsPerType];
  float prey_dx[kNearestTargetsPerType];
  float prey_dy[kNearestTargetsPerType];
  float prey_dist_sq[kNearestTargetsPerType];
  float food_dx[kNearestTargetsPerType];
  float food_dy[kNearestTargetsPerType];
  float food_dist_sq[kNearestTargetsPerType];

  for (int nearest_idx = 0; nearest_idx < kNearestTargetsPerType; ++nearest_idx) {
    predator_dx[nearest_idx] = 0.0f;
    predator_dy[nearest_idx] = 0.0f;
    predator_dist_sq[nearest_idx] = INFINITY;
    prey_dx[nearest_idx] = 0.0f;
    prey_dy[nearest_idx] = 0.0f;
    prey_dist_sq[nearest_idx] = INFINITY;
    food_dx[nearest_idx] = 0.0f;
    food_dy[nearest_idx] = 0.0f;
    food_dist_sq[nearest_idx] = INFINITY;
  }

  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const int cy = base_cy + dy_cell;
    if (cy < 0 || cy >= grid_rows) {
      continue;
    }
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int cx = base_cx + dx_cell;
      if (cx < 0 || cx >= grid_cols) {
        continue;
      }
      if (!cell_may_intersect_radius(cx, cy, grid_cell_size, px, py, vision_range)) {
        continue;
      }

      const int cell = cy * grid_cols + cx;
      for (int slot = predator_cell_offsets[cell]; slot < predator_cell_offsets[cell + 1]; ++slot) {
        const PopulationEntry entry = predator_entries[slot];
        if (SelfIsPredator && static_cast<int>(entry.id) == idx) {
          continue;
        }

        float dx = entry.pos_x - px;
        float dy = entry.pos_y - py;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq > vision_sq || dist_sq <= 0.0f) {
          continue;
        }

        insert_nearest_candidate(dx, dy, dist_sq, predator_dx, predator_dy, predator_dist_sq);
      }

      for (int slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1]; ++slot) {
        const PopulationEntry entry = prey_entries[slot];
        if (!SelfIsPredator && static_cast<int>(entry.id) == idx) {
          continue;
        }

        float dx = entry.pos_x - px;
        float dy = entry.pos_y - py;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq > vision_sq || dist_sq <= 0.0f) {
          continue;
        }

        insert_nearest_candidate(dx, dy, dist_sq, prey_dx, prey_dy, prey_dist_sq);
      }

      for (int slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1]; ++slot) {
        float dx = food_entries[slot].pos_x - px;
        float dy = food_entries[slot].pos_y - py;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq > vision_sq || dist_sq <= 0.0f) {
          continue;
        }

        insert_nearest_candidate(dx, dy, dist_sq, food_dx, food_dy, food_dist_sq);
      }
    }
  }

  encode_nearest_targets(predator_dx, predator_dy, predator_dist_sq, vision_range, out);
  encode_nearest_targets(prey_dx, prey_dy, prey_dist_sq, vision_range, out + kPerTypeSensorCount);
  encode_nearest_targets(food_dx, food_dy, food_dist_sq, vision_range, out + (2 * kPerTypeSensorCount));

  out[kSelfStateOffset] = clampf(self_energy[idx] / max_energy, 0.0f, 1.0f);
  if (agent_speed > 0.0f) {
    out[kSelfStateOffset + 1] = clampf(self_vel_x[idx] / agent_speed, -1.0f, 1.0f);
    out[kSelfStateOffset + 2] = clampf(self_vel_y[idx] / agent_speed, -1.0f, 1.0f);
  }

  // Boundary sensors: negative = left/top, positive = right/bottom.
  const float dist_left = px;
  const float dist_right = world_width - px;
  const float dist_top = py;
  const float dist_bottom = world_height - py;
  out[kWallSensorOffset] = encode_axis_wall_sensor(dist_left, dist_right, vision_range);
  out[kWallSensorOffset + 1] = encode_axis_wall_sensor(dist_top, dist_bottom, vision_range);
}

__global__ void kernel_update_vitals(float *__restrict__ energy, int *__restrict__ age, uint32_t *__restrict__ alive,
                                     int count, float energy_drain, int max_age) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count || alive[idx] == 0) {
    return;
  }

  age[idx] += 1;
  energy[idx] -= energy_drain;
  if (energy[idx] <= 0.0f || (max_age > 0 && age[idx] >= max_age)) {
    energy[idx] = 0.0f;
    alive[idx] = 0;
  }
}

__global__ void kernel_clamp_energy(float *__restrict__ energy, const uint32_t *__restrict__ alive, int count,
                                    float max_energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count || alive[idx] == 0) {
    return;
  }

  energy[idx] = clampf(energy[idx], 0.0f, max_energy);
}

__global__ void kernel_claim_food(const float *__restrict__ prey_pos_x, const float *__restrict__ prey_pos_y,
                                  const uint32_t *__restrict__ prey_alive, const int *__restrict__ food_cell_offsets,
                                  const FoodEntry *__restrict__ food_entries, int *__restrict__ food_consumed_by,
                                  int prey_count, int grid_cols, int grid_rows, float grid_cell_size,
                                  float pickup_range) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= prey_count || prey_alive[idx] == 0) {
    return;
  }

  const float px = prey_pos_x[idx];
  const float py = prey_pos_y[idx];
  const float range_sq = pickup_range * pickup_range;
  const int cells_to_check = static_cast<int>(pickup_range / grid_cell_size) + 1;
  const int base_cx = cell_coord(px, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(py, grid_cell_size, grid_rows);

  int best_food = -1;
  float best_dist_sq = range_sq;

  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const int cy = base_cy + dy_cell;
    if (cy < 0 || cy >= grid_rows) {
      continue;
    }
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int cx = base_cx + dx_cell;
      if (cx < 0 || cx >= grid_cols) {
        continue;
      }
      if (!cell_may_intersect_radius(cx, cy, grid_cell_size, px, py, pickup_range)) {
        continue;
      }

      const int cell = cy * grid_cols + cx;
      for (int slot = food_cell_offsets[cell]; slot < food_cell_offsets[cell + 1]; ++slot) {
        float dx = food_entries[slot].pos_x - px;
        float dy = food_entries[slot].pos_y - py;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq <= best_dist_sq) {
          best_dist_sq = dist_sq;
          best_food = static_cast<int>(food_entries[slot].id);
        }
      }
    }
  }

  if (best_food >= 0) {
    atomicMin(&food_consumed_by[best_food], idx);
  }
}

__global__ void kernel_finalize_food(float *__restrict__ prey_energy, const uint32_t *__restrict__ prey_alive,
                                     uint32_t *__restrict__ food_active, int *__restrict__ food_consumed_by,
                                     int food_count, float energy_gain_from_food) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int prey_idx = food_consumed_by[idx];
  if (prey_idx != kUnclaimed && prey_idx >= 0 && prey_alive[prey_idx] != 0) {
    food_active[idx] = 0;
    atomicAdd(&prey_energy[prey_idx], energy_gain_from_food);
  }
}

__global__ void kernel_claim_combat(const float *__restrict__ predator_pos_x, const float *__restrict__ predator_pos_y,
                                    const uint32_t *__restrict__ predator_alive,
                                    const int *__restrict__ prey_cell_offsets,
                                    const PopulationEntry *__restrict__ prey_entries, int *__restrict__ prey_claimed_by,
                                    int predator_count, int grid_cols, int grid_rows, float grid_cell_size,
                                    float interaction_range) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= predator_count || predator_alive[idx] == 0) {
    return;
  }

  const float px = predator_pos_x[idx];
  const float py = predator_pos_y[idx];
  const float range_sq = interaction_range * interaction_range;
  const int cells_to_check = static_cast<int>(interaction_range / grid_cell_size) + 1;
  const int base_cx = cell_coord(px, grid_cell_size, grid_cols);
  const int base_cy = cell_coord(py, grid_cell_size, grid_rows);

  int best_prey = -1;
  float best_dist_sq = range_sq;

  for (int dy_cell = -cells_to_check; dy_cell <= cells_to_check; ++dy_cell) {
    const int cy = base_cy + dy_cell;
    if (cy < 0 || cy >= grid_rows) {
      continue;
    }
    for (int dx_cell = -cells_to_check; dx_cell <= cells_to_check; ++dx_cell) {
      const int cx = base_cx + dx_cell;
      if (cx < 0 || cx >= grid_cols) {
        continue;
      }
      if (!cell_may_intersect_radius(cx, cy, grid_cell_size, px, py, interaction_range)) {
        continue;
      }

      const int cell = cy * grid_cols + cx;
      for (int slot = prey_cell_offsets[cell]; slot < prey_cell_offsets[cell + 1]; ++slot) {
        const PopulationEntry entry = prey_entries[slot];
        float dx = entry.pos_x - px;
        float dy = entry.pos_y - py;
        const float dist_sq = dx * dx + dy * dy;
        if (dist_sq <= best_dist_sq) {
          best_dist_sq = dist_sq;
          best_prey = static_cast<int>(entry.id);
        }
      }
    }
  }

  if (best_prey >= 0) {
    atomicMin(&prey_claimed_by[best_prey], idx);
  }
}

__global__ void kernel_finalize_combat(float *__restrict__ predator_energy, const uint32_t *__restrict__ predator_alive,
                                       uint32_t *__restrict__ predator_kill_counts, uint32_t *__restrict__ prey_alive,
                                       const int *__restrict__ prey_claimed_by, int prey_count,
                                       float energy_gain_from_kill) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= prey_count || prey_alive[idx] == 0) {
    return;
  }

  const int killer_idx = prey_claimed_by[idx];
  if (killer_idx != kUnclaimed && killer_idx >= 0 && predator_alive[killer_idx] != 0) {
    prey_alive[idx] = 0;
    atomicAdd(&predator_energy[killer_idx], energy_gain_from_kill);
    atomicAdd(&predator_kill_counts[killer_idx], 1U);
  }
}

__global__ void kernel_apply_movement(float *__restrict__ pos_x, float *__restrict__ pos_y, float *__restrict__ vel_x,
                                      float *__restrict__ vel_y, const uint32_t *__restrict__ alive,
                                      const float *__restrict__ brain_outputs, int count, float world_width,
                                      float world_height, float agent_speed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count || alive[idx] == 0) {
    return;
  }

  float dx = brain_outputs[idx * 2];
  float dy = brain_outputs[idx * 2 + 1];
  float len = sqrtf(dx * dx + dy * dy);
  if (len > 1e-6f) {
    dx /= len;
    dy /= len;
  } else {
    dx = 0.0f;
    dy = 0.0f;
  }

  vel_x[idx] = dx * agent_speed;
  vel_y[idx] = dy * agent_speed;

  pos_x[idx] += vel_x[idx];
  pos_y[idx] += vel_y[idx];

  if (pos_x[idx] < 0.0f) {
    pos_x[idx] = 0.0f;
    vel_x[idx] = 0.0f;
  } else if (pos_x[idx] >= world_width) {
    pos_x[idx] = world_width;
    vel_x[idx] = 0.0f;
  }
  if (pos_y[idx] < 0.0f) {
    pos_y[idx] = 0.0f;
    vel_y[idx] = 0.0f;
  } else if (pos_y[idx] >= world_height) {
    pos_y[idx] = world_height;
    vel_y[idx] = 0.0f;
  }
}

} // namespace

Batch::Batch() : Batch(0, 0, 0) {}

Batch::Batch(std::size_t max_predators, std::size_t max_prey, std::size_t max_food)
    : predator_buffer_(max_predators), prey_buffer_(max_prey), food_buffer_(max_food) {
  init_cuda_resources();
}

Batch::~Batch() {
  cleanup_cuda_resources();
}

void Batch::init_cuda_resources() {
  cudaStream_t stream = nullptr;
  const cudaError_t err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    spdlog::error("CUDA stream creation failed: {}", cudaGetErrorString(err));
    had_error_ = true;
    return;
  }
  stream_ = stream;
}

void Batch::cleanup_cuda_resources() {
  free_spatial_grid_buffers();
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

void Batch::free_spatial_grid_buffers() {
  if (d_predator_cell_counts_)
    cudaFree(d_predator_cell_counts_);
  if (d_predator_cell_offsets_)
    cudaFree(d_predator_cell_offsets_);
  if (d_predator_cell_write_offsets_)
    cudaFree(d_predator_cell_write_offsets_);
  if (d_predator_grid_entries_)
    cudaFree(d_predator_grid_entries_);
  if (d_prey_cell_counts_)
    cudaFree(d_prey_cell_counts_);
  if (d_prey_cell_offsets_)
    cudaFree(d_prey_cell_offsets_);
  if (d_prey_cell_write_offsets_)
    cudaFree(d_prey_cell_write_offsets_);
  if (d_prey_grid_entries_)
    cudaFree(d_prey_grid_entries_);
  if (d_food_cell_counts_)
    cudaFree(d_food_cell_counts_);
  if (d_food_cell_offsets_)
    cudaFree(d_food_cell_offsets_);
  if (d_food_cell_write_offsets_)
    cudaFree(d_food_cell_write_offsets_);
  if (d_food_grid_entries_)
    cudaFree(d_food_grid_entries_);

  d_predator_cell_counts_ = nullptr;
  d_predator_cell_offsets_ = nullptr;
  d_predator_cell_write_offsets_ = nullptr;
  d_predator_grid_entries_ = nullptr;
  d_prey_cell_counts_ = nullptr;
  d_prey_cell_offsets_ = nullptr;
  d_prey_cell_write_offsets_ = nullptr;
  d_prey_grid_entries_ = nullptr;
  d_food_cell_counts_ = nullptr;
  d_food_cell_offsets_ = nullptr;
  d_food_cell_write_offsets_ = nullptr;
  d_food_grid_entries_ = nullptr;
  grid_cell_capacity_ = 0;
}

void Batch::ensure_spatial_grid_capacity(std::size_t cell_count) {
  if (cell_count <= grid_cell_capacity_) {
    return;
  }

  free_spatial_grid_buffers();

  CUDA_CHECK(cudaMalloc(&d_predator_cell_counts_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_predator_cell_offsets_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_predator_cell_write_offsets_, cell_count * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_predator_grid_entries_, predator_buffer_.capacity() * sizeof(PopulationEntry)));

  CUDA_CHECK(cudaMalloc(&d_prey_cell_counts_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prey_cell_offsets_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prey_cell_write_offsets_, cell_count * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prey_grid_entries_, prey_buffer_.capacity() * sizeof(PopulationEntry)));

  CUDA_CHECK(cudaMalloc(&d_food_cell_counts_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_cell_offsets_, (cell_count + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_cell_write_offsets_, cell_count * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_food_grid_entries_, food_buffer_.capacity() * sizeof(FoodEntry)));

  grid_cell_capacity_ = cell_count;
}

void Batch::ensure_capacity(std::size_t predator_count, std::size_t prey_count, std::size_t food_count) {
  const bool predators_exceeded = predator_count > predator_buffer_.capacity();
  const bool prey_exceeded = prey_count > prey_buffer_.capacity();
  const bool food_exceeded = food_count > food_buffer_.capacity();

  if (!predators_exceeded && !prey_exceeded && !food_exceeded) {
    return;
  }

  const std::size_t new_predator_capacity =
      predators_exceeded ? next_power_of_2(std::max(predator_count, predator_buffer_.capacity() * 2))
                         : predator_buffer_.capacity();
  const std::size_t new_prey_capacity =
      prey_exceeded ? next_power_of_2(std::max(prey_count, prey_buffer_.capacity() * 2)) : prey_buffer_.capacity();
  const std::size_t new_food_capacity =
      food_exceeded ? next_power_of_2(std::max(food_count, food_buffer_.capacity() * 2)) : food_buffer_.capacity();

  predator_buffer_.reset(new_predator_capacity);
  prey_buffer_.reset(new_prey_capacity);
  food_buffer_.reset(new_food_capacity);
  free_spatial_grid_buffers();
}

void Batch::upload_async(std::size_t predator_count, std::size_t prey_count, std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("upload");

  const cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  predator_buffer_.upload_async(predator_count, stream);
  prey_buffer_.upload_async(prey_count, stream);
  food_buffer_.upload_async(food_count, stream);
}

void Batch::download_async(std::size_t predator_count, std::size_t prey_count, std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("download_enqueue");

  const cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  predator_buffer_.download_async(predator_count, stream);
  prey_buffer_.download_async(prey_count, stream);
  food_buffer_.download_async(food_count, stream);
}

void Batch::launch_build_sensors_async(const StepParams &params, std::size_t predator_count, std::size_t prey_count,
                                       std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("launch_sensors");

  grid_cell_size_ = std::max(params.vision_range, 1.0f);
  grid_cols_ = std::max(1, static_cast<int>(std::ceil(params.world_width / grid_cell_size_)));
  grid_rows_ = std::max(1, static_cast<int>(std::ceil(params.world_height / grid_cell_size_)));
  const std::size_t cell_count = static_cast<std::size_t>(grid_cols_) * static_cast<std::size_t>(grid_rows_);
  ensure_spatial_grid_capacity(cell_count);

  const cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  const int predator_blocks = (static_cast<int>(predator_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int prey_blocks = (static_cast<int>(prey_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int food_blocks = (static_cast<int>(food_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;

  {
    MOONAI_PROFILE_SCOPE("build_grid", stream);
    CUDA_CHECK(cudaMemsetAsync(d_predator_cell_counts_, 0, (cell_count + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_prey_cell_counts_, 0, (cell_count + 1) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_food_cell_counts_, 0, (cell_count + 1) * sizeof(int), stream));

    {
      MOONAI_PROFILE_SCOPE("grid_count", stream);
      if (predator_count > 0) {
        kernel_count_population_cells_from_positions<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
            predator_buffer_.device_positions_x(), predator_buffer_.device_positions_y(),
            predator_buffer_.device_alive(), d_predator_cell_counts_, static_cast<int>(predator_count), grid_cols_,
            grid_rows_, grid_cell_size_);
      }
      if (prey_count > 0) {
        kernel_count_population_cells_from_positions<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
            prey_buffer_.device_positions_x(), prey_buffer_.device_positions_y(), prey_buffer_.device_alive(),
            d_prey_cell_counts_, static_cast<int>(prey_count), grid_cols_, grid_rows_, grid_cell_size_);
      }
      if (food_count > 0) {
        kernel_count_food_cells_from_positions<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
            food_buffer_.device_positions_x(), food_buffer_.device_positions_y(), food_buffer_.device_active(),
            d_food_cell_counts_, static_cast<int>(food_count), grid_cols_, grid_rows_, grid_cell_size_);
      }
    }

    {
      MOONAI_PROFILE_SCOPE("grid_scan", stream);
      thrust::exclusive_scan(thrust::cuda::par.on(stream), d_predator_cell_counts_,
                             d_predator_cell_counts_ + cell_count + 1, d_predator_cell_offsets_);
      thrust::exclusive_scan(thrust::cuda::par.on(stream), d_prey_cell_counts_, d_prey_cell_counts_ + cell_count + 1,
                             d_prey_cell_offsets_);
      thrust::exclusive_scan(thrust::cuda::par.on(stream), d_food_cell_counts_, d_food_cell_counts_ + cell_count + 1,
                             d_food_cell_offsets_);

      CUDA_CHECK(cudaMemcpyAsync(d_predator_cell_write_offsets_, d_predator_cell_offsets_, cell_count * sizeof(int),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_prey_cell_write_offsets_, d_prey_cell_offsets_, cell_count * sizeof(int),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_food_cell_write_offsets_, d_food_cell_offsets_, cell_count * sizeof(int),
                                 cudaMemcpyDeviceToDevice, stream));
    }

    {
      MOONAI_PROFILE_SCOPE("grid_scatter", stream);
      if (predator_count > 0) {
        kernel_scatter_population_cells_from_positions<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
            predator_buffer_.device_positions_x(), predator_buffer_.device_positions_y(),
            predator_buffer_.device_alive(), d_predator_cell_write_offsets_, d_predator_grid_entries_,
            static_cast<int>(predator_count), grid_cols_, grid_rows_, grid_cell_size_);
      }
      if (prey_count > 0) {
        kernel_scatter_population_cells_from_positions<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
            prey_buffer_.device_positions_x(), prey_buffer_.device_positions_y(), prey_buffer_.device_alive(),
            d_prey_cell_write_offsets_, d_prey_grid_entries_, static_cast<int>(prey_count), grid_cols_, grid_rows_,
            grid_cell_size_);
      }
      if (food_count > 0) {
        kernel_scatter_food_cells_from_positions<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
            food_buffer_.device_positions_x(), food_buffer_.device_positions_y(), food_buffer_.device_active(),
            d_food_cell_write_offsets_, d_food_grid_entries_, static_cast<int>(food_count), grid_cols_, grid_rows_,
            grid_cell_size_);
      }
    }
  }

  {
    MOONAI_PROFILE_SCOPE("sensor_kernel", stream);
    if (predator_count > 0) {
      kernel_build_sensors<true><<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
          predator_buffer_.device_positions_x(), predator_buffer_.device_positions_y(),
          predator_buffer_.device_velocities_x(), predator_buffer_.device_velocities_y(),
          predator_buffer_.device_alive(), predator_buffer_.device_energy(), d_predator_cell_offsets_,
          d_predator_grid_entries_, d_prey_cell_offsets_, d_prey_grid_entries_, d_food_cell_offsets_,
          d_food_grid_entries_, predator_buffer_.device_sensor_inputs(), static_cast<int>(predator_count), grid_cols_,
          grid_rows_, grid_cell_size_, params.world_width, params.world_height, params.vision_range, params.max_energy,
          params.predator_speed);
    }
    if (prey_count > 0) {
      kernel_build_sensors<false><<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
          prey_buffer_.device_positions_x(), prey_buffer_.device_positions_y(), prey_buffer_.device_velocities_x(),
          prey_buffer_.device_velocities_y(), prey_buffer_.device_alive(), prey_buffer_.device_energy(),
          d_predator_cell_offsets_, d_predator_grid_entries_, d_prey_cell_offsets_, d_prey_grid_entries_,
          d_food_cell_offsets_, d_food_grid_entries_, prey_buffer_.device_sensor_inputs(), static_cast<int>(prey_count),
          grid_cols_, grid_rows_, grid_cell_size_, params.world_width, params.world_height, params.vision_range,
          params.max_energy, params.prey_speed);
    }
  }

  check_launch_error();
}

void Batch::launch_post_inference_async(const StepParams &params, std::size_t predator_count, std::size_t prey_count,
                                        std::size_t food_count) {
  MOONAI_PROFILE_SCOPE("post_inference");

  const cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  const int predator_blocks = (static_cast<int>(predator_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int prey_blocks = (static_cast<int>(prey_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int food_blocks = (static_cast<int>(food_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;

  if (predator_count > 0) {
    kernel_update_vitals<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
        predator_buffer_.device_energy(), predator_buffer_.device_age(), predator_buffer_.device_alive(),
        static_cast<int>(predator_count), params.energy_drain_per_step, params.max_age);
    CUDA_CHECK(cudaMemsetAsync(predator_buffer_.device_kill_counts(), 0, predator_count * sizeof(uint32_t), stream));
  }
  if (prey_count > 0) {
    kernel_update_vitals<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
        prey_buffer_.device_energy(), prey_buffer_.device_age(), prey_buffer_.device_alive(),
        static_cast<int>(prey_count), params.energy_drain_per_step, params.max_age);
    CUDA_CHECK(cudaMemsetAsync(prey_buffer_.device_claimed_by(), 0x7f, prey_count * sizeof(int), stream));
  }
  if (food_count > 0) {
    CUDA_CHECK(cudaMemsetAsync(food_buffer_.device_consumed_by(), 0x7f, food_count * sizeof(int), stream));
  }

  if (prey_count > 0 && food_count > 0) {
    kernel_claim_food<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
        prey_buffer_.device_positions_x(), prey_buffer_.device_positions_y(), prey_buffer_.device_alive(),
        d_food_cell_offsets_, d_food_grid_entries_, food_buffer_.device_consumed_by(), static_cast<int>(prey_count),
        grid_cols_, grid_rows_, grid_cell_size_, params.interaction_range);
    kernel_finalize_food<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
        prey_buffer_.device_energy(), prey_buffer_.device_alive(), food_buffer_.device_active(),
        food_buffer_.device_consumed_by(), static_cast<int>(food_count), params.energy_gain_from_food);
  }

  if (predator_count > 0 && prey_count > 0) {
    kernel_claim_combat<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
        predator_buffer_.device_positions_x(), predator_buffer_.device_positions_y(), predator_buffer_.device_alive(),
        d_prey_cell_offsets_, d_prey_grid_entries_, prey_buffer_.device_claimed_by(), static_cast<int>(predator_count),
        grid_cols_, grid_rows_, grid_cell_size_, params.interaction_range);
    kernel_finalize_combat<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
        predator_buffer_.device_energy(), predator_buffer_.device_alive(), predator_buffer_.device_kill_counts(),
        prey_buffer_.device_alive(), prey_buffer_.device_claimed_by(), static_cast<int>(prey_count),
        params.energy_gain_from_kill);
  }

  if (predator_count > 0) {
    kernel_clamp_energy<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
        predator_buffer_.device_energy(), predator_buffer_.device_alive(), static_cast<int>(predator_count),
        params.max_energy);
  }
  if (prey_count > 0) {
    kernel_clamp_energy<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
        prey_buffer_.device_energy(), prey_buffer_.device_alive(), static_cast<int>(prey_count), params.max_energy);
  }

  if (predator_count > 0) {
    kernel_apply_movement<<<predator_blocks, kThreadsPerBlock, 0, stream>>>(
        predator_buffer_.device_positions_x(), predator_buffer_.device_positions_y(),
        predator_buffer_.device_velocities_x(), predator_buffer_.device_velocities_y(), predator_buffer_.device_alive(),
        predator_buffer_.device_brain_outputs(), static_cast<int>(predator_count), params.world_width,
        params.world_height, params.predator_speed);
  }
  if (prey_count > 0) {
    kernel_apply_movement<<<prey_blocks, kThreadsPerBlock, 0, stream>>>(
        prey_buffer_.device_positions_x(), prey_buffer_.device_positions_y(), prey_buffer_.device_velocities_x(),
        prey_buffer_.device_velocities_y(), prey_buffer_.device_alive(), prey_buffer_.device_brain_outputs(),
        static_cast<int>(prey_count), params.world_width, params.world_height, params.prey_speed);
  }

  check_launch_error();
}

void Batch::synchronize() {
  MOONAI_PROFILE_SCOPE("synchronize");
  const cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  if (err != cudaSuccess) {
    spdlog::error("CUDA synchronize failed: {}", cudaGetErrorString(err));
    had_error_ = true;
  }
}

void Batch::mark_error() {
  had_error_ = true;
}

void Batch::check_launch_error() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("Kernel launch failed: {}", cudaGetErrorString(err));
    mark_error();
  }
}

} // namespace moonai::simulation
