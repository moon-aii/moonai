#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch_ecs.hpp"
#include <algorithm>
#include <cmath>

namespace moonai {
namespace gpu {

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr float kPi = 3.14159265f;
constexpr float kMaxDensity = 10.0f;

__device__ float clampf(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

__device__ void apply_wrap(float &dx, float &dy, float world_width,
                           float world_height, bool has_walls) {
  if (has_walls) {
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

__device__ float normalize_angle(float dx, float dy) {
  return atan2f(dy, dx) / kPi;
}
}

// Kernel: Build sensor inputs from agent data
__global__ void kernel_build_sensors(
    const float *__restrict__ pos_x, const float *__restrict__ pos_y,
    const float *__restrict__ vel_x, const float *__restrict__ vel_y,
    const float *__restrict__ speed, const uint8_t *__restrict__ types,
    const uint32_t *__restrict__ alive, const float *__restrict__ energy,
    float *__restrict__ sensor_inputs, int count, float world_width,
    float world_height, float vision_range, float max_energy, bool has_walls) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  float *out = sensor_inputs + idx * 15;

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

  if (alive[idx] == 0 || types[idx] == 2) {
    return;
  }

  const float self_x = pos_x[idx];
  const float self_y = pos_y[idx];
  const float vision_sq = vision_range * vision_range;
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

  for (int other = 0; other < count; ++other) {
    if (other == idx || alive[other] == 0) {
      continue;
    }

    float dx = pos_x[other] - self_x;
    float dy = pos_y[other] - self_y;
    apply_wrap(dx, dy, world_width, world_height, has_walls);
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq > vision_sq) {
      continue;
    }

    if (types[other] == 0) {
      ++local_predators;
      if (dist_sq < nearest_pred_dist_sq) {
        nearest_pred_dist_sq = dist_sq;
        pred_dx = dx;
        pred_dy = dy;
      }
    } else if (types[other] == 1) {
      ++local_prey;
      if (dist_sq < nearest_prey_dist_sq) {
        nearest_prey_dist_sq = dist_sq;
        prey_dx = dx;
        prey_dy = dy;
      }
    } else if (types[idx] == 1 && types[other] == 2 &&
               dist_sq < nearest_food_dist_sq) {
      nearest_food_dist_sq = dist_sq;
      food_dx = dx;
      food_dy = dy;
    }
  }

  if (nearest_pred_dist_sq < INFINITY) {
    out[0] = sqrtf(nearest_pred_dist_sq) / vision_range;
    out[1] = normalize_angle(pred_dx, pred_dy);
  }
  if (nearest_prey_dist_sq < INFINITY) {
    out[2] = sqrtf(nearest_prey_dist_sq) / vision_range;
    out[3] = normalize_angle(prey_dx, prey_dy);
  }
  if (nearest_food_dist_sq < INFINITY) {
    out[4] = sqrtf(nearest_food_dist_sq) / vision_range;
    out[5] = normalize_angle(food_dx, food_dy);
  }

  out[6] = clampf(energy[idx] / (max_energy * 2.0f), 0.0f, 1.0f);
  if (speed[idx] > 0.0f) {
    out[7] = clampf(vel_x[idx] / speed[idx], -1.0f, 1.0f);
    out[8] = clampf(vel_y[idx] / speed[idx], -1.0f, 1.0f);
  }
  out[9] = clampf(static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
  out[10] = clampf(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);

  if (has_walls) {
    out[11] = clampf(self_x / vision_range, 0.0f, 1.0f);
    out[12] = clampf((world_width - self_x) / vision_range, 0.0f, 1.0f);
    out[13] = clampf(self_y / vision_range, 0.0f, 1.0f);
    out[14] = clampf((world_height - self_y) / vision_range, 0.0f, 1.0f);
  }
}

// Kernel: Apply movement from brain outputs
__global__ void kernel_update_vitals(float *__restrict__ energy,
                                     int *__restrict__ age,
                                     int *__restrict__ reproduction_cooldown,
                                     uint32_t *__restrict__ alive,
                                     const uint8_t *__restrict__ types,
                                     int count, float energy_drain, int max_age,
                                     float max_energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count || !alive[idx] || types[idx] == 2) {
    return;
  }

  age[idx] += 1;
  if (reproduction_cooldown[idx] > 0) {
    reproduction_cooldown[idx] -= 1;
  }

  energy[idx] -= energy_drain;
  const bool died_of_starvation = energy[idx] <= 0.0f;
  const bool died_of_age = max_age > 0 && age[idx] >= max_age;
  if (died_of_starvation || died_of_age) {
    energy[idx] = 0.0f;
    alive[idx] = 0;
  } else if (energy[idx] > max_energy) {
    energy[idx] = max_energy;
  }
}

__global__ void
kernel_apply_movement(float *__restrict__ pos_x, float *__restrict__ pos_y,
                      float *__restrict__ vel_x, float *__restrict__ vel_y,
                      const float *__restrict__ speed,
                      uint32_t *__restrict__ alive,
                      const uint8_t *__restrict__ types,
                      float *__restrict__ distance_traveled,
                      const float *__restrict__ brain_outputs, int count,
                      float world_width, float world_height, bool has_walls) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  if (!alive[idx])
    return;

  // Skip food entities (type 2) - they don't move or consume energy
  if (types[idx] == 2)
    return;

  float dx = brain_outputs[idx * 2 + 0];
  float dy = brain_outputs[idx * 2 + 1];

  float len = sqrtf(dx * dx + dy * dy);
  if (len > 1e-6f) {
    dx /= len;
    dy /= len;
  } else {
    dx = 0.0f;
    dy = 0.0f;
  }

  vel_x[idx] = dx * speed[idx];
  vel_y[idx] = dy * speed[idx];

  const float old_x = pos_x[idx];
  const float old_y = pos_y[idx];

  pos_x[idx] += vel_x[idx];
  pos_y[idx] += vel_y[idx];

  // Boundary handling
  if (has_walls) {
    if (pos_x[idx] < 0.0f) {
      pos_x[idx] = 0.0f;
      vel_x[idx] = 0.0f;
    } else if (pos_x[idx] > world_width) {
      pos_x[idx] = world_width;
      vel_x[idx] = 0.0f;
    }

    if (pos_y[idx] < 0.0f) {
      pos_y[idx] = 0.0f;
      vel_y[idx] = 0.0f;
    } else if (pos_y[idx] > world_height) {
      pos_y[idx] = world_height;
      vel_y[idx] = 0.0f;
    }
  } else {
    while (pos_x[idx] < 0.0f)
      pos_x[idx] += world_width;
    while (pos_x[idx] >= world_width)
      pos_x[idx] -= world_width;
    while (pos_y[idx] < 0.0f)
      pos_y[idx] += world_height;
    while (pos_y[idx] >= world_height)
      pos_y[idx] -= world_height;
  }

  float dx_pos = pos_x[idx] - old_x;
  float dy_pos = pos_y[idx] - old_y;
  apply_wrap(dx_pos, dy_pos, world_width, world_height, has_walls);
  distance_traveled[idx] += sqrtf(dx_pos * dx_pos + dy_pos * dy_pos);

}

// Kernel: Process combat (predator attacks)
__global__ void kernel_process_combat(const float *__restrict__ pos_x,
                                      const float *__restrict__ pos_y,
                                      const uint8_t *__restrict__ types,
                                      float *__restrict__ energy,
                                      uint32_t *__restrict__ alive,
                                      uint32_t *__restrict__ kill_counts,
                                      int *__restrict__ killed_by,
                                      float attack_range, float energy_gain,
                                      int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  // Only predators can attack
  if (types[idx] != 0 || !alive[idx])
    return; // 0 = Predator

  float px = pos_x[idx];
  float py = pos_y[idx];
  float range_sq = attack_range * attack_range;

  // Find nearby prey
  for (int j = 0; j < count; ++j) {
    if (idx == j || types[j] != 1 || !alive[j])
      continue; // 1 = Prey

    float dx = pos_x[j] - px;
    float dy = pos_y[j] - py;
    float dist_sq = dx * dx + dy * dy;

    if (dist_sq < range_sq) {
      if (atomicCAS(&alive[j], 1U, 0U) == 1U) {
        atomicAdd(&energy[idx], energy_gain);
        atomicAdd(&kill_counts[idx], 1U);
        killed_by[j] = idx;
        break;
      }
    }
  }
}

GpuBatchECS::GpuBatchECS(std::size_t max_entities) : buffer_(max_entities) {
  init_cuda_resources();
  mapping_.resize(max_entities);
}

GpuBatchECS::~GpuBatchECS() {
  cleanup_cuda_resources();
}

void GpuBatchECS::init_cuda_resources() {
  const cudaError_t err =
      cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_));
  if (err != cudaSuccess) {
    had_error_ = true;
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
  }
}

void GpuBatchECS::cleanup_cuda_resources() {
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

void GpuBatchECS::upload_async(std::size_t agent_count) {
  buffer_.upload_async(agent_count, static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatchECS::download_async(std::size_t agent_count) {
  buffer_.download_async(agent_count, static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatchECS::launch_full_step_async(const GpuStepParams &params,
                                         std::size_t agent_count) {
  if (agent_count == 0 || had_error_)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  // 1. Build sensor inputs
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_velocities_x(), buffer_.device_velocities_y(),
      buffer_.device_speed(), buffer_.device_types(), buffer_.device_alive(),
      buffer_.device_energy(), buffer_.device_sensor_inputs(),
      static_cast<int>(agent_count), params.world_width, params.world_height,
      params.vision_range, params.max_energy, params.has_walls);
  check_launch_error();

  CUDA_CHECK(cudaMemsetAsync(buffer_.device_kill_counts(), 0,
                             agent_count * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_killed_by(), 0xFF,
                             agent_count * sizeof(int), stream));

  kernel_update_vitals<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_energy(), buffer_.device_age(),
      buffer_.device_reproduction_cooldown(), buffer_.device_alive(),
      buffer_.device_types(), static_cast<int>(agent_count),
      params.energy_drain_per_step, params.max_age, params.max_energy);
  check_launch_error();

  kernel_apply_movement<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_velocities_x(), buffer_.device_velocities_y(),
      buffer_.device_speed(), buffer_.device_alive(), buffer_.device_types(),
      buffer_.device_distance_traveled(), buffer_.device_brain_outputs(),
      static_cast<int>(agent_count), params.world_width, params.world_height,
      params.has_walls);
  check_launch_error();

  kernel_process_combat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_types(), buffer_.device_energy(),
      buffer_.device_alive(), buffer_.device_kill_counts(),
      buffer_.device_killed_by(), params.attack_range,
      params.energy_gain_from_kill, static_cast<int>(agent_count));
  check_launch_error();
}

void GpuBatchECS::synchronize() {
  const cudaError_t err =
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    had_error_ = true;
  }
}

void GpuBatchECS::launch_build_sensors_async(const GpuStepParams &params,
                                               std::size_t agent_count) {
  if (agent_count == 0 || had_error_)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_velocities_x(), buffer_.device_velocities_y(),
      buffer_.device_speed(), buffer_.device_types(), buffer_.device_alive(),
      buffer_.device_energy(), buffer_.device_sensor_inputs(),
      static_cast<int>(agent_count), params.world_width, params.world_height,
      params.vision_range, params.max_energy, params.has_walls);
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_brain_outputs(), 0,
                             agent_count * 2 * sizeof(float), stream));
  check_launch_error();
}

void GpuBatchECS::launch_apply_movement_async(const GpuStepParams &params,
                                                std::size_t agent_count) {
  if (agent_count == 0 || had_error_)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  kernel_apply_movement<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_velocities_x(), buffer_.device_velocities_y(),
      buffer_.device_speed(), buffer_.device_alive(), buffer_.device_types(),
      buffer_.device_distance_traveled(), buffer_.device_brain_outputs(),
      static_cast<int>(agent_count), params.world_width, params.world_height,
      params.has_walls);
  check_launch_error();
}

void GpuBatchECS::launch_update_vitals_async(const GpuStepParams &params,
                                             std::size_t agent_count) {
  if (agent_count == 0 || had_error_)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  kernel_update_vitals<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_energy(), buffer_.device_age(),
      buffer_.device_reproduction_cooldown(), buffer_.device_alive(),
      buffer_.device_types(), static_cast<int>(agent_count),
      params.energy_drain_per_step, params.max_age, params.max_energy);
  check_launch_error();
}

void GpuBatchECS::launch_process_combat_async(const GpuStepParams &params,
                                                std::size_t agent_count) {
  if (agent_count == 0 || had_error_)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  CUDA_CHECK(cudaMemsetAsync(buffer_.device_kill_counts(), 0,
                             agent_count * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_killed_by(), 0xFF,
                             agent_count * sizeof(int), stream));
  kernel_process_combat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_types(), buffer_.device_energy(), buffer_.device_alive(),
      buffer_.device_kill_counts(), buffer_.device_killed_by(),
      params.attack_range,
      params.energy_gain_from_kill, static_cast<int>(agent_count));
  check_launch_error();
}

void GpuBatchECS::mark_error() {
  had_error_ = true;
}

void GpuBatchECS::check_launch_error() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,
            cudaGetErrorString(err));
    mark_error();
  }
}

// Free function implementations
void launch_build_sensors_kernel(const float *d_pos_x, const float *d_pos_y,
                                 const float *d_vel_x, const float *d_vel_y,
                                 const float *d_speed, const uint8_t *d_types,
                                 const uint32_t *d_alive,
                                 const float *d_energy,
                                 float *d_sensor_inputs, std::size_t count,
                                 float world_width, float world_height,
                                 float vision_range, float max_energy,
                                 bool has_walls, cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_speed, d_types, d_alive, d_energy,
      d_sensor_inputs, static_cast<int>(count), world_width, world_height,
      vision_range, max_energy, has_walls);
}

void launch_apply_movement_kernel(
    float *d_pos_x, float *d_pos_y, float *d_vel_x, float *d_vel_y,
    const float *d_speed, uint32_t *d_alive, const uint8_t *d_types,
    float *d_distance_traveled, const float *d_brain_outputs,
    std::size_t count, float world_width, float world_height, bool has_walls,
    cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_apply_movement<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_speed, d_alive, d_types,
      d_distance_traveled, d_brain_outputs, static_cast<int>(count),
      world_width, world_height, has_walls);
}

void launch_update_vitals_kernel(float *d_energy, int *d_age,
                                 int *d_reproduction_cooldown,
                                 uint32_t *d_alive, const uint8_t *d_types,
                                 std::size_t count, float energy_drain,
                                 int max_age,
                                 float max_energy, cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_update_vitals<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_energy, d_age, d_reproduction_cooldown, d_alive, d_types,
      static_cast<int>(count), energy_drain, max_age, max_energy);
}

void launch_process_combat_kernel(const float *d_pos_x, const float *d_pos_y,
                                  const uint8_t *d_types, float *d_energy,
                                  uint32_t *d_alive, uint32_t *d_kill_counts,
                                  int *d_killed_by, float attack_range,
                                  float energy_gain, std::size_t count,
                                  cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_process_combat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_types, d_energy, d_alive, d_kill_counts,
      d_killed_by, attack_range, energy_gain, static_cast<int>(count));
}

} // namespace gpu
} // namespace moonai
