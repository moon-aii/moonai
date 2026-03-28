#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch_ecs.hpp"

#include <spdlog/spdlog.h>
#include <cmath>
#include <limits>

namespace moonai {
namespace gpu {

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr float kPi = 3.14159265f;
constexpr float kMaxDensity = 10.0f;
constexpr int kUnclaimed = 0x7f7f7f7f;

__device__ float clampf(float value, float min_value, float max_value) {
  return fminf(fmaxf(value, min_value), max_value);
}

__device__ void apply_wrap(float &dx, float &dy, float world_width,
                           float world_height) {
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

__global__ void kernel_build_sensors(
    const float *__restrict__ agent_pos_x, const float *__restrict__ agent_pos_y,
    const float *__restrict__ agent_vel_x, const float *__restrict__ agent_vel_y,
    const float *__restrict__ agent_speed,
    const uint8_t *__restrict__ agent_types,
    const uint32_t *__restrict__ agent_alive,
    const float *__restrict__ agent_energy,
    const float *__restrict__ food_pos_x, const float *__restrict__ food_pos_y,
    const uint32_t *__restrict__ food_active,
    float *__restrict__ sensor_inputs, int agent_count, int food_count,
    float world_width, float world_height, float vision_range,
    float max_energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count) {
    return;
  }

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

  if (agent_alive[idx] == 0) {
    return;
  }

  const float self_x = agent_pos_x[idx];
  const float self_y = agent_pos_y[idx];
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

  for (int other = 0; other < agent_count; ++other) {
    if (other == idx || agent_alive[other] == 0) {
      continue;
    }

    float dx = agent_pos_x[other] - self_x;
    float dy = agent_pos_y[other] - self_y;
    apply_wrap(dx, dy, world_width, world_height);
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq > vision_sq) {
      continue;
    }

    if (agent_types[other] == 0) {
      ++local_predators;
      if (dist_sq < nearest_pred_dist_sq) {
        nearest_pred_dist_sq = dist_sq;
        pred_dx = dx;
        pred_dy = dy;
      }
    } else if (agent_types[other] == 1) {
      ++local_prey;
      if (dist_sq < nearest_prey_dist_sq) {
        nearest_prey_dist_sq = dist_sq;
        prey_dx = dx;
        prey_dy = dy;
      }
    }
  }

  if (agent_types[idx] == 1) {
    for (int food_idx = 0; food_idx < food_count; ++food_idx) {
      if (food_active[food_idx] == 0) {
        continue;
      }

      float dx = food_pos_x[food_idx] - self_x;
      float dy = food_pos_y[food_idx] - self_y;
      apply_wrap(dx, dy, world_width, world_height);
      const float dist_sq = dx * dx + dy * dy;
      if (dist_sq <= vision_sq && dist_sq < nearest_food_dist_sq) {
        nearest_food_dist_sq = dist_sq;
        food_dx = dx;
        food_dy = dy;
      }
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

  out[6] = clampf(agent_energy[idx] / (max_energy * 2.0f), 0.0f, 1.0f);
  if (agent_speed[idx] > 0.0f) {
    out[7] = clampf(agent_vel_x[idx] / agent_speed[idx], -1.0f, 1.0f);
    out[8] = clampf(agent_vel_y[idx] / agent_speed[idx], -1.0f, 1.0f);
  }
  out[9] = clampf(static_cast<float>(local_predators) / kMaxDensity, 0.0f, 1.0f);
  out[10] = clampf(static_cast<float>(local_prey) / kMaxDensity, 0.0f, 1.0f);
}

__global__ void kernel_update_vitals(float *__restrict__ agent_energy,
                                     int *__restrict__ agent_age,
                                     int *__restrict__ agent_cooldown,
                                     uint32_t *__restrict__ agent_alive,
                                     int agent_count, float energy_drain,
                                     int max_age) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  agent_age[idx] += 1;
  if (agent_cooldown[idx] > 0) {
    agent_cooldown[idx] -= 1;
  }

  agent_energy[idx] -= energy_drain;
  if (agent_energy[idx] <= 0.0f || (max_age > 0 && agent_age[idx] >= max_age)) {
    agent_energy[idx] = 0.0f;
    agent_alive[idx] = 0;
  }
}

__global__ void kernel_claim_food(const float *__restrict__ agent_pos_x,
                                  const float *__restrict__ agent_pos_y,
                                  const uint8_t *__restrict__ agent_types,
                                  const uint32_t *__restrict__ agent_alive,
                                  const float *__restrict__ food_pos_x,
                                  const float *__restrict__ food_pos_y,
                                  const uint32_t *__restrict__ food_active,
                                  int *__restrict__ food_consumed_by,
                                  int agent_count, int food_count,
                                  float pickup_range, float world_width,
                                  float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 1) {
    return;
  }

  const float px = agent_pos_x[idx];
  const float py = agent_pos_y[idx];
  const float range_sq = pickup_range * pickup_range;

  int best_food = -1;
  float best_dist_sq = range_sq;

  for (int food_idx = 0; food_idx < food_count; ++food_idx) {
    if (food_active[food_idx] == 0) {
      continue;
    }
    float dx = food_pos_x[food_idx] - px;
    float dy = food_pos_y[food_idx] - py;
    apply_wrap(dx, dy, world_width, world_height);
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq <= best_dist_sq) {
      best_dist_sq = dist_sq;
      best_food = food_idx;
    }
  }

  if (best_food >= 0) {
    atomicMin(&food_consumed_by[best_food], idx);
  }
}

__global__ void kernel_finalize_food(float *__restrict__ agent_energy,
                                     const uint32_t *__restrict__ agent_alive,
                                     uint32_t *__restrict__ food_active,
                                     int *__restrict__ food_consumed_by,
                                     int food_count,
                                     float energy_gain_from_food) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= food_count || food_active[idx] == 0) {
    return;
  }

  const int prey_idx = food_consumed_by[idx];
  if (prey_idx != kUnclaimed && prey_idx >= 0 && agent_alive[prey_idx] != 0) {
    food_active[idx] = 0;
    atomicAdd(&agent_energy[prey_idx], energy_gain_from_food);
  }
}

__global__ void kernel_claim_combat(const float *__restrict__ agent_pos_x,
                                    const float *__restrict__ agent_pos_y,
                                    const uint8_t *__restrict__ agent_types,
                                    const uint32_t *__restrict__ agent_alive,
                                    int *__restrict__ agent_killed_by,
                                    int agent_count, float attack_range,
                                    float world_width, float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 0) {
    return;
  }

  const float px = agent_pos_x[idx];
  const float py = agent_pos_y[idx];
  const float range_sq = attack_range * attack_range;
  int best_prey = -1;
  float best_dist_sq = range_sq;

  for (int prey_idx = 0; prey_idx < agent_count; ++prey_idx) {
    if (agent_alive[prey_idx] == 0 || agent_types[prey_idx] != 1) {
      continue;
    }

    float dx = agent_pos_x[prey_idx] - px;
    float dy = agent_pos_y[prey_idx] - py;
    apply_wrap(dx, dy, world_width, world_height);
    const float dist_sq = dx * dx + dy * dy;
    if (dist_sq <= best_dist_sq) {
      best_dist_sq = dist_sq;
      best_prey = prey_idx;
    }
  }

  if (best_prey >= 0) {
    atomicMin(&agent_killed_by[best_prey], idx);
  }
}

__global__ void kernel_finalize_combat(float *__restrict__ agent_energy,
                                       uint32_t *__restrict__ agent_alive,
                                       const uint8_t *__restrict__ agent_types,
                                       uint32_t *__restrict__ agent_kill_counts,
                                       int *__restrict__ agent_killed_by,
                                       int agent_count,
                                       float energy_gain_from_kill) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0 || agent_types[idx] != 1) {
    return;
  }

  const int killer_idx = agent_killed_by[idx];
  if (killer_idx != kUnclaimed && killer_idx >= 0 && agent_alive[killer_idx] != 0) {
    agent_alive[idx] = 0;
    atomicAdd(&agent_energy[killer_idx], energy_gain_from_kill);
    atomicAdd(&agent_kill_counts[killer_idx], 1U);
  }
}

__global__ void kernel_apply_movement(
    float *__restrict__ agent_pos_x, float *__restrict__ agent_pos_y,
    float *__restrict__ agent_vel_x, float *__restrict__ agent_vel_y,
    const float *__restrict__ agent_speed,
    const uint32_t *__restrict__ agent_alive,
    float *__restrict__ agent_distance_traveled,
    const float *__restrict__ agent_brain_outputs, int agent_count,
    float world_width, float world_height) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= agent_count || agent_alive[idx] == 0) {
    return;
  }

  float dx = agent_brain_outputs[idx * 2 + 0];
  float dy = agent_brain_outputs[idx * 2 + 1];
  float len = sqrtf(dx * dx + dy * dy);
  if (len > 1e-6f) {
    dx /= len;
    dy /= len;
  } else {
    dx = 0.0f;
    dy = 0.0f;
  }

  agent_vel_x[idx] = dx * agent_speed[idx];
  agent_vel_y[idx] = dy * agent_speed[idx];

  const float old_x = agent_pos_x[idx];
  const float old_y = agent_pos_y[idx];

  agent_pos_x[idx] += agent_vel_x[idx];
  agent_pos_y[idx] += agent_vel_y[idx];

  while (agent_pos_x[idx] < 0.0f)
    agent_pos_x[idx] += world_width;
  while (agent_pos_x[idx] >= world_width)
    agent_pos_x[idx] -= world_width;
  while (agent_pos_y[idx] < 0.0f)
    agent_pos_y[idx] += world_height;
  while (agent_pos_y[idx] >= world_height)
    agent_pos_y[idx] -= world_height;

  float move_dx = agent_pos_x[idx] - old_x;
  float move_dy = agent_pos_y[idx] - old_y;
  apply_wrap(move_dx, move_dy, world_width, world_height);
  agent_distance_traveled[idx] += sqrtf(move_dx * move_dx + move_dy * move_dy);
}
} // namespace

GpuBatchECS::GpuBatchECS(std::size_t max_agents, std::size_t max_food)
    : buffer_(max_agents, max_food) {
  init_cuda_resources();
  agent_mapping_.resize(max_agents);
  food_mapping_.resize(max_food);
}

GpuBatchECS::~GpuBatchECS() {
  cleanup_cuda_resources();
}

void GpuBatchECS::init_cuda_resources() {
  const cudaError_t err =
      cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_));
  if (err != cudaSuccess) {
    had_error_ = true;
  }
}

void GpuBatchECS::cleanup_cuda_resources() {
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

void GpuBatchECS::upload_async(std::size_t agent_count, std::size_t food_count) {
  buffer_.upload_async(agent_count, food_count, static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatchECS::download_async(std::size_t agent_count,
                                 std::size_t food_count) {
  buffer_.download_async(agent_count, food_count,
                         static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatchECS::launch_build_sensors_async(const GpuStepParams &params,
                                             std::size_t agent_count,
                                             std::size_t food_count) {
  if (agent_count == 0 || had_error_) {
    return;
  }

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
      buffer_.device_agent_velocities_x(), buffer_.device_agent_velocities_y(),
      buffer_.device_agent_speed(), buffer_.device_agent_types(),
      buffer_.device_agent_alive(), buffer_.device_agent_energy(),
      buffer_.device_food_positions_x(), buffer_.device_food_positions_y(),
      buffer_.device_food_active(), buffer_.device_agent_sensor_inputs(),
      static_cast<int>(agent_count), static_cast<int>(food_count),
      params.world_width, params.world_height, params.vision_range,
      params.max_energy);
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_agent_brain_outputs(), 0,
                             agent_count * 2 * sizeof(float), stream));
  check_launch_error();
}

void GpuBatchECS::launch_post_inference_async(const GpuStepParams &params,
                                              std::size_t agent_count,
                                              std::size_t food_count) {
  launch_post_inference_kernel(
      buffer_.device_agent_positions_x(), buffer_.device_agent_positions_y(),
      buffer_.device_agent_velocities_x(), buffer_.device_agent_velocities_y(),
      buffer_.device_agent_speed(), buffer_.device_agent_energy(),
      buffer_.device_agent_age(), buffer_.device_agent_reproduction_cooldown(),
      buffer_.device_agent_alive(), buffer_.device_agent_types(),
      buffer_.device_agent_distance_traveled(), buffer_.device_agent_kill_counts(),
      buffer_.device_agent_killed_by(), buffer_.device_agent_brain_outputs(),
      buffer_.device_food_positions_x(), buffer_.device_food_positions_y(),
      buffer_.device_food_active(), buffer_.device_food_consumed_by(),
      agent_count, food_count, params.world_width, params.world_height,
      params.energy_drain_per_step, params.max_age, params.food_pickup_range,
      params.attack_range, params.energy_gain_from_food,
      params.energy_gain_from_kill, static_cast<cudaStream_t>(stream_));
  check_launch_error();
}

void GpuBatchECS::synchronize() {
  const cudaError_t err =
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
  if (err != cudaSuccess) {
    spdlog::error("GPU synchronize failed: {}", cudaGetErrorString(err));
    had_error_ = true;
  }
}

void GpuBatchECS::mark_error() {
  had_error_ = true;
}

void GpuBatchECS::check_launch_error() {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("GPU kernel launch failed: {}", cudaGetErrorString(err));
    mark_error();
  }
}

void launch_build_sensors_kernel(
    const float *d_pos_x, const float *d_pos_y, const float *d_vel_x,
    const float *d_vel_y, const float *d_speed, const uint8_t *d_agent_types,
    const uint32_t *d_agent_alive, const float *d_energy,
    const float *d_food_pos_x, const float *d_food_pos_y,
    const uint32_t *d_food_active, float *d_sensor_inputs,
    std::size_t agent_count, std::size_t food_count, float world_width,
    float world_height, float vision_range, float max_energy,
    cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_speed, d_agent_types, d_agent_alive,
      d_energy, d_food_pos_x, d_food_pos_y, d_food_active, d_sensor_inputs,
      static_cast<int>(agent_count), static_cast<int>(food_count), world_width,
      world_height, vision_range, max_energy);
}

void launch_post_inference_kernel(
    float *d_agent_pos_x, float *d_agent_pos_y, float *d_agent_vel_x,
    float *d_agent_vel_y, const float *d_agent_speed, float *d_agent_energy,
    int *d_agent_age, int *d_agent_reproduction_cooldown,
    uint32_t *d_agent_alive, const uint8_t *d_agent_types,
    float *d_agent_distance_traveled, uint32_t *d_agent_kill_counts,
    int *d_agent_killed_by, const float *d_agent_brain_outputs,
    float *d_food_pos_x, float *d_food_pos_y, uint32_t *d_food_active,
    int *d_food_consumed_by, std::size_t agent_count, std::size_t food_count,
    float world_width, float world_height, float energy_drain, int max_age,
    float food_pickup_range, float attack_range, float energy_gain_from_food,
    float energy_gain_from_kill, cudaStream_t stream) {
  if (agent_count == 0) {
    return;
  }

  const int agent_blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int food_blocks =
      (static_cast<int>(food_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;

  kernel_update_vitals<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_energy, d_agent_age, d_agent_reproduction_cooldown,
      d_agent_alive, static_cast<int>(agent_count), energy_drain, max_age);

  CUDA_CHECK(cudaMemsetAsync(d_agent_kill_counts, 0,
                             agent_count * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaMemsetAsync(d_agent_killed_by, 0x7F,
                             agent_count * sizeof(int), stream));

  if (food_count > 0) {
    CUDA_CHECK(cudaMemsetAsync(d_food_consumed_by, 0x7F,
                               food_count * sizeof(int), stream));
    kernel_claim_food<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
        d_agent_pos_x, d_agent_pos_y, d_agent_types, d_agent_alive, d_food_pos_x,
        d_food_pos_y, d_food_active, d_food_consumed_by,
        static_cast<int>(agent_count), static_cast<int>(food_count),
        food_pickup_range, world_width, world_height);
    kernel_finalize_food<<<food_blocks, kThreadsPerBlock, 0, stream>>>(
        d_agent_energy, d_agent_alive, d_food_active, d_food_consumed_by,
        static_cast<int>(food_count), energy_gain_from_food);
  }

  kernel_claim_combat<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_pos_x, d_agent_pos_y, d_agent_types, d_agent_alive,
      d_agent_killed_by, static_cast<int>(agent_count), attack_range,
      world_width, world_height);
  kernel_finalize_combat<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_energy, d_agent_alive, d_agent_types, d_agent_kill_counts,
      d_agent_killed_by, static_cast<int>(agent_count), energy_gain_from_kill);

  kernel_apply_movement<<<agent_blocks, kThreadsPerBlock, 0, stream>>>(
      d_agent_pos_x, d_agent_pos_y, d_agent_vel_x, d_agent_vel_y, d_agent_speed,
      d_agent_alive, d_agent_distance_traveled, d_agent_brain_outputs,
      static_cast<int>(agent_count), world_width, world_height);
}

} // namespace gpu
} // namespace moonai
