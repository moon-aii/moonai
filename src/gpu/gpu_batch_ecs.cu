#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch_ecs.hpp"
#include <algorithm>
#include <cub/cub.cuh>

namespace moonai {
namespace gpu {

namespace {
constexpr int kThreadsPerBlock = 256;
}

// Kernel: Build sensor inputs from agent data
__global__ void kernel_build_sensors(
    const float *__restrict__ pos_x, const float *__restrict__ pos_y,
    const uint8_t *__restrict__ types, const float *__restrict__ energy,
    float *__restrict__ sensor_inputs, int count, float world_width,
    float world_height, float max_energy, bool has_walls) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  // Sensor input layout (15 floats per agent):
  // 0-1: nearest predator (dist, angle)
  // 2-3: nearest prey (dist, angle)
  // 4-5: nearest food (dist, angle)
  // 6: energy level (normalized)
  // 7-8: velocity (x, y)
  // 9-10: local density (predators, prey)
  // 11-14: wall proximity (left, right, top, bottom)

  float *out = sensor_inputs + idx * 15;

  // Initialize sensors
  for (int i = 0; i < 15; ++i) {
    out[i] = 0.0f;
  }

  // Energy normalized to [0, 1]
  out[6] = energy[idx] / max_energy;

  // Wall proximity (if walls enabled)
  if (has_walls) {
    float x = pos_x[idx];
    float y = pos_y[idx];
    // Normalize by world size
    out[11] = x / world_width;                   // left
    out[12] = (world_width - x) / world_width;   // right
    out[13] = y / world_height;                  // top
    out[14] = (world_height - y) / world_height; // bottom
  }
}

// Kernel: Apply movement from brain outputs
__global__ void
kernel_apply_movement(float *__restrict__ pos_x, float *__restrict__ pos_y,
                      float *__restrict__ vel_x, float *__restrict__ vel_y,
                      float *__restrict__ energy, uint8_t *__restrict__ alive,
                      const float *__restrict__ brain_outputs, int count,
                      float dt, float world_width, float world_height,
                      bool has_walls, float energy_drain, float max_energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  if (!alive[idx])
    return;

  // Get brain outputs (2 outputs: x and y movement direction)
  float dx = brain_outputs[idx * 2 + 0];
  float dy = brain_outputs[idx * 2 + 1];

  // Update velocity based on brain decisions
  vel_x[idx] = dx;
  vel_y[idx] = dy;

  // Update position
  pos_x[idx] += vel_x[idx] * dt;
  pos_y[idx] += vel_y[idx] * dt;

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
    // Wrap-around
    pos_x[idx] = fmodf(pos_x[idx] + world_width, world_width);
    pos_y[idx] = fmodf(pos_y[idx] + world_height, world_height);
  }

  // Energy drain
  energy[idx] -= energy_drain;
  if (energy[idx] <= 0.0f) {
    energy[idx] = 0.0f;
    alive[idx] = 0;
  } else if (energy[idx] > max_energy) {
    energy[idx] = max_energy;
  }
}

// Kernel: Process combat (predator attacks)
__global__ void kernel_process_combat(const float *__restrict__ pos_x,
                                      const float *__restrict__ pos_y,
                                      const uint8_t *__restrict__ types,
                                      float *__restrict__ energy,
                                      uint8_t *__restrict__ alive,
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
      // Kill prey
      alive[j] = 0;
      // Give energy to predator
      atomicAdd(&energy[idx], energy_gain);
      break; // One kill per attack
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
  CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&stream_)));
}

void GpuBatchECS::cleanup_cuda_resources() {
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

void GpuBatchECS::upload_async(std::size_t agent_count) {
  buffer_.upload_async(agent_count, static_cast<cudaStream_t>(stream_));
}

void GpuBatchECS::download_async(std::size_t agent_count) {
  buffer_.download_async(agent_count, static_cast<cudaStream_t>(stream_));
}

void GpuBatchECS::launch_full_step_async(const GpuStepParams &params,
                                         std::size_t agent_count) {
  if (agent_count == 0)
    return;

  const int blocks =
      (static_cast<int>(agent_count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  // 1. Build sensor inputs
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_types(), buffer_.device_energy(),
      buffer_.device_sensor_inputs(), static_cast<int>(agent_count),
      params.world_width, params.world_height, params.max_energy,
      params.has_walls);

  // 2. Neural inference would go here (using network data)
  // For now, output zeros as placeholder
  CUDA_CHECK(cudaMemsetAsync(buffer_.device_brain_outputs(), 0,
                             agent_count * 2 * sizeof(float), stream));

  // 3. Apply movement
  kernel_apply_movement<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_outputs_velocities_x(),
      buffer_.device_outputs_velocities_y(), buffer_.device_outputs_energy(),
      buffer_.device_outputs_alive(), buffer_.device_brain_outputs(),
      static_cast<int>(agent_count), params.dt, params.world_width,
      params.world_height, params.has_walls, params.energy_drain_per_step,
      params.max_energy);

  // 4. Process combat
  kernel_process_combat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      buffer_.device_positions_x(), buffer_.device_positions_y(),
      buffer_.device_types(), buffer_.device_outputs_energy(),
      buffer_.device_outputs_alive(), params.attack_range,
      params.energy_gain_from_kill, static_cast<int>(agent_count));
}

void GpuBatchECS::synchronize() {
  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
}

// Free function implementations
void launch_build_sensors_kernel(const float *d_pos_x, const float *d_pos_y,
                                 const uint8_t *d_types, const float *d_energy,
                                 float *d_sensor_inputs, std::size_t count,
                                 float world_width, float world_height,
                                 float max_energy, bool has_walls,
                                 cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_build_sensors<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_types, d_energy, d_sensor_inputs,
      static_cast<int>(count), world_width, world_height, max_energy,
      has_walls);
}

void launch_apply_movement_kernel(
    float *d_pos_x, float *d_pos_y, float *d_vel_x, float *d_vel_y,
    float *d_energy, uint8_t *d_alive, const float *d_sensor_inputs,
    const float *d_brain_outputs, std::size_t count, float dt,
    float world_width, float world_height, bool has_walls, float energy_drain,
    float max_energy, cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_apply_movement<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_energy, d_alive, d_brain_outputs,
      static_cast<int>(count), dt, world_width, world_height, has_walls,
      energy_drain, max_energy);
}

void launch_process_combat_kernel(const float *d_pos_x, const float *d_pos_y,
                                  const uint8_t *d_types, float *d_energy,
                                  uint8_t *d_alive, float attack_range,
                                  float energy_gain, std::size_t count,
                                  cudaStream_t stream) {
  const int blocks =
      (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  kernel_process_combat<<<blocks, kThreadsPerBlock, 0, stream>>>(
      d_pos_x, d_pos_y, d_types, d_energy, d_alive, attack_range, energy_gain,
      static_cast<int>(count));
}

} // namespace gpu
} // namespace moonai
