#pragma once

#include <cstddef>
#include <cstdint>

// CUDA headers - only available when CUDA is enabled
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// Forward declarations for non-CUDA compilation
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {
namespace gpu {

class GpuDataBuffer {
public:
  GpuDataBuffer(std::size_t max_agents, std::size_t max_food);

  ~GpuDataBuffer();

  // Disable copy/move - buffers own CUDA resources
  GpuDataBuffer(const GpuDataBuffer &) = delete;
  GpuDataBuffer &operator=(const GpuDataBuffer &) = delete;
  GpuDataBuffer(GpuDataBuffer &&) = delete;
  GpuDataBuffer &operator=(GpuDataBuffer &&) = delete;

  // Host buffer accessors (for ECS packing)
  [[nodiscard]] float *host_agent_positions_x() const noexcept {
    return h_agent_pos_x_;
  }
  [[nodiscard]] float *host_agent_positions_y() const noexcept {
    return h_agent_pos_y_;
  }
  [[nodiscard]] float *host_agent_velocities_x() const noexcept {
    return h_agent_vel_x_;
  }
  [[nodiscard]] float *host_agent_velocities_y() const noexcept {
    return h_agent_vel_y_;
  }
  [[nodiscard]] float *host_agent_speed() const noexcept {
    return h_agent_speed_;
  }
  [[nodiscard]] float *host_agent_energy() const noexcept {
    return h_agent_energy_;
  }
  [[nodiscard]] int *host_agent_age() const noexcept {
    return h_agent_age_;
  }
  [[nodiscard]] uint32_t *host_agent_alive() const noexcept {
    return h_agent_alive_;
  }
  [[nodiscard]] uint8_t *host_agent_types() const noexcept {
    return h_agent_types_;
  }
  [[nodiscard]] int *host_agent_reproduction_cooldown() const noexcept {
    return h_agent_reproduction_cooldown_;
  }
  [[nodiscard]] float *host_agent_distance_traveled() const noexcept {
    return h_agent_distance_traveled_;
  }
  [[nodiscard]] uint32_t *host_agent_kill_counts() const noexcept {
    return h_agent_kill_counts_;
  }
  [[nodiscard]] int *host_agent_killed_by() const noexcept {
    return h_agent_killed_by_;
  }
  [[nodiscard]] float *host_agent_sensor_inputs() const noexcept {
    return h_agent_sensor_inputs_;
  }
  [[nodiscard]] float *host_agent_brain_outputs() const noexcept {
    return h_agent_brain_outputs_;
  }
  [[nodiscard]] float *host_food_positions_x() const noexcept {
    return h_food_pos_x_;
  }
  [[nodiscard]] float *host_food_positions_y() const noexcept {
    return h_food_pos_y_;
  }
  [[nodiscard]] uint32_t *host_food_active() const noexcept {
    return h_food_active_;
  }
  [[nodiscard]] int *host_food_consumed_by() const noexcept {
    return h_food_consumed_by_;
  }

  // Device buffer accessors (for kernel launches)
  // Kernels read/write these buffers in-place
  [[nodiscard]] float *device_agent_positions_x() const noexcept {
    return d_agent_pos_x_;
  }
  [[nodiscard]] float *device_agent_positions_y() const noexcept {
    return d_agent_pos_y_;
  }
  [[nodiscard]] float *device_agent_velocities_x() const noexcept {
    return d_agent_vel_x_;
  }
  [[nodiscard]] float *device_agent_velocities_y() const noexcept {
    return d_agent_vel_y_;
  }
  [[nodiscard]] float *device_agent_speed() const noexcept {
    return d_agent_speed_;
  }
  [[nodiscard]] float *device_agent_energy() const noexcept {
    return d_agent_energy_;
  }
  [[nodiscard]] int *device_agent_age() const noexcept {
    return d_agent_age_;
  }
  [[nodiscard]] uint32_t *device_agent_alive() const noexcept {
    return d_agent_alive_;
  }
  [[nodiscard]] uint8_t *device_agent_types() const noexcept {
    return d_agent_types_;
  }
  [[nodiscard]] int *device_agent_reproduction_cooldown() const noexcept {
    return d_agent_reproduction_cooldown_;
  }
  [[nodiscard]] float *device_agent_distance_traveled() const noexcept {
    return d_agent_distance_traveled_;
  }
  [[nodiscard]] uint32_t *device_agent_kill_counts() const noexcept {
    return d_agent_kill_counts_;
  }
  [[nodiscard]] int *device_agent_killed_by() const noexcept {
    return d_agent_killed_by_;
  }
  [[nodiscard]] float *device_agent_sensor_inputs() const noexcept {
    return d_agent_sensor_inputs_;
  }
  [[nodiscard]] float *device_agent_brain_outputs() const noexcept {
    return d_agent_brain_outputs_;
  }
  [[nodiscard]] float *device_food_positions_x() const noexcept {
    return d_food_pos_x_;
  }
  [[nodiscard]] float *device_food_positions_y() const noexcept {
    return d_food_pos_y_;
  }
  [[nodiscard]] uint32_t *device_food_active() const noexcept {
    return d_food_active_;
  }
  [[nodiscard]] int *device_food_consumed_by() const noexcept {
    return d_food_consumed_by_;
  }

  // Async transfer operations
  void upload_async(std::size_t agent_count, std::size_t food_count,
                    cudaStream_t stream);
  void download_async(std::size_t agent_count, std::size_t food_count,
                      cudaStream_t stream);

  [[nodiscard]] std::size_t agent_capacity() const noexcept {
    return agent_capacity_;
  }

  [[nodiscard]] std::size_t food_capacity() const noexcept {
    return food_capacity_;
  }

private:
  void allocate_buffers();
  void free_buffers();

  float *h_agent_pos_x_ = nullptr;
  float *h_agent_pos_y_ = nullptr;
  float *h_agent_vel_x_ = nullptr;
  float *h_agent_vel_y_ = nullptr;
  float *h_agent_speed_ = nullptr;
  float *h_agent_energy_ = nullptr;
  int *h_agent_age_ = nullptr;
  uint32_t *h_agent_alive_ = nullptr;
  uint8_t *h_agent_types_ = nullptr;
  int *h_agent_reproduction_cooldown_ = nullptr;
  float *h_agent_distance_traveled_ = nullptr;
  uint32_t *h_agent_kill_counts_ = nullptr;
  int *h_agent_killed_by_ = nullptr;
  float *h_agent_sensor_inputs_ = nullptr;
  float *h_agent_brain_outputs_ = nullptr;

  float *h_food_pos_x_ = nullptr;
  float *h_food_pos_y_ = nullptr;
  uint32_t *h_food_active_ = nullptr;
  int *h_food_consumed_by_ = nullptr;

  float *d_agent_pos_x_ = nullptr;
  float *d_agent_pos_y_ = nullptr;
  float *d_agent_vel_x_ = nullptr;
  float *d_agent_vel_y_ = nullptr;
  float *d_agent_speed_ = nullptr;
  float *d_agent_energy_ = nullptr;
  int *d_agent_age_ = nullptr;
  uint32_t *d_agent_alive_ = nullptr;
  uint8_t *d_agent_types_ = nullptr;
  int *d_agent_reproduction_cooldown_ = nullptr;
  float *d_agent_distance_traveled_ = nullptr;
  uint32_t *d_agent_kill_counts_ = nullptr;
  int *d_agent_killed_by_ = nullptr;
  float *d_agent_sensor_inputs_ = nullptr;
  float *d_agent_brain_outputs_ = nullptr;

  float *d_food_pos_x_ = nullptr;
  float *d_food_pos_y_ = nullptr;
  uint32_t *d_food_active_ = nullptr;
  int *d_food_consumed_by_ = nullptr;

  std::size_t agent_capacity_;
  std::size_t food_capacity_;

  static constexpr int kSensorInputsPerEntity = 12;
  static constexpr int kBrainOutputsPerEntity = 2;
};

} // namespace gpu
} // namespace moonai
