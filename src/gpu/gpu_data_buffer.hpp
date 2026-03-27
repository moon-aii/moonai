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

// Simplified GPU data buffer with single buffers for all fields
// Kernels modify data in-place for simplicity and performance
class GpuDataBuffer {
public:
  explicit GpuDataBuffer(std::size_t max_entities);

  ~GpuDataBuffer();

  // Disable copy/move - buffers own CUDA resources
  GpuDataBuffer(const GpuDataBuffer &) = delete;
  GpuDataBuffer &operator=(const GpuDataBuffer &) = delete;
  GpuDataBuffer(GpuDataBuffer &&) = delete;
  GpuDataBuffer &operator=(GpuDataBuffer &&) = delete;

  // Host buffer accessors (for ECS packing)
  [[nodiscard]] float *host_positions_x() const noexcept {
    return h_pos_x_;
  }
  [[nodiscard]] float *host_positions_y() const noexcept {
    return h_pos_y_;
  }
  [[nodiscard]] float *host_velocities_x() const noexcept {
    return h_vel_x_;
  }
  [[nodiscard]] float *host_velocities_y() const noexcept {
    return h_vel_y_;
  }
  [[nodiscard]] float *host_speed() const noexcept {
    return h_speed_;
  }
  [[nodiscard]] float *host_energy() const noexcept {
    return h_energy_;
  }
  [[nodiscard]] int *host_age() const noexcept {
    return h_age_;
  }
  [[nodiscard]] uint32_t *host_alive() const noexcept {
    return h_alive_;
  }
  [[nodiscard]] uint8_t *host_types() const noexcept {
    return h_types_;
  }
  [[nodiscard]] int *host_reproduction_cooldown() const noexcept {
    return h_reproduction_cooldown_;
  }
  [[nodiscard]] uint32_t *host_species_ids() const noexcept {
    return h_species_ids_;
  }
  [[nodiscard]] float *host_distance_traveled() const noexcept {
    return h_distance_traveled_;
  }
  [[nodiscard]] uint32_t *host_kill_counts() const noexcept {
    return h_kill_counts_;
  }
  [[nodiscard]] int *host_killed_by() const noexcept {
    return h_killed_by_;
  }
  [[nodiscard]] float *host_sensor_inputs() const noexcept {
    return h_sensor_inputs_;
  }
  [[nodiscard]] float *host_brain_outputs() const noexcept {
    return h_brain_outputs_;
  }

  // Device buffer accessors (for kernel launches)
  // Kernels read/write these buffers in-place
  [[nodiscard]] float *device_positions_x() const noexcept {
    return d_pos_x_;
  }
  [[nodiscard]] float *device_positions_y() const noexcept {
    return d_pos_y_;
  }
  [[nodiscard]] float *device_velocities_x() const noexcept {
    return d_vel_x_;
  }
  [[nodiscard]] float *device_velocities_y() const noexcept {
    return d_vel_y_;
  }
  [[nodiscard]] float *device_speed() const noexcept {
    return d_speed_;
  }
  [[nodiscard]] float *device_energy() const noexcept {
    return d_energy_;
  }
  [[nodiscard]] int *device_age() const noexcept {
    return d_age_;
  }
  [[nodiscard]] uint32_t *device_alive() const noexcept {
    return d_alive_;
  }
  [[nodiscard]] uint8_t *device_types() const noexcept {
    return d_types_;
  }
  [[nodiscard]] int *device_reproduction_cooldown() const noexcept {
    return d_reproduction_cooldown_;
  }
  [[nodiscard]] uint32_t *device_species_ids() const noexcept {
    return d_species_ids_;
  }
  [[nodiscard]] float *device_distance_traveled() const noexcept {
    return d_distance_traveled_;
  }
  [[nodiscard]] uint32_t *device_kill_counts() const noexcept {
    return d_kill_counts_;
  }
  [[nodiscard]] int *device_killed_by() const noexcept {
    return d_killed_by_;
  }
  [[nodiscard]] float *device_sensor_inputs() const noexcept {
    return d_sensor_inputs_;
  }
  [[nodiscard]] float *device_brain_outputs() const noexcept {
    return d_brain_outputs_;
  }

  // Async transfer operations
  void upload_async(std::size_t count, cudaStream_t stream);
  void download_async(std::size_t count, cudaStream_t stream);

  [[nodiscard]] std::size_t capacity() const noexcept {
    return capacity_;
  }

private:
  void allocate_buffers();
  void free_buffers();

  // Host buffers (pinned memory for fast transfers)
  float *h_pos_x_ = nullptr;
  float *h_pos_y_ = nullptr;
  float *h_vel_x_ = nullptr;
  float *h_vel_y_ = nullptr;
  float *h_speed_ = nullptr;
  float *h_energy_ = nullptr;
  int *h_age_ = nullptr;
  uint32_t *h_alive_ = nullptr;
  uint8_t *h_types_ = nullptr;
  int *h_reproduction_cooldown_ = nullptr;
  uint32_t *h_species_ids_ = nullptr;
  float *h_distance_traveled_ = nullptr;
  uint32_t *h_kill_counts_ = nullptr;
  int *h_killed_by_ = nullptr;
  float *h_sensor_inputs_ = nullptr;
  float *h_brain_outputs_ = nullptr;

  // Device buffers (kernels modify these in-place)
  float *d_pos_x_ = nullptr;
  float *d_pos_y_ = nullptr;
  float *d_vel_x_ = nullptr;
  float *d_vel_y_ = nullptr;
  float *d_speed_ = nullptr;
  float *d_energy_ = nullptr;
  int *d_age_ = nullptr;
  uint32_t *d_alive_ = nullptr;
  uint8_t *d_types_ = nullptr;
  int *d_reproduction_cooldown_ = nullptr;
  uint32_t *d_species_ids_ = nullptr;
  float *d_distance_traveled_ = nullptr;
  uint32_t *d_kill_counts_ = nullptr;
  int *d_killed_by_ = nullptr;
  float *d_sensor_inputs_ = nullptr;
  float *d_brain_outputs_ = nullptr;

  std::size_t capacity_;

  static constexpr int kSensorInputsPerEntity = 15;
  static constexpr int kBrainOutputsPerEntity = 2;
};

} // namespace gpu
} // namespace moonai
