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
  explicit GpuDataBuffer(std::size_t max_entities);

  ~GpuDataBuffer();

  // Disable copy/move - buffers own CUDA resources
  GpuDataBuffer(const GpuDataBuffer &) = delete;
  GpuDataBuffer &operator=(const GpuDataBuffer &) = delete;
  GpuDataBuffer(GpuDataBuffer &&) = delete;
  GpuDataBuffer &operator=(GpuDataBuffer &&) = delete;

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
  [[nodiscard]] float *host_energy() const noexcept {
    return h_energy_;
  }
  [[nodiscard]] float *host_age() const noexcept {
    return h_age_;
  }
  [[nodiscard]] uint8_t *host_alive() const noexcept {
    return h_alive_;
  }
  [[nodiscard]] uint8_t *host_types() const noexcept {
    return h_types_;
  }
  [[nodiscard]] uint32_t *host_species_ids() const noexcept {
    return h_species_ids_;
  }
  [[nodiscard]] float *host_sensor_inputs() const noexcept {
    return h_sensor_inputs_;
  }
  [[nodiscard]] float *host_brain_outputs() const noexcept {
    return h_brain_outputs_;
  }

  [[nodiscard]] const float *device_positions_x() const noexcept {
    return d_pos_x_;
  }
  [[nodiscard]] const float *device_positions_y() const noexcept {
    return d_pos_y_;
  }
  [[nodiscard]] float *device_positions_x() noexcept {
    return d_pos_x_;
  }
  [[nodiscard]] float *device_positions_y() noexcept {
    return d_pos_y_;
  }

  [[nodiscard]] const float *device_velocities_x() const noexcept {
    return d_vel_x_;
  }
  [[nodiscard]] const float *device_velocities_y() const noexcept {
    return d_vel_y_;
  }

  [[nodiscard]] const float *device_energy() const noexcept {
    return d_energy_;
  }
  [[nodiscard]] float *device_energy() noexcept {
    return d_energy_;
  }

  [[nodiscard]] const float *device_age() const noexcept {
    return d_age_;
  }

  [[nodiscard]] const uint8_t *device_alive() const noexcept {
    return d_alive_;
  }
  [[nodiscard]] uint8_t *device_alive() noexcept {
    return d_alive_;
  }

  [[nodiscard]] const uint8_t *device_types() const noexcept {
    return d_types_;
  }

  [[nodiscard]] const uint32_t *device_species_ids() const noexcept {
    return d_species_ids_;
  }

  [[nodiscard]] float *device_sensor_inputs() const noexcept {
    return d_sensor_inputs_;
  }
  [[nodiscard]] float *device_brain_outputs() const noexcept {
    return d_brain_outputs_;
  }

  [[nodiscard]] float *host_outputs_energy() const noexcept {
    return h_out_energy_;
  }
  [[nodiscard]] uint8_t *host_outputs_alive() const noexcept {
    return h_out_alive_;
  }
  [[nodiscard]] float *host_outputs_velocities_x() const noexcept {
    return h_out_vel_x_;
  }
  [[nodiscard]] float *host_outputs_velocities_y() const noexcept {
    return h_out_vel_y_;
  }
  [[nodiscard]] const float *device_outputs_energy() const noexcept {
    return d_out_energy_;
  }
  [[nodiscard]] const uint8_t *device_outputs_alive() const noexcept {
    return d_out_alive_;
  }
  [[nodiscard]] const float *device_outputs_velocities_x() const noexcept {
    return d_out_vel_x_;
  }
  [[nodiscard]] const float *device_outputs_velocities_y() const noexcept {
    return d_out_vel_y_;
  }

  // Non-const versions for kernel output
  [[nodiscard]] float *device_outputs_energy() noexcept {
    return d_out_energy_;
  }
  [[nodiscard]] uint8_t *device_outputs_alive() noexcept {
    return d_out_alive_;
  }
  [[nodiscard]] float *device_outputs_velocities_x() noexcept {
    return d_out_vel_x_;
  }
  [[nodiscard]] float *device_outputs_velocities_y() noexcept {
    return d_out_vel_y_;
  }

  // Async transfer operations
  /**
   * @brief Upload count entities from host to device
   * @param count Number of entities to upload
   * @param stream CUDA stream for async operation
   */
  void upload_async(std::size_t count, cudaStream_t stream);

  /**
   * @brief Download count entities from device to host
   * @param count Number of entities to download
   * @param stream CUDA stream for async operation
   */
  void download_async(std::size_t count, cudaStream_t stream);

  [[nodiscard]] std::size_t capacity() const noexcept {
    return capacity_;
  }

private:
  void allocate_buffers();
  void free_buffers();

  float *h_pos_x_ = nullptr;
  float *h_pos_y_ = nullptr;
  float *h_vel_x_ = nullptr;
  float *h_vel_y_ = nullptr;
  float *h_energy_ = nullptr;
  float *h_age_ = nullptr;
  uint8_t *h_alive_ = nullptr;
  uint8_t *h_types_ = nullptr;
  uint32_t *h_species_ids_ = nullptr;
  float *h_sensor_inputs_ = nullptr;
  float *h_brain_outputs_ = nullptr;

  float *d_pos_x_ = nullptr;
  float *d_pos_y_ = nullptr;
  float *d_vel_x_ = nullptr;
  float *d_vel_y_ = nullptr;
  float *d_energy_ = nullptr;
  float *d_age_ = nullptr;
  uint8_t *d_alive_ = nullptr;
  uint8_t *d_types_ = nullptr;
  uint32_t *d_species_ids_ = nullptr;
  float *d_sensor_inputs_ = nullptr;
  float *d_brain_outputs_ = nullptr;

  float *h_out_energy_ = nullptr;
  uint8_t *h_out_alive_ = nullptr;
  float *h_out_vel_x_ = nullptr;
  float *h_out_vel_y_ = nullptr;

  float *d_out_energy_ = nullptr;
  uint8_t *d_out_alive_ = nullptr;
  float *d_out_vel_x_ = nullptr;
  float *d_out_vel_y_ = nullptr;

  std::size_t capacity_;

  static constexpr int kSensorInputsPerEntity = 15;
  static constexpr int kBrainOutputsPerEntity = 2;
};

} // namespace gpu
} // namespace moonai
