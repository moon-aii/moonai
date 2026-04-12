#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai::gpu {

class GpuPopulationBuffer {
public:
  explicit GpuPopulationBuffer(std::size_t max_agents);
  ~GpuPopulationBuffer();

  GpuPopulationBuffer(const GpuPopulationBuffer &) = delete;
  GpuPopulationBuffer &operator=(const GpuPopulationBuffer &) = delete;
  GpuPopulationBuffer(GpuPopulationBuffer &&) = delete;
  GpuPopulationBuffer &operator=(GpuPopulationBuffer &&) = delete;

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
  [[nodiscard]] int *host_age() const noexcept {
    return h_age_;
  }
  [[nodiscard]] uint32_t *host_alive() const noexcept {
    return h_alive_;
  }
  [[nodiscard]] uint32_t *host_kill_counts() const noexcept {
    return h_kill_counts_;
  }
  [[nodiscard]] int *host_claimed_by() const noexcept {
    return h_claimed_by_;
  }
  [[nodiscard]] float *host_brain_outputs() const noexcept {
    return h_brain_outputs_;
  }

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
  [[nodiscard]] float *device_energy() const noexcept {
    return d_energy_;
  }
  [[nodiscard]] int *device_age() const noexcept {
    return d_age_;
  }
  [[nodiscard]] uint32_t *device_alive() const noexcept {
    return d_alive_;
  }
  [[nodiscard]] uint32_t *device_kill_counts() const noexcept {
    return d_kill_counts_;
  }
  [[nodiscard]] int *device_claimed_by() const noexcept {
    return d_claimed_by_;
  }
  [[nodiscard]] float *device_sensor_inputs() const noexcept {
    return d_sensor_inputs_;
  }
  [[nodiscard]] float *device_brain_outputs() const noexcept {
    return d_brain_outputs_;
  }

  void upload_async(std::size_t count, cudaStream_t stream);
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
  int *h_age_ = nullptr;
  uint32_t *h_alive_ = nullptr;
  uint32_t *h_kill_counts_ = nullptr;
  int *h_claimed_by_ = nullptr;
  float *h_brain_outputs_ = nullptr;

  float *d_pos_x_ = nullptr;
  float *d_pos_y_ = nullptr;
  float *d_vel_x_ = nullptr;
  float *d_vel_y_ = nullptr;
  float *d_energy_ = nullptr;
  int *d_age_ = nullptr;
  uint32_t *d_alive_ = nullptr;
  uint32_t *d_kill_counts_ = nullptr;
  int *d_claimed_by_ = nullptr;
  float *d_sensor_inputs_ = nullptr;
  float *d_brain_outputs_ = nullptr;

  std::size_t capacity_;

  static constexpr int kSensorInputsPerEntity = 12;
  static constexpr int kBrainOutputsPerEntity = 2;
};

class GpuFoodBuffer {
public:
  explicit GpuFoodBuffer(std::size_t max_food);
  ~GpuFoodBuffer();

  GpuFoodBuffer(const GpuFoodBuffer &) = delete;
  GpuFoodBuffer &operator=(const GpuFoodBuffer &) = delete;
  GpuFoodBuffer(GpuFoodBuffer &&) = delete;
  GpuFoodBuffer &operator=(GpuFoodBuffer &&) = delete;

  [[nodiscard]] float *host_positions_x() const noexcept {
    return h_pos_x_;
  }
  [[nodiscard]] float *host_positions_y() const noexcept {
    return h_pos_y_;
  }
  [[nodiscard]] uint32_t *host_active() const noexcept {
    return h_active_;
  }
  [[nodiscard]] int *host_consumed_by() const noexcept {
    return h_consumed_by_;
  }

  [[nodiscard]] float *device_positions_x() const noexcept {
    return d_pos_x_;
  }
  [[nodiscard]] float *device_positions_y() const noexcept {
    return d_pos_y_;
  }
  [[nodiscard]] uint32_t *device_active() const noexcept {
    return d_active_;
  }
  [[nodiscard]] int *device_consumed_by() const noexcept {
    return d_consumed_by_;
  }

  void upload_async(std::size_t count, cudaStream_t stream);
  void download_async(std::size_t count, cudaStream_t stream);

  [[nodiscard]] std::size_t capacity() const noexcept {
    return capacity_;
  }

private:
  void allocate_buffers();
  void free_buffers();

  float *h_pos_x_ = nullptr;
  float *h_pos_y_ = nullptr;
  uint32_t *h_active_ = nullptr;
  int *h_consumed_by_ = nullptr;

  float *d_pos_x_ = nullptr;
  float *d_pos_y_ = nullptr;
  uint32_t *d_active_ = nullptr;
  int *d_consumed_by_ = nullptr;

  std::size_t capacity_;
};

} // namespace moonai::gpu
