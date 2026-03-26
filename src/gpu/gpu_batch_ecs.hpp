#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_entity_mapping.hpp"
#include "gpu/gpu_types.hpp"
#include <cstddef>
#include <cstdint>

// CUDA forward declarations for non-CUDA compilation
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {
namespace gpu {

struct GpuStepParams {
  float dt = 0.0f;
  float world_width = 0.0f;
  float world_height = 0.0f;
  bool has_walls = false;
  float energy_drain_per_step = 0.0f;
  float max_energy = 150.0f;
  float food_pickup_range = 12.0f;
  float attack_range = 20.0f;
  float energy_gain_from_food = 40.0f;
  float energy_gain_from_kill = 60.0f;
  float food_respawn_rate = 0.02f;
  std::uint64_t seed = 0;
  int step_index = 0;
};

class GpuBatchECS {
public:
  GpuBatchECS(std::size_t max_entities);
  ~GpuBatchECS();

  GpuBatchECS(const GpuBatchECS &) = delete;
  GpuBatchECS &operator=(const GpuBatchECS &) = delete;
  GpuBatchECS(GpuBatchECS &&) = delete;
  GpuBatchECS &operator=(GpuBatchECS &&) = delete;

  [[nodiscard]] GpuDataBuffer &buffer() {
    return buffer_;
  }
  [[nodiscard]] const GpuDataBuffer &buffer() const {
    return buffer_;
  }

  [[nodiscard]] GpuEntityMapping &mapping() {
    return mapping_;
  }
  [[nodiscard]] const GpuEntityMapping &mapping() const {
    return mapping_;
  }

  void launch_full_step_async(const GpuStepParams &params,
                              std::size_t agent_count);

  void upload_async(std::size_t agent_count);

  void download_async(std::size_t agent_count);

  void synchronize();

  [[nodiscard]] bool ok() const noexcept {
    return !had_error_;
  }

  [[nodiscard]] cudaStream_t stream() const {
    return static_cast<cudaStream_t>(stream_);
  }

private:
  GpuDataBuffer buffer_;
  GpuEntityMapping mapping_;

  void *stream_ = nullptr;
  bool had_error_ = false;

  void init_cuda_resources();
  void cleanup_cuda_resources();
};

void launch_build_sensors_kernel(const float *d_pos_x, const float *d_pos_y,
                                 const uint8_t *d_types, const float *d_energy,
                                 float *d_sensor_inputs, std::size_t count,
                                 float world_width, float world_height,
                                 float max_energy, bool has_walls,
                                 cudaStream_t stream);

void launch_apply_movement_kernel(
    float *d_pos_x, float *d_pos_y, float *d_vel_x, float *d_vel_y,
    float *d_energy, uint8_t *d_alive, const float *d_sensor_inputs,
    const float *d_brain_outputs, std::size_t count, float dt,
    float world_width, float world_height, bool has_walls, float energy_drain,
    float max_energy, cudaStream_t stream);

void launch_process_combat_kernel(const float *d_pos_x, const float *d_pos_y,
                                  const uint8_t *d_types, float *d_energy,
                                  uint8_t *d_alive, float attack_range,
                                  float energy_gain, std::size_t count,
                                  cudaStream_t stream);

} // namespace gpu
} // namespace moonai
