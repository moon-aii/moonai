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

/**
 * @brief Parameters for GPU-accelerated simulation step
 *
 * All simulation parameters needed for GPU kernels.
 */
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

/**
 * @brief ECS-native GPU batch processing with clean buffer abstraction
 *
 * This is the new ECS-native GPU interface that uses GpuDataBuffer for
 * clean separation between ECS and GPU. Replaces the old GpuBatch which
 * used GpuAgentState structures.
 *
 * Features:
 * - Clean ECS-GPU boundary via GpuDataBuffer
 * - On-demand entity compaction via GpuEntityMapping
 * - Async H2D/D2H transfers with pinned memory
 * - Single CUDA stream for all operations
 */
class GpuBatchECS {
public:
  /**
   * @brief Create GPU batch for up to max_entities agents
   * @param max_entities Maximum number of entities this batch can process
   */
  GpuBatchECS(std::size_t max_entities);
  ~GpuBatchECS();

  // Disable copy/move - owns CUDA resources
  GpuBatchECS(const GpuBatchECS &) = delete;
  GpuBatchECS &operator=(const GpuBatchECS &) = delete;
  GpuBatchECS(GpuBatchECS &&) = delete;
  GpuBatchECS &operator=(GpuBatchECS &&) = delete;

  /**
   * @brief Get the data buffer for ECS population
   * @return Reference to GPU data buffer
   */
  [[nodiscard]] GpuDataBuffer &buffer() {
    return buffer_;
  }
  [[nodiscard]] const GpuDataBuffer &buffer() const {
    return buffer_;
  }

  /**
   * @brief Get the entity mapping
   * @return Reference to GPU entity mapping
   */
  [[nodiscard]] GpuEntityMapping &mapping() {
    return mapping_;
  }
  [[nodiscard]] const GpuEntityMapping &mapping() const {
    return mapping_;
  }

  /**
   * @brief Launch full simulation step on GPU
   *
   * Performs:
   * 1. Upload agent data to GPU
   * 2. Build spatial bins
   * 3. Build sensor inputs
   * 4. Run neural inference (if network data provided)
   * 5. Apply movement and energy drain
   * 6. Process food consumption and attacks
   * 7. Download results back to host buffers
   *
   * @param params Simulation parameters
   * @param agent_count Number of agents (from mapping.count())
   */
  void launch_full_step_async(const GpuStepParams &params,
                              std::size_t agent_count);

  /**
   * @brief Upload agent data from host to device
   * @param agent_count Number of agents to upload
   */
  void upload_async(std::size_t agent_count);

  /**
   * @brief Download results from device to host
   * @param agent_count Number of agents to download
   */
  void download_async(std::size_t agent_count);

  /**
   * @brief Wait for all async operations to complete
   */
  void synchronize();

  /**
   * @brief Check if GPU operations completed successfully
   */
  [[nodiscard]] bool ok() const noexcept {
    return !had_error_;
  }

  // Access to stream for external kernel launches
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

// Free functions for kernel launching
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
