#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_data_buffer.hpp"
#include <cstring>
#include <stdexcept>

namespace moonai {
namespace gpu {

GpuDataBuffer::GpuDataBuffer(std::size_t max_entities)
    : capacity_(max_entities) {
  allocate_buffers();
}

GpuDataBuffer::~GpuDataBuffer() {
  free_buffers();
}

void GpuDataBuffer::allocate_buffers() {
  if (capacity_ == 0) {
    return;
  }

  const std::size_t num_bytes = capacity_ * sizeof(float);
  const std::size_t int_bytes = capacity_ * sizeof(int);
  const std::size_t alive_bytes = capacity_ * sizeof(uint32_t);
  const std::size_t type_bytes = capacity_ * sizeof(uint8_t);
  const std::size_t sensor_bytes =
      capacity_ * kSensorInputsPerEntity * sizeof(float);
  const std::size_t brain_bytes =
      capacity_ * kBrainOutputsPerEntity * sizeof(float);

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&h_pos_x_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_pos_y_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_x_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_y_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_speed_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_energy_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_age_, int_bytes));
  CUDA_CHECK(cudaMallocHost(&h_alive_, alive_bytes));
  CUDA_CHECK(cudaMallocHost(&h_types_, type_bytes));
  CUDA_CHECK(cudaMallocHost(&h_reproduction_cooldown_, int_bytes));
  CUDA_CHECK(cudaMallocHost(&h_species_ids_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMallocHost(&h_distance_traveled_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_kill_counts_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMallocHost(&h_killed_by_, int_bytes));
  CUDA_CHECK(cudaMallocHost(&h_sensor_inputs_, sensor_bytes));
  CUDA_CHECK(cudaMallocHost(&h_brain_outputs_, brain_bytes));

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_pos_x_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_pos_y_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_x_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_y_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_speed_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_energy_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_age_, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_alive_, alive_bytes));
  CUDA_CHECK(cudaMalloc(&d_types_, type_bytes));
  CUDA_CHECK(cudaMalloc(&d_reproduction_cooldown_, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_species_ids_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_distance_traveled_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_kill_counts_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_killed_by_, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_sensor_inputs_, sensor_bytes));
  CUDA_CHECK(cudaMalloc(&d_brain_outputs_, brain_bytes));
}

void GpuDataBuffer::free_buffers() {
  // Free host memory
  if (h_pos_x_)
    cudaFreeHost(h_pos_x_);
  if (h_pos_y_)
    cudaFreeHost(h_pos_y_);
  if (h_vel_x_)
    cudaFreeHost(h_vel_x_);
  if (h_vel_y_)
    cudaFreeHost(h_vel_y_);
  if (h_speed_)
    cudaFreeHost(h_speed_);
  if (h_energy_)
    cudaFreeHost(h_energy_);
  if (h_age_)
    cudaFreeHost(h_age_);
  if (h_alive_)
    cudaFreeHost(h_alive_);
  if (h_types_)
    cudaFreeHost(h_types_);
  if (h_reproduction_cooldown_)
    cudaFreeHost(h_reproduction_cooldown_);
  if (h_species_ids_)
    cudaFreeHost(h_species_ids_);
  if (h_distance_traveled_)
    cudaFreeHost(h_distance_traveled_);
  if (h_kill_counts_)
    cudaFreeHost(h_kill_counts_);
  if (h_killed_by_)
    cudaFreeHost(h_killed_by_);
  if (h_sensor_inputs_)
    cudaFreeHost(h_sensor_inputs_);
  if (h_brain_outputs_)
    cudaFreeHost(h_brain_outputs_);

  // Free device memory
  if (d_pos_x_)
    cudaFree(d_pos_x_);
  if (d_pos_y_)
    cudaFree(d_pos_y_);
  if (d_vel_x_)
    cudaFree(d_vel_x_);
  if (d_vel_y_)
    cudaFree(d_vel_y_);
  if (d_speed_)
    cudaFree(d_speed_);
  if (d_energy_)
    cudaFree(d_energy_);
  if (d_age_)
    cudaFree(d_age_);
  if (d_alive_)
    cudaFree(d_alive_);
  if (d_types_)
    cudaFree(d_types_);
  if (d_reproduction_cooldown_)
    cudaFree(d_reproduction_cooldown_);
  if (d_species_ids_)
    cudaFree(d_species_ids_);
  if (d_distance_traveled_)
    cudaFree(d_distance_traveled_);
  if (d_kill_counts_)
    cudaFree(d_kill_counts_);
  if (d_killed_by_)
    cudaFree(d_killed_by_);
  if (d_sensor_inputs_)
    cudaFree(d_sensor_inputs_);
  if (d_brain_outputs_)
    cudaFree(d_brain_outputs_);

  // Reset pointers
  h_pos_x_ = h_pos_y_ = h_vel_x_ = h_vel_y_ = h_speed_ = h_energy_ = nullptr;
  h_age_ = nullptr;
  h_alive_ = nullptr;
  h_types_ = nullptr;
  h_reproduction_cooldown_ = nullptr;
  h_species_ids_ = nullptr;
  h_distance_traveled_ = nullptr;
  h_kill_counts_ = nullptr;
  h_killed_by_ = nullptr;
  h_sensor_inputs_ = h_brain_outputs_ = nullptr;
  d_pos_x_ = d_pos_y_ = d_vel_x_ = d_vel_y_ = d_speed_ = d_energy_ = nullptr;
  d_age_ = nullptr;
  d_alive_ = nullptr;
  d_types_ = nullptr;
  d_reproduction_cooldown_ = nullptr;
  d_species_ids_ = nullptr;
  d_distance_traveled_ = nullptr;
  d_kill_counts_ = nullptr;
  d_killed_by_ = nullptr;
  d_sensor_inputs_ = d_brain_outputs_ = nullptr;
}

void GpuDataBuffer::upload_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t num_bytes = count * sizeof(float);
  const std::size_t int_bytes = count * sizeof(int);
  const std::size_t alive_bytes = count * sizeof(uint32_t);
  const std::size_t type_bytes = count * sizeof(uint8_t);

  CUDA_CHECK(cudaMemcpyAsync(d_pos_x_, h_pos_x_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pos_y_, h_pos_y_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_x_, h_vel_x_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_y_, h_vel_y_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_speed_, h_speed_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_energy_, h_energy_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_age_, h_age_, int_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_alive_, h_alive_, alive_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_types_, h_types_, type_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_reproduction_cooldown_, h_reproduction_cooldown_,
                             int_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_species_ids_, h_species_ids_,
                             count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_distance_traveled_, h_distance_traveled_,
                             num_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_kill_counts_, h_kill_counts_,
                             count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_killed_by_, h_killed_by_, int_bytes,
                             cudaMemcpyHostToDevice, stream));
}

void GpuDataBuffer::download_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t num_bytes = count * sizeof(float);
  const std::size_t int_bytes = count * sizeof(int);
  const std::size_t alive_bytes = count * sizeof(uint32_t);

  // Download all modified fields back to host buffers
  CUDA_CHECK(cudaMemcpyAsync(h_pos_x_, d_pos_x_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_pos_y_, d_pos_y_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_vel_x_, d_vel_x_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_vel_y_, d_vel_y_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_energy_, d_energy_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_age_, d_age_, int_bytes, cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_alive_, d_alive_, alive_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_reproduction_cooldown_,
                             d_reproduction_cooldown_, int_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_distance_traveled_, d_distance_traveled_,
                             num_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_kill_counts_, d_kill_counts_,
                             count * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_killed_by_, d_killed_by_, int_bytes,
                             cudaMemcpyDeviceToHost, stream));
}

} // namespace gpu
} // namespace moonai
