#include "gpu/gpu_data_buffer.hpp"
#include "gpu/cuda_utils.cuh"
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
  const std::size_t bool_bytes = capacity_ * sizeof(uint8_t);
  const std::size_t sensor_bytes =
      capacity_ * kSensorInputsPerEntity * sizeof(float);
  const std::size_t brain_bytes =
      capacity_ * kBrainOutputsPerEntity * sizeof(float);

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&h_pos_x_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_pos_y_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_x_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_y_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_energy_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_age_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_alive_, bool_bytes));
  CUDA_CHECK(cudaMallocHost(&h_types_, bool_bytes));
  CUDA_CHECK(cudaMallocHost(&h_species_ids_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMallocHost(&h_sensor_inputs_, sensor_bytes));
  CUDA_CHECK(cudaMallocHost(&h_brain_outputs_, brain_bytes));

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_pos_x_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_pos_y_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_x_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_y_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_energy_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_age_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_alive_, bool_bytes));
  CUDA_CHECK(cudaMalloc(&d_types_, bool_bytes));
  CUDA_CHECK(cudaMalloc(&d_species_ids_, capacity_ * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sensor_inputs_, sensor_bytes));
  CUDA_CHECK(cudaMalloc(&d_brain_outputs_, brain_bytes));

  // Allocate output buffers (host)
  CUDA_CHECK(cudaMallocHost(&h_out_energy_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_out_alive_, bool_bytes));
  CUDA_CHECK(cudaMallocHost(&h_out_vel_x_, num_bytes));
  CUDA_CHECK(cudaMallocHost(&h_out_vel_y_, num_bytes));

  // Allocate output buffers (device)
  CUDA_CHECK(cudaMalloc(&d_out_energy_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_out_alive_, bool_bytes));
  CUDA_CHECK(cudaMalloc(&d_out_vel_x_, num_bytes));
  CUDA_CHECK(cudaMalloc(&d_out_vel_y_, num_bytes));
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
  if (h_energy_)
    cudaFreeHost(h_energy_);
  if (h_age_)
    cudaFreeHost(h_age_);
  if (h_alive_)
    cudaFreeHost(h_alive_);
  if (h_types_)
    cudaFreeHost(h_types_);
  if (h_species_ids_)
    cudaFreeHost(h_species_ids_);
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
  if (d_energy_)
    cudaFree(d_energy_);
  if (d_age_)
    cudaFree(d_age_);
  if (d_alive_)
    cudaFree(d_alive_);
  if (d_types_)
    cudaFree(d_types_);
  if (d_species_ids_)
    cudaFree(d_species_ids_);
  if (d_sensor_inputs_)
    cudaFree(d_sensor_inputs_);
  if (d_brain_outputs_)
    cudaFree(d_brain_outputs_);

  // Free output buffers
  if (h_out_energy_)
    cudaFreeHost(h_out_energy_);
  if (h_out_alive_)
    cudaFreeHost(h_out_alive_);
  if (h_out_vel_x_)
    cudaFreeHost(h_out_vel_x_);
  if (h_out_vel_y_)
    cudaFreeHost(h_out_vel_y_);
  if (d_out_energy_)
    cudaFree(d_out_energy_);
  if (d_out_alive_)
    cudaFree(d_out_alive_);
  if (d_out_vel_x_)
    cudaFree(d_out_vel_x_);
  if (d_out_vel_y_)
    cudaFree(d_out_vel_y_);

  // Reset pointers
  h_pos_x_ = h_pos_y_ = h_vel_x_ = h_vel_y_ = h_energy_ = h_age_ = nullptr;
  h_alive_ = h_types_ = nullptr;
  h_species_ids_ = nullptr;
  h_sensor_inputs_ = h_brain_outputs_ = nullptr;
  d_pos_x_ = d_pos_y_ = d_vel_x_ = d_vel_y_ = d_energy_ = d_age_ = nullptr;
  d_alive_ = d_types_ = nullptr;
  d_species_ids_ = nullptr;
  d_sensor_inputs_ = d_brain_outputs_ = nullptr;
  h_out_energy_ = nullptr;
  h_out_alive_ = nullptr;
  h_out_vel_x_ = nullptr;
  h_out_vel_y_ = nullptr;
  d_out_energy_ = nullptr;
  d_out_alive_ = nullptr;
  d_out_vel_x_ = nullptr;
  d_out_vel_y_ = nullptr;
}

void GpuDataBuffer::upload_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t num_bytes = count * sizeof(float);
  const std::size_t bool_bytes = count * sizeof(uint8_t);
  const std::size_t sensor_bytes =
      count * kSensorInputsPerEntity * sizeof(float);
  const std::size_t brain_bytes =
      count * kBrainOutputsPerEntity * sizeof(float);

  CUDA_CHECK(cudaMemcpyAsync(d_pos_x_, h_pos_x_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pos_y_, h_pos_y_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_x_, h_vel_x_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_y_, h_vel_y_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_energy_, h_energy_, num_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_age_, h_age_, num_bytes, cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_alive_, h_alive_, bool_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_types_, h_types_, bool_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_species_ids_, h_species_ids_,
                             count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_sensor_inputs_, h_sensor_inputs_, sensor_bytes,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_brain_outputs_, h_brain_outputs_, brain_bytes,
                             cudaMemcpyHostToDevice, stream));
}

void GpuDataBuffer::download_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t num_bytes = count * sizeof(float);
  const std::size_t bool_bytes = count * sizeof(uint8_t);

  CUDA_CHECK(cudaMemcpyAsync(h_out_energy_, d_out_energy_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out_alive_, d_out_alive_, bool_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out_vel_x_, d_out_vel_x_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out_vel_y_, d_out_vel_y_, num_bytes,
                             cudaMemcpyDeviceToHost, stream));
}

} // namespace gpu
} // namespace moonai
