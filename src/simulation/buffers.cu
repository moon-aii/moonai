#include "simulation/buffers.hpp"
#include "simulation/checks.cuh"

namespace moonai::simulation {

PopulationBuffer::PopulationBuffer(std::size_t max_agents) : capacity_(max_agents) {
  allocate_buffers();
}

PopulationBuffer::~PopulationBuffer() {
  free_buffers();
}

void PopulationBuffer::allocate_buffers() {
  if (capacity_ == 0) {
    return;
  }

  const std::size_t float_bytes = capacity_ * sizeof(float);
  const std::size_t int_bytes = capacity_ * sizeof(int);
  const std::size_t u32_bytes = capacity_ * sizeof(uint32_t);
  const std::size_t sensor_bytes = capacity_ * kSensorInputsPerEntity * sizeof(float);
  const std::size_t brain_bytes = capacity_ * kBrainOutputsPerEntity * sizeof(float);

  CUDA_CHECK(cudaMallocHost(&h_pos_x_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_pos_y_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_x_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_vel_y_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_energy_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_age_, int_bytes));
  CUDA_CHECK(cudaMallocHost(&h_alive_, u32_bytes));
  CUDA_CHECK(cudaMallocHost(&h_kill_counts_, u32_bytes));
  CUDA_CHECK(cudaMallocHost(&h_claimed_by_, int_bytes));
  CUDA_CHECK(cudaMallocHost(&h_brain_outputs_, brain_bytes));

  CUDA_CHECK(cudaMalloc(&d_pos_x_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_pos_y_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_x_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_vel_y_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_energy_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_age_, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_alive_, u32_bytes));
  CUDA_CHECK(cudaMalloc(&d_kill_counts_, u32_bytes));
  CUDA_CHECK(cudaMalloc(&d_claimed_by_, int_bytes));
  CUDA_CHECK(cudaMalloc(&d_sensor_inputs_, sensor_bytes));
  CUDA_CHECK(cudaMalloc(&d_brain_outputs_, brain_bytes));
}

void PopulationBuffer::free_buffers() {
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
  if (h_kill_counts_)
    cudaFreeHost(h_kill_counts_);
  if (h_claimed_by_)
    cudaFreeHost(h_claimed_by_);
  if (h_brain_outputs_)
    cudaFreeHost(h_brain_outputs_);

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
  if (d_kill_counts_)
    cudaFree(d_kill_counts_);
  if (d_claimed_by_)
    cudaFree(d_claimed_by_);
  if (d_sensor_inputs_)
    cudaFree(d_sensor_inputs_);
  if (d_brain_outputs_)
    cudaFree(d_brain_outputs_);

  h_pos_x_ = nullptr;
  h_pos_y_ = nullptr;
  h_vel_x_ = nullptr;
  h_vel_y_ = nullptr;
  h_energy_ = nullptr;
  h_age_ = nullptr;
  h_alive_ = nullptr;
  h_kill_counts_ = nullptr;
  h_claimed_by_ = nullptr;
  h_brain_outputs_ = nullptr;

  d_pos_x_ = nullptr;
  d_pos_y_ = nullptr;
  d_vel_x_ = nullptr;
  d_vel_y_ = nullptr;
  d_energy_ = nullptr;
  d_age_ = nullptr;
  d_alive_ = nullptr;
  d_kill_counts_ = nullptr;
  d_claimed_by_ = nullptr;
  d_sensor_inputs_ = nullptr;
  d_brain_outputs_ = nullptr;
}

void PopulationBuffer::upload_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t int_bytes = count * sizeof(int);
  const std::size_t u32_bytes = count * sizeof(uint32_t);

  CUDA_CHECK(cudaMemcpyAsync(d_pos_x_, h_pos_x_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pos_y_, h_pos_y_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_x_, h_vel_x_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_vel_y_, h_vel_y_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_energy_, h_energy_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_age_, h_age_, int_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_alive_, h_alive_, u32_bytes, cudaMemcpyHostToDevice, stream));
}

void PopulationBuffer::download_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t int_bytes = count * sizeof(int);
  const std::size_t u32_bytes = count * sizeof(uint32_t);
  const std::size_t brain_bytes = count * kBrainOutputsPerEntity * sizeof(float);

  CUDA_CHECK(cudaMemcpyAsync(h_pos_x_, d_pos_x_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_pos_y_, d_pos_y_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_vel_x_, d_vel_x_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_vel_y_, d_vel_y_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_energy_, d_energy_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_age_, d_age_, int_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_alive_, d_alive_, u32_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_kill_counts_, d_kill_counts_, u32_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_claimed_by_, d_claimed_by_, int_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_brain_outputs_, d_brain_outputs_, brain_bytes, cudaMemcpyDeviceToHost, stream));
}

void PopulationBuffer::reset(std::size_t capacity) {
  if (capacity == capacity_) {
    return;
  }

  free_buffers();
  capacity_ = capacity;
  allocate_buffers();
}

FoodBuffer::FoodBuffer(std::size_t max_food) : capacity_(max_food) {
  allocate_buffers();
}

FoodBuffer::~FoodBuffer() {
  free_buffers();
}

void FoodBuffer::allocate_buffers() {
  if (capacity_ == 0) {
    return;
  }

  const std::size_t float_bytes = capacity_ * sizeof(float);
  const std::size_t u32_bytes = capacity_ * sizeof(uint32_t);
  const std::size_t int_bytes = capacity_ * sizeof(int);

  CUDA_CHECK(cudaMallocHost(&h_pos_x_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_pos_y_, float_bytes));
  CUDA_CHECK(cudaMallocHost(&h_active_, u32_bytes));
  CUDA_CHECK(cudaMallocHost(&h_consumed_by_, int_bytes));

  CUDA_CHECK(cudaMalloc(&d_pos_x_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_pos_y_, float_bytes));
  CUDA_CHECK(cudaMalloc(&d_active_, u32_bytes));
  CUDA_CHECK(cudaMalloc(&d_consumed_by_, int_bytes));
}

void FoodBuffer::free_buffers() {
  if (h_pos_x_)
    cudaFreeHost(h_pos_x_);
  if (h_pos_y_)
    cudaFreeHost(h_pos_y_);
  if (h_active_)
    cudaFreeHost(h_active_);
  if (h_consumed_by_)
    cudaFreeHost(h_consumed_by_);

  if (d_pos_x_)
    cudaFree(d_pos_x_);
  if (d_pos_y_)
    cudaFree(d_pos_y_);
  if (d_active_)
    cudaFree(d_active_);
  if (d_consumed_by_)
    cudaFree(d_consumed_by_);

  h_pos_x_ = nullptr;
  h_pos_y_ = nullptr;
  h_active_ = nullptr;
  h_consumed_by_ = nullptr;

  d_pos_x_ = nullptr;
  d_pos_y_ = nullptr;
  d_active_ = nullptr;
  d_consumed_by_ = nullptr;
}

void FoodBuffer::upload_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t u32_bytes = count * sizeof(uint32_t);
  const std::size_t int_bytes = count * sizeof(int);

  CUDA_CHECK(cudaMemcpyAsync(d_pos_x_, h_pos_x_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pos_y_, h_pos_y_, float_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_active_, h_active_, u32_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_consumed_by_, h_consumed_by_, int_bytes, cudaMemcpyHostToDevice, stream));
}

void FoodBuffer::download_async(std::size_t count, cudaStream_t stream) {
  if (count == 0 || count > capacity_) {
    return;
  }

  const std::size_t float_bytes = count * sizeof(float);
  const std::size_t u32_bytes = count * sizeof(uint32_t);
  const std::size_t int_bytes = count * sizeof(int);

  CUDA_CHECK(cudaMemcpyAsync(h_pos_x_, d_pos_x_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_pos_y_, d_pos_y_, float_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_active_, d_active_, u32_bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_consumed_by_, d_consumed_by_, int_bytes, cudaMemcpyDeviceToHost, stream));
}

void FoodBuffer::reset(std::size_t capacity) {
  if (capacity == capacity_) {
    return;
  }

  free_buffers();
  capacity_ = capacity;
  allocate_buffers();
}

} // namespace moonai::simulation
