#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_data_buffer.hpp"

namespace moonai {
namespace gpu {

GpuDataBuffer::GpuDataBuffer(std::size_t max_agents, std::size_t max_food)
    : agent_capacity_(max_agents), food_capacity_(max_food) {
  allocate_buffers();
}

GpuDataBuffer::~GpuDataBuffer() {
  free_buffers();
}

void GpuDataBuffer::allocate_buffers() {
  const std::size_t agent_float_bytes = agent_capacity_ * sizeof(float);
  const std::size_t agent_int_bytes = agent_capacity_ * sizeof(int);
  const std::size_t agent_u32_bytes = agent_capacity_ * sizeof(uint32_t);
  const std::size_t agent_type_bytes = agent_capacity_ * sizeof(uint8_t);
  const std::size_t sensor_bytes =
      agent_capacity_ * kSensorInputsPerEntity * sizeof(float);
  const std::size_t brain_bytes =
      agent_capacity_ * kBrainOutputsPerEntity * sizeof(float);

  const std::size_t food_float_bytes = food_capacity_ * sizeof(float);
  const std::size_t food_u32_bytes = food_capacity_ * sizeof(uint32_t);
  const std::size_t food_int_bytes = food_capacity_ * sizeof(int);

  if (agent_capacity_ > 0) {
    CUDA_CHECK(cudaMallocHost(&h_agent_pos_x_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_pos_y_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_vel_x_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_vel_y_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_speed_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_energy_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_age_, agent_int_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_alive_, agent_u32_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_types_, agent_type_bytes));
    CUDA_CHECK(
        cudaMallocHost(&h_agent_reproduction_cooldown_, agent_int_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_distance_traveled_, agent_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_kill_counts_, agent_u32_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_killed_by_, agent_int_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_sensor_inputs_, sensor_bytes));
    CUDA_CHECK(cudaMallocHost(&h_agent_brain_outputs_, brain_bytes));

    CUDA_CHECK(cudaMalloc(&d_agent_pos_x_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_pos_y_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_vel_x_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_vel_y_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_speed_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_energy_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_age_, agent_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_alive_, agent_u32_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_types_, agent_type_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_reproduction_cooldown_, agent_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_distance_traveled_, agent_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_kill_counts_, agent_u32_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_killed_by_, agent_int_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_sensor_inputs_, sensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_agent_brain_outputs_, brain_bytes));
  }

  if (food_capacity_ > 0) {
    CUDA_CHECK(cudaMallocHost(&h_food_pos_x_, food_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_food_pos_y_, food_float_bytes));
    CUDA_CHECK(cudaMallocHost(&h_food_active_, food_u32_bytes));
    CUDA_CHECK(cudaMallocHost(&h_food_consumed_by_, food_int_bytes));

    CUDA_CHECK(cudaMalloc(&d_food_pos_x_, food_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_food_pos_y_, food_float_bytes));
    CUDA_CHECK(cudaMalloc(&d_food_active_, food_u32_bytes));
    CUDA_CHECK(cudaMalloc(&d_food_consumed_by_, food_int_bytes));
  }
}

void GpuDataBuffer::free_buffers() {
  if (h_agent_pos_x_)
    cudaFreeHost(h_agent_pos_x_);
  if (h_agent_pos_y_)
    cudaFreeHost(h_agent_pos_y_);
  if (h_agent_vel_x_)
    cudaFreeHost(h_agent_vel_x_);
  if (h_agent_vel_y_)
    cudaFreeHost(h_agent_vel_y_);
  if (h_agent_speed_)
    cudaFreeHost(h_agent_speed_);
  if (h_agent_energy_)
    cudaFreeHost(h_agent_energy_);
  if (h_agent_age_)
    cudaFreeHost(h_agent_age_);
  if (h_agent_alive_)
    cudaFreeHost(h_agent_alive_);
  if (h_agent_types_)
    cudaFreeHost(h_agent_types_);
  if (h_agent_reproduction_cooldown_)
    cudaFreeHost(h_agent_reproduction_cooldown_);
  if (h_agent_distance_traveled_)
    cudaFreeHost(h_agent_distance_traveled_);
  if (h_agent_kill_counts_)
    cudaFreeHost(h_agent_kill_counts_);
  if (h_agent_killed_by_)
    cudaFreeHost(h_agent_killed_by_);
  if (h_agent_sensor_inputs_)
    cudaFreeHost(h_agent_sensor_inputs_);
  if (h_agent_brain_outputs_)
    cudaFreeHost(h_agent_brain_outputs_);

  if (h_food_pos_x_)
    cudaFreeHost(h_food_pos_x_);
  if (h_food_pos_y_)
    cudaFreeHost(h_food_pos_y_);
  if (h_food_active_)
    cudaFreeHost(h_food_active_);
  if (h_food_consumed_by_)
    cudaFreeHost(h_food_consumed_by_);

  if (d_agent_pos_x_)
    cudaFree(d_agent_pos_x_);
  if (d_agent_pos_y_)
    cudaFree(d_agent_pos_y_);
  if (d_agent_vel_x_)
    cudaFree(d_agent_vel_x_);
  if (d_agent_vel_y_)
    cudaFree(d_agent_vel_y_);
  if (d_agent_speed_)
    cudaFree(d_agent_speed_);
  if (d_agent_energy_)
    cudaFree(d_agent_energy_);
  if (d_agent_age_)
    cudaFree(d_agent_age_);
  if (d_agent_alive_)
    cudaFree(d_agent_alive_);
  if (d_agent_types_)
    cudaFree(d_agent_types_);
  if (d_agent_reproduction_cooldown_)
    cudaFree(d_agent_reproduction_cooldown_);
  if (d_agent_distance_traveled_)
    cudaFree(d_agent_distance_traveled_);
  if (d_agent_kill_counts_)
    cudaFree(d_agent_kill_counts_);
  if (d_agent_killed_by_)
    cudaFree(d_agent_killed_by_);
  if (d_agent_sensor_inputs_)
    cudaFree(d_agent_sensor_inputs_);
  if (d_agent_brain_outputs_)
    cudaFree(d_agent_brain_outputs_);

  if (d_food_pos_x_)
    cudaFree(d_food_pos_x_);
  if (d_food_pos_y_)
    cudaFree(d_food_pos_y_);
  if (d_food_active_)
    cudaFree(d_food_active_);
  if (d_food_consumed_by_)
    cudaFree(d_food_consumed_by_);
}

void GpuDataBuffer::upload_async(std::size_t agent_count, std::size_t food_count,
                                 cudaStream_t stream) {
  if (agent_count > agent_capacity_ || food_count > food_capacity_) {
    return;
  }

  if (agent_count > 0) {
    const std::size_t agent_float_bytes = agent_count * sizeof(float);
    const std::size_t agent_int_bytes = agent_count * sizeof(int);
    const std::size_t agent_u32_bytes = agent_count * sizeof(uint32_t);
    const std::size_t agent_type_bytes = agent_count * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpyAsync(d_agent_pos_x_, h_agent_pos_x_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_pos_y_, h_agent_pos_y_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_vel_x_, h_agent_vel_x_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_vel_y_, h_agent_vel_y_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_speed_, h_agent_speed_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_energy_, h_agent_energy_,
                               agent_float_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_age_, h_agent_age_, agent_int_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_alive_, h_agent_alive_, agent_u32_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_types_, h_agent_types_, agent_type_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_reproduction_cooldown_,
                               h_agent_reproduction_cooldown_, agent_int_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_agent_distance_traveled_,
                               h_agent_distance_traveled_, agent_float_bytes,
                               cudaMemcpyHostToDevice, stream));
  }

  if (food_count > 0) {
    const std::size_t food_float_bytes = food_count * sizeof(float);
    const std::size_t food_u32_bytes = food_count * sizeof(uint32_t);
    const std::size_t food_int_bytes = food_count * sizeof(int);

    CUDA_CHECK(cudaMemcpyAsync(d_food_pos_x_, h_food_pos_x_, food_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_food_pos_y_, h_food_pos_y_, food_float_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_food_active_, h_food_active_, food_u32_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_food_consumed_by_, h_food_consumed_by_,
                               food_int_bytes, cudaMemcpyHostToDevice, stream));
  }
}

void GpuDataBuffer::download_async(std::size_t agent_count,
                                   std::size_t food_count,
                                   cudaStream_t stream) {
  if (agent_count > agent_capacity_ || food_count > food_capacity_) {
    return;
  }

  if (agent_count > 0) {
    const std::size_t agent_float_bytes = agent_count * sizeof(float);
    const std::size_t agent_int_bytes = agent_count * sizeof(int);
    const std::size_t agent_u32_bytes = agent_count * sizeof(uint32_t);
    const std::size_t sensor_bytes =
        agent_count * kSensorInputsPerEntity * sizeof(float);
    const std::size_t brain_bytes =
        agent_count * kBrainOutputsPerEntity * sizeof(float);

    CUDA_CHECK(cudaMemcpyAsync(h_agent_pos_x_, d_agent_pos_x_, agent_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_pos_y_, d_agent_pos_y_, agent_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_vel_x_, d_agent_vel_x_, agent_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_vel_y_, d_agent_vel_y_, agent_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_energy_, d_agent_energy_,
                               agent_float_bytes, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_age_, d_agent_age_, agent_int_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_alive_, d_agent_alive_, agent_u32_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_reproduction_cooldown_,
                               d_agent_reproduction_cooldown_, agent_int_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_distance_traveled_,
                               d_agent_distance_traveled_, agent_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_kill_counts_, d_agent_kill_counts_,
                               agent_u32_bytes, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_killed_by_, d_agent_killed_by_,
                               agent_int_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_sensor_inputs_, d_agent_sensor_inputs_,
                               sensor_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_agent_brain_outputs_, d_agent_brain_outputs_,
                               brain_bytes, cudaMemcpyDeviceToHost, stream));
  }

  if (food_count > 0) {
    const std::size_t food_float_bytes = food_count * sizeof(float);
    const std::size_t food_u32_bytes = food_count * sizeof(uint32_t);
    const std::size_t food_int_bytes = food_count * sizeof(int);

    CUDA_CHECK(cudaMemcpyAsync(h_food_pos_x_, d_food_pos_x_, food_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_food_pos_y_, d_food_pos_y_, food_float_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_food_active_, d_food_active_, food_u32_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_food_consumed_by_, d_food_consumed_by_,
                               food_int_bytes, cudaMemcpyDeviceToHost, stream));
  }
}

} // namespace gpu
} // namespace moonai
