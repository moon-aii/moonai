#include "gpu/cuda_utils.cuh"
#include "gpu/gpu_batch.hpp"

#include <cub/cub.cuh>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <tuple>

namespace moonai::gpu {

namespace {

bool check_cuda(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
            cudaGetErrorString(err));
    return false;
  }
  return true;
}

template <typename T>
bool malloc_checked(T **ptr, size_t bytes, const char *file, int line) {
  if (bytes == 0) {
    *ptr = nullptr;
    return true;
  }
  return check_cuda(cudaMalloc(reinterpret_cast<void **>(ptr), bytes), file,
                    line);
}

template <typename T>
bool malloc_host_checked(T **ptr, size_t bytes, const char *file, int line) {
  if (bytes == 0) {
    *ptr = nullptr;
    return true;
  }
  return check_cuda(cudaMallocHost(reinterpret_cast<void **>(ptr), bytes), file,
                    line);
}

bool memcpy_async_checked(void *dst, const void *src, size_t bytes,
                          cudaMemcpyKind kind, cudaStream_t stream,
                          const char *file, int line) {
  if (bytes == 0) {
    return true;
  }
  return check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), file, line);
}

__global__ void unpack_agent_states_kernel(
    const GpuAgentState *__restrict__ src, float *__restrict__ pos_x,
    float *__restrict__ pos_y, float *__restrict__ vel_x,
    float *__restrict__ vel_y, float *__restrict__ speed,
    float *__restrict__ vision, float *__restrict__ energy,
    float *__restrict__ distance_traveled, int *__restrict__ age,
    int *__restrict__ kills, int *__restrict__ food_eaten,
    unsigned int *__restrict__ ids, unsigned int *__restrict__ types,
    unsigned int *__restrict__ alive, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const GpuAgentState state = src[idx];
  pos_x[idx] = state.pos_x;
  pos_y[idx] = state.pos_y;
  vel_x[idx] = state.vel_x;
  vel_y[idx] = state.vel_y;
  speed[idx] = state.speed;
  vision[idx] = state.vision_range;
  energy[idx] = state.energy;
  distance_traveled[idx] = state.distance_traveled;
  age[idx] = state.age;
  kills[idx] = state.kills;
  food_eaten[idx] = state.food_eaten;
  ids[idx] = state.id;
  types[idx] = state.type;
  alive[idx] = state.alive;
}

__global__ void pack_agent_states_kernel(
    GpuAgentState *__restrict__ dst, const float *__restrict__ pos_x,
    const float *__restrict__ pos_y, const float *__restrict__ vel_x,
    const float *__restrict__ vel_y, const float *__restrict__ speed,
    const float *__restrict__ vision, const float *__restrict__ energy,
    const float *__restrict__ distance_traveled, const int *__restrict__ age,
    const int *__restrict__ kills, const int *__restrict__ food_eaten,
    const unsigned int *__restrict__ ids,
    const unsigned int *__restrict__ types,
    const unsigned int *__restrict__ alive, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  dst[idx] = GpuAgentState{
      pos_x[idx], pos_y[idx],  vel_x[idx],      vel_y[idx],
             speed[idx], vision[idx], energy[idx],     distance_traveled[idx],
      age[idx],   kills[idx],  food_eaten[idx], ids[idx],
      types[idx], alive[idx]};
}

__global__ void unpack_food_states_kernel(const GpuFoodState *__restrict__ src,
                                          float *__restrict__ pos_x,
                                          float *__restrict__ pos_y,
                                          unsigned int *__restrict__ active,
                                          int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const GpuFoodState state = src[idx];
  pos_x[idx] = state.pos_x;
  pos_y[idx] = state.pos_y;
  active[idx] = state.active;
}

__global__ void pack_food_states_kernel(GpuFoodState *__restrict__ dst,
                                        const float *__restrict__ pos_x,
                                        const float *__restrict__ pos_y,
                                        const unsigned int *__restrict__ active,
                                        int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  dst[idx] = GpuFoodState{pos_x[idx], pos_y[idx], active[idx]};
}

} // namespace

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs,
                   float world_width, float world_height)
    : num_agents_(num_agents), num_inputs_(num_inputs),
      num_outputs_(num_outputs), world_width_(world_width),
      world_height_(world_height) {
  const size_t in_bytes =
      static_cast<size_t>(num_agents) * num_inputs * sizeof(float);
  const size_t out_bytes =
      static_cast<size_t>(num_agents) * num_outputs * sizeof(float);
  const size_t agent_float_bytes =
      static_cast<size_t>(num_agents) * sizeof(float);
  const size_t agent_int_bytes = static_cast<size_t>(num_agents) * sizeof(int);
  const size_t agent_uint_bytes =
      static_cast<size_t>(num_agents) * sizeof(unsigned int);

  had_error_ |= !malloc_checked(&d_descs_, num_agents * sizeof(GpuNetDesc),
                                __FILE__, __LINE__);
  had_error_ |= !malloc_checked(&d_inputs_, in_bytes, __FILE__, __LINE__);
  had_error_ |= !malloc_checked(&d_outputs_, out_bytes, __FILE__, __LINE__);
  had_error_ |= !malloc_checked(
      &d_agent_states_, num_agents * sizeof(GpuAgentState), __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_pos_x_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_pos_y_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_vel_x_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_vel_y_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_speed_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_vision_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_energy_, agent_float_bytes, __FILE__, __LINE__);
  had_error_ |= !malloc_checked(&d_agent_distance_traveled_, agent_float_bytes,
                                __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_age_, agent_int_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_kills_, agent_int_bytes, __FILE__, __LINE__);
  had_error_ |= !malloc_checked(&d_agent_food_eaten_, agent_int_bytes, __FILE__,
                                __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_ids_, agent_uint_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_types_, agent_uint_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_checked(&d_agent_alive_, agent_uint_bytes, __FILE__, __LINE__);

  had_error_ |=
      !malloc_host_checked(&h_pinned_in_, in_bytes, __FILE__, __LINE__);
  had_error_ |=
      !malloc_host_checked(&h_pinned_out_, out_bytes, __FILE__, __LINE__);

  cudaStream_t s = nullptr;
  had_error_ |= !check_cuda(cudaStreamCreate(&s), __FILE__, __LINE__);
  stream_ = static_cast<void *>(s);
}

GpuBatch::~GpuBatch() {
  // Device arrays
  if (d_descs_)
    cudaFree(d_descs_);
  if (d_inputs_)
    cudaFree(d_inputs_);
  if (d_outputs_)
    cudaFree(d_outputs_);
  if (d_agent_states_)
    cudaFree(d_agent_states_);
  if (d_food_states_)
    cudaFree(d_food_states_);
  if (d_agent_cell_offsets_)
    cudaFree(d_agent_cell_offsets_);
  if (d_food_cell_offsets_)
    cudaFree(d_food_cell_offsets_);
  if (d_agent_cell_counts_)
    cudaFree(d_agent_cell_counts_);
  if (d_food_cell_counts_)
    cudaFree(d_food_cell_counts_);
  if (d_agent_cell_write_counts_)
    cudaFree(d_agent_cell_write_counts_);
  if (d_food_cell_write_counts_)
    cudaFree(d_food_cell_write_counts_);
  if (d_agent_cell_ids_)
    cudaFree(d_agent_cell_ids_);
  if (d_food_cell_ids_)
    cudaFree(d_food_cell_ids_);
  if (d_sensor_agent_entries_)
    cudaFree(d_sensor_agent_entries_);
  if (d_sensor_food_entries_)
    cudaFree(d_sensor_food_entries_);
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
  if (d_agent_vision_)
    cudaFree(d_agent_vision_);
  if (d_agent_energy_)
    cudaFree(d_agent_energy_);
  if (d_agent_distance_traveled_)
    cudaFree(d_agent_distance_traveled_);
  if (d_agent_age_)
    cudaFree(d_agent_age_);
  if (d_agent_kills_)
    cudaFree(d_agent_kills_);
  if (d_agent_food_eaten_)
    cudaFree(d_agent_food_eaten_);
  if (d_agent_ids_)
    cudaFree(d_agent_ids_);
  if (d_agent_types_)
    cudaFree(d_agent_types_);
  if (d_agent_alive_)
    cudaFree(d_agent_alive_);
  if (d_food_pos_x_)
    cudaFree(d_food_pos_x_);
  if (d_food_pos_y_)
    cudaFree(d_food_pos_y_);
  if (d_food_active_)
    cudaFree(d_food_active_);
  if (d_scan_temp_storage_)
    cudaFree(d_scan_temp_storage_);

  // Pinned host memory
  if (h_pinned_in_)
    cudaFreeHost(h_pinned_in_);
  if (h_pinned_out_)
    cudaFreeHost(h_pinned_out_);

  // Stream
  if (stream_)
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));

  // Topology arrays
  if (d_node_vals_)
    cudaFree(d_node_vals_);
  if (d_node_types_)
    cudaFree(d_node_types_);
  if (d_eval_order_)
    cudaFree(d_eval_order_);
  if (d_conn_ptr_)
    cudaFree(d_conn_ptr_);
  if (d_in_count_)
    cudaFree(d_in_count_);
  if (d_conn_from_)
    cudaFree(d_conn_from_);
  if (d_conn_w_)
    cudaFree(d_conn_w_);
  if (d_out_indices_)
    cudaFree(d_out_indices_);
}

bool GpuBatch::ensure_scan_temp_storage(size_t bytes) {
  if (bytes <= scan_temp_storage_bytes_) {
    return true;
  }
  if (d_scan_temp_storage_) {
    cudaFree(d_scan_temp_storage_);
    d_scan_temp_storage_ = nullptr;
    scan_temp_storage_bytes_ = 0;
  }
  if (!malloc_checked(reinterpret_cast<unsigned char **>(&d_scan_temp_storage_),
                      bytes, __FILE__, __LINE__)) {
    had_error_ = true;
    return false;
  }
  scan_temp_storage_bytes_ = bytes;
  return true;
}

// ── upload_network_data ──────────────────────────────────────────────────────

void GpuBatch::upload_network_data(const GpuNetworkData &data) {
  had_error_ = (stream_ == nullptr || d_descs_ == nullptr ||
                d_inputs_ == nullptr || d_outputs_ == nullptr ||
                h_pinned_in_ == nullptr || h_pinned_out_ == nullptr);
  activation_fn_id_ = (data.activation_fn_id >= 0 && data.activation_fn_id <= 2)
                          ? data.activation_fn_id
                          : 0;

  if (had_error_) {
    had_error_ = true;
    return;
  }

  if (static_cast<int>(data.descs.size()) != num_agents_) {
    had_error_ = true;
    fprintf(stderr,
            "GpuBatch upload rejected: descriptor count %zu != num_agents %d\n",
            data.descs.size(), num_agents_);
    return;
  }

  int total_nodes = static_cast<int>(data.node_types.size());
  int total_eval = static_cast<int>(data.eval_order.size());
  int total_conn = static_cast<int>(data.conn_from.size());
  int total_out = static_cast<int>(data.out_indices.size());

  int expected_nodes = 0;
  int expected_eval = 0;
  int expected_out = 0;
  for (const auto &desc : data.descs) {
    expected_nodes += desc.num_nodes;
    expected_eval += desc.num_eval;
    expected_out += desc.num_outputs;
  }
  if (expected_nodes != total_nodes || expected_eval != total_eval ||
      static_cast<int>(data.conn_w.size()) != total_conn ||
      static_cast<int>(data.conn_ptr.size()) != total_eval ||
      static_cast<int>(data.in_count.size()) != total_eval ||
      expected_out != total_out) {
    had_error_ = true;
    fprintf(stderr,
            "GpuBatch upload rejected: inconsistent flat topology sizes\n");
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  // ── Reallocate topology arrays only when capacity is exceeded ──────
  auto realloc_if_needed = [&](auto **ptr, int count, int &capacity,
                               size_t elem_size) {
    if (count > capacity) {
      if (*ptr)
        cudaFree(*ptr);
      *ptr = nullptr;
      if (!malloc_checked(ptr, static_cast<size_t>(count) * elem_size, __FILE__,
                          __LINE__)) {
        had_error_ = true;
        capacity = 0;
        return;
      }
      capacity = count;
    }
  };

  realloc_if_needed(&d_node_vals_, total_nodes, capacity_node_vals_,
                    sizeof(float));
  realloc_if_needed(&d_node_types_, total_nodes, capacity_node_types_,
                    sizeof(uint8_t));
  realloc_if_needed(&d_eval_order_, total_eval, capacity_eval_order_,
                    sizeof(int));
  realloc_if_needed(&d_conn_ptr_, total_eval, capacity_conn_ptr_, sizeof(int));
  realloc_if_needed(&d_in_count_, total_eval, capacity_in_count_, sizeof(int));
  if (total_conn > 0) {
    realloc_if_needed(&d_conn_from_, total_conn, capacity_conn_from_,
                      sizeof(int));
    realloc_if_needed(&d_conn_w_, total_conn, capacity_conn_w_, sizeof(float));
  }
  if (total_out > 0) {
    realloc_if_needed(&d_out_indices_, total_out, capacity_out_indices_,
                      sizeof(int));
  }
  if (had_error_) {
    return;
  }

  // Upload descriptors
  had_error_ |= !memcpy_async_checked(
      d_descs_, data.descs.data(), data.descs.size() * sizeof(GpuNetDesc),
      cudaMemcpyHostToDevice, s, __FILE__, __LINE__);

  // Upload topology arrays
  if (total_nodes > 0) {
    had_error_ |= !memcpy_async_checked(
        d_node_types_, data.node_types.data(),
        data.node_types.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, s,
        __FILE__, __LINE__);
  }
  if (total_eval > 0) {
    had_error_ |= !memcpy_async_checked(
        d_eval_order_, data.eval_order.data(),
        data.eval_order.size() * sizeof(int), cudaMemcpyHostToDevice, s,
        __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(
        d_conn_ptr_, data.conn_ptr.data(), data.conn_ptr.size() * sizeof(int),
        cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(
        d_in_count_, data.in_count.data(), data.in_count.size() * sizeof(int),
        cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
  }
  if (total_conn > 0) {
    had_error_ |= !memcpy_async_checked(
        d_conn_from_, data.conn_from.data(),
        data.conn_from.size() * sizeof(int), cudaMemcpyHostToDevice, s,
        __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(
        d_conn_w_, data.conn_w.data(), data.conn_w.size() * sizeof(float),
        cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
  }
  if (total_out > 0) {
    had_error_ |= !memcpy_async_checked(
        d_out_indices_, data.out_indices.data(),
        data.out_indices.size() * sizeof(int), cudaMemcpyHostToDevice, s,
        __FILE__, __LINE__);
  }
}

// ── Agent/Food State Upload/Download ─────────────────────────────────────────

void GpuBatch::upload_agent_states_async(const GpuAgentState *agents,
                                         int agent_count) {
  if (had_error_ || agents == nullptr || agent_count <= 0) {
    return;
  }
  if (agent_count > num_agents_) {
    had_error_ = true;
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  had_error_ |= !memcpy_async_checked(
      d_agent_states_, agents,
      static_cast<size_t>(agent_count) * sizeof(GpuAgentState),
      cudaMemcpyHostToDevice, s, __FILE__, __LINE__);

  // Unpack to SoA layout
  const int block_size = 256;
  const int grid_size = (agent_count + block_size - 1) / block_size;
  unpack_agent_states_kernel<<<grid_size, block_size, 0, s>>>(
      d_agent_states_, d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_,
      d_agent_vel_y_, d_agent_speed_, d_agent_vision_, d_agent_energy_,
      d_agent_distance_traveled_, d_agent_age_, d_agent_kills_,
      d_agent_food_eaten_, d_agent_ids_, d_agent_types_, d_agent_alive_,
      agent_count);
  had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
}

void GpuBatch::download_agent_states_async(GpuAgentState *agents,
                                           int agent_count) {
  if (had_error_ || agents == nullptr || agent_count <= 0) {
    return;
  }
  if (agent_count > num_agents_) {
    had_error_ = true;
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  // Pack from SoA layout
  const int block_size = 256;
  const int grid_size = (agent_count + block_size - 1) / block_size;
  pack_agent_states_kernel<<<grid_size, block_size, 0, s>>>(
      d_agent_states_, d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_,
      d_agent_vel_y_, d_agent_speed_, d_agent_vision_, d_agent_energy_,
      d_agent_distance_traveled_, d_agent_age_, d_agent_kills_,
      d_agent_food_eaten_, d_agent_ids_, d_agent_types_, d_agent_alive_,
      agent_count);
  had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);

  // Download to host
  had_error_ |= !memcpy_async_checked(
      agents, d_agent_states_,
      static_cast<size_t>(agent_count) * sizeof(GpuAgentState),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
}

void GpuBatch::download_agent_changes_async(float *pos_x, float *pos_y,
                                            float *energy, unsigned int *alive,
                                            int count) {
  if (had_error_ || pos_x == nullptr || pos_y == nullptr || energy == nullptr ||
      alive == nullptr || count <= 0) {
    return;
  }
  if (count > num_agents_) {
    had_error_ = true;
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  // Use staging buffers if needed, or download directly
  // For simplicity, we'll use cudaMemcpyAsync directly
  had_error_ |= !memcpy_async_checked(
      pos_x, d_agent_pos_x_, static_cast<size_t>(count) * sizeof(float),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      pos_y, d_agent_pos_y_, static_cast<size_t>(count) * sizeof(float),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      energy, d_agent_energy_, static_cast<size_t>(count) * sizeof(float),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      alive, d_agent_alive_, static_cast<size_t>(count) * sizeof(unsigned int),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
}

void GpuBatch::download_agent_counters_async(int *kills, int *food_eaten,
                                             int *age, float *distance_traveled,
                                             int count) {
  if (had_error_ || kills == nullptr || food_eaten == nullptr ||
      age == nullptr || distance_traveled == nullptr || count <= 0) {
    return;
  }
  if (count > num_agents_) {
    had_error_ = true;
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  had_error_ |= !memcpy_async_checked(
      kills, d_agent_kills_, static_cast<size_t>(count) * sizeof(int),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      food_eaten, d_agent_food_eaten_, static_cast<size_t>(count) * sizeof(int),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      age, d_agent_age_, static_cast<size_t>(count) * sizeof(int),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
  had_error_ |= !memcpy_async_checked(
      distance_traveled, d_agent_distance_traveled_,
      static_cast<size_t>(count) * sizeof(float), cudaMemcpyDeviceToHost, s,
      __FILE__, __LINE__);
}

void GpuBatch::upload_food_states_async(const GpuFoodState *food,
                                         int food_count) {
  if (had_error_ || food == nullptr || food_count <= 0) {
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  // Reallocate food arrays if needed
  if (food_count > food_capacity_) {
    if (d_food_states_)
      cudaFree(d_food_states_);
    if (d_food_pos_x_)
      cudaFree(d_food_pos_x_);
    if (d_food_pos_y_)
      cudaFree(d_food_pos_y_);
    if (d_food_active_)
      cudaFree(d_food_active_);
    if (d_food_cell_ids_)
      cudaFree(d_food_cell_ids_);

    had_error_ |= !malloc_checked(
        &d_food_states_,
        static_cast<size_t>(food_count) * sizeof(GpuFoodState), __FILE__,
        __LINE__);
    had_error_ |= !malloc_checked(
        &d_food_pos_x_, static_cast<size_t>(food_count) * sizeof(float),
        __FILE__, __LINE__);
    had_error_ |= !malloc_checked(
        &d_food_pos_y_, static_cast<size_t>(food_count) * sizeof(float),
        __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_food_active_,
                                  static_cast<size_t>(food_count) *
                                      sizeof(unsigned int),
                                  __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_food_cell_ids_,
                                  static_cast<size_t>(food_count) *
                                      sizeof(unsigned int),
                                  __FILE__, __LINE__);

    if (!had_error_) {
      food_capacity_ = food_count;
      food_bin_ids_capacity_ = food_count;
    }
  }

  // Upload food states
  had_error_ |= !memcpy_async_checked(
      d_food_states_, food,
      static_cast<size_t>(food_count) * sizeof(GpuFoodState),
      cudaMemcpyHostToDevice, s, __FILE__, __LINE__);

  // Unpack to SoA layout
  const int block_size = 256;
  const int grid_size = (food_count + block_size - 1) / block_size;
  unpack_food_states_kernel<<<grid_size, block_size, 0, s>>>(
      d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_,
      food_count);
  had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);

  food_count_ = food_count;
}

void GpuBatch::download_food_states_async(GpuFoodState *food, int food_count) {
  if (had_error_ || food == nullptr || food_count <= 0) {
    return;
  }
  if (food_count > food_count_) {
    had_error_ = true;
    return;
  }

  cudaStream_t s = static_cast<cudaStream_t>(stream_);

  // Pack from SoA layout
  const int block_size = 256;
  const int grid_size = (food_count + block_size - 1) / block_size;
  pack_food_states_kernel<<<grid_size, block_size, 0, s>>>(
      d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_,
      food_count);
  had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);

  // Download to host
  had_error_ |= !memcpy_async_checked(
      food, d_food_states_,
      static_cast<size_t>(food_count) * sizeof(GpuFoodState),
      cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
}

void GpuBatch::download_agent_states(std::vector<GpuAgentState> &agents) {
  agents.resize(static_cast<size_t>(num_agents_));
  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  const int block_size = 256;
  const int grid_size = (num_agents_ + block_size - 1) / block_size;
  pack_agent_states_kernel<<<grid_size, block_size, 0, s>>>(
      d_agent_states_, d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_,
      d_agent_vel_y_, d_agent_speed_, d_agent_vision_, d_agent_energy_,
      d_agent_distance_traveled_, d_agent_age_, d_agent_kills_,
      d_agent_food_eaten_, d_agent_ids_, d_agent_types_, d_agent_alive_,
      num_agents_);
  had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
  had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
  if (had_error_) {
    return;
  }
  had_error_ |= !check_cuda(
      cudaMemcpy(agents.data(), d_agent_states_,
                 static_cast<size_t>(num_agents_) * sizeof(GpuAgentState),
                 cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
}

void GpuBatch::download_food_states(std::vector<GpuFoodState> &food) {
  food.resize(static_cast<size_t>(food_count_));
  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  if (food_count_ > 0) {
    const int block_size = 256;
    const int grid_size = (food_count_ + block_size - 1) / block_size;
    pack_food_states_kernel<<<grid_size, block_size, 0, s>>>(
        d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_,
        food_count_);
    had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
  }
  had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
  if (had_error_) {
    return;
  }
  if (food_count_ > 0) {
    had_error_ |= !check_cuda(
        cudaMemcpy(food.data(), d_food_states_,
                   static_cast<size_t>(food_count_) * sizeof(GpuFoodState),
                   cudaMemcpyDeviceToHost),
        __FILE__, __LINE__);
  }
}

// ── Bin Allocation ───────────────────────────────────────────────────────────

void GpuBatch::allocate_bin_arrays() {
  if (num_agents_ <= 0)
    return;

  // Calculate grid dimensions using configured world size
  const float cell_size = 100.0f; // Default cell size

  agent_cols_ = static_cast<int>(world_width_ / cell_size) + 1;
  agent_rows_ = static_cast<int>(world_height_ / cell_size) + 1;
  agent_cell_size_ = cell_size;

  const int agent_cell_bins = agent_cols_ * agent_rows_;

  had_error_ |= !malloc_checked(
      &d_agent_cell_offsets_, (agent_cell_bins + 1) * sizeof(int), __FILE__,
      __LINE__);
  had_error_ |= !malloc_checked(&d_agent_cell_counts_,
                                agent_cell_bins * sizeof(int), __FILE__,
                                __LINE__);
  had_error_ |= !malloc_checked(&d_agent_cell_write_counts_,
                                agent_cell_bins * sizeof(int), __FILE__,
                                __LINE__);
  had_error_ |= !malloc_checked(
      &d_agent_cell_ids_, num_agents_ * sizeof(unsigned int), __FILE__,
      __LINE__);
  had_error_ |= !malloc_checked(
      &d_sensor_agent_entries_, num_agents_ * sizeof(GpuSensorAgentEntry),
      __FILE__, __LINE__);
}

void GpuBatch::ensure_bin_capacity(int agent_count, int food_count) {
  // Reallocate if needed - simplified version
  // Full implementation would check current capacity and grow as needed
}

// ── Inference I/O ────────────────────────────────────────────────────────────

void GpuBatch::pack_inputs_async(const float *flat_inputs, int count) {
  if (had_error_) {
    return;
  }
  if (!validate_copy_count(count, num_agents_ * num_inputs_, "input copy")) {
    return;
  }
  memcpy(h_pinned_in_, flat_inputs, static_cast<size_t>(count) * sizeof(float));
  pack_inputs_async(count);
}

void GpuBatch::pack_inputs_async(int count) {
  if (had_error_) {
    return;
  }
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  had_error_ |=
      !memcpy_async_checked(d_inputs_, h_pinned_in_, bytes,
                            cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
}

void GpuBatch::launch_inference_async() {
  if (had_error_) {
    return;
  }
  batch_neural_inference(*this);
}

void GpuBatch::start_unpack_async() {
  if (had_error_) {
    return;
  }
  size_t bytes = num_agents_ * num_outputs_ * sizeof(float);
  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  had_error_ |=
      !memcpy_async_checked(h_pinned_out_, d_outputs_, bytes,
                            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
}

void GpuBatch::finish_unpack() {
  cudaStream_t s = static_cast<cudaStream_t>(stream_);
  had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
}

void GpuBatch::finish_unpack(float *dst, int count) {
  if (!validate_copy_count(count, num_agents_ * num_outputs_, "output copy")) {
    return;
  }
  finish_unpack();
  if (had_error_) {
    return;
  }
  memcpy(dst, h_pinned_out_, static_cast<size_t>(count) * sizeof(float));
}

// ── Synchronization ──────────────────────────────────────────────────────────

void GpuBatch::synchronize() {
  if (stream_) {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    cudaStreamSynchronize(s);
  }
}

// ── Ecology Step Kernels
// These are implemented in gpu_simulation.cu to avoid ODR violations
// The declarations are in gpu_batch.hpp

bool GpuBatch::validate_copy_count(int count, int capacity, const char *label) {
  if (count < 0 || count > capacity) {
    had_error_ = true;
    fprintf(stderr, "GpuBatch %s rejected: count=%d capacity=%d\n", label,
            count, capacity);
    return false;
  }
  return true;
}

// ── CUDA initialization ──────────────────────────────────────────────────────

bool init_cuda() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return false;
  }

  int best_device = 0;
  int best_score = -1;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      continue;
    }
    int score = prop.multiProcessorCount * 100 + prop.major * 10 + prop.minor;
    if (score > best_score) {
      best_score = score;
      best_device = device;
    }
  }
  return cudaSetDevice(best_device) == cudaSuccess;
}

void print_device_info() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  spdlog::info("CUDA Device: {}", prop.name);
  spdlog::info("  Compute capability: {}.{}", prop.major, prop.minor);
  spdlog::info("  Total memory: {:.1f} MB",
               prop.totalGlobalMem / (1024.0 * 1024.0));
  spdlog::info("  SM count: {}", prop.multiProcessorCount);
}

} // namespace moonai::gpu
