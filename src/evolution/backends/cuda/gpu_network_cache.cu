#include "evolution/network_cache.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/backends/cuda/gpu_network_cache.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/spdlog.h>

namespace moonai {
namespace gpu {

namespace {

// Device-side activation functions
__device__ __forceinline__ float activate(float x) {
  return tanhf(x);
}

// CUDA error checking macro (logs error but doesn't throw for GPU cache)
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      spdlog::error("CUDA error in {} at {}: {}", #call, __FILE__, __LINE__, cudaGetErrorString(err));                 \
    }                                                                                                                  \
  } while (0)

} // anonymous namespace

// Neural inference kernel
// One thread per network (not per GPU buffer entity)
// Uses network_to_gpu mapping to access correct sensor inputs and write to
// correct outputs
__global__ void kernel_neural_inference(const GpuNetDescriptor *__restrict__ descriptors,
                                        float *__restrict__ node_values, const int *__restrict__ eval_order,
                                        const int *__restrict__ conn_from, const float *__restrict__ conn_weights,
                                        const int *__restrict__ conn_ptr, const int *__restrict__ out_indices,
                                        const int *__restrict__ network_to_gpu, const float *__restrict__ sensor_inputs,
                                        float *__restrict__ brain_outputs, int network_count) {
  const int network_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (network_idx >= network_count)
    return;

  // Get the GPU buffer index for this network
  const int gpu_idx = network_to_gpu[network_idx];
  const GpuNetDescriptor &desc = descriptors[network_idx];

  // Get network's slice of arrays
  float *my_nodes = node_values + desc.node_off;
  const int *my_eval = eval_order + desc.eval_off;
  const int *my_conn_from = conn_from + desc.conn_off;
  const float *my_weights = conn_weights + desc.conn_off;
  const int *my_conn_ptr_local = conn_ptr + desc.ptr_off;
  const int *my_out = out_indices + desc.out_off;

  // 1. Initialize nodes to zero
  for (int i = 0; i < desc.num_nodes; ++i) {
    my_nodes[i] = 0.0f;
  }

  // 2. Set sensor inputs (from GPU buffer index)
  const float *my_inputs = sensor_inputs + gpu_idx * 14;
  for (int i = 0; i < desc.num_inputs; ++i) {
    my_nodes[i] = my_inputs[i];
  }

  // Set bias node (input nodes are 0..num_inputs-1, bias is at num_inputs)
  my_nodes[desc.num_inputs] = 1.0f;

  // 3. Evaluate hidden/output nodes in topological order
  for (int e = 0; e < desc.num_eval; ++e) {
    int node_idx = my_eval[e];

    float sum = 0.0f;
    int start = my_conn_ptr_local[e];
    int end = my_conn_ptr_local[e + 1];

    for (int c = start; c < end; ++c) {
      int from_node = my_conn_from[c];
      float weight = my_weights[c];
      sum += my_nodes[from_node] * weight;
    }

    my_nodes[node_idx] = activate(sum);
  }

  // 4. Extract outputs (write to GPU buffer index)
  float *my_outputs = brain_outputs + gpu_idx * 2;
  for (int i = 0; i < desc.num_outputs && i < 2; ++i) {
    my_outputs[i] = my_nodes[my_out[i]];
  }
}

GpuNetworkCache::GpuNetworkCache() = default;

GpuNetworkCache::~GpuNetworkCache() {
  free_device_memory();
}

void GpuNetworkCache::allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity,
                                             std::size_t conn_capacity, std::size_t entity_capacity) {
  free_device_memory();

  const std::size_t ptr_capacity = eval_capacity + entity_capacity;
  const std::size_t output_capacity = entity_capacity * 2;

  CUDA_CHECK(cudaMalloc(&d_node_values_, node_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_eval_order_, eval_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_from_, conn_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_weights_, conn_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conn_ptr_, ptr_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_indices_, output_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_descriptors_, entity_capacity * sizeof(GpuNetDescriptor)));
  CUDA_CHECK(cudaMalloc(&d_network_to_gpu_, entity_capacity * sizeof(int)));

  entity_capacity_ = entity_capacity;
  node_capacity_ = node_capacity;
  eval_capacity_ = eval_capacity;
  conn_capacity_ = conn_capacity;
  ptr_capacity_ = ptr_capacity;
  output_capacity_ = output_capacity;
}

bool GpuNetworkCache::needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity,
                                         std::size_t conn_capacity, std::size_t entity_capacity) const {
  return entity_capacity > entity_capacity_ || node_capacity > node_capacity_ || eval_capacity > eval_capacity_ ||
         conn_capacity > conn_capacity_ || (eval_capacity + entity_capacity) > ptr_capacity_ ||
         (entity_capacity * 2) > output_capacity_;
}

void GpuNetworkCache::free_device_memory() {
  if (d_node_values_) {
    cudaFree(d_node_values_);
    d_node_values_ = nullptr;
  }
  if (d_eval_order_) {
    cudaFree(d_eval_order_);
    d_eval_order_ = nullptr;
  }
  if (d_conn_from_) {
    cudaFree(d_conn_from_);
    d_conn_from_ = nullptr;
  }
  if (d_conn_weights_) {
    cudaFree(d_conn_weights_);
    d_conn_weights_ = nullptr;
  }
  if (d_conn_ptr_) {
    cudaFree(d_conn_ptr_);
    d_conn_ptr_ = nullptr;
  }
  if (d_out_indices_) {
    cudaFree(d_out_indices_);
    d_out_indices_ = nullptr;
  }
  if (d_descriptors_) {
    cudaFree(d_descriptors_);
    d_descriptors_ = nullptr;
  }
  if (d_network_to_gpu_) {
    cudaFree(d_network_to_gpu_);
    d_network_to_gpu_ = nullptr;
  }
  entity_capacity_ = 0;
  node_capacity_ = 0;
  eval_capacity_ = 0;
  conn_capacity_ = 0;
  ptr_capacity_ = 0;
  output_capacity_ = 0;
}

void GpuNetworkCache::build_from(const NetworkCache &cpu_cache,
                                 const std::vector<std::pair<uint32_t, int>> &entities_with_indices) {
  if (entities_with_indices.empty()) {
    dirty_ = false;
    return;
  }

  spdlog::debug("Building GPU network cache for {} entities", entities_with_indices.size());

  // Clear host arrays
  h_node_values_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  h_descriptors_.clear();
  h_network_to_gpu_.clear();
  entity_to_gpu_.clear();
  entity_to_gpu_.reserve(entities_with_indices.size());

  // Reserve space (rough estimates)
  h_descriptors_.reserve(entities_with_indices.size());
  h_network_to_gpu_.reserve(entities_with_indices.size());

  int current_node_off = 0;
  int current_eval_off = 0;
  int current_conn_off = 0;
  int current_ptr_off = 0;
  int current_out_off = 0;

  // Build CSR format for each network
  for (const auto &[e, gpu_idx] : entities_with_indices) {
    const NeuralNetwork *network = cpu_cache.get_network(e);
    if (!network) {
      spdlog::warn("No network found for entity {}", e);
      continue;
    }

    entity_to_gpu_.push_back(e);
    h_network_to_gpu_.push_back(gpu_idx);

    GpuNetDescriptor desc;
    desc.num_inputs = network->num_input_nodes();
    desc.num_outputs = network->num_output_nodes();
    desc.num_nodes = network->num_nodes();
    desc.num_eval = desc.num_nodes - desc.num_inputs - 1; // Exclude inputs and bias

    // Offsets
    desc.node_off = current_node_off;
    desc.eval_off = current_eval_off;
    desc.conn_off = current_conn_off;
    desc.ptr_off = current_ptr_off;
    desc.out_off = current_out_off;
    desc.padding0 = 0;
    desc.padding = 0;

    // Build evaluation order and connections
    // Get topological order (excluding input and bias nodes)
    const auto &node_index_map = network->node_index_map();
    std::vector<int> eval_order;
    eval_order.reserve(network->eval_order().size());
    for (uint32_t node_id : network->eval_order()) {
      const auto it = node_index_map.find(node_id);
      if (it == node_index_map.end()) {
        spdlog::warn("Skipping missing node {} in GPU eval order", node_id);
        continue;
      }
      eval_order.push_back(it->second);
    }
    desc.num_eval = static_cast<int>(eval_order.size());

    // Build CSR representation of connections
    // For each node in eval_order, collect its incoming connections
    int ptr = 0;
    for (int node_idx : eval_order) {
      h_conn_ptr_.push_back(ptr);

      auto incoming = network->get_incoming_connections(node_idx);
      for (const auto &conn : incoming) {
        h_conn_from_.push_back(conn.from_node);
        h_conn_weights_.push_back(conn.weight);
        ptr++;
      }
    }
    h_conn_ptr_.push_back(ptr); // End pointer

    // Copy eval order
    for (int node_idx : eval_order) {
      h_eval_order_.push_back(node_idx);
    }

    // Output node indices
    std::vector<int> output_indices = network->get_output_indices();
    for (int idx : output_indices) {
      h_out_indices_.push_back(idx);
    }
    // Pad to at least 2 outputs
    while (h_out_indices_.size() < static_cast<size_t>(current_out_off + 2)) {
      h_out_indices_.push_back(0);
    }

    // Update offsets
    current_node_off += desc.num_nodes;
    current_eval_off += desc.num_eval;
    current_conn_off += ptr;
    current_ptr_off += desc.num_eval + 1;
    current_out_off += 2; // Always reserve 2 output slots

    h_descriptors_.push_back(desc);
  }

  // Reserve node values space (will be written during inference)
  h_node_values_.resize(current_node_off, 0.0f);

  // Allocate/resize device memory if needed
  if (needs_reallocation(current_node_off, current_eval_off, current_conn_off, h_descriptors_.size())) {
    allocate_device_memory(current_node_off, current_eval_off, current_conn_off, h_descriptors_.size());
  }

  // Upload to device
  if (!h_descriptors_.empty()) {
    CUDA_CHECK(cudaMemcpy(d_descriptors_, h_descriptors_.data(), h_descriptors_.size() * sizeof(GpuNetDescriptor),
                          cudaMemcpyHostToDevice));
  }
  if (!h_eval_order_.empty()) {
    CUDA_CHECK(
        cudaMemcpy(d_eval_order_, h_eval_order_.data(), h_eval_order_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  if (!h_conn_from_.empty()) {
    CUDA_CHECK(
        cudaMemcpy(d_conn_from_, h_conn_from_.data(), h_conn_from_.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_weights_, h_conn_weights_.data(), h_conn_weights_.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  if (!h_conn_ptr_.empty()) {
    CUDA_CHECK(cudaMemcpy(d_conn_ptr_, h_conn_ptr_.data(), h_conn_ptr_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  if (!h_out_indices_.empty()) {
    CUDA_CHECK(
        cudaMemcpy(d_out_indices_, h_out_indices_.data(), h_out_indices_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  if (!h_network_to_gpu_.empty()) {
    CUDA_CHECK(cudaMemcpy(d_network_to_gpu_, h_network_to_gpu_.data(), h_network_to_gpu_.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  dirty_ = false;
  spdlog::debug("GPU network cache built: {} entities, {} nodes, {} connections", h_descriptors_.size(),
                current_node_off, current_conn_off);
}

bool GpuNetworkCache::launch_inference_async(const float *d_sensor_inputs, float *d_brain_outputs, std::size_t count,
                                             cudaStream_t stream) {
  if (count == 0 || !is_valid()) {
    return true;
  }

  if (count > h_descriptors_.size()) {
    spdlog::error("Agent count exceeds cached network count");
    return false;
  }

  // Zero node values for this step
  int total_nodes = h_node_values_.size();
  CUDA_CHECK(cudaMemsetAsync(d_node_values_, 0, total_nodes * sizeof(float), stream));

  // Launch kernel
  const int block_size = 256;
  const int num_blocks = (static_cast<int>(count) + block_size - 1) / block_size;

  kernel_neural_inference<<<num_blocks, block_size, 0, stream>>>(
      d_descriptors_, d_node_values_, d_eval_order_, d_conn_from_, d_conn_weights_, d_conn_ptr_, d_out_indices_,
      d_network_to_gpu_, d_sensor_inputs, d_brain_outputs, static_cast<int>(count));

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("Neural inference kernel launch failed: {}", cudaGetErrorString(err));
    return false;
  }
  return true;
}

} // namespace gpu
} // namespace moonai
