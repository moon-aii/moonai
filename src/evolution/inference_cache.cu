#include "core/types.hpp"
#include "evolution/inference_cache.hpp"
#include "evolution/network_cache.hpp"
#include "evolution/neural_network.hpp"

#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/spdlog.h>

namespace moonai::evolution {

namespace {

constexpr int kBlockSize = 256;
constexpr int kOutputSlots = OUTPUT_COUNT;
constexpr int kSensorInputs = SENSOR_COUNT;

__device__ __forceinline__ float activate(float x) {
  return tanhf(x);
}

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      spdlog::error("CUDA error in {} at {}: {}", #call, __FILE__, __LINE__, cudaGetErrorString(err));                 \
    }                                                                                                                  \
  } while (0)

} // namespace

__global__ void kernel_neural_inference(const NetworkDescriptor *__restrict__ descriptors,
                                        float *__restrict__ node_values, const int *__restrict__ eval_order,
                                        const int *__restrict__ conn_from, const float *__restrict__ conn_weights,
                                        const int *__restrict__ conn_ptr, const int *__restrict__ out_indices,
                                        const int *__restrict__ network_to_slot,
                                        const float *__restrict__ sensor_inputs, float *__restrict__ brain_outputs,
                                        int network_count) {
  const int network_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (network_idx >= network_count) {
    return;
  }

  const int slot = network_to_slot[network_idx];
  const NetworkDescriptor &desc = descriptors[network_idx];

  float *my_nodes = node_values + desc.node_off;
  const int *my_eval = eval_order + desc.eval_off;
  const int *my_conn_from = conn_from + desc.conn_off;
  const float *my_weights = conn_weights + desc.conn_off;
  const int *my_conn_ptr_local = conn_ptr + desc.ptr_off;
  const int *my_out = out_indices + desc.out_off;

  for (int i = 0; i < desc.num_nodes; ++i) {
    my_nodes[i] = 0.0f;
  }

  const float *my_inputs = sensor_inputs + slot * kSensorInputs;
  for (int i = 0; i < desc.num_inputs; ++i) {
    my_nodes[i] = my_inputs[i];
  }
  my_nodes[desc.num_inputs] = 1.0f;

  for (int e = 0; e < desc.num_eval; ++e) {
    const int node_idx = my_eval[e];
    float sum = 0.0f;
    const int start = my_conn_ptr_local[e];
    const int end = my_conn_ptr_local[e + 1];

    for (int c = start; c < end; ++c) {
      sum += my_nodes[my_conn_from[c]] * my_weights[c];
    }

    my_nodes[node_idx] = activate(sum);
  }

  float *my_outputs = brain_outputs + slot * kOutputSlots;
  for (int i = 0; i < desc.num_outputs && i < kOutputSlots; ++i) {
    my_outputs[i] = my_nodes[my_out[i]];
  }
}

InferenceCache::InferenceCache() = default;

InferenceCache::~InferenceCache() {
  free_device_memory();
}

void InferenceCache::allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity,
                                            std::size_t conn_capacity, std::size_t entity_capacity) {
  free_device_memory();

  const std::size_t ptr_capacity = eval_capacity + entity_capacity;
  const std::size_t output_capacity = entity_capacity * kOutputSlots;

  CUDA_CHECK(cudaMalloc(&d_node_values_, node_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_eval_order_, eval_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_from_, conn_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_conn_weights_, conn_capacity * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_conn_ptr_, ptr_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_indices_, output_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_descriptors_, entity_capacity * sizeof(NetworkDescriptor)));
  CUDA_CHECK(cudaMalloc(&d_network_to_slot_, entity_capacity * sizeof(int)));

  entity_capacity_ = entity_capacity;
  node_capacity_ = node_capacity;
  eval_capacity_ = eval_capacity;
  conn_capacity_ = conn_capacity;
  ptr_capacity_ = ptr_capacity;
  output_capacity_ = output_capacity;
}

bool InferenceCache::needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                                        std::size_t entity_capacity) const {
  return entity_capacity > entity_capacity_ || node_capacity > node_capacity_ || eval_capacity > eval_capacity_ ||
         conn_capacity > conn_capacity_ || (eval_capacity + entity_capacity) > ptr_capacity_ ||
         (entity_capacity * kOutputSlots) > output_capacity_;
}

void InferenceCache::free_device_memory() {
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
  if (d_network_to_slot_) {
    cudaFree(d_network_to_slot_);
    d_network_to_slot_ = nullptr;
  }
  entity_capacity_ = 0;
  node_capacity_ = 0;
  eval_capacity_ = 0;
  conn_capacity_ = 0;
  ptr_capacity_ = 0;
  output_capacity_ = 0;
}

void InferenceCache::build_from(const NetworkCache &network_cache,
                                const std::vector<std::pair<uint32_t, int>> &entities_with_slots) {
  if (entities_with_slots.empty()) {
    dirty_ = false;
    return;
  }

  spdlog::debug("Building inference cache for {} entities", entities_with_slots.size());

  h_node_values_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  h_descriptors_.clear();
  h_network_to_slot_.clear();
  entity_mapping_.clear();
  entity_mapping_.reserve(entities_with_slots.size());

  h_descriptors_.reserve(entities_with_slots.size());
  h_network_to_slot_.reserve(entities_with_slots.size());

  int current_node_off = 0;
  int current_eval_off = 0;
  int current_conn_off = 0;
  int current_ptr_off = 0;
  int current_out_off = 0;

  for (const auto &[entity, slot] : entities_with_slots) {
    const NeuralNetwork *network = network_cache.get_network(entity);
    if (!network) {
      spdlog::warn("No network found for entity {}", entity);
      continue;
    }

    entity_mapping_.push_back(entity);
    h_network_to_slot_.push_back(slot);

    NetworkDescriptor descriptor;
    descriptor.num_inputs = network->num_input_nodes();
    descriptor.num_outputs = network->num_output_nodes();
    descriptor.num_nodes = network->num_nodes();
    descriptor.num_eval = descriptor.num_nodes - descriptor.num_inputs - 1;
    descriptor.node_off = current_node_off;
    descriptor.eval_off = current_eval_off;
    descriptor.conn_off = current_conn_off;
    descriptor.ptr_off = current_ptr_off;
    descriptor.out_off = current_out_off;
    descriptor.padding0 = 0;
    descriptor.padding = 0;

    const auto &node_index_map = network->node_index_map();
    std::vector<int> eval_order;
    eval_order.reserve(network->eval_order().size());
    for (uint32_t node_id : network->eval_order()) {
      const auto it = node_index_map.find(node_id);
      if (it == node_index_map.end()) {
        spdlog::warn("Skipping missing node {} in inference eval order", node_id);
        continue;
      }
      eval_order.push_back(it->second);
    }
    descriptor.num_eval = static_cast<int>(eval_order.size());

    int ptr = 0;
    for (int node_idx : eval_order) {
      h_conn_ptr_.push_back(ptr);

      const auto incoming = network->get_incoming_connections(node_idx);
      for (const auto &conn : incoming) {
        h_conn_from_.push_back(conn.from_node);
        h_conn_weights_.push_back(conn.weight);
        ++ptr;
      }
    }
    h_conn_ptr_.push_back(ptr);

    for (int node_idx : eval_order) {
      h_eval_order_.push_back(node_idx);
    }

    const std::vector<int> output_indices = network->get_output_indices();
    for (int idx : output_indices) {
      h_out_indices_.push_back(idx);
    }
    while (h_out_indices_.size() < static_cast<std::size_t>(current_out_off + kOutputSlots)) {
      h_out_indices_.push_back(0);
    }

    current_node_off += descriptor.num_nodes;
    current_eval_off += descriptor.num_eval;
    current_conn_off += ptr;
    current_ptr_off += descriptor.num_eval + 1;
    current_out_off += kOutputSlots;

    h_descriptors_.push_back(descriptor);
  }

  h_node_values_.resize(current_node_off, 0.0f);

  if (needs_reallocation(current_node_off, current_eval_off, current_conn_off, h_descriptors_.size())) {
    allocate_device_memory(current_node_off, current_eval_off, current_conn_off, h_descriptors_.size());
  }

  if (!h_descriptors_.empty()) {
    CUDA_CHECK(cudaMemcpy(d_descriptors_, h_descriptors_.data(), h_descriptors_.size() * sizeof(NetworkDescriptor),
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
  if (!h_network_to_slot_.empty()) {
    CUDA_CHECK(cudaMemcpy(d_network_to_slot_, h_network_to_slot_.data(), h_network_to_slot_.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  dirty_ = false;
  spdlog::debug("Inference cache built: {} entities, {} nodes, {} connections", h_descriptors_.size(), current_node_off,
                current_conn_off);
}

bool InferenceCache::launch_inference_async(const float *sensor_inputs, float *brain_outputs, std::size_t count,
                                            cudaStream_t stream) {
  if (count == 0 || !is_valid()) {
    return true;
  }

  if (count > h_descriptors_.size()) {
    spdlog::error("Agent count exceeds cached network count");
    return false;
  }

  const int total_nodes = static_cast<int>(h_node_values_.size());
  CUDA_CHECK(cudaMemsetAsync(d_node_values_, 0, total_nodes * sizeof(float), stream));

  const int num_blocks = (static_cast<int>(count) + kBlockSize - 1) / kBlockSize;

  kernel_neural_inference<<<num_blocks, kBlockSize, 0, stream>>>(
      d_descriptors_, d_node_values_, d_eval_order_, d_conn_from_, d_conn_weights_, d_conn_ptr_, d_out_indices_,
      d_network_to_slot_, sensor_inputs, brain_outputs, static_cast<int>(count));

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("Neural inference kernel launch failed: {}", cudaGetErrorString(err));
    return false;
  }

  return true;
}

} // namespace moonai::evolution
