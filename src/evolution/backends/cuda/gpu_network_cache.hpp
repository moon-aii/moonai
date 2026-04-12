#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// CUDA forward declarations for non-CUDA compilation
#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {

class NetworkCache;
class NeuralNetwork;

namespace gpu {

// Per-agent network descriptor for CSR-packed flat GPU layout
// Matches GpuNetDesc in gpu_types.hpp
struct alignas(16) GpuNetDescriptor {
  int num_nodes;   // Total nodes (input + bias + hidden + output)
  int num_eval;    // Hidden + output nodes to evaluate
  int num_inputs;  // Input count (excluding bias)
  int num_outputs; // Output count
  int node_off;    // Offset into d_node_values
  int eval_off;    // Offset into d_eval_order
  int conn_off;    // Offset into d_conn_from/weights
  int ptr_off;     // Offset into d_conn_ptr
  int out_off;     // Offset into d_out_indices
  int padding0;
  int padding; // Pad to 48 bytes for alignment
};

// GPU Network Cache for variable-topology NEAT networks
// Converts CPU NeuralNetworks to CSR format for efficient GPU inference
class GpuNetworkCache {
public:
  GpuNetworkCache();
  ~GpuNetworkCache();

  // Non-copyable, non-movable (CUDA resources)
  GpuNetworkCache(const GpuNetworkCache &) = delete;
  GpuNetworkCache &operator=(const GpuNetworkCache &) = delete;
  GpuNetworkCache(GpuNetworkCache &&) = delete;
  GpuNetworkCache &operator=(GpuNetworkCache &&) = delete;

  // Build GPU cache from CPU NetworkCache for given entities
  // entities_with_indices: pairs of (uint32_t, gpu_buffer_index)
  void build_from(const NetworkCache &cpu_cache, const std::vector<std::pair<uint32_t, int>> &entities_with_indices);

  // Launch neural inference kernel
  // d_sensor_inputs: [entity_count][14] floats
  // d_brain_outputs: [entity_count][2] floats
  bool launch_inference_async(const float *d_sensor_inputs, float *d_brain_outputs, std::size_t count,
                              cudaStream_t stream);

  // Invalidate cache (call when networks change)
  void invalidate() {
    dirty_ = true;
  }
  bool is_dirty() const {
    return dirty_;
  }
  bool is_valid() const {
    return !dirty_ && entity_capacity_ > 0;
  }

  // Get current entity mapping (network index -> uint32_t)
  const std::vector<uint32_t> &entity_mapping() const {
    return entity_to_gpu_;
  }

  // Get mapping from network index to GPU buffer index
  const std::vector<int> &network_to_gpu_mapping() const {
    return network_to_gpu_;
  }
  const int *device_network_to_gpu() const {
    return d_network_to_gpu_;
  }

  // Device array accessors (for kernel launches)
  const GpuNetDescriptor *device_descriptors() const {
    return d_descriptors_;
  }
  float *device_node_values() const {
    return d_node_values_;
  }
  const int *device_eval_order() const {
    return d_eval_order_;
  }
  const int *device_conn_from() const {
    return d_conn_from_;
  }
  const float *device_conn_weights() const {
    return d_conn_weights_;
  }
  const int *device_conn_ptr() const {
    return d_conn_ptr_;
  }
  const int *device_out_indices() const {
    return d_out_indices_;
  }

  std::size_t capacity() const {
    return entity_capacity_;
  }

private:
  // Device arrays (flat, all agents packed)
  float *d_node_values_ = nullptr;
  int *d_eval_order_ = nullptr;
  int *d_conn_from_ = nullptr;
  float *d_conn_weights_ = nullptr;
  int *d_conn_ptr_ = nullptr;
  int *d_out_indices_ = nullptr;
  GpuNetDescriptor *d_descriptors_ = nullptr;

  // Host arrays (for building/uploading)
  std::vector<float> h_node_values_;
  std::vector<int> h_eval_order_;
  std::vector<int> h_conn_from_;
  std::vector<float> h_conn_weights_;
  std::vector<int> h_conn_ptr_;
  std::vector<int> h_out_indices_;
  std::vector<GpuNetDescriptor> h_descriptors_;
  std::vector<int> h_network_to_gpu_;

  // uint32_t mapping (network index -> uint32_t)
  std::vector<uint32_t> entity_to_gpu_;
  // Mapping from network index to GPU buffer index
  std::vector<int> network_to_gpu_;
  int *d_network_to_gpu_ = nullptr;

  bool dirty_ = true;
  std::size_t entity_capacity_ = 0;
  std::size_t node_capacity_ = 0;
  std::size_t eval_capacity_ = 0;
  std::size_t conn_capacity_ = 0;
  std::size_t ptr_capacity_ = 0;
  std::size_t output_capacity_ = 0;

  // CUDA resource management
  void allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                              std::size_t entity_capacity);
  bool needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                          std::size_t entity_capacity) const;
  void free_device_memory();
};

} // namespace gpu
} // namespace moonai
