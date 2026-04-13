#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {

class NetworkCache;
class NeuralNetwork;

namespace evolution {

struct alignas(16) NetworkDescriptor {
  int num_nodes;
  int num_eval;
  int num_inputs;
  int num_outputs;
  int node_off;
  int eval_off;
  int conn_off;
  int ptr_off;
  int out_off;
  int padding0;
  int padding;
};

class InferenceCache {
public:
  InferenceCache();
  ~InferenceCache();

  InferenceCache(const InferenceCache &) = delete;
  InferenceCache &operator=(const InferenceCache &) = delete;
  InferenceCache(InferenceCache &&) = delete;
  InferenceCache &operator=(InferenceCache &&) = delete;

  void build_from(const NetworkCache &network_cache, const std::vector<std::pair<uint32_t, int>> &entities_with_slots);

  bool launch_inference_async(const float *sensor_inputs, float *brain_outputs, std::size_t count, cudaStream_t stream);

  void invalidate() {
    dirty_ = true;
  }
  bool is_dirty() const {
    return dirty_;
  }
  bool is_valid() const {
    return !dirty_ && entity_capacity_ > 0;
  }

  const std::vector<uint32_t> &entity_mapping() const {
    return entity_mapping_;
  }

  const std::vector<int> &network_to_slot_mapping() const {
    return network_to_slot_;
  }
  const int *device_network_to_slot() const {
    return d_network_to_slot_;
  }

  const NetworkDescriptor *device_descriptors() const {
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
  float *d_node_values_ = nullptr;
  int *d_eval_order_ = nullptr;
  int *d_conn_from_ = nullptr;
  float *d_conn_weights_ = nullptr;
  int *d_conn_ptr_ = nullptr;
  int *d_out_indices_ = nullptr;
  NetworkDescriptor *d_descriptors_ = nullptr;

  std::vector<float> h_node_values_;
  std::vector<int> h_eval_order_;
  std::vector<int> h_conn_from_;
  std::vector<float> h_conn_weights_;
  std::vector<int> h_conn_ptr_;
  std::vector<int> h_out_indices_;
  std::vector<NetworkDescriptor> h_descriptors_;
  std::vector<int> h_network_to_slot_;

  std::vector<uint32_t> entity_mapping_;
  std::vector<int> network_to_slot_;
  int *d_network_to_slot_ = nullptr;

  bool dirty_ = true;
  std::size_t entity_capacity_ = 0;
  std::size_t node_capacity_ = 0;
  std::size_t eval_capacity_ = 0;
  std::size_t conn_capacity_ = 0;
  std::size_t ptr_capacity_ = 0;
  std::size_t output_capacity_ = 0;

  void allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                              std::size_t entity_capacity);
  bool needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                          std::size_t entity_capacity) const;
  void free_device_memory();
};

} // namespace evolution
} // namespace moonai
