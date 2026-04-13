#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifndef __CUDACC__
typedef struct CUstream_st *cudaStream_t;
#endif

namespace moonai {

struct CompiledNetwork;
class NetworkCache;

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

  void clear();
  bool build_from(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream);
  void add_entity(uint32_t slot, const CompiledNetwork &compiled);
  void swap_remove_entity(uint32_t removed_slot, uint32_t last_slot);
  bool prepare_for_launch(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream);

  bool launch_inference_async(const float *sensor_inputs, float *brain_outputs, std::size_t count, cudaStream_t stream);

  void invalidate() {
    dirty_ = true;
  }
  bool is_dirty() const {
    return dirty_;
  }
  bool is_valid() const {
    return !dirty_ && d_descriptors_ != nullptr;
  }

private:
  struct Entry {
    NetworkDescriptor descriptor{};
    int node_capacity = 0;
    int eval_capacity = 0;
    int conn_capacity = 0;
    int ptr_capacity = 0;
    int conn_used = 0;
    int ptr_used = 0;
  };

  float *d_node_values_ = nullptr;
  int *d_eval_order_ = nullptr;
  int *d_conn_from_ = nullptr;
  float *d_conn_weights_ = nullptr;
  int *d_conn_ptr_ = nullptr;
  int *d_out_indices_ = nullptr;
  NetworkDescriptor *d_descriptors_ = nullptr;

  std::vector<Entry> entries_;
  std::vector<uint8_t> entry_in_use_;
  std::vector<uint32_t> slot_to_entry_;
  std::vector<uint32_t> free_entries_;
  std::vector<NetworkDescriptor> launch_descriptors_;
  std::vector<int> h_eval_order_;
  std::vector<int> h_conn_from_;
  std::vector<float> h_conn_weights_;
  std::vector<int> h_conn_ptr_;
  std::vector<int> h_out_indices_;
  std::vector<uint32_t> pending_entry_uploads_;
  std::vector<uint8_t> entry_upload_pending_;

  bool dirty_ = true;
  bool launch_dirty_ = true;
  bool full_upload_required_ = false;
  std::size_t entity_capacity_ = 0;
  std::size_t active_node_count_ = 0;
  std::size_t node_extent_ = 0;
  std::size_t node_capacity_ = 0;
  std::size_t active_eval_count_ = 0;
  std::size_t eval_extent_ = 0;
  std::size_t eval_capacity_ = 0;
  std::size_t active_conn_count_ = 0;
  std::size_t conn_extent_ = 0;
  std::size_t conn_capacity_ = 0;
  std::size_t active_ptr_count_ = 0;
  std::size_t ptr_extent_ = 0;
  std::size_t ptr_capacity_ = 0;
  std::size_t output_extent_ = 0;
  std::size_t output_capacity_ = 0;

  bool allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                              std::size_t ptr_capacity, std::size_t output_capacity, std::size_t entity_capacity);
  bool needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                          std::size_t ptr_capacity, std::size_t output_capacity, std::size_t entity_capacity) const;
  void free_device_memory();
  void assign_entry_data(Entry &entry, const CompiledNetwork &compiled);
  uint32_t acquire_entry(const CompiledNetwork &compiled);
  void mark_entry_for_upload(uint32_t entry_index);
  void rebuild_launch_descriptors();
  bool upload_full(cudaStream_t stream);
  bool upload_pending(cudaStream_t stream);
  void trim_free_tail();
  bool should_repack() const;
};

} // namespace evolution
} // namespace moonai
