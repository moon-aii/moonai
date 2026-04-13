#include "core/profiler_macros.hpp"
#include "core/types.hpp"
#include "evolution/inference_cache.hpp"
#include "evolution/network_cache.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/spdlog.h>

namespace moonai::evolution {

namespace {

constexpr int kBlockSize = 256;
constexpr int kOutputSlots = OUTPUT_COUNT;
constexpr int kSensorInputs = SENSOR_COUNT;
constexpr float kRepackSlackRatio = 1.5f;

__device__ __forceinline__ float activate(float x) {
  return __tanhf(x);
}

bool check_cuda(cudaError_t err, const char *call, const char *file, int line) {
  if (err == cudaSuccess) {
    return true;
  }

  spdlog::error("CUDA error in {} at {}:{}: {}", call, file, line, cudaGetErrorString(err));
  return false;
}

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

} // namespace

__global__ void kernel_neural_inference(const NetworkDescriptor *__restrict__ descriptors,
                                        float *__restrict__ node_values, const int *__restrict__ eval_order,
                                        const int *__restrict__ conn_from, const float *__restrict__ conn_weights,
                                        const int *__restrict__ conn_ptr, const int *__restrict__ out_indices,
                                        const float *__restrict__ sensor_inputs, float *__restrict__ brain_outputs,
                                        int network_count) {
  const int network_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (network_idx >= network_count) {
    return;
  }

  const int slot = network_idx;
  const NetworkDescriptor &desc = descriptors[network_idx];

  float *my_nodes = node_values + desc.node_off;
  const int *my_eval = eval_order + desc.eval_off;
  const int *my_conn_from = conn_from + desc.conn_off;
  const float *my_weights = conn_weights + desc.conn_off;
  const int *my_conn_ptr_local = conn_ptr + desc.ptr_off;
  const int *my_out = out_indices + desc.out_off;

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

void InferenceCache::clear() {
  entries_.clear();
  slot_to_entry_.clear();
  free_entries_.clear();
  launch_descriptors_.clear();
  entry_in_use_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  pending_entry_uploads_.clear();
  entry_upload_pending_.clear();

  dirty_ = true;
  launch_dirty_ = true;
  full_upload_required_ = false;
  active_node_count_ = 0;
  node_extent_ = 0;
  active_eval_count_ = 0;
  eval_extent_ = 0;
  active_conn_count_ = 0;
  conn_extent_ = 0;
  active_ptr_count_ = 0;
  ptr_extent_ = 0;
  output_extent_ = 0;

  free_device_memory();
}

bool InferenceCache::allocate_device_memory(std::size_t node_capacity, std::size_t eval_capacity,
                                            std::size_t conn_capacity, std::size_t ptr_capacity,
                                            std::size_t output_capacity, std::size_t entity_capacity) {
  free_device_memory();

  float *node_values = nullptr;
  int *eval_order = nullptr;
  int *conn_from = nullptr;
  float *conn_weights = nullptr;
  int *conn_ptr = nullptr;
  int *out_indices = nullptr;
  NetworkDescriptor *descriptors = nullptr;

  if (!CUDA_CHECK(cudaMalloc(&node_values, node_capacity * sizeof(float))) ||
      !CUDA_CHECK(cudaMalloc(&eval_order, eval_capacity * sizeof(int))) ||
      !CUDA_CHECK(cudaMalloc(&conn_from, conn_capacity * sizeof(int))) ||
      !CUDA_CHECK(cudaMalloc(&conn_weights, conn_capacity * sizeof(float))) ||
      !CUDA_CHECK(cudaMalloc(&conn_ptr, ptr_capacity * sizeof(int))) ||
      !CUDA_CHECK(cudaMalloc(&out_indices, output_capacity * sizeof(int))) ||
      !CUDA_CHECK(cudaMalloc(&descriptors, entity_capacity * sizeof(NetworkDescriptor)))) {
    if (node_values) {
      cudaFree(node_values);
    }
    if (eval_order) {
      cudaFree(eval_order);
    }
    if (conn_from) {
      cudaFree(conn_from);
    }
    if (conn_weights) {
      cudaFree(conn_weights);
    }
    if (conn_ptr) {
      cudaFree(conn_ptr);
    }
    if (out_indices) {
      cudaFree(out_indices);
    }
    if (descriptors) {
      cudaFree(descriptors);
    }
    return false;
  }

  d_node_values_ = node_values;
  d_eval_order_ = eval_order;
  d_conn_from_ = conn_from;
  d_conn_weights_ = conn_weights;
  d_conn_ptr_ = conn_ptr;
  d_out_indices_ = out_indices;
  d_descriptors_ = descriptors;

  entity_capacity_ = entity_capacity;
  node_capacity_ = node_capacity;
  eval_capacity_ = eval_capacity;
  conn_capacity_ = conn_capacity;
  ptr_capacity_ = ptr_capacity;
  output_capacity_ = output_capacity;
  return true;
}

bool InferenceCache::needs_reallocation(std::size_t node_capacity, std::size_t eval_capacity, std::size_t conn_capacity,
                                        std::size_t ptr_capacity, std::size_t output_capacity,
                                        std::size_t entity_capacity) const {
  return entity_capacity > entity_capacity_ || node_capacity > node_capacity_ || eval_capacity > eval_capacity_ ||
         conn_capacity > conn_capacity_ || ptr_capacity > ptr_capacity_ || output_capacity > output_capacity_;
}

void InferenceCache::free_device_memory() {
  if (d_node_values_) {
    CUDA_CHECK(cudaFree(d_node_values_));
    d_node_values_ = nullptr;
  }
  if (d_eval_order_) {
    CUDA_CHECK(cudaFree(d_eval_order_));
    d_eval_order_ = nullptr;
  }
  if (d_conn_from_) {
    CUDA_CHECK(cudaFree(d_conn_from_));
    d_conn_from_ = nullptr;
  }
  if (d_conn_weights_) {
    CUDA_CHECK(cudaFree(d_conn_weights_));
    d_conn_weights_ = nullptr;
  }
  if (d_conn_ptr_) {
    CUDA_CHECK(cudaFree(d_conn_ptr_));
    d_conn_ptr_ = nullptr;
  }
  if (d_out_indices_) {
    CUDA_CHECK(cudaFree(d_out_indices_));
    d_out_indices_ = nullptr;
  }
  if (d_descriptors_) {
    CUDA_CHECK(cudaFree(d_descriptors_));
    d_descriptors_ = nullptr;
  }
  entity_capacity_ = 0;
  active_node_count_ = 0;
  node_capacity_ = 0;
  active_eval_count_ = 0;
  eval_capacity_ = 0;
  active_conn_count_ = 0;
  conn_capacity_ = 0;
  active_ptr_count_ = 0;
  ptr_capacity_ = 0;
  output_capacity_ = 0;
}

namespace {

std::size_t grow_capacity(std::size_t current, std::size_t required) {
  if (required == 0) {
    return 0;
  }

  std::size_t capacity = std::max<std::size_t>(1, current);
  while (capacity < required) {
    capacity *= 2;
  }
  return capacity;
}

} // namespace

void InferenceCache::assign_entry_data(Entry &entry, const CompiledNetwork &compiled) {
  entry.descriptor.num_nodes = compiled.num_nodes;
  entry.descriptor.num_eval = compiled.num_eval();
  entry.descriptor.num_inputs = compiled.num_inputs;
  entry.descriptor.num_outputs = compiled.num_outputs;
  entry.conn_used = compiled.num_connections();
  entry.ptr_used = static_cast<int>(compiled.conn_ptr.size());
  entry.descriptor.padding0 = 0;
  entry.descriptor.padding = 0;

  std::copy(compiled.eval_order.begin(), compiled.eval_order.end(),
            h_eval_order_.begin() + entry.descriptor.eval_off);
  std::copy(compiled.conn_from.begin(), compiled.conn_from.end(), h_conn_from_.begin() + entry.descriptor.conn_off);
  std::copy(compiled.conn_weights.begin(), compiled.conn_weights.end(),
            h_conn_weights_.begin() + entry.descriptor.conn_off);
  std::copy(compiled.conn_ptr.begin(), compiled.conn_ptr.end(), h_conn_ptr_.begin() + entry.descriptor.ptr_off);
  std::copy(compiled.output_indices.begin(), compiled.output_indices.end(),
            h_out_indices_.begin() + entry.descriptor.out_off);
}

uint32_t InferenceCache::acquire_entry(const CompiledNetwork &compiled) {
  const auto free_it = std::find_if(free_entries_.begin(), free_entries_.end(), [&](const uint32_t entry_index) {
    const Entry &entry = entries_[entry_index];
    return compiled.num_nodes <= entry.node_capacity && compiled.num_eval() <= entry.eval_capacity &&
           compiled.num_connections() <= entry.conn_capacity && static_cast<int>(compiled.conn_ptr.size()) <= entry.ptr_capacity;
  });

  if (free_it != free_entries_.end()) {
    const uint32_t entry_index = *free_it;
    free_entries_.erase(free_it);
    assign_entry_data(entries_[entry_index], compiled);
    entry_in_use_[entry_index] = 1;
    mark_entry_for_upload(entry_index);
    active_node_count_ += static_cast<std::size_t>(entries_[entry_index].descriptor.num_nodes);
    active_eval_count_ += static_cast<std::size_t>(entries_[entry_index].descriptor.num_eval);
    active_conn_count_ += static_cast<std::size_t>(entries_[entry_index].conn_used);
    active_ptr_count_ += static_cast<std::size_t>(entries_[entry_index].ptr_used);
    return entry_index;
  }

  Entry entry;
  entry.descriptor.node_off = static_cast<int>(node_extent_);
  entry.descriptor.eval_off = static_cast<int>(eval_extent_);
  entry.descriptor.conn_off = static_cast<int>(conn_extent_);
  entry.descriptor.ptr_off = static_cast<int>(ptr_extent_);
  entry.descriptor.out_off = static_cast<int>(output_extent_);
  entry.node_capacity = compiled.num_nodes;
  entry.eval_capacity = compiled.num_eval();
  entry.conn_capacity = compiled.num_connections();
  entry.ptr_capacity = static_cast<int>(compiled.conn_ptr.size());

  node_extent_ += static_cast<std::size_t>(entry.node_capacity);
  eval_extent_ += static_cast<std::size_t>(entry.eval_capacity);
  conn_extent_ += static_cast<std::size_t>(entry.conn_capacity);
  ptr_extent_ += static_cast<std::size_t>(entry.ptr_capacity);
  output_extent_ += compiled.output_indices.size();

  h_eval_order_.resize(eval_extent_);
  h_conn_from_.resize(conn_extent_);
  h_conn_weights_.resize(conn_extent_);
  h_conn_ptr_.resize(ptr_extent_);
  h_out_indices_.resize(output_extent_);

  assign_entry_data(entry, compiled);

  const uint32_t entry_index = static_cast<uint32_t>(entries_.size());
  entries_.push_back(entry);
  entry_in_use_.push_back(1);
  entry_upload_pending_.push_back(0);
  mark_entry_for_upload(entry_index);
  active_node_count_ += static_cast<std::size_t>(entry.descriptor.num_nodes);
  active_eval_count_ += static_cast<std::size_t>(entry.descriptor.num_eval);
  active_conn_count_ += static_cast<std::size_t>(entry.conn_used);
  active_ptr_count_ += static_cast<std::size_t>(entry.ptr_used);
  return entry_index;
}

void InferenceCache::mark_entry_for_upload(uint32_t entry_index) {
  if (entry_index >= entry_upload_pending_.size()) {
    entry_upload_pending_.resize(static_cast<std::size_t>(entry_index) + 1, 0);
  }
  if (entry_upload_pending_[entry_index] != 0) {
    return;
  }
  entry_upload_pending_[entry_index] = 1;
  pending_entry_uploads_.push_back(entry_index);
}

void InferenceCache::rebuild_launch_descriptors() {
  launch_descriptors_.resize(slot_to_entry_.size());
  for (std::size_t slot = 0; slot < slot_to_entry_.size(); ++slot) {
    launch_descriptors_[slot] = entries_[slot_to_entry_[slot]].descriptor;
  }
}

bool InferenceCache::upload_full(cudaStream_t stream) {
  if (!h_eval_order_.empty()) {
    if (!CUDA_CHECK(cudaMemcpyAsync(d_eval_order_, h_eval_order_.data(), h_eval_order_.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream))) {
      return false;
    }
  }
  if (!h_conn_from_.empty()) {
    if (!CUDA_CHECK(cudaMemcpyAsync(d_conn_from_, h_conn_from_.data(), h_conn_from_.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream)) ||
        !CUDA_CHECK(cudaMemcpyAsync(d_conn_weights_, h_conn_weights_.data(), h_conn_weights_.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream))) {
      return false;
    }
  }
  if (!h_conn_ptr_.empty()) {
    if (!CUDA_CHECK(cudaMemcpyAsync(d_conn_ptr_, h_conn_ptr_.data(), h_conn_ptr_.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream))) {
      return false;
    }
  }
  if (!h_out_indices_.empty()) {
    if (!CUDA_CHECK(cudaMemcpyAsync(d_out_indices_, h_out_indices_.data(), h_out_indices_.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream))) {
      return false;
    }
  }

  return true;
}

bool InferenceCache::upload_pending(cudaStream_t stream) {
  if (pending_entry_uploads_.empty()) {
    return true;
  }

  MOONAI_PROFILE_SCOPE("inference_cache_upload", stream);

  for (const uint32_t entry_index : pending_entry_uploads_) {
    const Entry &entry = entries_[entry_index];

    if (entry.descriptor.num_eval > 0) {
      if (!CUDA_CHECK(cudaMemcpyAsync(d_eval_order_ + entry.descriptor.eval_off,
                                      h_eval_order_.data() + entry.descriptor.eval_off,
                                      static_cast<std::size_t>(entry.descriptor.num_eval) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream))) {
        return false;
      }
    }

    if (entry.conn_used > 0) {
      if (!CUDA_CHECK(cudaMemcpyAsync(d_conn_from_ + entry.descriptor.conn_off,
                                      h_conn_from_.data() + entry.descriptor.conn_off,
                                      static_cast<std::size_t>(entry.conn_used) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream)) ||
          !CUDA_CHECK(cudaMemcpyAsync(d_conn_weights_ + entry.descriptor.conn_off,
                                      h_conn_weights_.data() + entry.descriptor.conn_off,
                                      static_cast<std::size_t>(entry.conn_used) * sizeof(float),
                                      cudaMemcpyHostToDevice, stream))) {
        return false;
      }
    }

    if (entry.ptr_used > 0) {
      if (!CUDA_CHECK(cudaMemcpyAsync(d_conn_ptr_ + entry.descriptor.ptr_off,
                                      h_conn_ptr_.data() + entry.descriptor.ptr_off,
                                      static_cast<std::size_t>(entry.ptr_used) * sizeof(int), cudaMemcpyHostToDevice,
                                      stream))) {
        return false;
      }
    }

    if (!CUDA_CHECK(cudaMemcpyAsync(d_out_indices_ + entry.descriptor.out_off,
                                    h_out_indices_.data() + entry.descriptor.out_off,
                                    kOutputSlots * sizeof(int), cudaMemcpyHostToDevice, stream))) {
      return false;
    }

    entry_upload_pending_[entry_index] = 0;
  }

  pending_entry_uploads_.clear();
  return true;
}

void InferenceCache::trim_free_tail() {
  while (!entries_.empty() && !entry_in_use_.back()) {
    const uint32_t entry_index = static_cast<uint32_t>(entries_.size() - 1);
    const Entry &entry = entries_.back();

    free_entries_.erase(std::remove(free_entries_.begin(), free_entries_.end(), entry_index), free_entries_.end());
    pending_entry_uploads_.erase(std::remove(pending_entry_uploads_.begin(), pending_entry_uploads_.end(), entry_index),
                                 pending_entry_uploads_.end());
    if (!entry_upload_pending_.empty()) {
      entry_upload_pending_.pop_back();
    }

    node_extent_ -= static_cast<std::size_t>(entry.node_capacity);
    eval_extent_ -= static_cast<std::size_t>(entry.eval_capacity);
    conn_extent_ -= static_cast<std::size_t>(entry.conn_capacity);
    ptr_extent_ -= static_cast<std::size_t>(entry.ptr_capacity);
    output_extent_ -= kOutputSlots;

    entries_.pop_back();
    entry_in_use_.pop_back();
  }

  h_eval_order_.resize(eval_extent_);
  h_conn_from_.resize(conn_extent_);
  h_conn_weights_.resize(conn_extent_);
  h_conn_ptr_.resize(ptr_extent_);
  h_out_indices_.resize(output_extent_);
}

bool InferenceCache::should_repack() const {
  if (free_entries_.empty()) {
    return false;
  }

  const bool sparse_nodes = active_node_count_ > 0 &&
                            static_cast<float>(node_extent_) > static_cast<float>(active_node_count_) * kRepackSlackRatio;
  const bool sparse_conns = active_conn_count_ > 0 &&
                            static_cast<float>(conn_extent_) > static_cast<float>(active_conn_count_) * kRepackSlackRatio;
  const bool sparse_ptrs = active_ptr_count_ > 0 &&
                           static_cast<float>(ptr_extent_) > static_cast<float>(active_ptr_count_) * kRepackSlackRatio;
  const bool many_free_entries = free_entries_.size() * 3 >= entries_.size();
  return sparse_nodes || sparse_conns || sparse_ptrs || many_free_entries;
}

bool InferenceCache::build_from(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream) {
  entries_.clear();
  entry_in_use_.clear();
  slot_to_entry_.clear();
  free_entries_.clear();
  launch_descriptors_.clear();
  h_eval_order_.clear();
  h_conn_from_.clear();
  h_conn_weights_.clear();
  h_conn_ptr_.clear();
  h_out_indices_.clear();
  pending_entry_uploads_.clear();
  entry_upload_pending_.clear();

  active_node_count_ = 0;
  node_extent_ = 0;
  active_eval_count_ = 0;
  eval_extent_ = 0;
  active_conn_count_ = 0;
  conn_extent_ = 0;
  active_ptr_count_ = 0;
  ptr_extent_ = 0;
  output_extent_ = 0;

  entries_.reserve(count);
  entry_in_use_.reserve(count);
  slot_to_entry_.reserve(count);
  entry_upload_pending_.reserve(count);

  for (uint32_t slot = 0; slot < count; ++slot) {
    const CompiledNetwork *compiled = network_cache.get_compiled(slot);
    if (!compiled) {
      spdlog::error("Missing compiled network for slot {}", slot);
      clear();
      return false;
    }

    const uint32_t entry_index = acquire_entry(*compiled);
    slot_to_entry_.push_back(entry_index);
  }

  rebuild_launch_descriptors();

  if (needs_reallocation(node_extent_, eval_extent_, conn_extent_, ptr_extent_, output_extent_, slot_to_entry_.size())) {
    if (!allocate_device_memory(grow_capacity(node_capacity_, node_extent_), grow_capacity(eval_capacity_, eval_extent_),
                                grow_capacity(conn_capacity_, conn_extent_), grow_capacity(ptr_capacity_, ptr_extent_),
                                grow_capacity(output_capacity_, output_extent_),
                                grow_capacity(entity_capacity_, slot_to_entry_.size()))) {
      clear();
      return false;
    }
  }

  if (!upload_full(stream)) {
    clear();
    return false;
  }
  if (!launch_descriptors_.empty()) {
    if (!CUDA_CHECK(cudaMemcpyAsync(d_descriptors_, launch_descriptors_.data(),
                                    launch_descriptors_.size() * sizeof(NetworkDescriptor), cudaMemcpyHostToDevice,
                                    stream))) {
      clear();
      return false;
    }
  }

  pending_entry_uploads_.clear();
  std::fill(entry_upload_pending_.begin(), entry_upload_pending_.end(), 0);
  dirty_ = false;
  launch_dirty_ = false;
  full_upload_required_ = false;
  return true;
}

void InferenceCache::add_entity(uint32_t slot, const CompiledNetwork &compiled) {
  if (dirty_) {
    return;
  }

  if (slot != slot_to_entry_.size()) {
    dirty_ = true;
    return;
  }

  const uint32_t entry_index = acquire_entry(compiled);
  slot_to_entry_.push_back(entry_index);
  launch_dirty_ = true;
}

void InferenceCache::swap_remove_entity(uint32_t removed_slot, uint32_t last_slot) {
  if (dirty_) {
    return;
  }

  if (slot_to_entry_.empty() || last_slot >= slot_to_entry_.size() || removed_slot >= slot_to_entry_.size()) {
    dirty_ = true;
    return;
  }

  const uint32_t removed_entry = slot_to_entry_[removed_slot];
  const Entry &entry = entries_[removed_entry];
  active_node_count_ -= static_cast<std::size_t>(entry.descriptor.num_nodes);
  active_eval_count_ -= static_cast<std::size_t>(entry.descriptor.num_eval);
  active_conn_count_ -= static_cast<std::size_t>(entry.conn_used);
  active_ptr_count_ -= static_cast<std::size_t>(entry.ptr_used);
  entry_in_use_[removed_entry] = 0;
  free_entries_.push_back(removed_entry);
  if (removed_slot != last_slot) {
    slot_to_entry_[removed_slot] = slot_to_entry_[last_slot];
  }
  slot_to_entry_.pop_back();
  trim_free_tail();
  launch_dirty_ = true;
}

bool InferenceCache::prepare_for_launch(const NetworkCache &network_cache, std::size_t count, cudaStream_t stream) {
  if (count == 0) {
    return true;
  }

  if (dirty_ || slot_to_entry_.size() != count) {
    return build_from(network_cache, count, stream);
  }

  if (should_repack()) {
    return build_from(network_cache, count, stream);
  }

  if (needs_reallocation(node_extent_, eval_extent_, conn_extent_, ptr_extent_, output_extent_, count)) {
    if (!allocate_device_memory(grow_capacity(node_capacity_, node_extent_), grow_capacity(eval_capacity_, eval_extent_),
                                grow_capacity(conn_capacity_, conn_extent_), grow_capacity(ptr_capacity_, ptr_extent_),
                                grow_capacity(output_capacity_, output_extent_),
                                grow_capacity(entity_capacity_, count))) {
      clear();
      return false;
    }
    full_upload_required_ = true;
  }

  if (full_upload_required_) {
    if (!upload_full(stream)) {
      clear();
      return false;
    }
    std::fill(entry_upload_pending_.begin(), entry_upload_pending_.end(), 0);
    pending_entry_uploads_.clear();
  } else if (!upload_pending(stream)) {
    clear();
    return false;
  }

  if (launch_dirty_ || full_upload_required_) {
    rebuild_launch_descriptors();
    if (!launch_descriptors_.empty()) {
      if (!CUDA_CHECK(cudaMemcpyAsync(d_descriptors_, launch_descriptors_.data(),
                                      launch_descriptors_.size() * sizeof(NetworkDescriptor), cudaMemcpyHostToDevice,
                                      stream))) {
        clear();
        return false;
      }
    }
  }

  launch_dirty_ = false;
  full_upload_required_ = false;
  return true;
}

bool InferenceCache::launch_inference_async(const float *sensor_inputs, float *brain_outputs, std::size_t count,
                                            cudaStream_t stream) {
  if (count == 0 || !is_valid()) {
    return true;
  }

  if (count > slot_to_entry_.size()) {
    spdlog::error("Agent count exceeds cached network count");
    return false;
  }

  if (node_extent_ > 0) {
    if (!CUDA_CHECK(cudaMemsetAsync(d_node_values_, 0, node_extent_ * sizeof(float), stream))) {
      return false;
    }
  }

  const int num_blocks = (static_cast<int>(count) + kBlockSize - 1) / kBlockSize;

  MOONAI_PROFILE_SCOPE("inference_kernel", stream);

  kernel_neural_inference<<<num_blocks, kBlockSize, 0, stream>>>(
      d_descriptors_, d_node_values_, d_eval_order_, d_conn_from_, d_conn_weights_, d_conn_ptr_, d_out_indices_,
      sensor_inputs, brain_outputs, static_cast<int>(count));

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("Neural inference kernel launch failed: {}", cudaGetErrorString(err));
    return false;
  }

  return true;
}

} // namespace moonai::evolution
