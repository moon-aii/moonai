#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

#ifdef MOONAI_BUILD_PROFILER
#include "core/profiler.hpp"
#endif

#include <cub/cub.cuh>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <tuple>

namespace moonai::gpu {

namespace {

bool check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

template<typename T>
bool malloc_checked(T** ptr, size_t bytes, const char* file, int line) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    return check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), bytes), file, line);
}

template<typename T>
bool malloc_host_checked(T** ptr, size_t bytes, const char* file, int line) {
    if (bytes == 0) {
        *ptr = nullptr;
        return true;
    }
    return check_cuda(cudaMallocHost(reinterpret_cast<void**>(ptr), bytes), file, line);
}

bool memcpy_async_checked(void* dst, const void* src, size_t bytes,
                          cudaMemcpyKind kind, cudaStream_t stream,
                          const char* file, int line) {
    if (bytes == 0) {
        return true;
    }
    return check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), file, line);
}

__global__ void unpack_agent_states_kernel(
    const GpuAgentState* __restrict__ src,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ speed,
    float* __restrict__ vision,
    float* __restrict__ energy,
    float* __restrict__ distance_traveled,
    int* __restrict__ age,
    int* __restrict__ kills,
    int* __restrict__ food_eaten,
    unsigned int* __restrict__ ids,
    unsigned int* __restrict__ types,
    unsigned int* __restrict__ alive,
    int count) {
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
    GpuAgentState* __restrict__ dst,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ speed,
    const float* __restrict__ vision,
    const float* __restrict__ energy,
    const float* __restrict__ distance_traveled,
    const int* __restrict__ age,
    const int* __restrict__ kills,
    const int* __restrict__ food_eaten,
    const unsigned int* __restrict__ ids,
    const unsigned int* __restrict__ types,
    const unsigned int* __restrict__ alive,
    int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    dst[idx] = GpuAgentState{
        pos_x[idx],
        pos_y[idx],
        vel_x[idx],
        vel_y[idx],
        speed[idx],
        vision[idx],
        energy[idx],
        distance_traveled[idx],
        age[idx],
        kills[idx],
        food_eaten[idx],
        ids[idx],
        types[idx],
        alive[idx]
    };
}

__global__ void unpack_food_states_kernel(
    const GpuFoodState* __restrict__ src,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    unsigned int* __restrict__ active,
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

__global__ void pack_food_states_kernel(
    GpuFoodState* __restrict__ dst,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const unsigned int* __restrict__ active,
    int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    dst[idx] = GpuFoodState{pos_x[idx], pos_y[idx], active[idx]};
}

} // namespace

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs)
    : num_agents_(num_agents)
    , num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    const size_t in_bytes = static_cast<size_t>(num_agents) * num_inputs * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(num_agents) * num_outputs * sizeof(float);
    const size_t agent_float_bytes = static_cast<size_t>(num_agents) * sizeof(float);
    const size_t agent_int_bytes = static_cast<size_t>(num_agents) * sizeof(int);
    const size_t agent_uint_bytes = static_cast<size_t>(num_agents) * sizeof(unsigned int);

    had_error_ |= !malloc_checked(&d_descs_, num_agents * sizeof(GpuNetDesc), __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_inputs_, in_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_outputs_, out_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_states_, num_agents * sizeof(GpuAgentState), __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_pos_x_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_pos_y_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_vel_x_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_vel_y_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_speed_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_vision_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_energy_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_distance_traveled_, agent_float_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_age_, agent_int_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_kills_, agent_int_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_food_eaten_, agent_int_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_ids_, agent_uint_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_types_, agent_uint_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_agent_alive_, agent_uint_bytes, __FILE__, __LINE__);

    had_error_ |= !malloc_host_checked(&h_pinned_in_, in_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_host_checked(&h_pinned_out_, out_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_host_checked(&h_tick_index_, sizeof(int), __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_tick_index_, sizeof(int), __FILE__, __LINE__);
#ifdef MOONAI_BUILD_PROFILER
    had_error_ |= !malloc_checked(&d_resident_stage_accum_ns_,
        static_cast<size_t>(GpuStageTiming::Count) * sizeof(unsigned long long), __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_resident_stage_last_timestamp_ns_, sizeof(unsigned long long), __FILE__, __LINE__);
    had_error_ |= !malloc_host_checked(&h_resident_stage_accum_ns_,
        static_cast<size_t>(GpuStageTiming::Count) * sizeof(unsigned long long), __FILE__, __LINE__);
#endif

    cudaStream_t s = nullptr;
    had_error_ |= !check_cuda(cudaStreamCreate(&s), __FILE__, __LINE__);
    stream_ = static_cast<void*>(s);
}

GpuBatch::~GpuBatch() {
    invalidate_resident_graph();

    // Device arrays
    if (d_descs_)       cudaFree(d_descs_);
    if (d_inputs_)      cudaFree(d_inputs_);
    if (d_outputs_)     cudaFree(d_outputs_);
    if (d_agent_states_) cudaFree(d_agent_states_);
    if (d_food_states_) cudaFree(d_food_states_);
    if (d_agent_cell_offsets_) cudaFree(d_agent_cell_offsets_);
    if (d_food_cell_offsets_) cudaFree(d_food_cell_offsets_);
    if (d_agent_cell_counts_) cudaFree(d_agent_cell_counts_);
    if (d_food_cell_counts_) cudaFree(d_food_cell_counts_);
    if (d_agent_cell_write_counts_) cudaFree(d_agent_cell_write_counts_);
    if (d_food_cell_write_counts_) cudaFree(d_food_cell_write_counts_);
    if (d_agent_cell_ids_) cudaFree(d_agent_cell_ids_);
    if (d_food_cell_ids_) cudaFree(d_food_cell_ids_);
    if (d_agent_pos_x_) cudaFree(d_agent_pos_x_);
    if (d_agent_pos_y_) cudaFree(d_agent_pos_y_);
    if (d_agent_vel_x_) cudaFree(d_agent_vel_x_);
    if (d_agent_vel_y_) cudaFree(d_agent_vel_y_);
    if (d_agent_speed_) cudaFree(d_agent_speed_);
    if (d_agent_vision_) cudaFree(d_agent_vision_);
    if (d_agent_energy_) cudaFree(d_agent_energy_);
    if (d_agent_distance_traveled_) cudaFree(d_agent_distance_traveled_);
    if (d_agent_age_) cudaFree(d_agent_age_);
    if (d_agent_kills_) cudaFree(d_agent_kills_);
    if (d_agent_food_eaten_) cudaFree(d_agent_food_eaten_);
    if (d_agent_ids_) cudaFree(d_agent_ids_);
    if (d_agent_types_) cudaFree(d_agent_types_);
    if (d_agent_alive_) cudaFree(d_agent_alive_);
    if (d_food_pos_x_) cudaFree(d_food_pos_x_);
    if (d_food_pos_y_) cudaFree(d_food_pos_y_);
    if (d_food_active_) cudaFree(d_food_active_);
    if (d_inference_agent_indices_) cudaFree(d_inference_agent_indices_);
    if (d_tick_index_) cudaFree(d_tick_index_);
    if (d_scan_temp_storage_) cudaFree(d_scan_temp_storage_);
#ifdef MOONAI_BUILD_PROFILER
    if (d_resident_stage_accum_ns_) cudaFree(d_resident_stage_accum_ns_);
    if (d_resident_stage_last_timestamp_ns_) cudaFree(d_resident_stage_last_timestamp_ns_);
#endif

    // Pinned host memory
    if (h_pinned_in_)   cudaFreeHost(h_pinned_in_);
    if (h_pinned_out_)  cudaFreeHost(h_pinned_out_);
    if (h_tick_index_)  cudaFreeHost(h_tick_index_);
#ifdef MOONAI_BUILD_PROFILER
    if (h_resident_stage_accum_ns_) cudaFreeHost(h_resident_stage_accum_ns_);
#endif

    // Stream
    if (stream_)        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));

    // Topology arrays
    if (d_node_vals_)   cudaFree(d_node_vals_);
    if (d_node_types_)  cudaFree(d_node_types_);
    if (d_eval_order_)  cudaFree(d_eval_order_);
    if (d_conn_ptr_)    cudaFree(d_conn_ptr_);
    if (d_in_count_)    cudaFree(d_in_count_);
    if (d_conn_from_)   cudaFree(d_conn_from_);
    if (d_conn_w_)      cudaFree(d_conn_w_);
    if (d_out_indices_) cudaFree(d_out_indices_);
}

void GpuBatch::invalidate_resident_graph() {
    resident_graph_valid_ = false;
    if (resident_tick_graph_exec_) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(resident_tick_graph_exec_));
        resident_tick_graph_exec_ = nullptr;
    }
    if (resident_tick_graph_) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(resident_tick_graph_));
        resident_tick_graph_ = nullptr;
    }
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
    if (!malloc_checked(reinterpret_cast<unsigned char**>(&d_scan_temp_storage_), bytes,
                        __FILE__, __LINE__)) {
        had_error_ = true;
        return false;
    }
    scan_temp_storage_bytes_ = bytes;
    return true;
}

// ── upload_network_data ───────────────────────────────────────────────────────

void GpuBatch::upload_network_data(const GpuNetworkData& data) {
    had_error_ = (stream_ == nullptr
        || d_descs_ == nullptr
        || d_inputs_ == nullptr
        || d_outputs_ == nullptr
        || h_pinned_in_ == nullptr
        || h_pinned_out_ == nullptr);
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

    int n            = num_agents_;
    int total_nodes  = static_cast<int>(data.node_types.size());
    int total_eval   = static_cast<int>(data.eval_order.size());
    int total_conn   = static_cast<int>(data.conn_from.size());
    int total_out    = static_cast<int>(data.out_indices.size());

    int expected_nodes = 0;
    int expected_eval = 0;
    int expected_out = 0;
    for (const auto& desc : data.descs) {
        expected_nodes += desc.num_nodes;
        expected_eval += desc.num_eval;
        expected_out += desc.num_outputs;
    }
    if (expected_nodes != total_nodes
            || expected_eval != total_eval
            || static_cast<int>(data.conn_w.size()) != total_conn
            || static_cast<int>(data.conn_ptr.size()) != total_eval
            || static_cast<int>(data.in_count.size()) != total_eval
            || expected_out != total_out) {
        had_error_ = true;
        fprintf(stderr, "GpuBatch upload rejected: inconsistent flat topology sizes\n");
        return;
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream_);

    // ── Reallocate topology arrays only when capacity is exceeded ──────
    auto realloc_if_needed = [&](auto** ptr, int count, int& capacity, size_t elem_size) {
        if (count > capacity) {
            if (*ptr) cudaFree(*ptr);
            *ptr = nullptr;
            if (!malloc_checked(ptr, static_cast<size_t>(count) * elem_size, __FILE__, __LINE__)) {
                had_error_ = true;
                capacity = 0;
                return;
            }
            capacity = count;
        }
    };

    realloc_if_needed(&d_node_vals_,  total_nodes, capacity_node_vals_, sizeof(float));
    realloc_if_needed(&d_node_types_, total_nodes, capacity_node_types_, sizeof(uint8_t));
    realloc_if_needed(&d_eval_order_, total_eval,  capacity_eval_order_, sizeof(int));
    realloc_if_needed(&d_conn_ptr_,   total_eval,  capacity_conn_ptr_, sizeof(int));
    realloc_if_needed(&d_in_count_,   total_eval,  capacity_in_count_, sizeof(int));
    if (total_conn > 0) {
        realloc_if_needed(&d_conn_from_, total_conn, capacity_conn_from_, sizeof(int));
        realloc_if_needed(&d_conn_w_,    total_conn, capacity_conn_w_, sizeof(float));
    }
    if (total_out > 0) {
        realloc_if_needed(&d_out_indices_, total_out, capacity_out_indices_, sizeof(int));
    }
    if (had_error_) {
        return;
    }

    std::vector<int> ordered_agents(static_cast<size_t>(num_agents_));
    for (int i = 0; i < num_agents_; ++i) {
        ordered_agents[static_cast<size_t>(i)] = i;
    }
    std::sort(ordered_agents.begin(), ordered_agents.end(), [&](int lhs, int rhs) {
        const GpuNetDesc& a = data.descs[static_cast<size_t>(lhs)];
        const GpuNetDesc& b = data.descs[static_cast<size_t>(rhs)];
        return std::tie(a.num_eval, a.num_nodes, a.num_outputs, lhs)
            < std::tie(b.num_eval, b.num_nodes, b.num_outputs, rhs);
    });
    inference_bucket_offsets_.clear();
    inference_bucket_sizes_.clear();
    if (!ordered_agents.empty()) {
        int bucket_start = 0;
        for (int i = 1; i <= num_agents_; ++i) {
            const bool bucket_break = (i == num_agents_)
                || data.descs[static_cast<size_t>(ordered_agents[static_cast<size_t>(bucket_start)])].num_eval
                    != data.descs[static_cast<size_t>(ordered_agents[static_cast<size_t>(i)])].num_eval
                || data.descs[static_cast<size_t>(ordered_agents[static_cast<size_t>(bucket_start)])].num_nodes
                    != data.descs[static_cast<size_t>(ordered_agents[static_cast<size_t>(i)])].num_nodes;
            if (bucket_break) {
                inference_bucket_offsets_.push_back(bucket_start);
                inference_bucket_sizes_.push_back(i - bucket_start);
                bucket_start = i;
            }
        }
    }

    std::vector<GpuNetDesc> reordered_descs(static_cast<size_t>(num_agents_));
    std::vector<uint8_t> reordered_node_types(static_cast<size_t>(total_nodes));
    std::vector<int> reordered_eval_order(static_cast<size_t>(total_eval));
    std::vector<int> reordered_conn_ptr(static_cast<size_t>(total_eval));
    std::vector<int> reordered_in_count(static_cast<size_t>(total_eval));
    std::vector<int> reordered_conn_from(static_cast<size_t>(total_conn));
    std::vector<float> reordered_conn_w(static_cast<size_t>(total_conn));
    std::vector<int> reordered_out_indices(static_cast<size_t>(total_out));
    int next_node_off = 0;
    int next_eval_off = 0;
    int next_conn_off = 0;
    int next_out_off = 0;
    for (int sorted_idx = 0; sorted_idx < num_agents_; ++sorted_idx) {
        const int agent_idx = ordered_agents[static_cast<size_t>(sorted_idx)];
        const GpuNetDesc& src = data.descs[static_cast<size_t>(agent_idx)];
        GpuNetDesc dst = src;
        dst.node_off = next_node_off;
        dst.eval_off = next_eval_off;
        dst.conn_off = next_conn_off;
        dst.out_off = next_out_off;
        reordered_descs[static_cast<size_t>(sorted_idx)] = dst;

        std::copy_n(data.node_types.begin() + src.node_off, src.num_nodes,
                    reordered_node_types.begin() + next_node_off);
        std::copy_n(data.in_count.begin() + src.eval_off, src.num_eval,
                    reordered_in_count.begin() + next_eval_off);
        std::copy_n(data.conn_from.begin() + src.conn_off,
                    src.num_eval > 0 ? (data.conn_ptr[static_cast<size_t>(src.eval_off + src.num_eval - 1)]
                        + data.in_count[static_cast<size_t>(src.eval_off + src.num_eval - 1)]) : 0,
                    reordered_conn_from.begin() + next_conn_off);
        std::copy_n(data.conn_w.begin() + src.conn_off,
                    src.num_eval > 0 ? (data.conn_ptr[static_cast<size_t>(src.eval_off + src.num_eval - 1)]
                        + data.in_count[static_cast<size_t>(src.eval_off + src.num_eval - 1)]) : 0,
                    reordered_conn_w.begin() + next_conn_off);
        std::copy_n(data.out_indices.begin() + src.out_off, src.num_outputs,
                    reordered_out_indices.begin() + next_out_off);

        for (int eval_idx = 0; eval_idx < src.num_eval; ++eval_idx) {
            reordered_eval_order[static_cast<size_t>(next_eval_off + eval_idx)] =
                data.eval_order[static_cast<size_t>(src.eval_off + eval_idx)];
            reordered_conn_ptr[static_cast<size_t>(next_eval_off + eval_idx)] =
                data.conn_ptr[static_cast<size_t>(src.eval_off + eval_idx)];
        }

        next_node_off += src.num_nodes;
        next_eval_off += src.num_eval;
        next_conn_off += (src.num_eval > 0)
            ? data.conn_ptr[static_cast<size_t>(src.eval_off + src.num_eval - 1)]
                + data.in_count[static_cast<size_t>(src.eval_off + src.num_eval - 1)]
            : 0;
        next_out_off += src.num_outputs;
    }

    had_error_ |= !memcpy_async_checked(d_descs_, reordered_descs.data(),
        n * sizeof(GpuNetDesc), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_node_types_, reordered_node_types.data(),
        total_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_eval_order_, reordered_eval_order.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_conn_ptr_, reordered_conn_ptr.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_in_count_, reordered_in_count.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    if (total_conn > 0) {
        had_error_ |= !memcpy_async_checked(d_conn_from_, reordered_conn_from.data(),
            total_conn * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
        had_error_ |= !memcpy_async_checked(d_conn_w_, reordered_conn_w.data(),
            total_conn * sizeof(float), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    }
    if (total_out > 0) {
        had_error_ |= !memcpy_async_checked(d_out_indices_, reordered_out_indices.data(),
            total_out * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    }

    if (num_agents_ > inference_agent_indices_capacity_) {
        if (d_inference_agent_indices_) {
            cudaFree(d_inference_agent_indices_);
            d_inference_agent_indices_ = nullptr;
        }
        had_error_ |= !malloc_checked(&d_inference_agent_indices_,
            static_cast<size_t>(num_agents_) * sizeof(int), __FILE__, __LINE__);
        if (!had_error_) {
            inference_agent_indices_capacity_ = num_agents_;
        }
    }
    if (!had_error_ && !ordered_agents.empty()) {
        had_error_ |= !memcpy_async_checked(d_inference_agent_indices_, ordered_agents.data(),
            static_cast<size_t>(num_agents_) * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    }

    // Ensure upload completes before first inference tick
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
    if (!had_error_) {
        invalidate_resident_graph();
    }
}

// ── Per-tick I/O ─────────────────────────────────────────────────────────────

void GpuBatch::pack_inputs_async(const float* flat_inputs, int count) {
    if (had_error_) {
        return;
    }
    if (!validate_copy_count(count, num_agents_ * num_inputs_, "input copy")) {
        return;
    }
    size_t bytes = static_cast<size_t>(count) * sizeof(float);
    memcpy(h_pinned_in_, flat_inputs, bytes);
    pack_inputs_async(count);
}

void GpuBatch::pack_inputs_async(int count) {
    if (had_error_) {
        return;
    }
    if (!validate_copy_count(count, num_agents_ * num_inputs_, "input copy")) {
        return;
    }
    size_t bytes = static_cast<size_t>(count) * sizeof(float);
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    had_error_ |= !memcpy_async_checked(d_inputs_, h_pinned_in_,
        bytes, cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
}

void GpuBatch::upload_agent_states_async(const GpuAgentState* agents, int agent_count) {
    if (had_error_) {
        return;
    }
    if (agent_count != num_agents_ || agents == nullptr) {
        had_error_ = true;
        return;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    had_error_ |= !memcpy_async_checked(d_agent_states_, agents,
        static_cast<size_t>(agent_count) * sizeof(GpuAgentState), cudaMemcpyHostToDevice,
        s, __FILE__, __LINE__);
    invalidate_resident_graph();
}

void GpuBatch::upload_tick_state_async(
    const GpuAgentState* agents,
    int agent_count,
    const GpuFoodState* food,
    int food_count,
    int agent_cols,
    int agent_rows,
    float agent_cell_size,
    int food_cols,
    int food_rows,
    float food_cell_size) {
    if (had_error_) {
        return;
    }
    if (agent_count != num_agents_ || agents == nullptr
        || food_count < 0 || (food_count > 0 && food == nullptr)
        || agent_cols <= 0 || agent_rows <= 0 || agent_cell_size <= 0.0f
        || food_cols <= 0 || food_rows <= 0 || food_cell_size <= 0.0f
        ) {
        had_error_ = true;
        return;
    }

    auto checked_product = [&](int lhs, int rhs, int* out) {
        if (lhs < 0 || rhs < 0) {
            had_error_ = true;
            return false;
        }
        const long long product = static_cast<long long>(lhs) * static_cast<long long>(rhs);
        if (product <= 0 || product > static_cast<long long>(std::numeric_limits<int>::max() / 4)) {
            had_error_ = true;
            return false;
        }
        *out = static_cast<int>(product);
        return true;
    };

    auto realloc_if_needed = [&](auto** ptr, int count, int& capacity, size_t elem_size) {
        if (count > capacity) {
            if (*ptr) {
                cudaFree(*ptr);
                *ptr = nullptr;
            }
            if (!malloc_checked(ptr, static_cast<size_t>(count) * elem_size, __FILE__, __LINE__)) {
                had_error_ = true;
                capacity = 0;
                return;
            }
            capacity = count;
        }
    };

    int agent_cell_bins = 0;
    int food_cell_bins = 0;
    if (!checked_product(agent_cols, agent_rows, &agent_cell_bins)
        || !checked_product(food_cols, food_rows, &food_cell_bins)) {
        return;
    }

    realloc_if_needed(&d_agent_cell_offsets_, agent_cell_bins + 1, agent_cell_capacity_, sizeof(int));
    realloc_if_needed(&d_food_cell_offsets_, food_cell_bins + 1, food_cell_capacity_, sizeof(int));
    realloc_if_needed(&d_agent_cell_counts_, agent_cell_bins, agent_bin_capacity_, sizeof(int));
    realloc_if_needed(&d_food_cell_counts_, food_cell_bins, food_bin_capacity_, sizeof(int));
    realloc_if_needed(&d_agent_cell_write_counts_, agent_cell_bins, agent_write_capacity_, sizeof(int));
    realloc_if_needed(&d_food_cell_write_counts_, food_cell_bins, food_write_capacity_, sizeof(int));
    realloc_if_needed(&d_agent_cell_ids_, num_agents_, agent_bin_ids_capacity_, sizeof(unsigned int));
    realloc_if_needed(&d_food_cell_ids_, food_count > 0 ? food_count : 1, food_bin_ids_capacity_, sizeof(unsigned int));
    if (had_error_) {
        return;
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    size_t agent_scan_bytes = 0;
    size_t food_scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, agent_scan_bytes,
        d_agent_cell_counts_, d_agent_cell_offsets_ + 1, agent_cell_bins, s));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, food_scan_bytes,
        d_food_cell_counts_, d_food_cell_offsets_ + 1, food_cell_bins, s));
    agent_scan_temp_bytes_ = agent_scan_bytes;
    food_scan_temp_bytes_ = food_scan_bytes;
    if (!ensure_scan_temp_storage(std::max(agent_scan_bytes, food_scan_bytes))) {
        return;
    }

    if (food_count > food_capacity_) {
        if (d_food_states_) cudaFree(d_food_states_);
        if (d_food_pos_x_) cudaFree(d_food_pos_x_);
        if (d_food_pos_y_) cudaFree(d_food_pos_y_);
        if (d_food_active_) cudaFree(d_food_active_);
        d_food_states_ = nullptr;
        d_food_pos_x_ = nullptr;
        d_food_pos_y_ = nullptr;
        d_food_active_ = nullptr;
        had_error_ |= !malloc_checked(&d_food_states_, static_cast<size_t>(food_count) * sizeof(GpuFoodState), __FILE__, __LINE__);
        had_error_ |= !malloc_checked(&d_food_pos_x_, static_cast<size_t>(food_count) * sizeof(float), __FILE__, __LINE__);
        had_error_ |= !malloc_checked(&d_food_pos_y_, static_cast<size_t>(food_count) * sizeof(float), __FILE__, __LINE__);
        had_error_ |= !malloc_checked(&d_food_active_, static_cast<size_t>(food_count) * sizeof(unsigned int), __FILE__, __LINE__);
        food_capacity_ = had_error_ ? 0 : food_count;
    }
    if (had_error_) {
        return;
    }

    had_error_ |= !memcpy_async_checked(d_agent_states_, agents,
        static_cast<size_t>(agent_count) * sizeof(GpuAgentState), cudaMemcpyHostToDevice,
        s, __FILE__, __LINE__);
    const int block_size = 256;
    const int grid_size = (agent_count + block_size - 1) / block_size;
    unpack_agent_states_kernel<<<grid_size, block_size, 0, s>>>(
        d_agent_states_,
        d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_, d_agent_vel_y_,
        d_agent_speed_, d_agent_vision_, d_agent_energy_, d_agent_distance_traveled_,
        d_agent_age_, d_agent_kills_, d_agent_food_eaten_,
        d_agent_ids_, d_agent_types_, d_agent_alive_,
        agent_count);
    had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
    if (food_count > 0) {
        had_error_ |= !memcpy_async_checked(d_food_states_, food,
            static_cast<size_t>(food_count) * sizeof(GpuFoodState), cudaMemcpyHostToDevice,
            s, __FILE__, __LINE__);
        const int food_grid_size = (food_count + block_size - 1) / block_size;
        unpack_food_states_kernel<<<food_grid_size, block_size, 0, s>>>(
            d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_, food_count);
        had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
    }

    agent_cell_count_ = agent_cell_bins;
    agent_grid_entry_count_ = agent_count;
    food_cell_count_ = food_cell_bins;
    food_grid_entry_count_ = food_count;
    agent_cols_ = agent_cols;
    agent_rows_ = agent_rows;
    agent_cell_size_ = agent_cell_size;
    food_cols_ = food_cols;
    food_rows_ = food_rows;
    food_cell_size_ = food_cell_size;
    food_count_ = food_count;
    invalidate_resident_graph();
}

void GpuBatch::launch_sensor_build_async(float world_width, float world_height,
                                         float max_energy, bool has_walls) {
    if (had_error_) {
        return;
    }
    batch_build_sensors(*this, world_width, world_height, max_energy, has_walls);
}

void GpuBatch::upload_resident_food_states_async(const GpuFoodState* food, int food_count) {
    if (had_error_) {
        return;
    }
    if (food_count < 0 || (food_count > 0 && food == nullptr)) {
        had_error_ = true;
        return;
    }
    if (food_count > food_capacity_) {
        if (d_food_states_) {
            cudaFree(d_food_states_);
            d_food_states_ = nullptr;
        }
        if (d_food_pos_x_) {
            cudaFree(d_food_pos_x_);
            d_food_pos_x_ = nullptr;
        }
        if (d_food_pos_y_) {
            cudaFree(d_food_pos_y_);
            d_food_pos_y_ = nullptr;
        }
        if (d_food_active_) {
            cudaFree(d_food_active_);
            d_food_active_ = nullptr;
        }
        if (d_food_cell_ids_) {
            cudaFree(d_food_cell_ids_);
            d_food_cell_ids_ = nullptr;
        }
        if (!malloc_checked(&d_food_states_, static_cast<size_t>(food_count) * sizeof(GpuFoodState),
                            __FILE__, __LINE__)) {
            had_error_ = true;
            food_capacity_ = 0;
            return;
        }
        food_capacity_ = food_count;
        if (food_count > 0) {
            had_error_ |= !malloc_checked(&d_food_pos_x_, static_cast<size_t>(food_count) * sizeof(float), __FILE__, __LINE__);
            had_error_ |= !malloc_checked(&d_food_pos_y_, static_cast<size_t>(food_count) * sizeof(float), __FILE__, __LINE__);
            had_error_ |= !malloc_checked(&d_food_active_, static_cast<size_t>(food_count) * sizeof(unsigned int), __FILE__, __LINE__);
            had_error_ |= !malloc_checked(&d_food_cell_ids_, static_cast<size_t>(food_count) * sizeof(unsigned int), __FILE__, __LINE__);
            if (!had_error_) {
                food_bin_ids_capacity_ = food_count;
            }
        }
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    if (food_count > 0) {
        had_error_ |= !memcpy_async_checked(d_food_states_, food,
            static_cast<size_t>(food_count) * sizeof(GpuFoodState), cudaMemcpyHostToDevice,
            s, __FILE__, __LINE__);
        const int block_size = 256;
        const int grid_size = (food_count + block_size - 1) / block_size;
        unpack_food_states_kernel<<<grid_size, block_size, 0, s>>>(
            d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_, food_count);
        had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
    }
    food_count_ = food_count;
    invalidate_resident_graph();
}

void GpuBatch::launch_resident_sensor_build_async(float world_width, float world_height,
                                                  float max_energy, bool has_walls) {
    if (had_error_) {
        return;
    }
    batch_build_sensors_resident(*this, world_width, world_height, max_energy, has_walls);
}

void GpuBatch::prepare_resident_tick_graph(const ResidentTickParams& params) {
    if (had_error_) {
        return;
    }
    if (resident_graph_valid_
        && resident_tick_params_.dt == params.dt
        && resident_tick_params_.world_width == params.world_width
        && resident_tick_params_.world_height == params.world_height
        && resident_tick_params_.has_walls == params.has_walls
        && resident_tick_params_.energy_drain_per_tick == params.energy_drain_per_tick
        && resident_tick_params_.target_fps == params.target_fps
        && resident_tick_params_.food_pickup_range == params.food_pickup_range
        && resident_tick_params_.attack_range == params.attack_range
        && resident_tick_params_.max_energy == params.max_energy
        && resident_tick_params_.energy_gain_from_food == params.energy_gain_from_food
        && resident_tick_params_.energy_gain_from_kill == params.energy_gain_from_kill
        && resident_tick_params_.food_respawn_rate == params.food_respawn_rate
        && resident_tick_params_.seed == params.seed) {
        return;
    }
    invalidate_resident_graph();
    resident_tick_params_ = params;
    batch_prepare_resident_tick_graph(*this, params);
}

void GpuBatch::launch_resident_inference_tick_async(float dt, float world_width, float world_height,
                                                    bool has_walls, float energy_drain_per_tick,
                                                    int target_fps, float food_pickup_range,
                                                    float attack_range, float max_energy,
                                                    float energy_gain_from_food,
                                                    float energy_gain_from_kill, float food_respawn_rate,
                                                    std::uint64_t seed, int tick_index) {
    if (had_error_) {
        return;
    }
    ResidentTickParams params{
        dt,
        world_width,
        world_height,
        has_walls,
        energy_drain_per_tick,
        target_fps,
        food_pickup_range,
        attack_range,
        max_energy,
        energy_gain_from_food,
        energy_gain_from_kill,
        food_respawn_rate,
        seed,
    };
    prepare_resident_tick_graph(params);
    if (had_error_) {
        return;
    }
    batch_simulate_tick_resident(*this, dt, world_width, world_height, has_walls,
                                 energy_drain_per_tick, target_fps, food_pickup_range,
                                 attack_range, energy_gain_from_food, energy_gain_from_kill,
                                 food_respawn_rate, seed, tick_index);
}

#ifdef MOONAI_BUILD_PROFILER
void GpuBatch::reset_resident_stage_timings_async() {
    if (had_error_) {
        return;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    had_error_ |= !check_cuda(cudaMemsetAsync(d_resident_stage_accum_ns_, 0,
        static_cast<size_t>(GpuStageTiming::Count) * sizeof(unsigned long long), s), __FILE__, __LINE__);
    had_error_ |= !check_cuda(cudaMemsetAsync(d_resident_stage_last_timestamp_ns_, 0,
        sizeof(unsigned long long), s), __FILE__, __LINE__);
}

void GpuBatch::flush_resident_stage_timings_to_profiler() {
    if (had_error_ || !Profiler::instance().enabled()) {
        return;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    had_error_ |= !memcpy_async_checked(h_resident_stage_accum_ns_, d_resident_stage_accum_ns_,
        static_cast<size_t>(GpuStageTiming::Count) * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
        s, __FILE__, __LINE__);
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
    if (had_error_) {
        return;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(GpuStageTiming::Count); ++i) {
        Profiler::instance().add_gpu_stage_duration(static_cast<GpuStageTiming>(i),
            static_cast<std::int64_t>(h_resident_stage_accum_ns_[i]));
    }
}
#endif

void GpuBatch::download_agent_states(std::vector<GpuAgentState>& agents) {
    agents.resize(static_cast<size_t>(num_agents_));
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int block_size = 256;
    const int grid_size = (num_agents_ + block_size - 1) / block_size;
    pack_agent_states_kernel<<<grid_size, block_size, 0, s>>>(
        d_agent_states_,
        d_agent_pos_x_, d_agent_pos_y_, d_agent_vel_x_, d_agent_vel_y_,
        d_agent_speed_, d_agent_vision_, d_agent_energy_, d_agent_distance_traveled_,
        d_agent_age_, d_agent_kills_, d_agent_food_eaten_,
        d_agent_ids_, d_agent_types_, d_agent_alive_,
        num_agents_);
    had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
    if (had_error_) {
        return;
    }
    had_error_ |= !check_cuda(cudaMemcpy(agents.data(), d_agent_states_,
        static_cast<size_t>(num_agents_) * sizeof(GpuAgentState), cudaMemcpyDeviceToHost),
        __FILE__, __LINE__);
}

void GpuBatch::download_food_states(std::vector<GpuFoodState>& food) {
    food.resize(static_cast<size_t>(food_count_));
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    if (food_count_ > 0) {
        const int block_size = 256;
        const int grid_size = (food_count_ + block_size - 1) / block_size;
        pack_food_states_kernel<<<grid_size, block_size, 0, s>>>(
            d_food_states_, d_food_pos_x_, d_food_pos_y_, d_food_active_, food_count_);
        had_error_ |= !check_cuda(cudaGetLastError(), __FILE__, __LINE__);
    }
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
    if (had_error_) {
        return;
    }
    if (food_count_ > 0) {
        had_error_ |= !check_cuda(cudaMemcpy(food.data(), d_food_states_,
            static_cast<size_t>(food_count_) * sizeof(GpuFoodState), cudaMemcpyDeviceToHost),
            __FILE__, __LINE__);
    }
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
    had_error_ |= !memcpy_async_checked(h_pinned_out_, d_outputs_,
        bytes, cudaMemcpyDeviceToHost, s, __FILE__, __LINE__);
}

void GpuBatch::finish_unpack() {
    cudaStream_t s = static_cast<cudaStream_t>(stream_);
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
}

void GpuBatch::finish_unpack(float* dst, int count) {
    if (!validate_copy_count(count, num_agents_ * num_outputs_, "output copy")) {
        return;
    }
    finish_unpack();
    if (had_error_) {
        return;
    }
    memcpy(dst, h_pinned_out_, static_cast<size_t>(count) * sizeof(float));
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
    spdlog::info("  Total memory: {:.1f} MB", prop.totalGlobalMem / (1024.0 * 1024.0));
    spdlog::info("  SM count: {}", prop.multiProcessorCount);
}

bool GpuBatch::validate_copy_count(int count, int capacity, const char* label) {
    if (count < 0 || count > capacity) {
        had_error_ = true;
        fprintf(stderr, "GpuBatch %s rejected: count=%d capacity=%d\n", label, count, capacity);
        return false;
    }
    return true;
}

} // namespace moonai::gpu
