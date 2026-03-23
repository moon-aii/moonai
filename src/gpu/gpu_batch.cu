#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

#include <spdlog/spdlog.h>

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

} // namespace

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs)
    : num_agents_(num_agents)
    , num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    const size_t in_bytes = static_cast<size_t>(num_agents) * num_inputs * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(num_agents) * num_outputs * sizeof(float);

    had_error_ |= !malloc_checked(&d_descs_, num_agents * sizeof(GpuNetDesc), __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_inputs_, in_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_checked(&d_outputs_, out_bytes, __FILE__, __LINE__);

    had_error_ |= !malloc_host_checked(&h_pinned_in_, in_bytes, __FILE__, __LINE__);
    had_error_ |= !malloc_host_checked(&h_pinned_out_, out_bytes, __FILE__, __LINE__);

    cudaStream_t s = nullptr;
    had_error_ |= !check_cuda(cudaStreamCreate(&s), __FILE__, __LINE__);
    stream_ = static_cast<void*>(s);
}

GpuBatch::~GpuBatch() {
    // Device arrays
    if (d_descs_)       cudaFree(d_descs_);
    if (d_inputs_)      cudaFree(d_inputs_);
    if (d_outputs_)     cudaFree(d_outputs_);

    // Pinned host memory
    if (h_pinned_in_)   cudaFreeHost(h_pinned_in_);
    if (h_pinned_out_)  cudaFreeHost(h_pinned_out_);

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

    // ── Upload to device (async on batch stream) ──────────────────────
    had_error_ |= !memcpy_async_checked(d_descs_, data.descs.data(),
        n * sizeof(GpuNetDesc), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_node_types_, data.node_types.data(),
        total_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_eval_order_, data.eval_order.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_conn_ptr_, data.conn_ptr.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    had_error_ |= !memcpy_async_checked(d_in_count_, data.in_count.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    if (total_conn > 0) {
        had_error_ |= !memcpy_async_checked(d_conn_from_, data.conn_from.data(),
            total_conn * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
        had_error_ |= !memcpy_async_checked(d_conn_w_, data.conn_w.data(),
            total_conn * sizeof(float), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    }
    if (total_out > 0) {
        had_error_ |= !memcpy_async_checked(d_out_indices_, data.out_indices.data(),
            total_out * sizeof(int), cudaMemcpyHostToDevice, s, __FILE__, __LINE__);
    }

    // Ensure upload completes before first inference tick
    had_error_ |= !check_cuda(cudaStreamSynchronize(s), __FILE__, __LINE__);
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
