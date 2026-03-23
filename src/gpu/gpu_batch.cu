#include "gpu/gpu_batch.hpp"
#include "gpu/cuda_utils.cuh"

namespace moonai::gpu {

// ── Constructor / Destructor ─────────────────────────────────────────────────

GpuBatch::GpuBatch(int num_agents, int num_inputs, int num_outputs)
    : num_agents_(num_agents)
    , num_inputs_(num_inputs)
    , num_outputs_(num_outputs) {
    size_t in_bytes  = num_agents * num_inputs  * sizeof(float);
    size_t out_bytes = num_agents * num_outputs * sizeof(float);

    // Sync path arrays
    CUDA_CHECK_ABORT(cudaMalloc(&d_descs_,   num_agents * sizeof(GpuNetDesc)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_inputs_,  in_bytes));
    CUDA_CHECK_ABORT(cudaMalloc(&d_outputs_, out_bytes));

    // Async path: double-buffered device arrays, pinned host memory, streams
    for (int b = 0; b < kNumBuffers; ++b) {
        CUDA_CHECK_ABORT(cudaMalloc(&d_inputs_db_[b],  in_bytes));
        CUDA_CHECK_ABORT(cudaMalloc(&d_outputs_db_[b], out_bytes));
        CUDA_CHECK_ABORT(cudaMallocHost(&h_pinned_in_[b],  in_bytes));
        CUDA_CHECK_ABORT(cudaMallocHost(&h_pinned_out_[b], out_bytes));
        cudaStream_t s;
        CUDA_CHECK_ABORT(cudaStreamCreate(&s));
        streams_[b] = static_cast<void*>(s);
    }
}

GpuBatch::~GpuBatch() {
    // Sync path
    if (d_descs_)       cudaFree(d_descs_);
    if (d_inputs_)      cudaFree(d_inputs_);
    if (d_outputs_)     cudaFree(d_outputs_);

    // Async path: double buffers, pinned memory, streams
    for (int b = 0; b < kNumBuffers; ++b) {
        if (d_inputs_db_[b])  cudaFree(d_inputs_db_[b]);
        if (d_outputs_db_[b]) cudaFree(d_outputs_db_[b]);
        if (h_pinned_in_[b])  cudaFreeHost(h_pinned_in_[b]);
        if (h_pinned_out_[b]) cudaFreeHost(h_pinned_out_[b]);
        if (streams_[b])      cudaStreamDestroy(static_cast<cudaStream_t>(streams_[b]));
    }

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
    activation_fn_id_ = data.activation_fn_id;

    int n            = num_agents_;
    int total_nodes  = static_cast<int>(data.node_types.size());
    int total_eval   = static_cast<int>(data.eval_order.size());
    int total_conn   = static_cast<int>(data.conn_from.size());
    int total_out    = static_cast<int>(data.out_indices.size());

    // ── Free old topology allocations ────────────────────────────────────
    if (d_node_vals_)   { cudaFree(d_node_vals_);   d_node_vals_   = nullptr; }
    if (d_node_types_)  { cudaFree(d_node_types_);  d_node_types_  = nullptr; }
    if (d_eval_order_)  { cudaFree(d_eval_order_);  d_eval_order_  = nullptr; }
    if (d_conn_ptr_)    { cudaFree(d_conn_ptr_);    d_conn_ptr_    = nullptr; }
    if (d_in_count_)    { cudaFree(d_in_count_);    d_in_count_    = nullptr; }
    if (d_conn_from_)   { cudaFree(d_conn_from_);   d_conn_from_   = nullptr; }
    if (d_conn_w_)      { cudaFree(d_conn_w_);      d_conn_w_      = nullptr; }
    if (d_out_indices_) { cudaFree(d_out_indices_); d_out_indices_ = nullptr; }

    // ── Allocate device topology arrays ──────────────────────────────────
    CUDA_CHECK_ABORT(cudaMalloc(&d_node_vals_,  total_nodes * sizeof(float)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_node_types_, total_nodes * sizeof(uint8_t)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_eval_order_, total_eval  * sizeof(int)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_conn_ptr_,   total_eval  * sizeof(int)));
    CUDA_CHECK_ABORT(cudaMalloc(&d_in_count_,   total_eval  * sizeof(int)));
    if (total_conn > 0) {
        CUDA_CHECK_ABORT(cudaMalloc(&d_conn_from_, total_conn * sizeof(int)));
        CUDA_CHECK_ABORT(cudaMalloc(&d_conn_w_,    total_conn * sizeof(float)));
    }
    if (total_out > 0) {
        CUDA_CHECK_ABORT(cudaMalloc(&d_out_indices_, total_out * sizeof(int)));
    }

    // ── Upload to device ─────────────────────────────────────────────────
    CUDA_CHECK_ABORT(cudaMemcpy(d_descs_, data.descs.data(),
        n * sizeof(GpuNetDesc), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_node_types_, data.node_types.data(),
        total_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_eval_order_, data.eval_order.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_conn_ptr_, data.conn_ptr.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ABORT(cudaMemcpy(d_in_count_, data.in_count.data(),
        total_eval * sizeof(int), cudaMemcpyHostToDevice));
    if (total_conn > 0) {
        CUDA_CHECK_ABORT(cudaMemcpy(d_conn_from_, data.conn_from.data(),
            total_conn * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK_ABORT(cudaMemcpy(d_conn_w_, data.conn_w.data(),
            total_conn * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (total_out > 0) {
        CUDA_CHECK_ABORT(cudaMemcpy(d_out_indices_, data.out_indices.data(),
            total_out * sizeof(int), cudaMemcpyHostToDevice));
    }
    // d_node_vals_ is scratch — no initial upload needed; kernel initializes it each tick
}

// ── Per-tick I/O ─────────────────────────────────────────────────────────────

void GpuBatch::pack_inputs(const std::vector<float>& flat_inputs) {
    CUDA_CHECK(cudaMemcpy(d_inputs_, flat_inputs.data(),
        flat_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void GpuBatch::unpack_outputs(std::vector<float>& flat_out) const {
    flat_out.resize(num_agents_ * num_outputs_);
    CUDA_CHECK(cudaMemcpy(flat_out.data(), d_outputs_,
        flat_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

// ── Async per-tick I/O ───────────────────────────────────────────────────────

void GpuBatch::pack_inputs_async(const float* flat_inputs, int count, int buf) {
    size_t bytes = count * sizeof(float);
    memcpy(h_pinned_in_[buf], flat_inputs, bytes);
    cudaStream_t s = static_cast<cudaStream_t>(streams_[buf]);
    CUDA_CHECK(cudaMemcpyAsync(d_inputs_db_[buf], h_pinned_in_[buf],
        bytes, cudaMemcpyHostToDevice, s));
}

void GpuBatch::launch_inference_async(int buf) {
    batch_neural_inference(*this, buf);
}

void GpuBatch::start_unpack_async(int buf) {
    size_t bytes = num_agents_ * num_outputs_ * sizeof(float);
    cudaStream_t s = static_cast<cudaStream_t>(streams_[buf]);
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out_[buf], d_outputs_db_[buf],
        bytes, cudaMemcpyDeviceToHost, s));
}

void GpuBatch::finish_unpack(float* dst, int count, int buf) {
    cudaStream_t s = static_cast<cudaStream_t>(streams_[buf]);
    CUDA_CHECK(cudaStreamSynchronize(s));
    memcpy(dst, h_pinned_out_[buf], count * sizeof(float));
}

void GpuBatch::sync_stream(int buf) {
    cudaStream_t s = static_cast<cudaStream_t>(streams_[buf]);
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ── CUDA initialization (moved from fitness_eval.cu) ─────────────────────────

bool init_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(0));
    return true;
}

void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  SM count: %d\n", prop.multiProcessorCount);
}

} // namespace moonai::gpu
