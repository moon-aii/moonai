#pragma once

// gpu_batch.hpp — C++ header with NO CUDA dependency.
// Can be included from regular .cpp files when MOONAI_ENABLE_CUDA is ON.
// The CUDA implementation lives in gpu_batch.cu.

#include "gpu/gpu_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moonai::gpu {

// Pre-extracted flat network data (built on CPU, then uploaded to device).
// Separates the NeuralNetwork-specific packing logic from the GPU batch class
// so that moonai_gpu has no link-time dependency on moonai_evolution.
struct GpuNetworkData {
    std::vector<GpuNetDesc> descs;       // one per agent
    std::vector<uint8_t>    node_types;  // flat: 0=Input,1=Bias,2=Hidden,3=Output
    std::vector<int>        eval_order;  // flat: node indices in topo order per agent
    std::vector<int>        conn_ptr;    // flat: start idx in conn arrays per eval step
    std::vector<int>        in_count;    // flat: # incoming edges per eval step
    std::vector<int>        conn_from;   // flat: source node index per edge
    std::vector<float>      conn_w;      // flat: edge weight per edge
    std::vector<int>        out_indices; // flat: output node positions per agent
    int activation_fn_id = 0;           // 0=sigmoid, 1=tanh, 2=relu
};

// RAII wrapper owning all device memory for a generation's batch inference.
// Fixed-size arrays (inputs, outputs, descs) are allocated in the constructor.
// Topology arrays are freed and reallocated each generation by upload_network_data().
//
// Uses a single CUDA stream with pinned host memory for async H2D/D2H transfers.
class GpuBatch {
public:
    GpuBatch(int num_agents, int num_inputs, int num_outputs);
    ~GpuBatch();

    GpuBatch(const GpuBatch&)            = delete;
    GpuBatch& operator=(const GpuBatch&) = delete;

    // ── Per-generation setup ─────────────────────────────────────────────
    // Accepts pre-extracted network data (built by caller) and uploads to GPU.
    void upload_network_data(const GpuNetworkData& data);

    // ── Per-tick I/O (async — uses pinned memory + single CUDA stream) ──
    // Copy flat_inputs into pinned host buffer, then async H2D.
    void pack_inputs_async(const float* flat_inputs, int count);
    // Launch H2D using the existing pinned host input buffer.
    void pack_inputs_async(int count);
    // Launch kernel on the stream.
    void launch_inference_async();
    // Async D2H into pinned host buffer.
    void start_unpack_async();
    // Block until stream completes, then copy from pinned buffer to dst.
    void finish_unpack();
    void finish_unpack(float* dst, int count);
    bool ok() const { return !had_error_; }

    float*       host_inputs()       { return h_pinned_in_; }
    const float* host_outputs() const { return h_pinned_out_; }

    // ── Kernel-facing accessors (device pointers) ────────────────────────
    const GpuNetDesc* d_descs()       const { return d_descs_; }
    float*            d_node_vals()         { return d_node_vals_; }
    const uint8_t*    d_node_types()  const { return d_node_types_; }
    const int*        d_eval_order()  const { return d_eval_order_; }
    const int*        d_conn_ptr()    const { return d_conn_ptr_; }
    const int*        d_in_count()    const { return d_in_count_; }
    const int*        d_conn_from()   const { return d_conn_from_; }
    const float*      d_conn_w()      const { return d_conn_w_; }
    const int*        d_out_indices() const { return d_out_indices_; }
    float*            d_inputs()            { return d_inputs_; }
    float*            d_outputs()           { return d_outputs_; }
    void*             stream_handle() const { return stream_; }

    int num_agents()       const { return num_agents_; }
    int num_inputs()       const { return num_inputs_; }
    int num_outputs()      const { return num_outputs_; }
    int activation_fn_id() const { return activation_fn_id_; }

private:
    // Device arrays (allocated in constructor, freed in destructor)
    GpuNetDesc*    d_descs_     = nullptr;  // [num_agents]
    float*         d_inputs_    = nullptr;  // [num_agents * num_inputs]
    float*         d_outputs_   = nullptr;  // [num_agents * num_outputs]

    // Pinned host memory (required for cudaMemcpyAsync)
    float*         h_pinned_in_  = nullptr;
    float*         h_pinned_out_ = nullptr;

    // Single CUDA stream (stored as void* to keep CUDA types out of header)
    void*          stream_ = nullptr;

    // Topology arrays (reallocated each generation by upload_network_data)
    float*   d_node_vals_   = nullptr;  // [Σ num_nodes]   — scratch per tick
    uint8_t* d_node_types_  = nullptr;  // [Σ num_nodes]   — node kind
    int*     d_eval_order_  = nullptr;  // [Σ num_eval]    — node indices in topo order
    int*     d_conn_ptr_    = nullptr;  // [Σ num_eval]    — start in conn arrays
    int*     d_in_count_    = nullptr;  // [Σ num_eval]    — # incoming edges
    int*     d_conn_from_   = nullptr;  // [Σ total_conn]  — source node index
    float*   d_conn_w_      = nullptr;  // [Σ total_conn]  — edge weight
    int*     d_out_indices_ = nullptr;  // [Σ num_outputs] — output node positions

    // Topology capacity tracking (avoid reallocation when sizes fit)
    int capacity_node_vals_   = 0;
    int capacity_node_types_  = 0;
    int capacity_eval_order_  = 0;
    int capacity_conn_ptr_    = 0;
    int capacity_in_count_    = 0;
    int capacity_conn_from_   = 0;
    int capacity_conn_w_      = 0;
    int capacity_out_indices_ = 0;

    int num_agents_;
    int num_inputs_;
    int num_outputs_;
    int activation_fn_id_ = 0;  // 0=sigmoid, 1=tanh, 2=relu
    bool had_error_ = false;

    bool validate_copy_count(int count, int capacity, const char* label);
};

// ── Free functions (implemented in neural_inference.cu / gpu_batch.cu) ────
void batch_neural_inference(GpuBatch& batch);  // launches on batch's stream
bool init_cuda();
void print_device_info();

} // namespace moonai::gpu
