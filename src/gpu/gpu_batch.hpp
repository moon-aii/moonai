#pragma once

// gpu_batch.hpp — C++ header with NO CUDA dependency.
// Can be included from regular .cpp files when MOONAI_ENABLE_CUDA is ON.
// The CUDA implementation lives in gpu_batch.cu.

#include "gpu/gpu_types.hpp"

#include <cstdint>
#include <cstddef>
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

struct ResidentTickParams {
    float dt = 0.0f;
    float world_width = 0.0f;
    float world_height = 0.0f;
    bool has_walls = false;
    float energy_drain_per_tick = 0.0f;
    int target_fps = 0;
    float food_pickup_range = 0.0f;
    float attack_range = 0.0f;
    float max_energy = 0.0f;
    float energy_gain_from_food = 0.0f;
    float energy_gain_from_kill = 0.0f;
    float food_respawn_rate = 0.0f;
    std::uint64_t seed = 0;
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
    void upload_agent_states_async(const GpuAgentState* agents, int agent_count);
    void upload_tick_state_async(
        const GpuAgentState* agents,
        int agent_count,
        const GpuFoodState* food,
        int food_count,
        int agent_cols,
        int agent_rows,
        float agent_cell_size,
        int food_cols,
        int food_rows,
        float food_cell_size);
    void launch_sensor_build_async(float world_width, float world_height,
                                   float max_energy, bool has_walls);
    // Launch kernel on the stream.
    void launch_inference_async();
    // Async D2H into pinned host buffer.
    void start_unpack_async();
    // Block until stream completes, then copy from pinned buffer to dst.
    void finish_unpack();
    void finish_unpack(float* dst, int count);
    void upload_resident_food_states_async(const GpuFoodState* food, int food_count);
    void launch_resident_sensor_build_async(float world_width, float world_height,
                                            float max_energy, bool has_walls);
    void prepare_resident_tick_graph(const ResidentTickParams& params);
    void launch_resident_inference_tick_async(float dt, float world_width, float world_height,
                                              bool has_walls, float energy_drain_per_tick,
                                              int target_fps, float food_pickup_range,
                                              float attack_range, float max_energy,
                                              float energy_gain_from_food,
                                              float energy_gain_from_kill, float food_respawn_rate,
                                              std::uint64_t seed, int tick_index);
    void download_agent_states(std::vector<GpuAgentState>& agents);
    void download_food_states(std::vector<GpuFoodState>& food);
    bool ok() const { return !had_error_; }

    float*       host_inputs()       { return h_pinned_in_; }
    const float* host_outputs() const { return h_pinned_out_; }

    // ── Kernel-facing accessors (device pointers) ────────────────────────
    const GpuNetDesc* d_descs()       const { return d_descs_; }
    const GpuAgentState* d_agent_states() const { return d_agent_states_; }
    GpuAgentState* d_agent_states() { return d_agent_states_; }
    const GpuFoodState* d_food_states() const { return d_food_states_; }
    GpuFoodState* d_food_states() { return d_food_states_; }
    const int* d_agent_cell_offsets() const { return d_agent_cell_offsets_; }
    const int* d_food_cell_offsets() const { return d_food_cell_offsets_; }
    int* d_agent_cell_offsets() { return d_agent_cell_offsets_; }
    int* d_food_cell_offsets() { return d_food_cell_offsets_; }
    int* d_agent_cell_counts() { return d_agent_cell_counts_; }
    int* d_food_cell_counts() { return d_food_cell_counts_; }
    int* d_agent_cell_write_counts() { return d_agent_cell_write_counts_; }
    int* d_food_cell_write_counts() { return d_food_cell_write_counts_; }
    unsigned int* d_agent_cell_ids() { return d_agent_cell_ids_; }
    unsigned int* d_food_cell_ids() { return d_food_cell_ids_; }
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
    int food_count()       const { return food_count_; }
    int agent_cell_count() const { return agent_cell_count_; }
    int agent_grid_entry_count() const { return agent_grid_entry_count_; }
    int food_cell_count()  const { return food_cell_count_; }
    int food_grid_entry_count() const { return food_grid_entry_count_; }
    int agent_cols() const { return agent_cols_; }
    int agent_rows() const { return agent_rows_; }
    float agent_cell_size() const { return agent_cell_size_; }
    int food_cols() const { return food_cols_; }
    int food_rows() const { return food_rows_; }
    float food_cell_size() const { return food_cell_size_; }
    int agent_bin_stride() const { return num_agents_; }
    int food_bin_stride() const { return food_count_; }
    int num_inputs()       const { return num_inputs_; }
    int num_outputs()      const { return num_outputs_; }
    int activation_fn_id() const { return activation_fn_id_; }
    const float* d_agent_pos_x() const { return d_agent_pos_x_; }
    const float* d_agent_pos_y() const { return d_agent_pos_y_; }
    const float* d_agent_vel_x() const { return d_agent_vel_x_; }
    const float* d_agent_vel_y() const { return d_agent_vel_y_; }
    const float* d_agent_speed() const { return d_agent_speed_; }
    const float* d_agent_vision() const { return d_agent_vision_; }
    const float* d_agent_energy() const { return d_agent_energy_; }
    const float* d_agent_distance_traveled() const { return d_agent_distance_traveled_; }
    const int* d_agent_age() const { return d_agent_age_; }
    const int* d_agent_kills() const { return d_agent_kills_; }
    const int* d_agent_food_eaten() const { return d_agent_food_eaten_; }
    const unsigned int* d_agent_ids() const { return d_agent_ids_; }
    const unsigned int* d_agent_types() const { return d_agent_types_; }
    const unsigned int* d_agent_alive() const { return d_agent_alive_; }
    float* d_agent_pos_x() { return d_agent_pos_x_; }
    float* d_agent_pos_y() { return d_agent_pos_y_; }
    float* d_agent_vel_x() { return d_agent_vel_x_; }
    float* d_agent_vel_y() { return d_agent_vel_y_; }
    float* d_agent_speed() { return d_agent_speed_; }
    float* d_agent_vision() { return d_agent_vision_; }
    float* d_agent_energy() { return d_agent_energy_; }
    float* d_agent_distance_traveled() { return d_agent_distance_traveled_; }
    int* d_agent_age() { return d_agent_age_; }
    int* d_agent_kills() { return d_agent_kills_; }
    int* d_agent_food_eaten() { return d_agent_food_eaten_; }
    unsigned int* d_agent_ids() { return d_agent_ids_; }
    unsigned int* d_agent_types() { return d_agent_types_; }
    unsigned int* d_agent_alive() { return d_agent_alive_; }
    const float* d_food_pos_x() const { return d_food_pos_x_; }
    const float* d_food_pos_y() const { return d_food_pos_y_; }
    const unsigned int* d_food_active() const { return d_food_active_; }
    float* d_food_pos_x() { return d_food_pos_x_; }
    float* d_food_pos_y() { return d_food_pos_y_; }
    unsigned int* d_food_active() { return d_food_active_; }
    const int* d_inference_agent_indices() const { return d_inference_agent_indices_; }
    int* d_tick_index() { return d_tick_index_; }
    void* d_scan_temp_storage() const { return d_scan_temp_storage_; }
    int* host_tick_index() { return h_tick_index_; }
    size_t agent_scan_temp_bytes() const { return agent_scan_temp_bytes_; }
    size_t food_scan_temp_bytes() const { return food_scan_temp_bytes_; }
    int inference_bucket_count() const { return static_cast<int>(inference_bucket_offsets_.size()); }
    int inference_bucket_start(int bucket_index) const { return inference_bucket_offsets_[static_cast<size_t>(bucket_index)]; }
    int inference_bucket_size(int bucket_index) const { return inference_bucket_sizes_[static_cast<size_t>(bucket_index)]; }
    bool ensure_scan_temp_storage(size_t bytes);

private:
    friend void batch_prepare_resident_tick_graph(GpuBatch& batch, const ResidentTickParams& params);
    friend void batch_simulate_tick_resident(GpuBatch& batch, float dt, float world_width,
                                             float world_height, bool has_walls,
                                             float energy_drain_per_tick, int target_fps,
                                             float food_pickup_range, float attack_range,
                                             float energy_gain_from_food, float energy_gain_from_kill,
                                             float food_respawn_rate, std::uint64_t seed,
                                             int tick_index);

    // Device arrays (allocated in constructor, freed in destructor)
    GpuNetDesc*    d_descs_     = nullptr;  // [num_agents]
    float*         d_inputs_    = nullptr;  // [num_agents * num_inputs]
    float*         d_outputs_   = nullptr;  // [num_agents * num_outputs]
    GpuAgentState* d_agent_states_ = nullptr; // [num_agents]
    GpuFoodState*  d_food_states_ = nullptr;
    int*           d_agent_cell_offsets_ = nullptr;
    int*           d_food_cell_offsets_ = nullptr;
    int*           d_agent_cell_counts_ = nullptr;
    int*           d_food_cell_counts_ = nullptr;
    int*           d_agent_cell_write_counts_ = nullptr;
    int*           d_food_cell_write_counts_ = nullptr;
    unsigned int*  d_agent_cell_ids_ = nullptr;
    unsigned int*  d_food_cell_ids_ = nullptr;

    float*         d_agent_pos_x_ = nullptr;
    float*         d_agent_pos_y_ = nullptr;
    float*         d_agent_vel_x_ = nullptr;
    float*         d_agent_vel_y_ = nullptr;
    float*         d_agent_speed_ = nullptr;
    float*         d_agent_vision_ = nullptr;
    float*         d_agent_energy_ = nullptr;
    float*         d_agent_distance_traveled_ = nullptr;
    int*           d_agent_age_ = nullptr;
    int*           d_agent_kills_ = nullptr;
    int*           d_agent_food_eaten_ = nullptr;
    unsigned int*  d_agent_ids_ = nullptr;
    unsigned int*  d_agent_types_ = nullptr;
    unsigned int*  d_agent_alive_ = nullptr;

    float*         d_food_pos_x_ = nullptr;
    float*         d_food_pos_y_ = nullptr;
    unsigned int*  d_food_active_ = nullptr;

    int*           d_inference_agent_indices_ = nullptr;
    int*           d_tick_index_ = nullptr;
    void*          d_scan_temp_storage_ = nullptr;

    // Pinned host memory (required for cudaMemcpyAsync)
    float*         h_pinned_in_  = nullptr;
    float*         h_pinned_out_ = nullptr;
    int*           h_tick_index_ = nullptr;

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
    int food_capacity_ = 0;
    int food_count_ = 0;
    int agent_cell_capacity_ = 0;
    int food_cell_capacity_ = 0;
    int agent_bin_capacity_ = 0;
    int food_bin_capacity_ = 0;
    int agent_write_capacity_ = 0;
    int food_write_capacity_ = 0;
    int agent_bin_ids_capacity_ = 0;
    int food_bin_ids_capacity_ = 0;
    int inference_agent_indices_capacity_ = 0;
    int agent_cell_count_ = 0;
    int agent_grid_entry_count_ = 0;
    int food_cell_count_ = 0;
    int food_grid_entry_count_ = 0;
    int agent_cols_ = 0;
    int agent_rows_ = 0;
    float agent_cell_size_ = 1.0f;
    int food_cols_ = 0;
    int food_rows_ = 0;
    float food_cell_size_ = 1.0f;
    int num_inputs_;
    int num_outputs_;
    int activation_fn_id_ = 0;  // 0=sigmoid, 1=tanh, 2=relu
    bool had_error_ = false;
    bool resident_graph_valid_ = false;
    ResidentTickParams resident_tick_params_{};
    void* resident_tick_graph_ = nullptr;
    void* resident_tick_graph_exec_ = nullptr;
    std::vector<int> inference_bucket_offsets_;
    std::vector<int> inference_bucket_sizes_;
    size_t scan_temp_storage_bytes_ = 0;
    size_t agent_scan_temp_bytes_ = 0;
    size_t food_scan_temp_bytes_ = 0;

    bool validate_copy_count(int count, int capacity, const char* label);
    void invalidate_resident_graph();
};

// ── Free functions (implemented in neural_inference.cu / gpu_batch.cu) ────
void batch_neural_inference(GpuBatch& batch);  // launches on batch's stream
void batch_build_sensors(GpuBatch& batch, float world_width, float world_height,
                         float max_energy, bool has_walls);
void batch_build_sensors_resident(GpuBatch& batch, float world_width, float world_height,
                                  float max_energy, bool has_walls);
void batch_rebuild_compact_bins(GpuBatch& batch);
void batch_prepare_resident_tick_graph(GpuBatch& batch, const ResidentTickParams& params);
void batch_simulate_tick_resident(GpuBatch& batch, float dt, float world_width,
                                  float world_height, bool has_walls,
                                  float energy_drain_per_tick, int target_fps,
                                  float food_pickup_range, float attack_range,
                                  float energy_gain_from_food, float energy_gain_from_kill,
                                  float food_respawn_rate, std::uint64_t seed,
                                  int tick_index);
bool init_cuda();
void print_device_info();

} // namespace moonai::gpu
