#pragma once

// gpu_batch.hpp — C++ header with NO CUDA dependency.
// Can be included from regular .cpp files when MOONAI_ENABLE_CUDA is ON.
// The CUDA implementation lives in gpu_batch.cu.

#include "gpu/gpu_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moonai::gpu {

// Pre-extracted flat network data (built on CPU, then uploaded to device).
// Separates the NeuralNetwork-specific packing logic from the GPU batch class
// so that moonai_gpu has no link-time dependency on moonai_evolution.
struct GpuNetworkData {
  std::vector<GpuNetDesc> descs;   // one per agent
  std::vector<uint8_t> node_types; // flat: 0=Input,1=Bias,2=Hidden,3=Output
  std::vector<int> eval_order;     // flat: node indices in topo order per agent
  std::vector<int> conn_ptr;    // flat: start idx in conn arrays per eval step
  std::vector<int> in_count;    // flat: # incoming edges per eval step
  std::vector<int> conn_from;   // flat: source node index per edge
  std::vector<float> conn_w;    // flat: edge weight per edge
  std::vector<int> out_indices; // flat: output node positions per agent
  int activation_fn_id = 0;     // 0=sigmoid, 1=tanh, 2=relu
};

// Parameters for ecology step kernel sequence
struct EcologyStepParams {
  float dt = 0.0f;
  float world_width = 0.0f;
  float world_height = 0.0f;
  bool has_walls = false;
  float energy_drain_per_step = 0.0f;
  int target_fps = 60;
  float food_pickup_range = 12.0f;
  float attack_range = 20.0f;
  float max_energy = 150.0f;
  float energy_gain_from_food = 40.0f;
  float energy_gain_from_kill = 60.0f;
  float food_respawn_rate = 0.02f;
  std::uint64_t seed = 0;
  int step_index = 0;
};

// RAII wrapper owning all device memory for GPU-accelerated ecology.
// Supports two modes:
//   1. Inference-only: GPU runs neural networks, CPU runs ecology
//   2. Full ecology: GPU runs sensing, inference, movement, food, attacks
//
// Uses a single CUDA stream with pinned host memory for async H2D/D2H
// transfers.
class GpuBatch {
public:
  GpuBatch(int num_agents, int num_inputs, int num_outputs,
           float world_width = 3000.0f, float world_height = 3000.0f);
  ~GpuBatch();

  GpuBatch(const GpuBatch &) = delete;
  GpuBatch &operator=(const GpuBatch &) = delete;

  // ── Per-generation setup ─────────────────────────────────────────────
  // Accepts pre-extracted network data (built by caller) and uploads to GPU.
  void upload_network_data(const GpuNetworkData &data);

  // ── Full Ecology Step (NEW) ──────────────────────────────────────────
  // Upload agent states from CPU to GPU (positions, energy, age, etc.)
  void upload_agent_states_async(const GpuAgentState *agents, int agent_count);

  // Download agent states from GPU to CPU after ecology step
  void download_agent_states_async(GpuAgentState *agents, int agent_count);

  // Download only changed fields (positions, energy, alive flags)
  void download_agent_changes_async(float *pos_x, float *pos_y, float *energy,
                                    unsigned int *alive, int count);

  // Download agent counters (kills, food_eaten, age, distance_traveled)
  void download_agent_counters_async(int *kills, int *food_eaten, int *age,
                                     float *distance_traveled, int count);

  // Upload food states from CPU to GPU
  void upload_food_states_async(const GpuFoodState *food, int food_count);

  // Download food states from GPU to CPU
  void download_food_states_async(GpuFoodState *food, int food_count);

  // Launch the full ecology step sequence on GPU
  // This replaces individual kernel calls for maximum efficiency
  void launch_ecology_step_async(const EcologyStepParams &params);

  // ── Individual Kernel Stages (for flexibility/debugging) ─────────────
  // Rebuild spatial bins for all agents
  void rebuild_bins_async(float world_width, float world_height);

  // Rebuild spatial bins for food only (used after agent movement)
  void rebuild_food_bins_async(float world_width, float world_height);

  // Rebuild spatial bins for prey only (used after movement)
  void rebuild_prey_bins_async(float world_width, float world_height);

  // Build sensor inputs for all agents
  void build_sensors_async(float world_width, float world_height,
                           float max_energy, bool has_walls);

  // Run neural inference (forward pass)
  void launch_inference_async();

  // Apply movement, energy drain, boundary handling, death check
  void apply_movement_async(const EcologyStepParams &params);

  // Process prey food consumption
  void process_prey_food_async(float world_width, float world_height,
                               bool has_walls, float food_pickup_range,
                               float energy_gain_from_food);

  // Process predator attacks
  void process_predator_attacks_async(float world_width, float world_height,
                                      bool has_walls, float attack_range,
                                      float energy_gain_from_kill);

  // Respawn food
  void respawn_food_async(float world_width, float world_height,
                          float respawn_rate, std::uint64_t seed,
                          int step_index);

  // ── Synchronization ──────────────────────────────────────────────────
  // Wait for all async operations to complete
  void synchronize();

  // ── Per-tick I/O (async — uses pinned memory + single CUDA stream) ──
  // Copy flat_inputs into pinned host buffer, then async H2D.
  void pack_inputs_async(const float *flat_inputs, int count);
  // Launch H2D using the existing pinned host input buffer.
  void pack_inputs_async(int count);

  // Async D2H into pinned host buffer.
  void start_unpack_async();
  // Block until stream completes, then copy from pinned buffer to dst.
  void finish_unpack();
  void finish_unpack(float *dst, int count);

  // Download agent states (legacy interface)
  void download_agent_states(std::vector<GpuAgentState> &agents);

  // Download food states (legacy interface)
  void download_food_states(std::vector<GpuFoodState> &food);

  bool ok() const {
    return !had_error_;
  }

  // ── Host accessors ───────────────────────────────────────────────────
  float *host_inputs() {
    return h_pinned_in_;
  }
  const float *host_outputs() const {
    return h_pinned_out_;
  }

  // ── Kernel-facing accessors (device pointers) ────────────────────────
  const GpuNetDesc *d_descs() const {
    return d_descs_;
  }
  const GpuAgentState *d_agent_states() const {
    return d_agent_states_;
  }
  GpuAgentState *d_agent_states() {
    return d_agent_states_;
  }
  const GpuFoodState *d_food_states() const {
    return d_food_states_;
  }
  GpuFoodState *d_food_states() {
    return d_food_states_;
  }
  const int *d_agent_cell_offsets() const {
    return d_agent_cell_offsets_;
  }
  const int *d_food_cell_offsets() const {
    return d_food_cell_offsets_;
  }
  int *d_agent_cell_offsets() {
    return d_agent_cell_offsets_;
  }
  int *d_food_cell_offsets() {
    return d_food_cell_offsets_;
  }
  int *d_agent_cell_counts() {
    return d_agent_cell_counts_;
  }
  int *d_food_cell_counts() {
    return d_food_cell_counts_;
  }
  int *d_agent_cell_write_counts() {
    return d_agent_cell_write_counts_;
  }
  int *d_food_cell_write_counts() {
    return d_food_cell_write_counts_;
  }
  unsigned int *d_agent_cell_ids() {
    return d_agent_cell_ids_;
  }
  unsigned int *d_food_cell_ids() {
    return d_food_cell_ids_;
  }
  GpuSensorAgentEntry *d_sensor_agent_entries() {
    return d_sensor_agent_entries_;
  }
  const GpuSensorAgentEntry *d_sensor_agent_entries() const {
    return d_sensor_agent_entries_;
  }
  GpuSensorFoodEntry *d_sensor_food_entries() {
    return d_sensor_food_entries_;
  }
  const GpuSensorFoodEntry *d_sensor_food_entries() const {
    return d_sensor_food_entries_;
  }
  float *d_node_vals() {
    return d_node_vals_;
  }
  const uint8_t *d_node_types() const {
    return d_node_types_;
  }
  const int *d_eval_order() const {
    return d_eval_order_;
  }
  const int *d_conn_ptr() const {
    return d_conn_ptr_;
  }
  const int *d_in_count() const {
    return d_in_count_;
  }
  const int *d_conn_from() const {
    return d_conn_from_;
  }
  const float *d_conn_w() const {
    return d_conn_w_;
  }
  const int *d_out_indices() const {
    return d_out_indices_;
  }
  float *d_inputs() {
    return d_inputs_;
  }
  float *d_outputs() {
    return d_outputs_;
  }
  void *stream_handle() const {
    return stream_;
  }

  int num_agents() const {
    return num_agents_;
  }
  int food_count() const {
    return food_count_;
  }
  int agent_cell_count() const {
    return agent_cell_count_;
  }
  int agent_grid_entry_count() const {
    return agent_grid_entry_count_;
  }
  int food_cell_count() const {
    return food_cell_count_;
  }
  int food_grid_entry_count() const {
    return food_grid_entry_count_;
  }
  int agent_cols() const {
    return agent_cols_;
  }
  int agent_rows() const {
    return agent_rows_;
  }
  float agent_cell_size() const {
    return agent_cell_size_;
  }
  int food_cols() const {
    return food_cols_;
  }
  int food_rows() const {
    return food_rows_;
  }
  float food_cell_size() const {
    return food_cell_size_;
  }
  int agent_bin_stride() const {
    return num_agents_;
  }
  int food_bin_stride() const {
    return food_count_;
  }
  int num_inputs() const {
    return num_inputs_;
  }
  int num_outputs() const {
    return num_outputs_;
  }
  int activation_fn_id() const {
    return activation_fn_id_;
  }
  const float *d_agent_pos_x() const {
    return d_agent_pos_x_;
  }
  const float *d_agent_pos_y() const {
    return d_agent_pos_y_;
  }
  const float *d_agent_vel_x() const {
    return d_agent_vel_x_;
  }
  const float *d_agent_vel_y() const {
    return d_agent_vel_y_;
  }
  const float *d_agent_speed() const {
    return d_agent_speed_;
  }
  const float *d_agent_vision() const {
    return d_agent_vision_;
  }
  const float *d_agent_energy() const {
    return d_agent_energy_;
  }
  const float *d_agent_distance_traveled() const {
    return d_agent_distance_traveled_;
  }
  const int *d_agent_age() const {
    return d_agent_age_;
  }
  const int *d_agent_kills() const {
    return d_agent_kills_;
  }
  const int *d_agent_food_eaten() const {
    return d_agent_food_eaten_;
  }
  const unsigned int *d_agent_ids() const {
    return d_agent_ids_;
  }
  const unsigned int *d_agent_types() const {
    return d_agent_types_;
  }
  const unsigned int *d_agent_alive() const {
    return d_agent_alive_;
  }
  float *d_agent_pos_x() {
    return d_agent_pos_x_;
  }
  float *d_agent_pos_y() {
    return d_agent_pos_y_;
  }
  float *d_agent_vel_x() {
    return d_agent_vel_x_;
  }
  float *d_agent_vel_y() {
    return d_agent_vel_y_;
  }
  float *d_agent_speed() {
    return d_agent_speed_;
  }
  float *d_agent_vision() {
    return d_agent_vision_;
  }
  float *d_agent_energy() {
    return d_agent_energy_;
  }
  float *d_agent_distance_traveled() {
    return d_agent_distance_traveled_;
  }
  int *d_agent_age() {
    return d_agent_age_;
  }
  int *d_agent_kills() {
    return d_agent_kills_;
  }
  int *d_agent_food_eaten() {
    return d_agent_food_eaten_;
  }
  unsigned int *d_agent_ids() {
    return d_agent_ids_;
  }
  unsigned int *d_agent_types() {
    return d_agent_types_;
  }
  unsigned int *d_agent_alive() {
    return d_agent_alive_;
  }
  const float *d_food_pos_x() const {
    return d_food_pos_x_;
  }
  const float *d_food_pos_y() const {
    return d_food_pos_y_;
  }
  const unsigned int *d_food_active() const {
    return d_food_active_;
  }
  float *d_food_pos_x() {
    return d_food_pos_x_;
  }
  float *d_food_pos_y() {
    return d_food_pos_y_;
  }
  unsigned int *d_food_active() {
    return d_food_active_;
  }
  void *d_scan_temp_storage() const {
    return d_scan_temp_storage_;
  }
  size_t agent_scan_temp_bytes() const {
    return agent_scan_temp_bytes_;
  }
  size_t food_scan_temp_bytes() const {
    return food_scan_temp_bytes_;
  }
  bool ensure_scan_temp_storage(size_t bytes);

private:
  // Device arrays (allocated in constructor, freed in destructor)
  GpuNetDesc *d_descs_ = nullptr;           // [num_agents]
  float *d_inputs_ = nullptr;               // [num_agents * num_inputs]
  float *d_outputs_ = nullptr;              // [num_agents * num_outputs]
  GpuAgentState *d_agent_states_ = nullptr; // [num_agents]
  GpuFoodState *d_food_states_ = nullptr;
  int *d_agent_cell_offsets_ = nullptr;
  int *d_food_cell_offsets_ = nullptr;
  int *d_agent_cell_counts_ = nullptr;
  int *d_food_cell_counts_ = nullptr;
  int *d_agent_cell_write_counts_ = nullptr;
  int *d_food_cell_write_counts_ = nullptr;
  unsigned int *d_agent_cell_ids_ = nullptr;
  unsigned int *d_food_cell_ids_ = nullptr;
  GpuSensorAgentEntry *d_sensor_agent_entries_ = nullptr;
  GpuSensorFoodEntry *d_sensor_food_entries_ = nullptr;

  float *d_agent_pos_x_ = nullptr;
  float *d_agent_pos_y_ = nullptr;
  float *d_agent_vel_x_ = nullptr;
  float *d_agent_vel_y_ = nullptr;
  float *d_agent_speed_ = nullptr;
  float *d_agent_vision_ = nullptr;
  float *d_agent_energy_ = nullptr;
  float *d_agent_distance_traveled_ = nullptr;
  int *d_agent_age_ = nullptr;
  int *d_agent_kills_ = nullptr;
  int *d_agent_food_eaten_ = nullptr;
  unsigned int *d_agent_ids_ = nullptr;
  unsigned int *d_agent_types_ = nullptr;
  unsigned int *d_agent_alive_ = nullptr;

  float *d_food_pos_x_ = nullptr;
  float *d_food_pos_y_ = nullptr;
  unsigned int *d_food_active_ = nullptr;

  void *d_scan_temp_storage_ = nullptr;

  // Pinned host memory (required for cudaMemcpyAsync)
  float *h_pinned_in_ = nullptr;
  float *h_pinned_out_ = nullptr;

  // Single CUDA stream (stored as void* to keep CUDA types out of header)
  void *stream_ = nullptr;

  // Topology arrays (reallocated each generation by upload_network_data)
  float *d_node_vals_ = nullptr;    // [Σ num_nodes]   — scratch per tick
  uint8_t *d_node_types_ = nullptr; // [Σ num_nodes]   — node kind
  int *d_eval_order_ = nullptr;  // [Σ num_eval]    — node indices in topo order
  int *d_conn_ptr_ = nullptr;    // [Σ num_eval]    — start in conn arrays
  int *d_in_count_ = nullptr;    // [Σ num_eval]    — # incoming edges
  int *d_conn_from_ = nullptr;   // [Σ total_conn]  — source node index
  float *d_conn_w_ = nullptr;    // [Σ total_conn]  — edge weight
  int *d_out_indices_ = nullptr; // [Σ num_outputs] — output node positions

  // Topology capacity tracking (avoid reallocation when sizes fit)
  int capacity_node_vals_ = 0;
  int capacity_node_types_ = 0;
  int capacity_eval_order_ = 0;
  int capacity_conn_ptr_ = 0;
  int capacity_in_count_ = 0;
  int capacity_conn_from_ = 0;
  int capacity_conn_w_ = 0;
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
  int sensor_agent_entry_capacity_ = 0;
  int sensor_food_entry_capacity_ = 0;
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
  float world_width_ = 3000.0f; // Default world size (square)
  float world_height_ = 3000.0f;
  int num_inputs_;
  int num_outputs_;
  int activation_fn_id_ = 0; // 0=sigmoid, 1=tanh, 2=relu
  bool had_error_ = false;
  size_t scan_temp_storage_bytes_ = 0;
  size_t agent_scan_temp_bytes_ = 0;
  size_t food_scan_temp_bytes_ = 0;

  bool validate_copy_count(int count, int capacity, const char *label);
  void allocate_bin_arrays();
  void ensure_bin_capacity(int agent_count, int food_count);
};

// ── Free functions (implemented in neural_inference.cu / gpu_simulation.cu)
// ────
void batch_neural_inference(GpuBatch &batch); // launches on batch's stream
void batch_build_sensors(GpuBatch &batch, float world_width, float world_height,
                         float max_energy, bool has_walls);
void batch_rebuild_compact_bins(GpuBatch &batch);
bool init_cuda();
void print_device_info();

} // namespace moonai::gpu
