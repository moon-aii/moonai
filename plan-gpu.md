# GPU Integration Plan for MoonAI - Final

**Status:** Final Design Approved  
**Target:** Reconnect existing GPU infrastructure to post-ECS-migration architecture  
**Last Updated:** 2026-03-27

---

## Executive Summary

**Context:** CUDA infrastructure already exists in `src/gpu/` and was previously working. A major ECS migration broke the integration between GPU kernels and the simulation loop. This plan reconnects the existing GPU code.

**Scope:**
- ✅ Reconnect existing GPU kernels (`kernel_build_sensors`, `kernel_apply_movement`, `kernel_process_combat`)
- ✅ Implement missing `kernel_neural_inference` for NEAT networks
- ✅ Keep food/death/spatial systems on CPU as requested
- ❌ No new CUDA setup required - infrastructure exists

**Key Challenge:** NEAT networks have variable topology (different node/connection counts per agent), requiring CSR (Compressed Sparse Row) format conversion for GPU.

**Architecture Decision:** SimulationManager owns the full GPU flow, delegates neural inference to EvolutionManager.

---

## Architecture Overview

### Component Responsibilities

```
Session (High-Level Orchestrator)
  └── step()
        ├── if (use_gpu_) simulation_.step_gpu_ecs(registry, dt)
        │       └── [GPU PATH - see below]
        └── else simulation_.step_ecs(registry, dt)
                └── [CPU PATH - existing code]
        
        └── evolution_.compute_actions_ecs(registry, actions) [CPU fallback only]

SimulationManager (GPU Flow Owner)
  ├── step_ecs(registry, dt)           [CPU - existing]
  │
  └── step_gpu_ecs(registry, dt)       [NEW - GPU orchestration]
        ├── Pack ECS → GPU buffers
        ├── Upload to GPU
        ├── Launch sensors kernel
        ├── Delegate neural to EvolutionManager
        ├── Launch movement kernel  
        ├── Launch combat kernel
        ├── Download from GPU
        ├── Unpack GPU → ECS
        └── CPU systems: food, death

EvolutionManager (Neural Data Owner)
  ├── compute_actions_ecs()            [CPU - existing]
  │
  └── launch_gpu_neural(gpu_batch)     [NEW - GPU kernel launch]
        ├── Rebuild GPU cache if dirty
        └── Launch neural inference kernel

GpuBatchECS (GPU Buffer & Kernel Manager)
  ├── upload_async()
  ├── launch_build_sensors_async()
  ├── launch_apply_movement_async()
  ├── launch_process_combat_async()
  ├── download_async()
  └── synchronize()
```

### Data Flow

**GPU Path (per step):**

```
┌────────────────────────────────────────────────────────────────────┐
│  CPU: Pack ECS Data                                                │
│  Registry (SoA) → GpuDataBuffer (contiguous)                      │
│  O(n) copy: positions, velocities, energy, age, type, sensors     │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  GPU: Upload (H2D)                                                 │
│  cudaMemcpyAsync: buffer_ → device memory                         │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  GPU: Kernel Pipeline                                              │
│                                                                    │
│  1. kernel_build_sensors                                           │
│     Input:  positions, types, spatial data                        │
│     Output: sensor_inputs[15] per agent                           │
│                                                                    │
│  2. kernel_neural_inference (delegated to EvolutionManager)       │
│     Input:  sensor_inputs[15], CSR network data                   │
│     Output: brain_outputs[2] per agent                            │
│                                                                    │
│  3. kernel_apply_movement                                          │
│     Input:  brain_outputs[2], current positions/velocities        │
│     Output: updated positions, velocities, energy                 │
│                                                                    │
│  4. kernel_process_combat                                          │
│     Input:  positions, types                                      │
│     Output: updated alive status, energy                          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  GPU: Download (D2H)                                               │
│  cudaMemcpyAsync: device → buffer_                                │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  CPU: Unpack to ECS                                                │
│  GpuDataBuffer → Registry (SoA)                                   │
│  O(n) copy: positions, velocities, energy, alive                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  CPU: Remaining Systems                                            │
│  - process_food_ecs()       (prey eating)                         │
│  - process_step_deaths_ecs() (energy <= 0, max age)               │
│  - rebuild_spatial_grid_ecs() (spatial indexing)                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. CSR Format for Variable NEAT Networks

Since each agent has a different network topology, we pack all networks into flat CSR arrays:

```cpp
// Per-agent network descriptor
struct alignas(16) GpuNetDesc {
    int num_nodes;      // Total nodes (input + bias + hidden + output)
    int num_eval;       // Hidden + output nodes to evaluate
    int num_inputs;     // Input count (excluding bias)
    int num_outputs;    // Output count
    int node_off;       // Offset into d_node_values
    int eval_off;       // Offset into d_eval_order
    int conn_off;       // Offset into d_conn_from/weights
    int ptr_off;        // Offset into d_conn_ptr
    int out_off;        // Offset into d_out_indices
    int activation_fn;  // 0=sigmoid, 1=tanh, 2=relu
    int padding;        // Pad to 48 bytes for alignment
};

// GPU Network Cache (owned by EvolutionManager)
class GpuNetworkCache {
public:
    void build_from(const NetworkCache& cpu_cache, 
                    const std::vector<Entity>& entities);
    
    void launch_inference_async(const float* d_sensor_inputs,
                                 float* d_brain_outputs,
                                 std::size_t count,
                                 cudaStream_t stream);
    
    void invalidate() { dirty_ = true; }
    bool is_dirty() const { return dirty_; }
    
private:
    // Device arrays (flat, all agents packed)
    float* d_node_values_ = nullptr;
    int* d_eval_order_ = nullptr;
    int* d_conn_from_ = nullptr;
    float* d_conn_weights_ = nullptr;
    int* d_conn_ptr_ = nullptr;
    int* d_out_indices_ = nullptr;
    GpuNetDesc* d_descriptors_ = nullptr;
    
    // Host arrays (for building/uploading)
    std::vector<float> h_node_values_;
    std::vector<int> h_eval_order_;
    std::vector<int> h_conn_from_;
    std::vector<float> h_conn_weights_;
    std::vector<int> h_conn_ptr_;
    std::vector<int> h_out_indices_;
    std::vector<GpuNetDesc> h_descriptors_;
    
    // Entity mapping
    std::vector<Entity> entity_to_gpu_;
    
    bool dirty_ = true;
    std::size_t capacity_ = 0;
};
```

### 2. Neural Inference Kernel

```cuda
// One thread per agent
__global__ void kernel_neural_inference(
    const GpuNetDesc* __restrict__ descriptors,
    float* __restrict__ node_values,
    const int* __restrict__ eval_order,
    const int* __restrict__ conn_from,
    const float* __restrict__ conn_weights,
    const int* __restrict__ conn_ptr,
    const int* __restrict__ out_indices,
    const float* __restrict__ sensor_inputs,
    float* __restrict__ brain_outputs,
    int agent_count
) {
    const int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= agent_count) return;
    
    const GpuNetDesc& desc = descriptors[agent_idx];
    
    // Get agent's slice of arrays
    float* my_nodes = node_values + desc.node_off;
    const int* my_eval = eval_order + desc.eval_off;
    const int* my_conn_from = conn_from + desc.conn_off;
    const float* my_weights = conn_weights + desc.conn_off;
    const int* my_conn_ptr_local = conn_ptr + desc.ptr_off;
    const int* my_out = out_indices + desc.out_off;
    
    // 1. Initialize nodes
    for (int i = 0; i < desc.num_nodes; ++i) {
        my_nodes[i] = 0.0f;
    }
    
    // 2. Set sensor inputs
    const float* my_inputs = sensor_inputs + agent_idx * 15;
    for (int i = 0; i < desc.num_inputs; ++i) {
        my_nodes[i] = my_inputs[i];
    }
    
    // Set bias node
    my_nodes[desc.num_inputs] = 1.0f;
    
    // 3. Evaluate hidden/output nodes in topological order
    for (int e = 0; e < desc.num_eval; ++e) {
        int node_idx = my_eval[e];
        
        float sum = 0.0f;
        int start = my_conn_ptr_local[e];
        int end = my_conn_ptr_local[e + 1];
        
        for (int c = start; c < end; ++c) {
            int from_node = my_conn_from[c];
            float weight = my_weights[c];
            sum += my_nodes[from_node] * weight;
        }
        
        my_nodes[node_idx] = activate(sum, desc.activation_fn);
    }
    
    // 4. Extract outputs
    float* my_outputs = brain_outputs + agent_idx * 2;
    for (int i = 0; i < desc.num_outputs; ++i) {
        my_outputs[i] = my_nodes[my_out[i]];
    }
}

__device__ __forceinline__ float activate(float x, int fn) {
    if (fn == 0) {          // Sigmoid
        return 1.0f / (1.0f + expf(-4.9f * x));
    } else if (fn == 1) { // Tanh
        return tanhf(x);
    } else {                // ReLU
        return fmaxf(0.0f, x);
    }
}
```

### 3. SimulationManager GPU Step

```cpp
void SimulationManager::step_gpu_ecs(Registry& registry, float dt) {
    MOONAI_PROFILE_SCOPE("simulation_step_gpu");
    
    // 1. Prepare GPU buffers
    const std::size_t count = gpu::prepare_ecs_for_gpu(
        registry, 
        gpu_batch_->mapping(), 
        gpu_batch_->buffer()
    );
    
    if (count == 0) {
        return;
    }
    
    // 2. Upload to GPU
    gpu_batch_->upload_async(count);
    
    // 3. Launch GPU kernels
    GpuStepParams params;
    params.grid_size = config_.grid_size;
    params.dt = dt;
    params.energy_drain = config_.energy_drain_per_step;
    params.vision_range = config_.vision_range;
    // ... other params
    
    gpu_batch_->launch_build_sensors_async(params, count);
    
    // Delegate neural inference to EvolutionManager
    // EvolutionManager uses GpuBatchECS's buffers for input/output
    evolution_manager_->launch_gpu_neural(*gpu_batch_, count);
    
    gpu_batch_->launch_apply_movement_async(params, count);
    gpu_batch_->launch_process_combat_async(params, count);
    
    // 4. Download results
    gpu_batch_->download_async(count);
    gpu_batch_->synchronize();
    
    // 5. Unpack to ECS
    gpu::apply_gpu_results(
        gpu_batch_->buffer(), 
        gpu_batch_->mapping(), 
        registry
    );
    
    // 6. CPU-only systems work on updated data
    process_food_ecs(registry);
    process_step_deaths_ecs(registry);
    
    // Update alive counters
    count_alive_ecs(registry);
}
```

### 4. EvolutionManager Neural Launch

```cpp
void EvolutionManager::launch_gpu_neural(GpuBatchECS& gpu_batch, 
                                          std::size_t agent_count) {
    // Rebuild GPU cache if networks changed
    if (!gpu_network_cache_ || gpu_network_cache_->is_dirty()) {
        if (!gpu_network_cache_) {
            gpu_network_cache_ = std::make_unique<gpu::GpuNetworkCache>();
        }
        
        std::vector<Entity> living;
        living.reserve(entity_genomes_.size());
        for (const auto& [entity, _] : entity_genomes_) {
            living.push_back(entity);
        }
        
        gpu_network_cache_->build_from(network_cache_, living);
    }
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = (agent_count + block_size - 1) / block_size;
    
    kernel_neural_inference<<<num_blocks, block_size, 0, gpu_batch.stream()>>>(
        gpu_network_cache_->descriptors(),
        gpu_network_cache_->node_values(),
        gpu_network_cache_->eval_order(),
        gpu_network_cache_->conn_from(),
        gpu_network_cache_->conn_weights(),
        gpu_network_cache_->conn_ptr(),
        gpu_network_cache_->out_indices(),
        gpu_batch.buffer().device_sensor_inputs(),
        gpu_batch.buffer().device_brain_outputs(),
        static_cast<int>(agent_count)
    );
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw GpuException("Neural inference kernel failed: " + 
                          std::string(cudaGetErrorString(err)));
    }
}

// Invalidate GPU cache when networks change
void EvolutionManager::create_offspring_ecs(...) {
    // ... existing code ...
    if (gpu_network_cache_) {
        gpu_network_cache_->invalidate();
    }
}
```

### 5. Session Integration

```cpp
void Session::step(float dt) {
    MOONAI_PROFILE_SCOPE("session_step");
    
    try {
        if (use_gpu_) {
            simulation_.step_gpu_ecs(registry_, dt);
        } else {
            // CPU path: existing flow
            actions_buffer_.clear();
            evolution_.compute_actions_ecs(registry_, actions_buffer_);
            
            // Apply actions to entities
            apply_actions_to_entities(dt);
            
            simulation_.step_ecs(registry_, dt);
        }
        
        // Reproduction (CPU only)
        auto pairs = simulation_.find_reproduction_pairs_ecs(registry_);
        for (const auto& pair : pairs) {
            Entity child = evolution_.create_offspring_ecs(
                registry_, pair.parent_a, pair.parent_b, pair.spawn_position
            );
            if (child != INVALID_ENTITY) {
                simulation_.record_event(
                    SimEvent{SimEvent::Birth, child, child, 
                            pair.parent_a, pair.parent_b, pair.spawn_position}
                );
            }
        }
        
        // Refresh fitness
        evolution_.refresh_fitness_ecs(registry_);
        
        // Update species periodically
        if (should_update_species()) {
            evolution_.refresh_species_ecs(registry_);
        }
        
    } catch (const GpuException& e) {
        spdlog::error("GPU execution failed: {}. Aborting.", e.what());
        throw;  // Abort as requested
    }
    
    ++steps_executed_;
}
```

---

## Implementation Tasks

### Task 1: GpuNetworkCache Foundation (Day 1)

**Files:**
- `src/gpu/gpu_network_cache.hpp` (new)
- `src/gpu/gpu_network_cache.cu` (new)

**Work:**
1. Define `GpuNetDesc` structure
2. Implement `GpuNetworkCache` class
3. Implement `build_from()` to convert CPU networks to CSR format
4. Memory management (allocation/deallocation)

**Success Criteria:**
- Can convert multiple networks to GPU format
- Memory allocations succeed
- No memory leaks

### Task 2: Neural Inference Kernel (Day 2)

**File:**
- `src/gpu/gpu_network_cache.cu`

**Work:**
1. Implement `kernel_neural_inference`
2. Device-side activation functions
3. Kernel launch wrapper
4. Add to `GpuBatchECS` if needed for coordination

**Success Criteria:**
- Kernel compiles and launches
- Single network produces identical CPU vs GPU results
- All activation functions work

### Task 3: Network Cache Integration (Day 3)

**Files:**
- `src/evolution/network_cache.hpp` (modify)
- `src/evolution/network_cache.cpp` (modify)
- `src/evolution/evolution_manager.hpp` (modify)
- `src/evolution/evolution_manager.cpp` (modify)

**Work:**
1. Add `GpuNetworkCache` member to `EvolutionManager`
2. Implement `launch_gpu_neural()` method
3. Add cache invalidation hooks (`create_offspring_ecs`, `on_entity_destroyed`)
4. Add `enable_gpu()` method to initialize GPU cache

**Success Criteria:**
- Cache builds correctly for living entities
- Rebuilds only when dirty
- Integrates with `GpuBatchECS` buffers

### Task 4: SimulationManager GPU Step (Day 4)

**Files:**
- `src/simulation/simulation_manager.hpp` (modify)
- `src/simulation/simulation_manager.cpp` (modify)

**Work:**
1. Add `GpuBatchECS` member
2. Implement `step_gpu_ecs()` method
3. Integrate pack/upload/kernels/download/unpack flow
4. Wire up `launch_gpu_neural()` delegation
5. Keep CPU systems (food, death) after GPU step

**Success Criteria:**
- GPU step runs end-to-end
- Results applied to ECS correctly
- CPU systems see updated data

### Task 5: Session Integration (Day 5)

**Files:**
- `src/simulation/session.hpp` (modify)
- `src/simulation/session.cpp` (modify)

**Work:**
1. Modify `step()` to choose GPU vs CPU path
2. Pass `EvolutionManager` reference to `SimulationManager` for GPU coordination
3. Error handling (abort on GPU failure)
4. Configuration flag handling

**Success Criteria:**
- GPU path selected when `enable_gpu=true`
- Falls back/abort on error as configured
- Full simulation runs correctly

### Task 6: Testing & Validation (Day 6-7)

**Files:**
- `tests/test_gpu_neural.cpp` (new)
- `tests/test_gpu_integration.cpp` (new)

**Work:**
1. Unit tests: CPU vs GPU neural inference comparison
2. Integration test: Full GPU step vs CPU step
3. Performance benchmarks
4. Memory leak testing

**Success Criteria:**
- 100% test pass rate
- Numerical results within 1e-5 tolerance
- Performance improvement for >1000 agents
- No memory leaks

---

## Memory Budget

For 10K agents with avg 50 nodes, 100 connections:

| Component | Size |
|-----------|------|
| Node values | 2 MB |
| Connections | 8 MB |
| CSR pointers | 2 MB |
| Eval order | 1.9 MB |
| Output indices | 80 KB |
| Descriptors | 280 KB |
| **Network Total** | **~15 MB** |
| ECS Buffers | ~5 MB |
| **Grand Total** | **~20 MB** |

Very reasonable for modern GPUs.

---

## Performance Targets

| Population | CPU Time | GPU Target | Speedup |
|------------|----------|------------|---------|
| 1,000 | ~10 ms | ~5 ms | 2x |
| 5,000 | ~50 ms | ~15 ms | 3x |
| 10,000 | ~100 ms | ~25 ms | 4x |
| 20,000 | ~200 ms | ~50 ms | 4x |

---

## Error Handling

**Policy:** Abort on GPU failure (as requested)

```cpp
try {
    simulation_.step_gpu_ecs(registry_, dt);
} catch (const GpuException& e) {
    spdlog::error("GPU execution failed: {}", e.what());
    throw;  // Abort the simulation
}
```

**GPU failures include:**
- CUDA out of memory
- Kernel launch failure
- CUDA error during execution
- Numerical divergence (if detected)

---

## Files to Create/Modify

### New Files
```
src/gpu/
├── gpu_network_cache.hpp      # GpuNetDesc, GpuNetworkCache
└── gpu_network_cache.cu       # CSR conversion and kernel

tests/
├── test_gpu_neural.cpp        # Neural inference tests
└── test_gpu_integration.cpp   # Full GPU step tests
```

### Modified Files
```
src/gpu/
├── gpu_batch_ecs.hpp          # May need neural coordination interface
└── gpu_types.hpp              # Add GpuNetDesc

src/evolution/
├── network_cache.hpp          # Add GPU cache support
├── network_cache.cpp          # Integration hooks
├── evolution_manager.hpp      # Add launch_gpu_neural()
└── evolution_manager.cpp      # Implementation

src/simulation/
├── simulation_manager.hpp     # Add step_gpu_ecs()
├── simulation_manager.cpp     # Implementation
├── session.hpp                # GPU path selection
└── session.cpp                # Implementation
```

---

## Key Design Decisions Summary

1. **SimulationManager owns GPU flow** - Coordinates full pipeline, delegates neural to EvolutionManager
2. **Full GPU pipeline then download** - Minimize CPU-GPU transfers
3. **Food/death/spatial stay on CPU** - As requested
4. **EvolutionManager owns neural data** - Provides kernel launch service
5. **Abort on GPU failure** - As requested
6. **CSR format for variable networks** - Efficient GPU representation
7. **One-thread-per-agent** - Simple, sufficient for typical NEAT networks

---

## Timeline

- **Day 1-2:** GpuNetworkCache and CSR conversion
- **Day 3-4:** Neural kernel and integration
- **Day 5:** SimulationManager GPU step
- **Day 6:** Session integration
- **Day 7:** Testing and validation

**Total: 7 days**

---

**Ready for implementation. Review and approve to begin.**
