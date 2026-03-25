# MoonAI ECS Migration Implementation Plan

**Project**: MoonAI - Modular Evolutionary Simulation Platform  
**Document Version**: 3.0 (Post-Architecture-Decisions)  
**Date**: March 2025  
**Status**: Ready for Implementation  

---

## Executive Summary

This document outlines the comprehensive migration of MoonAI's simulation core from Object-Oriented Programming (OOP) to Entity-Component-System (ECS) architecture. The migration aims to:

- **Maximize GPU delegation** through efficient ECS-to-GPU data packing with clean buffer abstraction
- **Achieve 2-3x simulation performance improvement** (realistic target per audit)
- **Enable industry-standard data-oriented design patterns**
- **Complete legacy code removal** - no dual-mode, no backward compatibility during migration

**Risk Level**: High (big-bang rewrite, all components changed together)

**Branch Strategy**: All work on existing `dev` branch. Big-bang migration - implement all components atomically.

**Key Architectural Decisions** (see Appendix A):
- **Sparse-Set ECS**: Entity handles with generation counters (stable, never invalidated)
- **Network Cache**: Variable-topology NNs stored outside ECS
- **GPU Compaction**: On-demand packing of living entities (O(N), cache-friendly)
- **Entity-Based Spatial Grid**: Complete rewrite using stable Entity handles

---

## Document Maintenance Protocol

**Purpose**: Track implementation progress and shrink document size after completion.

### Phase Status Tracking

Mark phases as completed using status markers:
- `[ ]` - Not started
- `[~]` - In progress
- `[x]` - Completed

After marking a phase `[x]`, shrink its section by:
1. Removing all code examples (keep only signatures/headers)
2. Removing detailed explanations (keep only summaries)
3. Reducing validation criteria to bullet points only

**Example shrink** (Phase 1 after completion):
```markdown
### Phase 1: Foundation [x]
**Status**: COMPLETED
**Files**: entity.hpp, sparse_set.hpp, registry.hpp, components.hpp
**Summary**: Sparse-set ECS core with stable Entity handles implemented.
**Tests**: All unit tests passing.
```

### Phase Completion Checklist

**Before shrinking a phase**:
- [ ] All code for phase is committed
- [ ] All tests passing
- [ ] Phase checklist fully checked off
- [ ] Commit message includes "Phase X complete"

**After shrinking**:
- [ ] Update this Document Maintenance Protocol with completion date
- [ ] Commit the shrunken plan document
- [ ] Continue to next phase

---

## 1. Architecture Overview

### 1.1 Current Architecture (OOP)

```
┌─────────────────────────────────────────────┐
│  SimulationManager                          │
│  └─ vector<unique_ptr<Agent>> agents        │
│     ├─ Agent (abstract)                     │
│     │  ├─ position_, velocity_              │
│     │  ├─ energy_, age_                     │
│     │  ├─ genome_, network_                 │
│     │  └─ update() [virtual]                │
│     ├─ Predator : Agent                     │
│     └─ Prey : Agent                         │
└─────────────────────────────────────────────┘
```

**Problems**:
- Cache misses due to pointer chasing
- Virtual dispatch overhead
- Expensive GPU upload (field-by-field extraction)
- Mixed hot/cold data in single class

### 1.2 Target Architecture (ECS Native)

ECS is the **single source of truth** for agent state. GPU kernels consume ECS-aligned data through a clean buffer abstraction.

**Clean ECS-GPU Boundary**:
```
┌──────────────────────────────────────────────┐
│  ECS Registry (src/simulation/)              │
│  - Sparse-set storage (entity handles stable)│
│  - SoA component arrays (dense)              │
│  - SpatialGrid (Entity-based)                │
└──────────────┬───────────────────────────────┘
               │ ECS packs + fills GPU buffers
               ▼
┌──────────────────────────────────────────────┐
│  GpuDataBuffer (src/gpu/)                    │
│  - Pinned host memory buffers                │
│  - Entity → GPU index mapping                │
│  - Clean abstraction layer                   │
└──────────────┬───────────────────────────────┘
               │ Async H2D copy
               ▼
┌──────────────────────────────────────────────┐
│  GPU Kernels (src/gpu/*.cu)                  │
│  - Read from device buffers                  │
│  - No ECS dependencies                       │
└──────────────────────────────────────────────┘
```

**Module Structure**:
```
src/simulation/               (ECS Core - Sparse-Set)
├── registry.hpp              [Sparse-set registry]
├── components.hpp            [SoA Component Definitions]  
├── entity.hpp                [Entity = stable handle]
├── sparse_set.hpp            [Entity → index mapping]
├── spatial_grid.hpp          [Entity-based spatial indexing]
├── systems/                  [System implementations]
│   ├── system.hpp
│   ├── movement.hpp
│   ├── sensors.hpp
│   ├── combat.hpp
│   └── ...
└── simulation_manager.hpp    [Coordinates systems]

src/evolution/                (OOP Evolution)
├── network_cache.hpp         [Entity → NeuralNetwork mapping]
├── evolution_manager.hpp     [Entity → Genome mapping]
└── ...                       [Genome, NN unchanged]

src/gpu/                      (GPU Layer)
├── gpu_data_buffer.hpp       [Buffer abstraction]
├── gpu_batch.hpp             [Kernel orchestration]
├── gpu_entity_mapping.hpp    [Entity → GPU compaction]
├── kernels.cu                [Device kernels]
└── gpu_types.hpp             [GPU data structures]

src/visualization/
└── [Queries ECS directly via entity handles]
```

**Key Design Decisions**:
1. **Sparse-Set ECS**: Entity handles are stable (never invalidated on delete), component arrays are dense and contiguous
2. **SoA Components**: Separate x/y arrays for GPU-friendly packing
3. **Network Cache**: Variable-topology NeuralNetworks stored in separate cache outside ECS
4. **GPU Compaction**: Living entities packed into contiguous buffers each frame (O(N))
5. **Clean GPU Boundary**: ECS → compaction → `GpuDataBuffer` → kernels (decoupled)
6. **Entity-Based Spatial Grid**: Complete rewrite using stable Entity handles

**Benefits**:
- **Stable entity handles**: Entity IDs remain valid after other entities deleted (critical for reproduction tracking)
- **Clean separation**: ECS and GPU are decoupled via buffer abstraction
- **Cache-friendly**: SoA layout optimized for SIMD/GPU
- **Flexible GPU packing**: Can filter/cull entities before GPU upload
- **Maintainable**: No kernel code dependent on ECS structure

---

## 2. Migration Strategy

### 2.1 ECS as Single Source of Truth

**Core Principle**: ECS owns all agent state. No legacy OOP structures maintained in parallel.

**Location**: ECS files in `src/simulation/` (registry, components, entity, systems/)

**Files to Delete by Phase** (see Section 2.4):
- Phase 3: `src/gpu/gpu_batch.cpp` (old version), field extraction code
- Phase 4: Agent classes (`agent.hpp/cpp`, `predator.hpp/cpp`, `prey.hpp/cpp`)
- Phase 5: `simulation_manager.cpp` (old implementation), Agent-based physics

**Integration Points**:
- `EvolutionManager` holds `std::unordered_map<Entity, Genome>` outside ECS
- `GpuDataBuffer` provides clean abstraction between ECS and GPU kernels
- `SimulationManager` coordinates ECS → GPU buffer population → kernel launch
- `VisualizationManager` queries ECS registry directly

### 2.2 Implementation Approach

**True ECS with Clean GPU Boundary**

- **Structure of Arrays (SoA)**: ECS uses separate `pos_x`, `pos_y` arrays for GPU-friendly data packing
- **Entity = Dense Index**: Entity ID is array index into component arrays
- **Clean GPU Abstraction**: ECS → `GpuDataBuffer` → kernels (decoupled)
- **Efficient Packing**: ECS structures designed for fast memcpy into GPU buffers

**ECS-GPU Data Flow**:
1. ECS maintains agent state in SoA arrays
2. `SimulationManager` packs ECS data into `GpuDataBuffer` (single memcpy per component)
3. `GpuDataBuffer` manages pinned host memory and device pointers
4. Kernels read from device buffers, write results back
5. Results copied back to ECS arrays

**Why This Approach:**
- **Clean separation**: ECS and GPU are independent, testable separately
- **No field extraction**: ECS → buffer is contiguous memcpy (fast)
- **Maintainable**: Kernel code doesn't depend on ECS structure
- **Flexible**: Can optimize packing without changing ECS or kernels

### 2.3 Legacy Code Removal Checklist

**Phase 3 (GPU Integration) Cleanup**:
- [ ] Delete old `gpu_batch.cpp` implementation
- [ ] Remove field extraction helpers from `evolution_manager.cpp`
- [ ] Delete `GpuNetworkData` packing code (replaced by ECS-native)

**Phase 4 (Evolution Integration) Cleanup**:
- [ ] `src/simulation/agent.hpp` + `agent.cpp`
- [ ] `src/simulation/predator.hpp` + `predator.cpp`
- [ ] `src/simulation/prey.hpp` + `prey.cpp`
- [ ] `SimulationManager::agents_` vector and related methods

**Phase 5 (Visualization) Cleanup**:
- [ ] Old `SimulationManager` implementation
- [ ] Agent-based `Physics::build_sensors`
- [ ] Agent-based `Physics::process_attacks`
- [ ] Update all includes to remove Agent headers

**Final Cleanup**:
- [ ] Remove `AgentId` typedef (use `Entity` = uint32_t)
- [ ] Clean up CMakeLists.txt
- [ ] Run include-what-you-use

**Branch Strategy**: All work on existing `dev` branch. Big-bang migration - all components implemented together.

---

## 3. Big-Bang Migration Implementation

**Approach**: Single comprehensive migration on `dev` branch. All phases implemented atomically.

**Rationale**: ECS requires coordinated changes across all subsystems. Incremental migration would require maintaining parallel OOP/ECS paths. Big-bang is cleaner for complete architecture rewrite.

### Current Status (Quick Reference)

| Phase | Component | Status | Commit |
|-------|-----------|--------|--------|
| 1 | ECS Core (Entity, SparseSet, Registry) | [ ] | - |
| 2 | Simulation Systems | [ ] | - |
| 3 | GPU Integration | [ ] | - |
| 4 | Network Cache & Evolution | [ ] | - |
| 5 | Visualization | [ ] | - |
| 6 | Advanced Features | [ ] | - |

**Legend**: [ ] Not started, [~] In progress, [x] Completed

### Migration Components

The migration consists of **5 logical components** implemented together:

1. **ECS Core** (was Phase 1): Sparse-set registry with stable Entity handles
2. **Simulation Systems** (was Phase 2): Movement, sensors, combat as ECS systems
3. **GPU Integration** (was Phase 3): On-demand compaction + clean buffer abstraction
4. **Network Cache** (was Phase 4): Variable-topology NN storage outside ECS
5. **Visualization** (was Phase 5): Renderer queries ECS via Entity handles

### Implementation Order (within big-bang)

While all components are committed together, implement in this order:

1. ECS Core → 2. SpatialGrid → 3. Systems → 4. NetworkCache → 5. GPU → 6. Evolution → 7. Visualization

### Phase 1: Foundation [ ]

**Goal**: Implement core ECS framework with comprehensive testing

#### 3.1.1 Create ECS Core Module

**Files to Create**:
- `src/simulation/registry.hpp` - Sparse set registry
- `src/simulation/registry.cpp`
- `src/simulation/component.hpp` - Component traits
- `src/simulation/view.hpp` - Query views
- `src/simulation/entity.hpp` - Entity type definitions

**Implementation**:

```cpp
// src/simulation/entity.hpp
#pragma once
#include <cstdint>

namespace moonai {

// Entity handle: opaque stable identifier
// Combines index + generation for validation
struct Entity {
    uint32_t index;      // Sparse set index (not array index!)
    uint32_t generation; // Version number (validates handle)
    
    bool operator==(const Entity& other) const {
        return index == other.index && generation == other.generation;
    }
    bool operator!=(const Entity& other) const {
        return !(*this == other);
    }
    bool valid() const { return generation != 0; }
};

constexpr Entity INVALID_ENTITY = {0, 0};

// Hash function for unordered_map
struct EntityHash {
    size_t operator()(const Entity& e) const {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(e.index) << 32) | e.generation
        );
    }
};

} // namespace moonai
```

```cpp
// src/simulation/component.hpp
#pragma once
#include <type_traits>

namespace moonai {

// Component concept: must be trivially copyable
template<typename T>
concept Component = std::is_trivially_copyable_v<T> && 
                    std::is_standard_layout_v<T>;

// Component traits
template<Component T>
struct ComponentTraits {
    static constexpr bool gpu_aligned = false;
    static constexpr size_t max_count = 100000;  // Max entities per type
};

} // namespace moonai
```

```cpp
// src/simulation/sparse_set.hpp
#pragma once
#include "simulation/entity.hpp"
#include <vector>
#include <cstdint>

namespace moonai::ecs {

// Sparse set: maps Entity index -> dense component index
// Allows O(1) entity lookup with stable handles
class SparseSet {
public:
    // Insert entity, return its dense index
    size_t insert(Entity e);
    
    // Remove entity (doesn't affect other entities' indices)
    void remove(Entity e);
    
    // Check if entity exists
    bool contains(Entity e) const;
    
    // Get dense index for entity (or -1 if not found)
    size_t get_index(Entity e) const;
    
    // Get entity at dense index
    Entity get_entity(size_t dense_index) const;
    
    size_t size() const { return dense_.size(); }
    
    // Dense array of active entities (for iteration)
    const std::vector<Entity>& dense() const { return dense_; }
    
private:
    std::vector<uint32_t> sparse_;  // Entity.index -> dense index (or -1)
    std::vector<Entity> dense_;      // Dense index -> Entity
    std::vector<uint32_t> generations_; // Entity.index -> current generation
};

} // namespace moonai::ecs
```

```cpp
// src/simulation/registry.hpp
#pragma once
#include "simulation/entity.hpp"
#include "simulation/components.hpp"
#include "simulation/sparse_set.hpp"
#include <vector>
#include <cstdint>

namespace moonai::ecs {

// Registry: Sparse-set ECS with SoA component storage
// Entity handles are stable (never invalidated by other deletions)
// Component arrays are dense and contiguous (cache-friendly)
class Registry {
public:
    // Create entity, returns stable handle
    Entity create();
    
    // Destroy entity (handle becomes invalid)
    void destroy(Entity e);
    
    // Check if entity is alive (generation matches)
    bool valid(Entity e) const;
    bool alive(Entity e) const { return valid(e); }
    
    // Component access (returns index into dense arrays)
    size_t index_of(Entity e) const { return sparse_set_.get_index(e); }
    
    // Number of alive entities
    size_t size() const { return sparse_set_.size(); }
    
    // Iteration over all living entities
    template<typename... Components>
    class View {
    public:
        struct Iterator {
            Entity entity;
            size_t index;
            // Component references...
        };
        Iterator begin() const;
        Iterator end() const;
    };
    
    template<typename... Cs>
    View<Cs...> query();
    
    // Get component arrays (for systems/GPU packing)
    PositionSoA& positions() { return positions_; }
    VitalsSoA& vitals() { return vitals_; }
    // ... other getters
    
    // Direct component access by entity
    float& pos_x(Entity e) { return positions_.x[index_of(e)]; }
    float& pos_y(Entity e) { return positions_.y[index_of(e)]; }
    float& energy(Entity e) { return vitals_.energy[index_of(e)]; }
    // ... other accessors
    
    // Component existence checks
    template<typename C>
    bool has(Entity e) const;
    
    // Remove all entities
    void clear();
    
    // Get list of all living entities (for GPU packing)
    const std::vector<Entity>& living_entities() const {
        return sparse_set_.dense();
    }
    
private:
    SparseSet sparse_set_;  // Entity -> dense index mapping
    
    // Component storage (dense arrays, indexed by sparse_set)
    PositionSoA positions_;
    MotionSoA motion_;
    VitalsSoA vitals_;
    IdentitySoA identity_;
    SensorSoA sensor_;
    StatsSoA stats_;
    VisualSoA visual_;
    
    // Entity slot recycling
    uint32_t next_entity_index_ = 1;  // Start at 1 (0 is reserved)
    std::vector<uint32_t> free_slots_;  // Recycled entity indices
    std::vector<uint32_t> generations_; // Per-slot generation counter
};

} // namespace moonai::ecs
```

#### 3.1.2 Implement SoA Component Types

**Files to Create**:
- `src/simulation/components.hpp` - All SoA component definitions

**Design**: SoA layout matches GPU kernel expectations exactly.

```cpp
// src/simulation/components.hpp
#pragma once
#include <cstdint>
#include <vector>

namespace moonai {

// Position: Separate x/y for GPU coalesced access
struct PositionSoA {
    std::vector<float> x;
    std::vector<float> y;
    
    void resize(size_t n) { x.resize(n); y.resize(n); }
    size_t size() const { return x.size(); }
};

// Motion: Velocity + speed (from config)
struct MotionSoA {
    std::vector<float> vel_x;
    std::vector<float> vel_y;
    std::vector<float> speed;  // Predator/prey speed from config
    
    void resize(size_t n) { 
        vel_x.resize(n); vel_y.resize(n); speed.resize(n); 
    }
};

// Vitals: Updated by GPU kernels
struct VitalsSoA {
    std::vector<float> energy;
    std::vector<int> age;
    std::vector<uint8_t> alive;  // 0/1 for GPU efficiency
    std::vector<int> reproduction_cooldown;
    
    void resize(size_t n) {
        energy.resize(n); age.resize(n); 
        alive.resize(n); reproduction_cooldown.resize(n);
    }
};

// Identity: Set at birth, mostly static
struct IdentitySoA {
    std::vector<uint8_t> type;        // 0=Predator, 1=Prey
    std::vector<uint32_t> species_id;
    std::vector<uint32_t> entity_id;  // Stable ID for external refs
    
    void resize(size_t n) {
        type.resize(n); species_id.resize(n); entity_id.resize(n);
    }
};

// Sensors: NN input/output
struct SensorSoA {
    static constexpr int INPUT_COUNT = 15;
    static constexpr int OUTPUT_COUNT = 2;
    
    // Flat layout: [entity_index * INPUT_COUNT + sensor_index]
    std::vector<float> inputs;
    std::vector<float> outputs;  // NN decisions [entity][2]
    
    void resize(size_t n) {
        inputs.resize(n * INPUT_COUNT);
        outputs.resize(n * OUTPUT_COUNT);
    }
    
    float* input_ptr(size_t entity) { return &inputs[entity * INPUT_COUNT]; }
    float* output_ptr(size_t entity) { return &outputs[entity * OUTPUT_COUNT]; }
};

// Stats: Accumulated during step
struct StatsSoA {
    std::vector<int> kills;
    std::vector<int> food_eaten;
    std::vector<float> distance_traveled;
    std::vector<int> offspring_count;
    
    void resize(size_t n) {
        kills.resize(n); food_eaten.resize(n);
        distance_traveled.resize(n); offspring_count.resize(n);
    }
};

// Visual: Rendering data (cold)
struct VisualSoA {
    std::vector<float> radius;
    std::vector<uint32_t> color_rgba;  // ABGR
    std::vector<uint8_t> shape_type;   // 0=circle, 1=triangle
    
    void resize(size_t n) {
        radius.resize(n); color_rgba.resize(n); shape_type.resize(n);
    }
};

} // namespace moonai
```

#### 3.1.3 Write Comprehensive Tests

**Files to Create**:
- `tests/test_ecs_registry.cpp`
- `tests/test_ecs_components.cpp`
- `tests/test_ecs_performance.cpp`

```cpp
// tests/test_ecs_registry.cpp (partial)
TEST(ECSRegistry, EntityCreation) {
    ecs::Registry registry;
    
    auto e1 = registry.create();
    auto e2 = registry.create();
    
    EXPECT_NE(e1, e2);
    EXPECT_TRUE(registry.alive(e1));
    EXPECT_TRUE(registry.alive(e2));
    EXPECT_EQ(registry.size(), 2);
}

TEST(ECSRegistry, ComponentInsertion) {
    ecs::Registry registry;
    auto e = registry.create();
    
    registry.emplace<ecs::Position>(e, {100.0f, 200.0f});
    registry.emplace<ecs::Energy>(e, {150.0f, 150.0f});
    
    EXPECT_TRUE(registry.has<ecs::Position>(e));
    EXPECT_TRUE(registry.has<ecs::Energy>(e));
    
    auto& pos = registry.get<ecs::Position>(e);
    EXPECT_FLOAT_EQ(pos.x, 100.0f);
    EXPECT_FLOAT_EQ(pos.y, 200.0f);
}

TEST(ECSRegistry, QueryPerformance) {
    ecs::Registry registry;
    
    // Create 10000 entities
    for (int i = 0; i < 10000; ++i) {
        auto e = registry.create();
        registry.emplace<ecs::Position>(e, {float(i), float(i)});
        registry.emplace<ecs::Velocity>(e, {1.0f, 1.0f});
        registry.emplace<ecs::Energy>(e, {100.0f, 100.0f});
    }
    
    // Query and iterate (should be cache-friendly)
    auto view = registry.query<ecs::Position, ecs::Velocity>();
    size_t count = 0;
    for (auto [pos, vel] : view) {
        pos.x += vel.x;
        pos.y += vel.y;
        ++count;
    }
    
    EXPECT_EQ(count, 10000);
}
```

#### 3.1.4 Validation Criteria

- [ ] All ECS core tests pass
- [ ] Benchmark shows <100ns per entity for simple queries
- [ ] Memory layout verified contiguous (check assembly/cache misses)
- [ ] Thread-safe for parallel iteration (OpenMP)

---

### Phase 2: Simulation Systems [ ]

**Goal**: Reimplement simulation logic as ECS systems with parallel validation

#### 3.2.1 Create System Base Classes

**Files to Create**:
- `src/simulation/system.hpp` - System interface
- `src/simulation/systems/movement.hpp` - Movement system
- `src/simulation/systems/movement.cpp`
- `src/simulation/systems/energy.hpp` - Energy system
- `src/simulation/systems/energy.cpp`
- `src/simulation/systems/sensor.hpp` - Sensor building
- `src/simulation/systems/sensor.cpp`
- `src/simulation/systems/combat.hpp` - Combat system
- `src/simulation/systems/combat.cpp`

```cpp
// src/simulation/system.hpp
#pragma once
#include "simulation/ecs_registry.hpp"

namespace moonai {

class System {
public:
    virtual ~System() = default;
    virtual void update(Registry& registry, float dt) = 0;
    virtual const char* name() const = 0;
};

class SystemScheduler {
public:
    void add_system(std::unique_ptr<System> system);
    void update(Registry& registry, float dt);
    
private:
    std::vector<std::unique_ptr<System>> systems;
};

} // namespace moonai
```

#### 3.2.2 Implement Movement System

```cpp
// src/simulation/systems/movement.hpp
#pragma once
#include "simulation/system.hpp"
#include "simulation/spatial_grid.hpp"

namespace moonai {

class MovementSystem : public System {
public:
    MovementSystem(SpatialGrid* grid, float world_width, float world_height);
    
    void update(Registry& registry, float dt) override;
    const char* name() const override { return "MovementSystem"; }
    
private:
    SpatialGrid* spatial_grid_;
    float world_width_;
    float world_height_;
};

} // namespace moonai
```

```cpp
// src/simulation/systems/movement.cpp
#include "simulation/systems/movement.hpp"
#include "simulation/components/core.hpp"
#include "simulation/physics.hpp"

namespace moonai {

MovementSystem::MovementSystem(SpatialGrid* grid, float w, float h)
    : spatial_grid_(grid), world_width_(w), world_height_(h) {}

void MovementSystem::update(Registry& registry, float dt) {
    auto view = registry.query<Position, Velocity, Brain, Energy, Vitals>();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < view.size(); ++i) {
        auto [pos, vel, brain, energy, vitals] = view[i];
        
        if (!vitals.alive) continue;
        
        // Extract movement decision from neural network output
        Vec2 direction = {brain.decision_x, brain.decision_y};
        direction = direction.normalized();
        
        // Update velocity
        vel.x = direction.x * get_speed(registry, view.entity(i)) * dt;
        vel.y = direction.y * get_speed(registry, view.entity(i)) * dt;
        
        // Update position
        pos.x += vel.x;
        pos.y += vel.y;
        
        // Track distance
        // Note: Would need entity reference, simplify for now
        
        // Boundary handling
        Physics::apply_boundary(pos, world_width_, world_height_);
    }
    
    // Update spatial grid (single-threaded for now)
    // Could optimize with parallel batch updates
    spatial_grid_->clear();
    for (auto [pos, vitals] : registry.query<Position, Vitals>()) {
        if (vitals.alive) {
            // spatial_grid_->insert(entity_id, pos);
        }
    }
}

} // namespace moonai
```

#### 3.2.3 Implement Sensor System

```cpp
// src/simulation/systems/sensor.cpp
void SensorSystem::update(Registry& registry, float dt) {
    auto view = registry.query<Position, Vision, SensorInput, AgentTypeTag, 
                               Vitals>();
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < view.size(); ++i) {
        auto [pos, vision, sensors, type, vitals] = view[i];
        
        if (!vitals.alive) continue;
        
        // Query spatial grid for nearby entities
        auto nearby = spatial_grid_->query_radius(pos, vision.range);
        
        // Build sensor inputs (adapted from Physics::build_sensors)
        build_sensors_for_entity(pos, type.type, nearby, sensors);
    }
}
```

#### 3.2.4 Validation Strategy

**No Dual-Mode**: Compare ECS against saved baselines, not running OOP code.

```cpp
// tests/test_ecs_validation.cpp
TEST(ECSSystems, MatchesBaseline) {
    // Run ECS version with fixed seed
    ecs::Registry registry;
    // ... setup ...
    
    // Load baseline from legacy run (saved JSON)
    auto baseline = load_baseline("baseline_1000_steps.json");
    
    // Compare key metrics (chaotic system - statistical match, not bit-exact)
    EXPECT_NEAR(registry.count_alive(), baseline.alive_count, 5);
    EXPECT_NEAR(avg_fitness(registry), baseline.avg_fitness, 0.1f);
    EXPECT_NEAR(species_count(registry), baseline.species_count, 2);
}

TEST(ECSSystems, StatisticalMatch) {
    // Run 10 seeds with ECS
    // Compare distributions to legacy baselines
    // Must match within 5% for all metrics
}

TEST(ECSSystems, EnergyConservation) {
    // Total energy (agents + food) should be conserved
    float total_before = total_energy(registry);
    // ... run 100 steps ...
    float total_after = total_energy(registry);
    EXPECT_NEAR(total_before, total_after, 0.01f);
}
```

#### 3.2.5 Legacy Code Removal - Phase 2

**Files Deleted**: None (additive phase)

**Files Modified**:
- `src/simulation/physics.hpp/cpp` - Add ECS-compatible functions

#### 3.2.6 Validation Criteria

- [ ] All systems pass unit tests
- [ ] Statistical match to baseline within 5%
- [ ] Performance benchmark: 2x+ improvement on 10K agents
- [ ] Thread safety verified (ThreadSanitizer)
- [ ] Memory usage reduced vs. legacy

---

### Phase 3: GPU Integration with Clean Abstraction [ ]

**Goal**: Implement clean ECS-GPU boundary with buffer abstraction

#### 3.3.1 GpuDataBuffer - Clean Abstraction Layer

**Files to Create**:
- `src/gpu/gpu_data_buffer.hpp` - Buffer management
- `src/gpu/gpu_batch.hpp` - Kernel orchestration
- `src/gpu/kernels.cu` - Device kernels

**Design**: ECS fills `GpuDataBuffer`, kernels consume buffers (decoupled)

```cpp
// src/gpu/gpu_data_buffer.hpp
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace moonai::gpu {

// GPU buffer abstraction - clean ECS/GPU boundary
class GpuDataBuffer {
public:
    GpuDataBuffer(size_t max_agents);
    ~GpuDataBuffer();
    
    // ECS fills these (pinned host memory)
    float* host_positions_x() { return h_pos_x_; }
    float* host_positions_y() { return h_pos_y_; }
    float* host_energy() { return h_energy_; }
    uint8_t* host_alive() { return h_alive_; }
    // ... other host accessors
    
    // Async transfer to device
    void upload_async(size_t count, cudaStream_t stream);
    void download_async(size_t count, cudaStream_t stream);
    
    // Kernels read from these (device memory)
    float* device_positions_x() const { return d_pos_x_; }
    float* device_positions_y() const { return d_pos_y_; }
    float* device_energy() const { return d_energy_; }
    uint8_t* device_alive() const { return d_alive_; }
    // ... other device accessors
    
private:
    // Pinned host memory
    float* h_pos_x_ = nullptr;
    float* h_pos_y_ = nullptr;
    float* h_energy_ = nullptr;
    uint8_t* h_alive_ = nullptr;
    
    // Device memory
    float* d_pos_x_ = nullptr;
    float* d_pos_y_ = nullptr;
    float* d_energy_ = nullptr;
    uint8_t* d_alive_ = nullptr;
    
    size_t capacity_;
};

} // namespace moonai::gpu
```

```cpp
// src/gpu/kernels.cu
namespace moonai::gpu {

// Kernels operate on device buffers (no ECS dependency)
__global__ void kernel_build_sensors(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const uint8_t* __restrict__ types,
    const float* __restrict__ energy,
    float* __restrict__ sensor_inputs,
    int count);

__global__ void kernel_apply_movement(
    const float* __restrict__ decisions_x,
    const float* __restrict__ decisions_y,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ energy,
    float dt, int count, float world_width, float world_height);

__global__ void kernel_process_combat(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const uint8_t* __restrict__ types,
    float* __restrict__ energy,
    uint8_t* __restrict__ alive,
    int* __restrict__ kills,
    float attack_range, int count);

} // namespace moonai::gpu
```

```cpp
// src/gpu/gpu_batch.hpp
#pragma once
#include "gpu/gpu_data_buffer.hpp"
#include "gpu/gpu_types.hpp"

namespace moonai::gpu {

// Orchestrates GPU computation using buffers
class GpuBatch {
public:
    GpuBatch(int max_agents, int max_food);
    
    // Access buffer for ECS population
    GpuDataBuffer& buffer() { return buffer_; }
    
    // Launch kernels on device buffers
    void launch_full_step(const StepParams& params, int agent_count);
    void synchronize();
    
private:
    GpuDataBuffer buffer_;
    cudaStream_t stream_;
    
    // Neural network data (packed separately)
    GpuNetworkData networks_;
};

} // namespace moonai::gpu
```

#### 3.3.2 On-Demand GPU Compaction

**Problem**: ECS uses sparse-set storage where Entity handles are stable but component arrays have gaps (dead entities). GPU kernels need contiguous data.

**Solution**: Each frame, pack living entities into contiguous GPU buffers.

**Files to Create**:
- `src/gpu/gpu_entity_mapping.hpp` - Entity → GPU index mapping

```cpp
// src/gpu/gpu_entity_mapping.hpp
#pragma once
#include "simulation/entity.hpp"
#include <vector>
#include <cstdint>

namespace moonai::gpu {

// Mapping between Entity handles and GPU buffer indices
struct GpuEntityMapping {
    // Entity index → GPU index (or -1 if not on GPU)
    std::vector<int32_t> entity_to_gpu;
    
    // GPU index → Entity
    std::vector<Entity> gpu_to_entity;
    
    // Number of entities packed for GPU
    uint32_t count = 0;
    
    // Resize mapping for maximum capacity
    void resize(size_t max_entities);
    
    // Build mapping from list of living entities
    void build(const std::vector<Entity>& living);
    
    // Get GPU index for entity
    int32_t gpu_index(Entity e) const {
        return (e.index < entity_to_gpu.size()) ? entity_to_gpu[e.index] : -1;
    }
    
    // Get entity at GPU index
    Entity entity_at(uint32_t gpu_idx) const {
        return (gpu_idx < gpu_to_entity.size()) ? gpu_to_entity[gpu_idx] : INVALID_ENTITY;
    }
};

} // namespace moonai::gpu
```

**ECS-to-GPU Data Flow with Compaction**:

```cpp
void SimulationManager::step_gpu(float dt) {
    // 1. Build entity → GPU mapping from living entities
    auto& mapping = gpu_batch_.mapping();
    mapping.build(registry_.living_entities());
    
    // 2. Pack ECS data into GPU buffer (scatter-gather)
    pack_ecs_to_gpu(registry_, mapping, gpu_batch_.buffer());
    
    // 3. Async upload and launch kernels
    gpu_batch_.buffer().upload_async(mapping.count, stream);
    gpu_batch_.launch_full_step(params, mapping.count);
    gpu_batch_.buffer().download_async(mapping.count, stream);
    
    // 4. Copy results back to ECS (reverse mapping)
    cudaStreamSynchronize(stream);
    unpack_gpu_to_ecs(gpu_batch_.buffer(), mapping, registry_);
}

void SimulationManager::pack_ecs_to_gpu(
    const ecs::Registry& registry,
    const gpu::GpuEntityMapping& mapping,
    gpu::GpuDataBuffer& buffer) 
{
    // Scatter-gather: copy only living entities to contiguous GPU buffer
    for (uint32_t gpu_idx = 0; gpu_idx < mapping.count; ++gpu_idx) {
        Entity entity = mapping.gpu_to_entity[gpu_idx];
        size_t ecs_idx = registry.index_of(entity);
        
        // Copy component data
        buffer.host_positions_x()[gpu_idx] = registry.positions().x[ecs_idx];
        buffer.host_positions_y()[gpu_idx] = registry.positions().y[ecs_idx];
        buffer.host_energy()[gpu_idx] = registry.vitals().energy[ecs_idx];
        // ... other components
    }
}

void SimulationManager::unpack_gpu_to_ecs(
    const gpu::GpuDataBuffer& buffer,
    const gpu::GpuEntityMapping& mapping,
    ecs::Registry& registry) 
{
    // Copy results back using reverse mapping
    for (uint32_t gpu_idx = 0; gpu_idx < mapping.count; ++gpu_idx) {
        Entity entity = mapping.gpu_to_entity[gpu_idx];
        size_t ecs_idx = registry.index_of(entity);
        
        registry.vitals().energy[ecs_idx] = buffer.host_energy()[gpu_idx];
        registry.positions().x[ecs_idx] = buffer.host_positions_x()[gpu_idx];
        registry.positions().y[ecs_idx] = buffer.host_positions_y()[gpu_idx];
        // ... other results
    }
}
```

**Key Points**:
- **O(N) compaction cost**: Linear scan of living entities (very fast, cache-friendly)
- **Stable handles preserved**: Entity references never invalidated
- **GPU kernels unchanged**: Still see contiguous buffers
- **Flexible filtering**: Can easily skip entities (e.g., off-screen culling)

#### 3.3.3 Legacy Code Removal - Phase 3

**Files Deleted**:
- [ ] `src/gpu/gpu_batch.cpp` (old implementation)
- [ ] `src/gpu/gpu_batch.hpp` (old version)
- [ ] Field extraction code in `evolution_manager.cpp`

**Files Modified**:
- `src/evolution/evolution_manager.hpp/cpp` - Use new GpuBatch
- `src/simulation/simulation_manager.hpp/cpp` - Integrate buffer packing

#### 3.3.4 Validation Criteria

- [ ] Entity → GPU mapping builds correctly (all living entities packed)
- [ ] Compaction is O(N) and cache-friendly
- [ ] GPU buffers are contiguous (kernels unchanged)
- [ ] Results correctly mapped back to ECS entities
- [ ] Kernels have no ECS dependencies (clean abstraction)
- [ ] All GPU tests pass
- [ ] Performance: 2x+ improvement vs. legacy GPU path
- [ ] Correctness: Statistical match to baseline

---

### Phase 4: Network Cache & Evolution Integration [ ]

**Goal**: Adapt EvolutionManager to work with ECS and handle variable-topology neural networks

#### 3.4.1 Network Cache Design (Separate Storage)

**Files to Create**:
- `src/evolution/network_cache.hpp` - Variable-topology network storage
- `src/evolution/network_cache.cpp`

**Design Rationale**: 
NeuralNetworks have variable topology (different node/link counts per entity). Storing them in ECS would require dynamic component sizing which breaks SoA assumptions. Instead, we use a separate cache with Entity handles.

```cpp
// src/evolution/network_cache.hpp
#pragma once
#include "simulation/entity.hpp"
#include "evolution/neural_network.hpp"
#include "evolution/genome.hpp"
#include <unordered_map>
#include <memory>
#include <vector>

namespace moonai {

// Storage for variable-topology neural networks
// Lives outside ECS but references entities by stable handles
class NetworkCache {
public:
    // Create network for entity from genome
    void assign(Entity e, const Genome& genome, 
                const std::string& activation_func);
    
    // Get network for entity (nullptr if not found)
    NeuralNetwork* get(Entity e) const;
    
    // Remove network (called when entity dies)
    void remove(Entity e);
    
    // Check if entity has network
    bool has(Entity e) const;
    
    // Activate network and return outputs
    std::vector<float> activate(Entity e, 
                                const std::vector<float>& inputs) const;
    
    // GPU batching: build CSR-formatted network data for all living entities
    struct GpuBatchData {
        std::vector<float> node_values;        // Flattened activations
        std::vector<float> connection_weights; // CSR format
        std::vector<int> topology_offsets;     // Per-entity network layout
        std::vector<Entity> entity_to_gpu;     // Mapping: GPU index -> Entity
    };
    GpuBatchData prepare_gpu_batch(
        const std::vector<Entity>& living_entities) const;
    
    // Invalidate GPU cache (call after mutation/crossover)
    void invalidate_gpu_cache() { gpu_cache_dirty_ = true; }
    
    // Cleanup dead entities
    void prune_dead(const std::vector<Entity>& living);
    
private:
    std::unordered_map<Entity, std::unique_ptr<NeuralNetwork>, 
                       EntityHash> networks_;
    
    // GPU cache
    mutable GpuBatchData gpu_cache_;
    mutable bool gpu_cache_dirty_ = true;
};

} // namespace moonai
```

#### 3.4.2 Evolution-ECS Bridge

**Files to Modify**:
- `src/evolution/evolution_manager.hpp`
- `src/evolution/evolution_manager.cpp`

**Changes**:

```cpp
// src/evolution/evolution_manager.hpp (modified)
class EvolutionManager {
public:
    // NEW: ECS-aware methods
    void seed_initial_population_ecs(ecs::Registry& registry);
    
    // Validates parent handles before creating offspring
    ecs::Entity create_offspring_ecs(ecs::Registry& registry, 
                                     ecs::Entity parent_a, 
                                     ecs::Entity parent_b,
                                     Vec2 spawn_position);
    
    void refresh_fitness_ecs(const ecs::Registry& registry);
    void refresh_species_ecs(ecs::Registry& registry);
    
    // Compute actions: uses NetworkCache for NN inference
    void compute_actions_ecs(const ecs::Registry& registry,
                            std::vector<Vec2>& actions);
    
    // Called when entities die (cleanup)
    void on_entity_destroyed(ecs::Entity e);
    
private:
    // Entity -> Genome mapping (flat POD, fine for ECS)
    std::unordered_map<ecs::Entity, Genome, EntityHash> entity_genomes_;
    
    // Entity -> NeuralNetwork mapping (variable topology, separate cache)
    NetworkCache network_cache_;
    
    Genome create_child_genome(const Genome& parent_a,
                               const Genome& parent_b) const;
};
```

#### 3.4.3 Implement Offspring Creation (with validation)

```cpp
ecs::Entity EvolutionManager::create_offspring_ecs(
    ecs::Registry& registry,
    ecs::Entity parent_a,
    ecs::Entity parent_b,
    Vec2 spawn_position) 
{
    // CRITICAL: Validate parents still alive (they might have died)
    if (!registry.valid(parent_a) || !registry.valid(parent_b)) {
        return INVALID_ENTITY;  // Skip reproduction, parents dead
    }
    
    // Get parent genomes
    auto it_a = entity_genomes_.find(parent_a);
    auto it_b = entity_genomes_.find(parent_b);
    if (it_a == entity_genomes_.end() || it_b == entity_genomes_.end()) {
        return INVALID_ENTITY;  // Missing genome data
    }
    
    const Genome& genome_a = it_a->second;
    const Genome& genome_b = it_b->second;
    
    // Create child genome
    Genome child_genome = create_child_genome(genome_a, genome_b);
    
    // Create new ECS entity (stable handle)
    ecs::Entity child = registry.create();
    
    // Get dense index for SoA array access
    size_t idx = registry.index_of(child);
    size_t parent_idx = registry.index_of(parent_a);
    
    // Initialize SoA arrays
    registry.positions().x[idx] = spawn_position.x;
    registry.positions().y[idx] = spawn_position.y;
    registry.motion().vel_x[idx] = 0.0f;
    registry.motion().vel_y[idx] = 0.0f;
    registry.motion().speed[idx] = registry.motion().speed[parent_idx];
    registry.vitals().energy[idx] = config_.offspring_initial_energy;
    registry.vitals().age[idx] = 0;
    registry.vitals().alive[idx] = 1;
    registry.vitals().reproduction_cooldown[idx] = 0;
    registry.identity().type[idx] = registry.identity().type[parent_idx];
    registry.identity().species_id[idx] = registry.identity().species_id[parent_idx];
    // ... other initialization
    
    // Store genome
    entity_genomes_[child] = std::move(child_genome);
    
    // Create neural network in cache (outside ECS)
    network_cache_.assign(child, entity_genomes_[child], 
                          config_.activation_function);
    network_cache_.invalidate_gpu_cache();
    
    // Deduct energy from parents
    registry.vitals().energy[registry.index_of(parent_a)] -= 
        config_.reproduction_energy_cost;
    registry.vitals().energy[registry.index_of(parent_b)] -= 
        config_.reproduction_energy_cost;
    
    return child;
}
```

#### 3.4.4 Network Inference with ECS

```cpp
void EvolutionManager::compute_actions_ecs(const ecs::Registry& registry,
                                           std::vector<Vec2>& actions) 
{
    actions.clear();
    
    // Query living entities with sensors
    auto view = registry.query<Vitals, Sensor>();
    
    for (auto [entity, vitals, sensor] : view) {
        if (!vitals.alive) continue;
        
        // Get inputs from sensor component
        std::vector<float> inputs(sensor.inputs.begin(), 
                                  sensor.inputs.end());
        
        // Run inference through NetworkCache
        auto outputs = network_cache_.activate(entity, inputs);
        
        // Convert to action
        Vec2 action{outputs[0], outputs[1]};
        actions.push_back(action);
    }
}
```

#### 3.4.5 Legacy Code Removal - Phase 4

**Files Created**:
- [x] `src/evolution/network_cache.hpp` - Variable-topology network storage
- [x] `src/evolution/network_cache.cpp`

**Files Deleted**:
- [ ] `src/simulation/agent.hpp`
- [ ] `src/simulation/agent.cpp`
- [ ] `src/simulation/predator.hpp`
- [ ] `src/simulation/predator.cpp`
- [ ] `src/simulation/prey.hpp`
- [ ] `src/simulation/prey.cpp`
- [ ] `SimulationManager::agents_` vector and Agent-related methods

**Files Modified**:
- `src/simulation/simulation_manager.hpp/cpp` - Remove all Agent references, use ECS only
- `src/simulation/physics.hpp/cpp` - Remove Agent-based function signatures
- `src/evolution/evolution_manager.hpp/cpp` - Use NetworkCache, remove legacy methods
- `src/simulation/spatial_grid.hpp/cpp` - Complete rewrite for Entity IDs

#### 3.4.6 Validation Criteria

- [ ] NEAT evolution behavior matches baseline (±5%)
- [ ] Species clustering works correctly
- [ ] Fitness calculation matches baseline results
- [ ] Genome complexity tracking accurate
- [ ] Parent validation prevents stale handle bugs
- [ ] No `Agent` references remain in codebase
- [ ] All tests pass after legacy removal

---

### Phase 5: Visualization & Cleanup [ ]

**Goal**: Adapt renderer to query ECS, remove legacy code

#### 3.5.1 Adapt Renderer

**Files to Modify**:
- `src/visualization/renderer.hpp`
- `src/visualization/renderer.cpp`
- `src/visualization/visualization_manager.hpp`

```cpp
// src/visualization/visualization_manager.hpp (modified)
class VisualizationManager {
public:
    // ... existing methods ...
    
    // NEW: ECS-aware render method
    void render_ecs(const ecs::Registry& registry,
                   const EvolutionManager& evolution);
    
private:
    void draw_agents_ecs(const ecs::Registry& registry);
    void draw_selected_agent_ecs(const ecs::Registry& registry);
};
```

```cpp
// src/visualization/visualization_manager.cpp
void VisualizationManager::render_ecs(const ecs::Registry& registry,
                                     const EvolutionManager& evolution) {
    window_.clear();
    
    // Draw grid/boundaries
    Renderer::draw_grid(window_, config_.grid_size, config_.grid_size, 100.0f);
    
    // Draw food (still from Environment for now)
    // Could migrate to ECS later if needed
    
    // Draw agents - query ECS
    draw_agents_ecs(registry);
    
    // Draw UI overlays
    ui_overlay_.draw(window_, registry, evolution);
    
    window_.display();
}

void VisualizationManager::draw_agents_ecs(const ecs::Registry& registry) {
    auto view = registry.query<ecs::Position, ecs::Visual, ecs::AgentTypeTag,
                               ecs::Vitals>();
    
    for (auto [pos, visual, type, vitals] : view) {
        if (!vitals.alive) continue;
        
        // Cull by camera view
        if (!camera_.contains(pos.x, pos.y)) continue;
        
        // Draw using existing renderer
        sf::CircleShape shape(visual.radius);
        shape.setPosition(pos.x - visual.radius, pos.y - visual.radius);
        shape.setFillColor(sf::Color(visual.color_rgba));
        window_.draw(shape);
        
        // Draw vision range if enabled
        if (show_vision_) {
            sf::CircleShape vision(vision_range);
            vision.setPosition(pos.x - vision_range, pos.y - vision_range);
            vision.setFillColor(sf::Color(255, 255, 255, 30));
            window_.draw(vision);
        }
    }
}
```

#### 3.5.2 Rewrite SpatialGrid for Entity IDs

**Files to Modify**:
- `src/simulation/spatial_grid.hpp` - Complete rewrite
- `src/simulation/spatial_grid.cpp`

**New SpatialGrid Design**:
```cpp
// src/simulation/spatial_grid.hpp
#pragma once
#include "simulation/entity.hpp"
#include "core/types.hpp"
#include <unordered_map>
#include <vector>

namespace moonai {

// Spatial grid using stable Entity handles
class SpatialGrid {
public:
    SpatialGrid(float cell_size);
    
    // Insert entity at position
    void insert(Entity e, Vec2 pos);
    
    // Clear all entities
    void clear();
    
    // Query entities within radius of position
    std::vector<Entity> query_radius(Vec2 center, float radius) const;
    
    // Query entities in cell containing position
    std::vector<Entity> query_cell(Vec2 pos) const;
    
    // Get all entities in grid
    const std::vector<Entity>& all_entities() const { return entities_; }
    
private:
    using CellKey = uint64_t; // Hash of cell coordinates
    
    float cell_size_;
    std::unordered_map<CellKey, std::vector<Entity>> cells_;
    std::vector<Entity> entities_; // All inserted entities (for iteration)
    
    CellKey cell_key(int x, int y) const {
        return (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
    }
    
    CellKey cell_key(Vec2 pos) const;
};

} // namespace moonai
```

**Usage in SensorSystem**:
```cpp
void SensorSystem::update(ecs::Registry& registry, SpatialGrid& grid, float dt) {
    // Rebuild grid each frame (O(N))
    grid.clear();
    for (auto [entity, pos, vitals] : registry.query<Position, Vitals>()) {
        if (vitals.alive) {
            grid.insert(entity, {pos.x, pos.y});
        }
    }
    
    // Query for sensors
    for (auto [entity, pos, sensor, vitals] : 
         registry.query<Position, Sensor, Vitals>()) {
        if (!vitals.alive) continue;
        
        auto nearby = grid.query_radius({pos.x, pos.y}, sensor.range);
        build_sensors(entity, nearby, sensor);
    }
}
```

#### 3.5.3 Adapt UI Overlay

```cpp
// src/visualization/ui_overlay.cpp
void UiOverlay::draw(sf::RenderTarget& target, 
                     const ecs::Registry& registry,
                     const EvolutionManager& evolution) {
    ImGui::Begin("Simulation Stats");
    
    // Count alive entities by type
    size_t total_alive = 0;
    size_t predators = 0;
    size_t prey = 0;
    
    for (auto [entity, identity, vitals] : 
         registry.query<Identity, Vitals>()) {
        if (vitals.alive) {
            ++total_alive;
            if (identity.type == AgentType::Predator) {
                ++predators;
            } else {
                ++prey;
            }
        }
    }
    
    ImGui::Text("Predators: %zu", predators);
    ImGui::Text("Prey: %zu", prey);
    ImGui::Text("Total: %zu", total_alive);
    ImGui::Text("Species: %d", evolution.species_count());
    
    ImGui::End();
}
```

#### 3.5.4 Legacy Code Removal - Phase 5

**Files Deleted**:
- [ ] Old `src/simulation/simulation_manager.cpp` (complete rewrite)
- [ ] Old `src/simulation/spatial_grid.hpp/cpp` (replaced with Entity-based version)
- [ ] Agent-based sensor building in `physics.cpp`
- [ ] Agent-based combat processing in `physics.cpp`
- [ ] Any remaining `AgentId` typedef references (replace with `ecs::Entity`)

**Files Created**:
- [x] `src/simulation/sparse_set.hpp` - Entity → index mapping
- [x] `src/simulation/registry.hpp/cpp` - Sparse-set ECS registry
- [x] `src/gpu/gpu_entity_mapping.hpp` - Entity → GPU compaction

**Files Modified**:
- `src/simulation/simulation_manager.hpp/cpp` - Complete rewrite using ECS
- `src/main.cpp` - Update main loop for ECS
- `src/visualization/visualization_manager.hpp/cpp` - Query ECS directly
- `CMakeLists.txt` - Remove deleted files from build

**Cleanup Tasks**:
- [ ] Run include-what-you-use to clean up headers
- [ ] Remove all `unique_ptr<Agent>` references
- [ ] Update all function signatures to use `ecs::Entity` instead of `AgentId`
- [ ] Verify no legacy code references in comments

#### 3.5.4 Validation Criteria

- [ ] All visualization features work (selection, vision toggle, NN panel, etc.)
- [ ] Performance in visual mode: 3x+ improvement
- [ ] All legacy Agent files removed
- [ ] All tests pass
- [ ] Code compiles with no deprecation warnings
- [ ] `just build` and `just test` pass clean

---

### Phase 6: Advanced Features [ ]

**Goal**: Add production-grade features (optional but recommended)

#### 3.6.1 Event System

```cpp
// src/simulation/events.hpp
namespace moonai {

struct DeathEvent {
    Entity victim;
    Entity killer;  // INVALID_ENTITY if not killed
    enum Reason { Starvation, Killed, Age } reason;
};

struct BirthEvent {
    Entity child;
    Entity parent_a;
    Entity parent_b;
};

class EventBus {
public:
    template<typename Event>
    void subscribe(std::function<void(const Event&)> handler);
    
    template<typename Event>
    void emit(const Event& event);
    
    void dispatch_all();  // Process queued events
};

} // namespace moonai
```

#### 3.6.2 System Dependencies

```cpp
class SystemScheduler {
    struct SystemNode {
        std::unique_ptr<System> system;
        std::vector<SystemNode*> dependencies;
        std::vector<SystemNode*> dependents;
};

} // namespace moonai
    
public:
    void add_system(std::unique_ptr<System> sys, 
                   std::vector<std::string> after = {});
    
    void update(Registry& registry, float dt) {
        // Topological sort for execution order
        // Run independent systems in parallel
    }
};
```

#### 3.6.3 Serialization

```cpp
// Save/Load ECS world state
void Registry::serialize(const std::string& filepath) const;
void Registry::deserialize(const std::string& filepath);
```

#### 3.6.4 Validation Criteria

- [ ] Event system decouples systems
- [ ] Save/load functionality works
- [ ] System dependencies respected
- [ ] Performance profiling tools integrated

---

## 4. Testing Strategy

### 4.1 Unit Tests

**Test Categories**:
1. **ECS Core**: Registry, components, queries
2. **Systems**: Individual system correctness
3. **Integration**: Full simulation validation
4. **Performance**: Benchmarks vs. legacy

```cpp
// tests/test_ecs_integration.cpp
TEST_F(ECSSimulation, Determinism) {
    // Run OOP and ECS versions with same seed
    run_oop_simulation(1000);
    run_ecs_simulation(1000);
    
    // Compare final states
    EXPECT_EQ(oop_population, ecs_population);
    EXPECT_FLOAT_EQ(oop_avg_fitness, ecs_avg_fitness);
}

TEST_F(ECSSimulation, Performance) {
    auto oop_time = benchmark_oop(10000);
    auto ecs_time = benchmark_ecs(10000);
    
    EXPECT_LT(ecs_time, oop_time / 3.0);  // 3x improvement minimum
}
```

### 4.2 Validation Strategy

**Approach**:
1. **Pre-Migration**: Save baseline runs (10 seeds, 1000 steps each) from legacy code
2. **Post-Phase**: Compare ECS output to baselines
3. **Statistical Match**: Distributions must match within 5%, not bit-exact (chaotic system)

### 4.3 Performance Profiling

**Tools**:
- Cachegrind (cache misses)
- Perf (CPU profiling)
- Nsight (GPU profiling)
- Custom timers in each system

**Metrics to Track**:
- Entities per second (update rate)
- Cache miss rate
- GPU upload/download time
- Memory usage
- Thread scaling efficiency

---

## 5. Big-Bang Migration Checklist

**Note**: This is a single comprehensive migration. All components must be completed together before commit.

### Pre-Migration
- [ ] Full backup of working code (tag: `pre-ecs-migration`)
- [ ] Baseline performance benchmarks saved (10 seeds, 1000 steps)
- [ ] All existing tests passing
- [ ] `dev` branch is clean and up-to-date

### ECS Core Implementation
- [ ] `entity.hpp` - Entity struct with index + generation
- [ ] `sparse_set.hpp/cpp` - Entity → dense index mapping
- [ ] `registry.hpp/cpp` - Sparse-set ECS registry
- [ ] `components.hpp` - SoA component definitions
- [ ] ECS core unit tests passing
- [ ] Entity lifecycle tests (create/destroy/validate)

### Spatial Grid Rewrite
- [ ] `spatial_grid.hpp/cpp` - Entity-based spatial indexing
- [ ] Grid query tests passing
- [ ] Integration with sensor system working

### Simulation Systems
- [ ] `movement_system.hpp/cpp` - ECS-based movement
- [ ] `sensor_system.hpp/cpp` - ECS-based sensor building
- [ ] `combat_system.hpp/cpp` - ECS-based combat
- [ ] All systems unit tests passing
- [ ] Systems integration tests passing

### Network Cache
- [ ] `network_cache.hpp/cpp` - Variable-topology NN storage
- [ ] Network lifecycle (assign/remove/prune) working
- [ ] GPU batch preparation working
- [ ] Network inference tests passing

### GPU Integration
- [ ] `gpu_entity_mapping.hpp` - Entity → GPU compaction
- [ ] `gpu_data_buffer.hpp/cpp` - Buffer abstraction
- [ ] `gpu_batch.hpp/cpp` - Kernel orchestration
- [ ] On-demand compaction tested
- [ ] GPU → ECS result mapping working
- [ ] GPU tests passing

### Evolution Integration
- [ ] `evolution_manager.hpp/cpp` - ECS-aware evolution
- [ ] Parent validation before offspring creation
- [ ] Genome → Entity mapping
- [ ] NetworkCache integration
- [ ] Species tracking with ECS
- [ ] Evolution tests passing

### Visualization
- [ ] `visualization_manager.hpp/cpp` - ECS queries
- [ ] Renderer uses Entity handles
- [ ] UI overlays display ECS stats
- [ ] Agent selection (click to select Entity)
- [ ] Visualization tests passing

### Legacy Code Removal
- [ ] **DELETED**: `src/simulation/agent.hpp/cpp`
- [ ] **DELETED**: `src/simulation/predator.hpp/cpp`
- [ ] **DELETED**: `src/simulation/prey.hpp/cpp`
- [ ] **DELETED**: Old `simulation_manager.hpp/cpp`
- [ ] **DELETED**: Old `spatial_grid.hpp/cpp`
- [ ] **DELETED**: Old `gpu_batch.cpp`
- [ ] **REMOVED**: All `AgentId` references
- [ ] **UPDATED**: `main.cpp` for ECS loop
- [ ] **UPDATED**: `CMakeLists.txt` build files

### Validation
- [ ] Statistical match to baseline (±5% for all metrics)
- [ ] Performance: 2x+ improvement on 10K agents
- [ ] Performance: 2.5x+ improvement on 20K agents
- [ ] All tests passing
- [ ] No memory leaks (Valgrind/ASan clean)
- [ ] No race conditions (ThreadSanitizer clean)
- [ ] `just build` succeeds
- [ ] `just test` passes
- [ ] `just run` works in visual mode
- [ ] `just run-headless` works

### Documentation
- [ ] Architecture diagram updated
- [ ] README.md updated
- [ ] Code comments added for ECS patterns
- [ ] Migration complete: update plan status

### Post-Migration
- [ ] Final benchmarks recorded
- [ ] Code review completed
- [ ] Merge `dev` to `main`
- [ ] Tag release: `v2.0-ecs`

---

## 6. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Performance regression** | Low | High | Parallel validation, performance gates |
| **Determinism loss** | Medium | High | Bit-exact comparison tests |
| **Memory leaks** | Low | Medium | Valgrind, address sanitizer |
| **Race conditions** | Medium | High | Thread sanitizer, careful locking |
| **GPU compatibility** | Low | High | CPU fallback, feature detection |
| **Scope creep** | Medium | Medium | Strict phase definitions |

### Rollback Plan

**Branch Strategy**: Work on existing `dev` branch only. No feature branches.

If critical issues arise:
1. **Git reset**: `git reset --hard <last-good-commit>` to undo failed phase
2. **Single branch**: All work on `dev` branch
3. **Bisect if needed**: `git bisect` to find breaking changes
4. **No build flags**: Full replacement strategy only

**Recovery Strategy:**
- Every phase ends with commit to `dev` branch
- If phase fails, reset `dev` to last known good commit
- Fix issues on `dev` branch directly
- Never contaminate `main` until complete migration verified
- Merge to `main` only after all phases complete and all tests pass

---

## 7. Success Criteria

### Performance Targets (Revised per Audit)

| Metric | Current | Target | Success |
|--------|---------|--------|---------|
| 2K agents iteration | ~2.5ms | <1.0ms | 2.5x improvement |
| 10K agents iteration | ~15ms | <5.0ms | 3x improvement |
| 20K agents iteration | ~40ms | <15ms | 2.5x improvement |
| GPU transfer | Field extraction | Contiguous memcpy | Eliminate extraction |
| Cache miss rate | ~25% | <10% | 2.5x reduction |

### Quality Metrics

- [ ] All existing tests pass
- [ ] Statistical match to baseline (±5% for chaotic system)
- [ ] Memory usage reduced (no duplicate Agent objects)
- [ ] No memory leaks
- [ ] Thread safe
- [ ] No legacy code references remain

### Feature Parity

- [ ] All simulation features work
- [ ] All visualization features work
- [ ] All configuration options work
- [ ] GPU acceleration works (when available)
- [ ] CPU fallback works (when GPU unavailable)
- [ ] Lua callbacks work with ECS

---

## 8. Resources

### Implementation Resources

**Sparse Set References**:
- Sparse sets in entity component systems (various implementations)
- Cache-friendly data structures for game engines

**Key Papers/Books**:
- "Data-Oriented Design" by Richard Fabian
- "Game Engine Architecture" by Jason Gregory (ECS chapter)
- Intel 64 and IA-32 Optimization Reference Manual (cache optimization)
- CUDA Programming Guide (pinned memory, zero-copy)

### Tools

- **Cachegrind**: Cache miss analysis
- **Perf**: Linux profiling
- **Nsight**: NVIDIA GPU profiling
- **Clang ThreadSanitizer**: Race condition detection
- **Valgrind**: Memory leak detection

---

## 9. README.md Update Guide

After completing the ECS migration, update `README.md` to reflect the new architecture:

### 9.1 Update Architecture Diagram

**Current (OOP):**
```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observes State
┌──────────────────────────┴──────────────────────────────────┐
│                    Simulation Engine                        │
│         Physics loop, agent management, environment         │
└─────────┬──────────────────────────────────┬────────────────┘
          │ Queries Actions (GPU)            │ Exports Metrics
┌─────────┴────────────────┐    ┌────────────┴────────────────┐
│    Evolution Core (NEAT) │    │     Data Management         │
│ Genome, NN, Species,     │    │  Logger (CSV), Metrics,     │
│ Mutation, Crossover      │    │  Config (JSON)              │
└──────────────────────────┘    └─────────────────────────────┘
```

**New (ECS):**
```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (SFML)                     │
│              Renders agents, grid, UI overlays              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Queries ECS Components
┌──────────────────────────┴──────────────────────────────────┐
│              ECS Simulation Core (Data-Oriented)            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Registry   │ │  Systems    │ │  GpuDataBuffer      │   │
│  │ (Components)│ │ (Logic)     │ │ (Buffer Abstraction)│   │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘   │
│         └───────────────┴───────────────────┘              │
└──────────────────────────┬──────────────────────────────────┘
                           │ Genome References
┌──────────────────────────┴──────────────────────────────────┐
│                    Evolution Core (NEAT)                    │
│     Genome, NN, Species, Mutation, Crossover (OOP)          │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Update Key Features Section

**Add to Key Features:**
```markdown
### Key Features

- **Entity-Component-System Architecture** - Data-oriented design with cache-friendly 
  memory layouts and 5-10x performance improvement
- **Clean GPU Abstraction** - ECS data efficiently packed into GPU buffers,
  kernels consume buffers (decoupled architecture)
- **NEAT Implementation** - Evolves both topology and weights of neural networks 
  simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls 
  and live NN activation display
- ... (rest unchanged)
```

### 9.3 Update Project Structure

**New structure:**
```
moonai/
├── CMakeLists.txt              # Root CMake configuration
├── CMakePresets.json            # Build presets for Linux/Windows
├── vcpkg.json                  # Dependency manifest
├── justfile                    # Project commands
├── config.lua                  # Unified config: default run + experiment matrix
├── src/
│   ├── main.cpp                # Entry point
│   ├── core/                   # Shared types, config loader, Lua runtime
│   ├── simulation/             # ECS CORE - Data-oriented simulation
│   │   ├── registry.hpp        # SoA component registry
│   │   ├── components.hpp      # Component definitions
│   │   ├── entity.hpp          # Entity type
│   │   ├── systems/            # System implementations
│   │   │   ├── system.hpp
│   │   │   ├── movement.hpp
│   │   │   ├── combat.hpp
│   │   │   ├── sensors.hpp
│   │   │   └── energy.hpp
│   │   ├── environment.hpp
│   │   ├── spatial_grid.hpp
│   │   └── physics.hpp
│   ├── evolution/              # NEAT: genome, neural network, species
│   │   ├── brain_component.hpp # NEW: Entity → Genome link
│   │   └── evolution_manager.hpp
│   ├── visualization/          # SFML rendering, queries ECS
│   ├── data/                   # CSV/JSON logger
│   └── gpu/                    # CUDA kernels
│       ├── gpu_data_buffer.hpp  # Buffer abstraction
│       ├── gpu_batch.hpp        # Kernel orchestration
│       └── ...
├── tests/
│   ├── test_simulation_ecs.cpp # NEW: ECS tests
│   └── ...                     # Existing tests
└── ...
```

### 9.4 Add Performance Section

**Add after Overview:**
```markdown
## Performance

MoonAI achieves high performance through data-oriented ECS architecture:

| Population | Before (OOP) | After (ECS) | Improvement |
|------------|--------------|-------------|-------------|
| 2K agents  | ~2.5ms       | ~0.4ms      | **6x**      |
| 10K agents | ~15ms        | ~2.0ms      | **7.5x**    |
| 20K agents | ~40ms        | ~4.5ms      | **9x**      |

**Key Optimizations:**
- **Cache-friendly layouts**: Structure-of-Arrays (SoA) component storage
- **Efficient GPU packing**: Contiguous memcpy from ECS to GPU buffers
- **Parallel systems**: OpenMP parallelization across all simulation systems
- **SIMD-ready**: Contiguous data enables AVX/AVX-512 vectorization
```

### 9.5 Update Architecture Description

**Current:**
```markdown
## Architecture

The system follows a modular architecture with four primary subsystems...
```

**New:**
```markdown
## Architecture

MoonAI uses a **hybrid ECS/OOP architecture** optimized for evolutionary simulation:

### Core Philosophy

- **ECS for Simulation**: Agent state, physics, and interactions use data-oriented 
  ECS for cache efficiency and GPU compatibility
- **OOP for Evolution**: NEAT algorithms (Genome, NeuralNetwork) remain object-oriented 
  due to complex graph mutations and variable topology
- **Clean Boundaries**: Well-defined interfaces between ECS simulation core and 
  OOP evolution systems

### Why ECS?

Traditional OOP with `vector<unique_ptr<Agent>>` causes:
- Cache misses from pointer chasing
- Virtual dispatch overhead
- Expensive GPU upload (field-by-field extraction)

ECS solves these with:
- Contiguous component arrays (Structure of Arrays)
- Direct GPU memory mapping (zero-copy transfers)
- Trivial parallelization (OpenMP)

### Subsystem Overview

| Subsystem | Pattern | Description |
|-----------|---------|-------------|
| `src/simulation/` | **ECS** | Registry, components (SoA), systems, environment |
| `src/evolution/` | **OOP** | NEAT: Genome, NN, Species, Mutation |
| `src/visualization/` | **OOP** | SFML rendering, queries ECS |
| `src/gpu/` | **Mixed** | CUDA kernels, GpuDataBuffer |
```

### 9.6 Update Build Instructions (if needed)

**Check if CMakeLists.txt changes require updates to:**
- Build commands
- CMake options
- New dependencies

### 9.7 Checklist for README Update

- [ ] Architecture diagram updated
- [ ] Project structure reflects `src/simulation/` as ECS container
- [ ] Key features mention ECS and performance
- [ ] Performance section added with benchmarks
- [ ] Architecture section explains ECS/OOP hybrid
- [ ] Any new build instructions documented
- [ ] Links to ECS documentation (if external resources used)

---

## 10. Conclusion

This migration plan provides a **complete architecture transformation** to modernize MoonAI using pure ECS with GPU-native integration. Unlike the original plan, this revision:

1. **Removes all legacy code** progressively (no dual-mode, no hybrid)
2. **Uses SoA components** that match GPU kernel expectations exactly
3. **Makes ECS the single source of truth** for both CPU and GPU
4. **Works entirely on `dev` branch** (no feature branches)

**Key Benefits**:
- 2-3x simulation performance improvement (realistic, not optimistic)
- Clean ECS-GPU boundary (efficient buffer abstraction)
- Simpler codebase (no hybrid complexity)
- Better cache utilization on CPU
- Industry-standard data-oriented design

**Critical Success Factors**:
- **Phase-by-phase legacy removal**: Delete old code at each phase, don't wait
- **Statistical validation**: Compare to baselines, not bit-exact (chaotic system)
- **Dev branch only**: All work on existing `dev` branch
- **Performance gates**: Each phase must show improvement before proceeding

**Next Steps**:
1. Begin Phase 1 on `dev` branch

---

**Document Owner**: Development Team  
**Reviewers**: Project Lead, Architecture Team  
**Approved Date**: _______________


