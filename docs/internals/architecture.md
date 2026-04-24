---
description: System Architecture.
---

# Architecture

MoonAI is a CUDA-first predator-prey simulation platform with NEAT-based neural evolution.
It uses ECS-style SoA data for simulation state and GPU kernels, and OOP-style graph/genetics logic for NEAT mutation, crossover, and speciation.

The diagrams below describe execution, dataflow, layouts, ECS, spatial indexing, NEAT, GPU execution, and module dependencies.

## 1. Runtime Architecture

Primary references: `src/main.cpp`, `src/app/app.cpp`, `src/core/app_state.hpp`, `src/simulation/batch.cu`, `src/evolution/evolution_manager.cpp`.

```mermaid
flowchart LR
  subgraph Host[CPU Host Runtime]
    Main[main.cpp CLI]
    App[App Orchestrator]
    Core[Core Config + State]
    Evo[Evolution Manager]
    Metrics[Metrics + Logger]
    Viz[Visualization]
  end

  subgraph Device[GPU Device Runtime]
    SimKernels[Simulation Kernels]
    InfKernel[Inference Kernel]
    Grid[Spatial Grid Buffers]
  end

  Main --> App
  App --> Core
  App --> Evo
  App --> Metrics
  App --> Viz
  App --> SimKernels
  Evo --> InfKernel
  SimKernels --> Grid
  SimKernels --> Metrics
  Evo --> Metrics
```

## 2. Execution and Dataflow

### 2.1 Program Entry and Experiment Selection

```mermaid
flowchart TB
  Start[Process Start]
  Parse[parse_main_args]
  Load[load_all_configs_lua]
  Route{Mode}
  List[list experiments]
  Validate[validate config]
  RunOne[run_experiment selected]
  RunAll[run_experiment for each config]
  End[Exit]

  Start --> Parse --> Load --> Route
  Route -->|list| List --> End
  Route -->|validate| Validate --> End
  Route -->|run single| RunOne --> End
  Route -->|run all| RunAll --> End
```

### 2.2 App Construction Path

Source: `src/app/app.cpp:33-72`.

```mermaid
flowchart TB
  Ctor[App constructor]
  Seed{"seed equals 0"}
  SetSeed[set seed from clock]
  InitState[create app state]
  Validate[validate config]
  SimInit[initialize food]
  EvoInit[initialize evolution]
  SeedPop[seed initial population]
  Species[refresh species]
  InfInit[initialize inference]
  MetricsInit[refresh metrics]
  LogInit[initialize logger]
  VizCheck{headless mode}
  VizInit[initialize visualization]
  Ready[run loop ready]

  Ctor --> Seed
  Seed -->|yes| SetSeed --> InitState
  Seed -->|no| InitState
  InitState --> Validate --> SimInit --> EvoInit --> SeedPop --> Species --> InfInit --> MetricsInit --> LogInit --> VizCheck
  VizCheck -->|true| Ready
  VizCheck -->|false| VizInit --> Ready
```

### 2.3 Main Loop

Source: `src/app/app.cpp:138-194`.

```mermaid
flowchart TB
  LoopStart[while max_steps not reached]
  Signal{signal stop?}
  HandleEvents[handle visualization events]
  Window{window close?}
  StepCount[compute steps to run]
  StepIter[iterate step count]
  StepCall[run simulation step]
  StepOk{step ok?}
  Report{"step is on report interval"}
  Record[record and log]
  Render[render frame]
  Flush[flush logger]
  End[return success or failure]

  LoopStart --> Signal
  Signal -->|yes| Flush
  Signal -->|no| HandleEvents
  HandleEvents --> Window
  Window -->|yes| Flush
  Window -->|no| StepCount --> StepIter --> StepCall --> StepOk
  StepOk -->|no| Flush
  StepOk -->|yes| Report
  Report -->|yes| Record --> StepIter
  Report -->|no| StepIter
  StepIter --> Render --> LoopStart
  Flush --> End
```

### 2.4 Five-Phase Step Pipeline

Sources: `src/app/app.cpp:74-98`, `src/simulation/simulation.cpp:159-221`, `src/evolution/evolution_manager.cpp:124-127`, `src/evolution/evolution_manager.cpp:368-373`.

```mermaid
sequenceDiagram
  participant App as App::step
  participant Sim as simulation
  participant Batch as simulation::Batch
  participant Evo as EvolutionManager
  participant GPU as CUDA Stream

  App->>Sim: prepare_step(state, config)
  Sim->>Batch: ensure_capacity
  Sim->>Batch: pack_state
  Batch->>GPU: upload_async
  Batch->>GPU: launch_build_sensors_async

  App->>Evo: run_inference(state)
  Evo->>GPU: launch_population_inference predators
  Evo->>GPU: launch_population_inference prey

  App->>Sim: resolve_step(state, config)
  Sim->>Batch: launch_post_inference_async
  Sim->>Batch: download_async
  Sim->>Batch: synchronize
  Sim->>Sim: apply_results
  Sim->>Sim: collect_step_events

  App->>Sim: post_step(state, config)
  Sim->>Sim: predator.compact
  Sim->>Sim: prey.compact
  Sim->>Sim: food.respawn_step

  App->>Evo: post_step(state)
  Evo->>Evo: reproduce_population predators
  Evo->>Evo: reproduce_population prey
```

## 3. Failure and Recovery Paths

### 3.1 Step Failure Propagation

Sources: `src/simulation/simulation.cpp:168-176`, `src/simulation/simulation.cpp:192-209`, `src/evolution/evolution_manager.cpp:391-410`, `src/app/app.cpp:167-187`.

```mermaid
flowchart TB
  StepCall[App step]
  Prepare[simulation prepare_step]
  Inf[evolution run_inference]
  Resolve[simulation resolve_step]
  Post[post_step phases]
  Success[step returns true]

  PrepErr[return false]
  InfErr[batch.mark_error and return false]
  ResErr[return false]
  LoopFail[App run sets failed=true]
  Stop[break loop and report Simulation step failed]

  StepCall --> Prepare
  Prepare -->|ok| Inf
  Prepare -->|error state or ensure_capacity fail| PrepErr --> LoopFail
  Inf -->|ok| Resolve
  Inf -->|launch failure| InfErr --> LoopFail
  Resolve -->|ok| Post --> Success
  Resolve -->|batch error before or after sync| ResErr --> LoopFail
  LoopFail --> Stop
```

## 4. Critical Data Layouts

### 4.1 AppState Composition

Source: `src/core/app_state.hpp:111-124`.

```mermaid
classDiagram
  class AppState {
    +UiState ui
    +AgentRegistry predator
    +AgentRegistry prey
    +Food food
    +MetricsSnapshot metrics
    +RuntimeState runtime
    +StepBuffers step_buffers
    +simulation::Batch batch
  }

  class UiState {
    +bool paused
    +bool step_requested
    +int speed_multiplier
    +uint32 selected_agent_id
  }

  class RuntimeState {
    +Random rng
    +uint32 next_agent_id
    +int step
  }

  AppState --> UiState
  AppState --> RuntimeState
```

### 4.2 AgentRegistry SoA Layout

Source: `src/core/app_state.hpp:38-67`.

```mermaid
flowchart TB
  subgraph EntityIndexSpace[Entity Index Space]
    E0[0]
    E1[1]
    EN[N]
  end

  subgraph Components[SoA Arrays]
    PX[pos_x]
    PY[pos_y]
    VX[vel_x]
    VY[vel_y]
    ENE[energy]
    AGE[age]
    ALV[alive]
    SID[species_id]
    AID[entity_id]
    GEN[generation]
  end

  E0 --> PX
  E0 --> PY
  E0 --> VX
  E0 --> VY
  E0 --> ENE
  E0 --> AGE
  E0 --> ALV
  E0 --> SID
  E0 --> AID
  E0 --> GEN
  E1 --> PX
  EN --> GEN
```

### 4.3 Registry API Semantics

Sources: `src/core/app_state.cpp:50-56`.

```mermaid
flowchart TB
  Valid[valid entity check] --> Expr1[entity is not INVALID_ENTITY]
  Valid --> Expr2[entity is less than registry size]
  Size[size query] --> Expr3[returns position array size]
```

### 4.4 Host and Device Buffers

Source: `src/simulation/buffers.hpp`.

```mermaid
flowchart LR
  subgraph HostPinned[Host Pinned Buffers]
    HPX[h_pos_x]
    HPY[h_pos_y]
    HVX[h_vel_x]
    HVY[h_vel_y]
    HE[h_energy]
    HA[h_age]
    HAL[h_alive]
    HK[h_kill_counts]
    HC[h_claimed_by]
    HO[h_brain_outputs]
  end

  subgraph DeviceBuffers[Device Buffers]
    DPX[d_pos_x]
    DPY[d_pos_y]
    DVX[d_vel_x]
    DVY[d_vel_y]
    DE[d_energy]
    DA[d_age]
    DAL[d_alive]
    DK[d_kill_counts]
    DC[d_claimed_by]
    DS[d_sensor_inputs]
    DO[d_brain_outputs]
  end

  HPX --> DPX
  HPY --> DPY
  HVX --> DVX
  HVY --> DVY
  HE --> DE
  HA --> DA
  HAL --> DAL
  DK --> HK
  DC --> HC
  DO --> HO
```

### 4.5 Spatial Entry Layout

Source: `src/simulation/layout.hpp:5-17`.

```mermaid
classDiagram
  class PopulationEntry {
    +unsigned id
    +float pos_x
    +float pos_y
    +float padding
  }
  class FoodEntry {
    +unsigned id
    +float pos_x
    +float pos_y
    +float padding
  }
```

## 5. ECS Architecture

### 5.1 Lifecycle for Agents

```mermaid
stateDiagram-v2
  [*] --> Created: create()
  Created --> Alive: alive=1
  Alive --> Dead: energy <= 0 or age >= max_age
  Dead --> Removed: compact()
  Removed --> [*]
```

### 5.2 Food Lifecycle

```mermaid
stateDiagram-v2
  [*] --> Active
  Active --> Inactive: consumed by prey
  Inactive --> Active: respawn_step()
```

### 5.3 Compaction Procedure

Source: `src/core/app_state.cpp:62-98`.

```mermaid
flowchart TB
  Start[compact loop from first index]
  AliveCheck{"alive flag at index is not zero"}
  NextI[increment index]
  Last[last index is size minus one]
  SwapNeeded{"i is not last"}
  SwapComp[swap entities]
  MoveGenome[move genome from last to current]
  MoveNet[move network cache entry]
  SwapInf[swap remove inference cache entry]
  RemoveLastNet[remove last network cache entry]
  Pop[remove last SoA entries]
  Loop[continue while index is in range]

  Start --> AliveCheck
  AliveCheck -->|yes| NextI --> Loop --> AliveCheck
  AliveCheck -->|no| Last --> SwapNeeded
  SwapNeeded -->|yes| SwapComp --> MoveGenome --> MoveNet --> SwapInf --> RemoveLastNet --> Pop --> Loop
  SwapNeeded -->|no| SwapInf --> RemoveLastNet --> Pop --> Loop
```

### 5.4 ECS + Evolution Cache Consistency

```mermaid
sequenceDiagram
  participant ECS as AgentRegistry
  participant NC as NetworkCache
  participant IC as InferenceCache

  ECS->>ECS: compact removes dead slot
  ECS->>NC: move_entity(last, i)
  ECS->>IC: swap_remove_entity(i, last)
  ECS->>NC: remove(last)
```

## 6. Spatial Grid

### 6.1 Grid Resources

Source: `src/simulation/batch.hpp:91-109`.

```mermaid
flowchart TB
  subgraph PredatorGrid
    PCount[d_predator_cell_counts]
    POff[d_predator_cell_offsets]
    PWrite[d_predator_cell_write_offsets]
    PEntries[d_predator_grid_entries]
  end

  subgraph PreyGrid
    YCount[d_prey_cell_counts]
    YOff[d_prey_cell_offsets]
    YWrite[d_prey_cell_write_offsets]
    YEntries[d_prey_grid_entries]
  end

  subgraph FoodGrid
    FCount[d_food_cell_counts]
    FOff[d_food_cell_offsets]
    FWrite[d_food_cell_write_offsets]
    FEntries[d_food_grid_entries]
  end

  Meta[grid_cols, grid_rows, grid_cell_size]

  Meta --> PredatorGrid
  Meta --> PreyGrid
  Meta --> FoodGrid
```

### 6.2 Count-Scan-Scatter Build

Source: `src/simulation/batch.cu:131-192`, `src/simulation/batch.cu:671-776`.

```mermaid
flowchart LR
  Input[positions + alive]
  Count[kernel_count_*_cells_from_positions]
  Scan[thrust exclusive_scan]
  Scatter[kernel_scatter_*_cells_from_positions]
  Entries[cell entries ready]

  Input --> Count --> Scan --> Scatter --> Entries
```

### 6.3 Sensor Query Pipeline

Source: `src/simulation/batch.cu:194-325`.

```mermaid
flowchart TB
  Agent[one agent thread]
  BaseCell[compute base cell]
  Radius[compute cell search radius]
  NeighborLoop[iterate neighboring cells]
  Cull[cell intersects vision radius]
  ReadEntries[iterate entries in cell]
  TrackNearest[keep 5 nearest per target type]
  Encode[encode_nearest_targets]
  SelfFeatures[append self energy and velocity]
  WallFeatures[append wall sensors]
  Write[write 35 sensor values]

  Agent --> BaseCell --> Radius --> NeighborLoop --> Cull
  Cull -->|intersects| ReadEntries --> TrackNearest --> NeighborLoop
  Cull -->|skip| NeighborLoop
  NeighborLoop --> Encode --> SelfFeatures --> WallFeatures --> Write
```

### 6.4 Sensor Layout (35 Inputs)

Sources: `src/core/types.hpp:21`, `src/simulation/batch.cu:16-29`.

```mermaid
flowchart LR
  subgraph PredatorTargets[10 values]
    P1[predator 1 dx dy]
    P2[predator 2 dx dy]
    P3[predator 3 dx dy]
    P4[predator 4 dx dy]
    P5[predator 5 dx dy]
  end

  subgraph PreyTargets[10 values]
    Y1[prey 1 dx dy]
    Y2[prey 2 dx dy]
    Y3[prey 3 dx dy]
    Y4[prey 4 dx dy]
    Y5[prey 5 dx dy]
  end

  subgraph FoodTargets[10 values]
    F1[food 1 dx dy]
    F2[food 2 dx dy]
    F3[food 3 dx dy]
    F4[food 4 dx dy]
    F5[food 5 dx dy]
  end

  subgraph SelfState[3 values]
    E[self energy]
    VX[self vel x]
    VY[self vel y]
  end

  subgraph WallState[2 values]
    WX[wall x signed proximity]
    WY[wall y signed proximity]
  end
```

## 7. NEAT Evolution System

### 7.1 Genome Model

Source: `src/evolution/genome.hpp`.

```mermaid
classDiagram
  class Genome {
    +num_inputs
    +num_outputs
    +nodes
    +connections
    +add_node()
    +add_connection()
    +has_connection()
    +has_node()
    +compatibility_distance()
  }

  class NodeGene {
    +id
    +type
  }

  class ConnectionGene {
    +in_node
    +out_node
    +weight
    +enabled
    +innovation
  }

  class NodeType {
    Input
    Hidden
    Output
    Bias
  }

  Genome --> NodeGene
  Genome --> ConnectionGene
  NodeGene --> NodeType
```

### 7.2 Neural Compilation and Launch Path

Sources: `src/evolution/network_cache.hpp`, `src/evolution/inference_cache.hpp`, `src/evolution/evolution_manager.cpp:377-388`.

```mermaid
flowchart LR
  GenomeIn[Genome]
  BuildNN[NeuralNetwork construction]
  Compile[CompiledNetwork arrays]
  NetCache[NetworkCache assign]
  InfCache[InferenceCache prepare_for_launch]
  Launch[kernel_neural_inference]

  GenomeIn --> BuildNN --> Compile --> NetCache --> InfCache --> Launch
```

### 7.3 Innovation Tracker

Source: `src/evolution/mutation.cpp:30-55`.

```mermaid
flowchart TB
  Pair[node pair in_node out_node]
  HasInnov{innovation exists}
  ReturnInnov[return existing innovation]
  NewInnov[create innovation_counter++]
  SplitPair[split pair in_node out_node]
  HasSplit{split node id exists}
  ReturnSplit[return existing split node id]
  NewSplit[create next_node_id]

  Pair --> HasInnov
  HasInnov -->|yes| ReturnInnov
  HasInnov -->|no| NewInnov

  SplitPair --> HasSplit
  HasSplit -->|yes| ReturnSplit
  HasSplit -->|no| NewSplit
```

### 7.4 Mutation Pipeline

Source: `src/evolution/mutation.cpp:178-198`.

```mermaid
flowchart TB
  Start[Mutation::mutate]
  Weights{rng < mutation_rate}
  AddConn{rng < add_connection_rate}
  AddNode{rng < add_node_rate}
  DelConn{rng < delete_connection_rate}
  EnsureEnabled[if none enabled then enable random connection]
  End[return mutated genome]

  Start --> Weights --> AddConn --> AddNode --> DelConn --> EnsureEnabled --> End
```

### 7.5 Crossover Path

Source: `src/evolution/crossover.cpp:8-76`.

```mermaid
flowchart TB
  Parents[parent A and parent B]
  Index[map genes by innovation]
  Iterate[iterate all innovation ids]
  Match{gene in both parents}
  MatchPick[pick one parent gene]
  Disabled{either gene disabled}
  DisableRule[75 percent chance child gene disabled]
  Single{gene in one parent only}
  KeepSingle[50 percent chance keep]
  EnsureOne[if child empty choose one random gene]
  HiddenNodes[add required hidden nodes]
  Child[child genome]

  Parents --> Index --> Iterate --> Match
  Match -->|yes| MatchPick --> Disabled
  Disabled -->|yes| DisableRule --> Single
  Disabled -->|no| Single
  Match -->|no| Single
  Single --> KeepSingle --> Iterate
  Iterate --> EnsureOne --> HiddenNodes --> Child
```

### 7.6 Speciation Path

Sources: `src/evolution/species.hpp`, `src/evolution/evolution_manager.cpp:253-300`.

```mermaid
flowchart TB
  Clear[clear all species members]
  ForGenome[for each genome]
  FindCompat[check species representative compatibility]
  Compatible{"distance is at most threshold"}
  AddMember[add member to species]
  NewSpecies[create new species with representative]
  WriteId[write species_id for entity]
  Refresh[refresh species summaries]
  Prune[remove empty species]

  Formula[distance uses excess disjoint normalization and weight diff terms]

  Clear --> ForGenome --> FindCompat --> Compatible
  Compatible -->|yes| AddMember --> WriteId --> ForGenome
  Compatible -->|no| NewSpecies --> WriteId --> ForGenome
  ForGenome --> Refresh --> Prune
  Formula --> FindCompat
```

### 7.7 Reproduction Path

Source: `src/evolution/evolution_manager.cpp:310-365`.

```mermaid
flowchart TB
  BuildGrid[DenseReproductionGrid build]
  ForEntity[for each entity]
  EnergyCheck{energy >= reproduction threshold}
  UsedCheck{already used this step}
  CandidateSearch[search nearby candidates within mate_range]
  MateFound{best mate found}
  Offspring[create_offspring]
  ChildGenome[crossover and mutation]
  ChildCache[network_cache assign and inference_cache add_entity]
  EnergyCost[subtract reproduction energy cost from both parents]
  MarkUsed[mark both parents used]

  BuildGrid --> ForEntity --> EnergyCheck
  EnergyCheck -->|no| ForEntity
  EnergyCheck -->|yes| UsedCheck
  UsedCheck -->|yes| ForEntity
  UsedCheck -->|no| CandidateSearch --> MateFound
  MateFound -->|no| ForEntity
  MateFound -->|yes| Offspring --> ChildGenome --> ChildCache --> EnergyCost --> MarkUsed --> ForEntity
```

## 8. GPU Execution

### 8.1 Simulation Kernel Order

Source: `src/simulation/batch.cu:778-849`.

```mermaid
flowchart LR
  VPred[kernel_update_vitals predators]
  VPrey[kernel_update_vitals prey]
  ClaimFood[kernel_claim_food]
  FinalFood[kernel_finalize_food]
  ClaimCombat[kernel_claim_combat]
  FinalCombat[kernel_finalize_combat]
  ClampPred[kernel_clamp_energy predators]
  ClampPrey[kernel_clamp_energy prey]
  MovePred[kernel_apply_movement predators]
  MovePrey[kernel_apply_movement prey]

  VPred --> VPrey --> ClaimFood --> FinalFood --> ClaimCombat --> FinalCombat --> ClampPred --> ClampPrey --> MovePred --> MovePrey
```

### 8.2 Inference Kernel Dataflow

Sources: `src/evolution/inference_cache.cu:39-83`, `src/evolution/inference_cache.hpp:19-31`.

```mermaid
flowchart TB
  Slot[network slot thread]
  Desc[NetworkDescriptor offsets]
  LoadIn[load SENSOR_COUNT inputs]
  Bias[set bias node to 1]
  EvalLoop[for node in eval_order]
  Sum[sum incoming weighted edges]
  Act[apply tanh]
  WriteOut[write OUTPUT_COUNT outputs]

  Slot --> Desc --> LoadIn --> Bias --> EvalLoop --> Sum --> Act --> EvalLoop --> WriteOut
```

### 8.3 Host-Device Transfer Timeline

Sources: `src/simulation/simulation.cpp:179-184`, `src/simulation/simulation.cpp:202-205`, `src/simulation/buffers.cu`.

```mermaid
sequenceDiagram
  participant CPU
  participant Stream as CUDA stream

  CPU->>Stream: upload_async predator, prey, food
  CPU->>Stream: launch_build_sensors_async
  CPU->>Stream: launch_inference_async predator and prey
  CPU->>Stream: launch_post_inference_async
  CPU->>Stream: download_async predator, prey, food
  CPU->>Stream: synchronize
  CPU->>CPU: apply_results and collect_step_events
```

### 8.4 Inference Cache Allocation and Repack Policy

Sources: `src/evolution/inference_cache.cu:266-321`, `src/evolution/inference_cache.cu:455-468`, `src/evolution/inference_cache.cu:585-596`.

```mermaid
flowchart TB
  Acquire[acquire_entry compiled network]
  FindFree[search free_entries for capacity fit]
  Reuse{fit found}
  ReuseEntry[reuse entry and mark upload pending]
  NewEntry[append new entry and extend extents]
  LaunchPrep[prepare_for_launch]
  Repack{should_repack}
  Rebuild[build_from network_cache]
  UploadPending[upload only pending entries]
  UploadFull[full upload after reallocation]

  Acquire --> FindFree --> Reuse
  Reuse -->|yes| ReuseEntry --> LaunchPrep
  Reuse -->|no| NewEntry --> LaunchPrep

  LaunchPrep --> Repack
  Repack -->|yes| Rebuild --> UploadFull
  Repack -->|no| UploadPending
```

## 9. Module Dependencies

### 9.1 CMake Link Graph

Sources: `src/core/CMakeLists.txt`, `src/simulation/CMakeLists.txt`, `src/evolution/CMakeLists.txt`, `src/metrics/CMakeLists.txt`, `src/visualization/CMakeLists.txt`, `src/app/CMakeLists.txt`.

```mermaid
flowchart TB
  Core[moonai_core]
  Sim[moonai_simulation]
  Evo[moonai_evolution]
  Metrics[moonai_metrics]
  Viz[moonai_visualization]
  App[moonai_app]

  Sim --> Core
  Evo --> Core
  Evo --> Sim
  Metrics --> Core
  Viz --> Core
  Viz --> Sim
  Viz --> Evo
  App --> Core
  App --> Sim
  App --> Evo
  App --> Metrics
  App --> Viz
```
