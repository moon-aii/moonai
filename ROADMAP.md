# MoonAI - Implementation Roadmap

A step-by-step guide to fully implementing MoonAI. Each phase builds on the previous one. Phases are designed to produce a testable, working system at each stage.

> **Status as of 2026-03-22:** Phases 1–7 complete. Phase 5 (GPU) fully implemented: CSR-packed kernels, RAII device memory, runtime CPU fallback. Phase 8 (Experimentation) partially complete: experiment matrix defined, 60+ runs executed, plots and summary generated. Remaining: final report and presentation.

---

## Phase 1: Core Infrastructure ✅ Complete

**Goal:** Get a compiling, configurable project with a basic simulation loop.

### 1.1 Build System Verification
- [x] Bootstrap vcpkg on all team members' machines (Linux + Windows)
- [x] Verify `just configure && just build` works on both platforms
- [x] Verify `just test` runs and all skeleton tests pass
- [x] Set up IDE integration (`just compdb` for clangd/VSCode)

### 1.2 Configuration System
- [x] Finalize all configuration parameters in `default_config.json`
- [x] Add config validation (bounds checking, sanity checks) in `config.cpp`
- [x] Add command-line argument parsing (config path, seed override, headless flag)
- [x] Write tests for config loading with various valid/invalid inputs

### 1.3 Random Number Generator
- [x] Verify deterministic behavior: same seed produces identical simulation runs
- [x] Add RNG utilities needed for evolution: weighted random selection, shuffle
- [x] Write tests confirming reproducibility across platforms

### 1.4 Logging Foundation
- [x] Implement `Logger` with timestamped file creation in output directory
- [x] Add master seed to the header of every output file
- [x] Implement JSON export for genome snapshots (`genomes.json`)
- [x] Test that log files are correctly formatted and parseable by Python

**Deliverable:** Project builds on Linux and Windows. Configuration loads, RNG is deterministic, logs are written.

---

## Phase 2: Simulation Environment ✅ Complete

**Goal:** Agents move around in a 2D world with collision detection and sensing.

### 2.1 Environment
- [x] Implement the 2D grid with configurable dimensions
- [x] Decide boundary behavior (wrap-around vs. wall clamping) - make configurable
- [x] Add resource nodes / food spawn points for prey (optional complexity lever)
- [x] Implement spatial partitioning (uniform grid) for O(1) neighbor lookups instead of O(n^2)

### 2.2 Agent System
- [x] Finalize Agent base class with all attributes: position, velocity, stamina, energy, age
- [x] Implement Predator-specific behavior: energy drain from moving, energy gain from eating prey
- [x] Implement Prey-specific behavior: energy gain from resources, fleeing mechanics
- [x] Add death conditions: stamina depletion, being caught by predator
- [x] Add age tracking for fitness calculation

### 2.3 Physics and Sensing
- [x] Implement efficient collision detection using spatial grid
- [x] Implement agent sensing: each agent detects nearest N neighbors within vision range
- [x] Build sensor input vector for neural network:
  - Distance and angle to nearest predator(s)
  - Distance and angle to nearest prey/food
  - Own stamina/energy level
  - Own velocity
  - Wall distances (if applicable)
- [x] Write tests for collision detection edge cases
- [x] Write tests for sensor input generation

### 2.4 Simulation Loop
- [x] Implement the tick-based simulation loop
- [x] Add fixed timestep with accumulator pattern for consistent physics
- [x] Implement generation boundary: after N ticks, end generation and trigger evolution
- [x] Add population tracking: count alive predators/prey each tick

**Deliverable:** Agents spawn, move (randomly for now), detect neighbors, eat/die. Population stats are tracked per tick.

---

## Phase 3: NEAT Evolution Core ✅ Complete

**Goal:** Full NEAT algorithm implementation. Agents evolve neural networks that control their behavior.

### 3.1 Genome Encoding
- [x] Finalize node gene structure (id, type, activation function)
- [x] Finalize connection gene structure (in, out, weight, enabled, innovation)
- [x] Implement global innovation number tracking with historical markings
  - Use a lookup table: (in_node, out_node) -> innovation number
  - This prevents duplicate innovations from getting different numbers
- [x] Implement genome serialization to/from JSON for checkpointing
- [x] Write comprehensive tests for genome operations

### 3.2 Neural Network (Phenotype)
- [x] Implement proper topological sort for feed-forward evaluation
  - Recurrent connections rejected via DFS cycle detection in `add_connection`
- [x] Support multiple activation functions: sigmoid, tanh, ReLU *(configurable via `activation_function` field)*
- [x] Optimize forward pass for networks with many nodes *(precomputed incoming_ adjacency list)*
- [x] Write tests: verify output for known network topologies with hand-calculated values
- [x] Benchmark: measure forward pass time for 1000 networks *(debug timing via `--verbose`, `just bench-nn`)*

### 3.3 Mutation Operators
- [x] **Weight mutation**: perturb existing weight (90%) vs. replace with random (10%)
- [x] **Add connection**: connect two previously unconnected nodes
- [x] **Add node**: split an existing connection with a new hidden node
- [x] **Toggle connection**: enable/disable a connection
- [x] **Delete connection**: remove a connection entirely *(configurable `delete_connection_rate`, keeps ≥1 connection)*
- [x] Ensure all mutations maintain valid network structure (no dangling nodes)
- [x] Write tests: mutated genomes produce valid networks that can be evaluated

### 3.4 Crossover
- [x] Implement multipoint crossover following the NEAT paper:
  - Matching genes: randomly pick from either parent
  - Disjoint/excess genes: take from fitter parent
- [x] Handle disabled genes: if disabled in either parent, 75% chance of being disabled in child
- [x] Write tests: crossover produces valid genomes, fitness-based parent preference works

### 3.5 Speciation
- [x] Implement compatibility distance (c1 * excess + c2 * disjoint + c3 * avg_weight_diff)
- [x] Implement species assignment: each genome goes to the first compatible species
- [x] Update species representative each generation (random member or best member)
- [x] Implement dynamic compatibility threshold adjustment to maintain target species count
- [x] Remove stagnant species (no fitness improvement for N generations)
- [x] Write tests: genomes are correctly grouped, stagnant species are culled

### 3.6 Selection and Reproduction
- [x] Implement explicit fitness sharing (divide fitness by species size)
- [x] Calculate offspring allocation per species (proportional to adjusted fitness sum) *(largest-remainder method)*
- [x] Implement elitism: preserve champion of each species unchanged
- [x] Implement tournament selection within species for parent selection
- [x] Fill population with crossover (configured rate) + mutation
- [x] Maintain constant population size across generations
- [x] Write tests: population size is maintained, elitism preserves best genomes

### 3.7 Fitness Function
- [x] Design predator fitness: weighted sum of kills, survival, energy efficiency
- [x] Design prey fitness: weighted sum of survival, food eaten, energy efficiency
- [x] Distance-traveled component: `fitness_distance_weight` now tracked in `Agent.distance_traveled_` and used in `compute_fitness()`
- [x] Make fitness weights configurable in JSON
- [x] Complexity penalty: small per-connection/node deduction discourages bloat

### 3.8 Integration: Wire NEAT to Simulation
- [x] Assign one genome per agent at generation start
- [x] Build neural network from genome, use it for agent decision-making each tick
- [x] After generation ends, collect fitness from agents, assign to genomes
- [x] Run evolution (speciation, selection, reproduction, mutation)
- [x] Spawn new agents with new genomes for next generation
- [x] End-to-end test: run 10 generations, verify fitness improves (or at least doesn't crash)

**Deliverable:** Complete NEAT loop running. Agents controlled by evolving neural networks. Fitness improves over generations.

---

## Phase 4: Visualization ✅ Mostly Complete

**Goal:** Real-time SFML 3.x visualization for observing and debugging the simulation.

### 4.1 Window and Rendering
- [x] Create SFML window with configurable resolution (1280×720 default)
- [x] Implement camera with pan and zoom (mouse drag + scroll wheel)
- [x] Render environment grid/boundaries
- [x] Render predators as red triangles (pointing in movement direction)
- [x] Render prey as green circles
- [x] Render dead agents as faded/transparent
- [x] Color-code agents by species (deterministic hue from species ID)

### 4.2 UI Overlay
- [x] Display generation number, tick counter
- [x] Display FPS counter
- [x] Display population counts (alive predators / prey)
- [x] Display best fitness, average fitness
- [x] Display number of species
- [x] Show selected agent info on click (genome complexity, fitness, energy, kills, food eaten)

### 4.3 Simulation Controls
- [x] Pause/resume (Space)
- [x] Speed control: 1x–10x (arrow keys)
- [x] Step-by-step mode (advance one tick at a time)
- [x] Toggle visualization elements (show/hide vision ranges, sensor lines)
- [x] Reset simulation (R key)
- [x] Screenshot (S key) *(saves screenshot_gen<N>_tick<T>.png)*
- [x] Toggle headless fast-forward from within visual mode *(H key toggles FF mode; generation runs at max speed)*

### 4.4 Debug Visualization
- [x] Draw vision range circle for selected agent
- [x] Draw lines from agent to detected neighbors
- [x] Show neural network topology of selected agent (small side panel) *(250×300px right-edge panel)*
- [x] Show neural network activation values in real time *(activation-based node coloring in NN panel)*

**Deliverable:** Full interactive visualization. Can observe agent behavior, track evolution, control simulation speed.

---

## Phase 5: GPU Acceleration ✅ Implemented

**Goal:** CUDA backend for batch neural network inference and fitness evaluation.

> **Current state:** Full CUDA implementation. CSR-packed variable-topology networks, GPU neural inference kernel, GPU fitness kernel, RAII device memory management (`GpuBatch`), runtime CPU fallback, `--no-gpu` flag.

### 5.1 Data Layout for GPU
- [x] Design flattened data structures for batch GPU transfer:
  - CSR-packed variable-topology layout (`GpuNetDesc` + flat arrays)
  - `GpuBatch` RAII class manages all device memory
- [x] Implement host↔device memory management with RAII wrappers (`GpuBatch`)
- [x] CSR-packed approach: no padding waste, each agent's network stored contiguously

### 5.2 Batch Neural Inference
- [x] Implement CUDA kernel for parallel feed-forward pass (`neural_forward_kernel`)
  - One thread per agent
  - Topological evaluation order, supports sigmoid/tanh/relu
- [x] Handle variable-topology networks via CSR offset descriptors
- [x] Benchmark GPU vs CPU inference: measure speedup (`just cuda-bench`)
- [x] Validate GPU results match CPU results exactly (`just cuda-validate`)

### 5.3 Batch Fitness Evaluation
- [x] Implement CUDA kernel for parallel fitness computation (`fitness_eval_kernel`)
- [x] Agent stats pre-packed on CPU, fitness evaluated on GPU without simulation round-trip
- [x] Benchmark GPU vs CPU fitness evaluation *(included in `just cuda-bench`)*

### 5.4 CPU Fallback
- [x] Compile-time flag: build without CUDA entirely (`MOONAI_ENABLE_CUDA=OFF`)
- [x] Runtime detection: if no CUDA device, automatically uses CPU path
- [x] `--no-gpu` flag to force CPU path even when CUDA is available
- [x] Verify identical results between GPU and CPU code paths (`just cuda-validate`)

**Deliverable:** GPU-accelerated simulation runs significantly faster with large populations. CPU fallback works identically.

---

## Phase 6: Data Collection and Analysis ✅ Complete

**Goal:** Research-grade data logging and Python analysis pipeline.

### 6.1 Per-Generation Logging
- [x] Log to `stats.csv`: generation, predator_count, prey_count, best_fitness, avg_fitness, num_species, avg_complexity
- [x] Log to `genomes.json`: full genome of best agent each generation (topology + weights)
- [x] Log to `species.csv`: species_id, size, avg_fitness, stagnation_count per generation
- [x] Include master seed and full config in output metadata

### 6.2 Per-Tick Logging (Optional, High-Volume)
- [x] Agent trajectories: position, velocity, energy per tick (configurable: every N ticks) *(`tick_log_enabled`, `tick_log_interval` config fields; writes `ticks.csv`)*
- [x] Interaction events: kills and food consumption logged to `events.csv` *(buffered writes; near-misses excluded for performance)*
- [x] Buffer writes and flush periodically to avoid I/O bottleneck *(buffered with flush every 500 rows)*

### 6.3 Python Analysis Suite
- [x] `plot_fitness.py`: fitness curves over generations (best, average)
- [x] `plot_population.py`: predator/prey population dynamics
- [x] `plot_species.py`: species count and diversity over time
- [x] `plot_complexity.py`: genome complexity (nodes, connections) over generations
- [x] `compare_experiments.py`: overlay metrics from multiple runs
- [x] `analyze_genome.py`: visualize a genome's neural network topology (using networkx)

### 6.4 Experiment Management
- [x] Implement experiment naming: output dir includes timestamp + seed
- [x] Add checkpoint/resume: `--resume <path>` and `--checkpoint <N>` CLI flags; full state serialization implemented in `EvolutionManager::save_checkpoint/load_checkpoint`
- [x] Add batch experiment runner: `analysis/run_experiments.py` runs N seeds automatically

**Deliverable:** Complete data pipeline. Every experiment produces structured, reproducible data. Python scripts generate publication-quality plots.

---

## Phase 7: Polish and Optimization ✅ Mostly Complete

**Goal:** Performance optimization, edge case handling, code quality.

### 7.1 Performance Profiling
- [x] Profile simulation with large populations (1000+ agents) *(`just profile` uses perf)*
- [x] Identify and fix bottlenecks:
  - Spatial grid cell size halved (vision/2) for denser indexing
  - Spatial grid stores positions for O(1) distance filtering
  - NN activate() hot path: precomputed incoming adjacency list, no per-call map allocation
  - Food lookup uses dedicated `food_grid_` SpatialGrid (O(1) per agent instead of O(n))
  - OpenMP parallel sensor+NN compute phase in `evaluate_generation()`
- [ ] Object pool for agent allocation *(low priority — see NOTES.md)*
- [ ] Target: 30+ FPS with 500 agents with visualization, 1000+ agents headless *(not benchmarked)*

### 7.2 Edge Cases and Robustness
- [x] Handle population extinction (all predators or all prey die) *(early-exit in evaluate_generation)*
- [x] Handle species collapse (all genomes in one species) *(always keep ≥1 species)*
- [x] Handle network degeneracy (all connections disabled) *(mutation ensures ≥1 enabled)*
- [x] Graceful shutdown on SIGINT/SIGTERM
- [x] Input validation for all config values

### 7.3 Code Quality
- [x] Run static analysis (cppcheck, clang-tidy) and fix warnings *(`just lint` using cppcheck)*
- [x] Ensure no memory leaks (valgrind / AddressSanitizer) *(`just check-memory` uses ASan+UBSan)*
- [x] Add documentation comments to all public APIs *(config.hpp fields fully annotated)*
- [ ] Review and clean up TODO markers in code

### 7.4 Cross-Platform Testing
- [x] Full test suite passes on Linux (GCC + Clang) *(94/94 tests)*
- [ ] Full test suite passes on Windows (MSVC) *(not verified)*
- [ ] CUDA path tested on both platforms *(stub only)*
- [x] Verify determinism: same seed produces identical output *(tested on Linux)*

**Deliverable:** Production-quality code. Fast, robust, clean, cross-platform verified.

---

## Phase 8: Experimentation and Report 🟡 In Progress

**Goal:** Run experiments, collect results, write the final report.

### 8.1 Experiment Design
- [x] Define experiment matrix (8 conditions in `config/experiments/`):
  - Vary mutation rates (0.1, 0.3, 0.5)
  - Vary population sizes (small: 25+75, large: 100+300)
  - Compare with/without speciation (`compatibility_threshold: 100`)
  - Compare activation functions (sigmoid vs tanh)
  - Compare crossover rates (0.25 vs 0.75)
- [x] Each experiment: 5 runs with different seeds for statistical significance

### 8.2 Run Experiments
- [x] Execute experiment matrix (use `analysis/run_experiments.py`) — 60+ runs completed
- [x] Collect all output data in `output/`
- [x] Generate analysis plots for each experiment (51 PNGs in `output/plots/`)

### 8.3 Analysis
- [x] Analyze fitness convergence rates across configurations (`output/plots/summary.md`)
- [ ] Analyze emergent behaviors: do predators develop hunting strategies? Do prey develop flocking?
- [ ] Analyze genome complexity: does NEAT grow complexity minimally?
- [ ] Compare GPU vs CPU performance at various population sizes
- [ ] Statistical analysis: mean, std, confidence intervals across runs

### 8.4 Final Report
- [ ] Document methodology, results, and conclusions
- [ ] Include representative visualizations and plots
- [ ] Discuss limitations and future work
- [ ] Prepare demo/presentation

**Deliverable:** Completed senior design project with experimental results and final report.

---

## Summary Timeline

| Phase | Status | Core Deliverable |
|-------|--------|-----------------|
| 1. Core Infrastructure | ✅ Complete | Building, config, logging, RNG |
| 2. Simulation Environment | ✅ Complete | Moving agents with collision and sensing |
| 3. NEAT Evolution | ✅ Complete | Full evolutionary loop with improving fitness |
| 4. Visualization | ✅ Mostly Complete | Interactive SFML 3.x viewer |
| 5. GPU Acceleration | ✅ Implemented | CSR-packed kernels, GpuBatch RAII, runtime CPU fallback, --no-gpu flag |
| 6. Data & Analysis | ✅ Complete | Research-grade logging + Python pipeline |
| 7. Polish | ✅ Mostly Complete | Performance, robustness, code quality |
| 8. Experimentation | 🟡 In Progress | Results, analysis, final report |

## Development Principles

1. **Test as you go** - Write tests for each component before moving to the next phase
2. **CPU first, GPU second** - Get everything working on CPU, then accelerate with CUDA
3. **Small commits** - Each completed task gets its own commit on the dev branch
4. **Branch per feature** - Use feature branches off `dev`, merge via PR with code review
5. **Reproduce everything** - Every result must be reproducible with a seed + config
