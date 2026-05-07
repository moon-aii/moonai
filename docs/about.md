# About

## Overview

MoonAI uses a predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve continuously through births and deaths using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

## Features

- **Entity-Component-System Architecture** - Data-oriented design with sparse-set ECS, cache-friendly SoA memory layouts, and 5-10x performance improvement
- **NEAT Implementation** - Evolves both topology and weights of neural networks simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls and live NN activation display
- **GPU Acceleration** - CUDA backend for sensing, neural inference, and simulation systems in both visual and headless modes
- **Cross-Platform** - Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible Experiments** - Seeded RNG with deterministic behavior within the CUDA execution path on a fixed runtime environment
- **Lua Configuration** - Define named experiments and parameter sweeps in `config.lua` without recompilation
- **Data Export** - CSV/JSON output (including optional per-tick trajectories) compatible with Python analysis tools

### Real-Time Analytics

Researchers observe emergent behaviors through an SFML-based real-time visualization layer. The system concurrently logs extensive telemetry, including population metrics and genome histories, exporting structured data for rigorous offline analysis using Python-based tools.

### Heterogeneous Architecture

To achieve high-performance execution, MoonAI uses a CUDA-first runtime. The host orchestrates lifecycle and data flow, while NVIDIA CUDA executes neural inference and simulation kernels for large agent populations.

### Simulation Environment

The simulation operates within a deterministic, time-ticked 2D world. This virtual ecosystem imposes selective pressures on agents—predators and prey—with configurable attributes including speed, vision, stamina, and reproduction rates. Each agent is controlled by a neural network that reads 35 local inputs: the 5 closest predators, prey, and food items as signed proximity-weighted dx and dy pairs, plus self energy, velocity x/y, and signed wall proximity x/y.

### Evolutionary Core

MoonAI implements the NeuroEvolution of Augmenting Topologies (NEAT) algorithm to optimize agent behaviors. By evolving both neural network weights and topological structures, the system enables emergence of complex behavioral strategies through mutation, crossover, and speciation across successive generations.

### NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. It was chosen because it simultaneously evolves both the topology and weights of networks, allowing complex structures to emerge from simple beginnings without requiring manual architecture design.

### Configuration

Simulation parameters are defined in the Lua-based `config.lua` experiment file, covering population sizes, mutation rates, NEAT parameters, and energy system settings. UI configuration (colors, sizes, panel layout, window settings) is defined separately in `settings.json`. Custom experiment sets can be specified at runtime without recompilation.
