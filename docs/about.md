# About

## Overview

MoonAI uses a predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve continuously through births and deaths using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The key point is that MoonAI is primarily a simulation environment for studying machine learning,
not a single-task model package. Its value comes from the combination of configurable
experiments, large evolving populations, real-time observation, structured artifact export, and
post-run analysis.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

It also provides a practical workflow for experimentation: choose a preset, run or queue it from
the UI, inspect live behavior, export structured results, and compare conditions later through the
HTML analysis pipeline.

## Features

### Configuration

- **Simulation**: Simulation parameters are defined in the Lua-based `experiments.lua` experiment file, covering population sizes, mutation rates, NEAT parameters, and energy system settings. Presets are selected, edited, queued, and run through the application UI.
- **Visualization**: UI configuration (colors, sizes, panel layout, window settings) is defined in `settings.json` under `ui`.

### Reproducible Experiments

Seeded RNG with deterministic behavior on a fixed runtime environment

### Real-Time Analytics

Researchers observe emergent behaviors through a real-time visualization layer.

### Data Export

The system concurrently logs extensive telemetry (CSV/JSON output), including population metrics and genome histories, exporting structured data for rigorous offline analysis using Python-based tools.

### Experiment Management

The application includes a queue-driven workflow for selecting presets, editing drafts, starting or
replacing runs, and preserving run history without editing source files.

### Cross-Platform

Runs on Linux and Windows with matched features and stable runtime behavior

### High Performance

To achieve high-performance execution, MoonAI uses a CUDA for all of the simulation, evolution, and neural network calculations. The host orchestrates lifecycle and data flow, while NVIDIA CUDA executes neural inference and simulation kernels for large agent populations.

### Simulation Environment

The simulation operates within a deterministic, time-ticked 2D world. This virtual ecosystem imposes selective pressures on agents—predators and prey—with configurable attributes including speed, vision, stamina, and reproduction rates. Each agent is controlled by a neural network that reads 35 local inputs: the 5 closest predators, prey, and food items as signed proximity-weighted dx and dy pairs, plus self energy, velocity x/y, and signed wall proximity x/y.

The purpose of this environment is not only to animate agents, but to create repeatable pressure
under which adaptation, diversity, topology growth, and ecological balance can be studied across
many parameter settings.

### NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. It was chosen because it simultaneously evolves both the topology and weights of networks, allowing complex structures to emerge from simple beginnings without requiring manual architecture design. MoonAI implements the NeuroEvolution of Augmenting Topologies (NEAT) algorithm to optimize agent behaviors. By evolving both neural network weights and topological structures, the system enables emergence of complex behavioral strategies through mutation, crossover, and speciation across successive generations.
