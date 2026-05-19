---
title: Home
hide:
  - navigation
  - toc
---

<p align="center"> <img src="_assets/logo.svg" alt="MoonAI Logo" width="120em" /> </p>

# MoonAI</br>NeuroEvolution Predator-Prey Simulation {: align="center" }

<div class="grid cards" markdown>

-   ## Project

    Simulation platform for studying neural network evolution through predator-prey dynamics using the NEAT algorithm.

    **Team**:

    - Caner Aras
    - Emir Irkılata
    - Oğuzhan Özkaya

    **Supervisor**:

    - Ayşenur Birtürk

    _CMPE 491/492 Senior Design Project - TED University_

-   ## Reports

    [:fontawesome-regular-file-lines: Project Proposal](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/ProjectProposal.pdf)

    [:fontawesome-regular-file-lines: Project Specification](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/ProjectSpecification.pdf)

    [:fontawesome-regular-file-lines: Analysis Report](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/AnalysisReport.pdf)

    [:fontawesome-regular-file-lines: High-Level Design Report](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/HighLevelDesignReport.pdf)

    [:fontawesome-regular-file-lines: Poster](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/Poster.pdf)

    [:fontawesome-regular-file-lines: Low-Level Design Report](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/LowLevelDesignReport.pdf)

    [:fontawesome-regular-file-lines: Test Plan Report](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/TestPlanReport.pdf)

    [:fontawesome-regular-file-lines: Final Report](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/FinalReport.pdf)

    [:fontawesome-regular-file-lines: GBYF Poster](https://raw.githubusercontent.com/moon-aii/moonai/main/papers/GBYF-Poster.pdf)

</div>

<div class="grid cards" markdown>

-   ### Motivation

    Modern artificial intelligence training often requires vast amounts of real-world data, which do not scale efficiently. MoonAI addresses this limitation by providing self-generating training environments for studying artificial intelligence without external data dependencies.

-   ### Objective

    Develop a robust simulation environment to research and optimize evolutionary algorithms. By decoupling training from real-world data dependencies, we investigate how genetic representations influence learning efficiency and adaptability in dynamic environments.

-   ### Approach

    The system employs a high-fidelity predator-prey simulation to generate evolutionary and genetic data. This synthetic ecosystem serves as a dynamic benchmark for evaluating evolutionary computation techniques in adaptive behavior modeling.

</div>

---

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

MoonAI's main deliverable is the simulation environment itself. The runtime, queueing model,
exports, and analysis tooling are designed to help study evolutionary machine learning behavior,
not to present one fixed model checkpoint as the final outcome.

## Concepts

### Simulation Environment

The simulation operates within a time-ticked 2D world. This virtual ecosystem imposes selective
pressures on agents—predators and prey—with configurable attributes including speed, vision,
stamina, and reproduction rates. Each agent is controlled by a neural network that reads 35 local
inputs: the 5 closest predators, prey, and food items as signed proximity-weighted dx and dy
pairs, plus self energy, velocity x/y, and signed wall proximity x/y.

The purpose of this environment is not only to animate agents, but to create repeatable pressure
under which adaptation, diversity, topology growth, and ecological balance can be studied across
many parameter settings.

### NEAT

NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. It was chosen because it simultaneously evolves both the topology and weights of networks, allowing complex structures to emerge from simple beginnings without requiring manual architecture design. MoonAI implements the NeuroEvolution of Augmenting Topologies (NEAT) algorithm to optimize agent behaviors. By evolving both neural network weights and topological structures, the system enables emergence of complex behavioral strategies through mutation, crossover, and speciation across successive generations.

## Features

- **Configuration**: Simulation parameters are defined in the Lua-based `experiments.lua` experiment file, covering population sizes, mutation rates, NEAT parameters, and energy system settings. Presets are selected, edited, queued, and run through the application UI.
- **High Performance**: To achieve high-performance execution, MoonAI uses a CUDA for all of the simulation, evolution, and neural network calculations. The host orchestrates lifecycle and data flow, while NVIDIA CUDA executes neural inference and simulation kernels for large agent populations.
- **Telemetry**: The system concurrently logs extensive telemetry (CSV/JSON output), including population metrics and genome histories, exporting structured data for rigorous offline analysis using Python-based tools.
- **Cross-Platform**: Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible**: Experiments Seeded runs provide comparable experiment setup and initial conditions.
