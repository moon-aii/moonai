# Project Analysis Notes

This file contains synthesized notes from the project's PDF documentation.

## Project Proposal Summary

### Objectives
1.  **Simulation Environment:** Develop a predator-prey simulation to generate data for training ML algorithms without needing real-world data.
2.  **Investigation:** Study how genetic representations influence efficiency and adaptability in dynamic environments.
3.  **Evaluation:** Assess the potential of evolutionary methods (specifically NeuroEvolution of Augmenting Topologies - NEAT) for solving adaptive behavior problems.

### Methodology
-   **Algorithm:** Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve both population-level parameters (movement speed, vision, stamina, reproduction rate) and individual neural network topologies.
-   **Approach:** Gradient-free optimization suitable for nonlinear, multimodal search spaces.

### Simulation Setup
-   **Environment:** 2D space with time-step-based dynamics.
-   **Agents:** Predators and Prey with configurable attributes.
-   **Optimization Goal:** Maximize survival rate, optimize population size and stability.
-   **Data Collection:** Population statistics, behavior trajectories, network topologies, and genetic representations.

### Technologies
-   **Simulation & Visualization:** C++ with SFML (Simple and Fast Multimedia Library).
-   **Machine Learning:** CUDA for calculation and C++ for algorithm implementation.
-   **Data Analysis:** Python (Pandas, Matplotlib).

## Project Specification Summary

### Description
A virtual ecosystem to optimize genetic and evolutionary algorithms. It focuses on evolving population-level parameters (speed, vision, reproduction) and individual behavioral strategies (neural networks) using NEAT.

### Constraints
-   **Economic:** Relies on open-source tools (C++, SFML, Python). Simulation scale limited by GPU resources.
-   **Ethical:** Adheres to ACM Code of Ethics; commits to transparency and fairness.
-   **Technical:** Software maintainability (modular architecture), portability (Windows/Linux), and sustainability (reusable code).

### Requirements
-   **Functional:**
    -   **FR-1:** Time-stepped 2D simulation environment with configurable agents.
    -   **FR-5:** Implementation of NEAT algorithm for dynamic evolution.
    -   **FR-9:** Real-time visualization using SFML.
    -   **FR-11:** Comprehensive data logging (metrics, trajectories).
    -   **FR-10:** User-adjustable configuration via JSON/TOML/YAML.
-   **Non-Functional:**
    -   **Performance:** Support for hundreds/thousands of agents (>30 FPS) using GPU acceleration (CUDA).
    -   **Scalability:** Extensible design for agents and environment.
    -   **Usability:** Interpretable visualization and analysis outputs.
    -   **Portability:** Windows and Linux support.

## Analysis Report Summary

### System Models
-   **Core Components:** Simulation (Loop), Environment (Spatial Context), Agents (Predator/Prey entities).
-   **Architecture:**
    -   **Agents:** Possess a Neural Network (Brain) for decision-making and a Genome for evolutionary traits.
    -   **EvolutionManager:** Handles NEAT operations including mutation, crossover, and speciation.
-   **Flow:** The "Agent Decision Loop" involves sensing the environment, processing inputs via the neural network, and executing actions.

### Refined Requirements
-   **Functional:**
    -   Time-stepped 2D environment with obstacles and resources.
    -   Distinct Predator and Prey roles with specific behaviors.
    -   Neural network-based behavior control evolved via NEAT.
    -   GPU-accelerated fitness evaluation and neural inference (CUDA).
-   **User Interface:**
    -   **Screens:** Main Menu, Configuration (Environment, Agent, Evolution settings), Live Visualization, Data & Logs (Graphs, Export).
    -   **Principles:** Minimalism, Transparency, Reproducibility, and Real-time Interpretability.

## High-Level Design Summary

### Software Architecture
-   **Pattern:** Object-Oriented, Modular Architecture.
-   **Subsystems:**
    1.  **Simulation Engine:** Manages the core loop, physics updates, and entity states. Key classes include `SimulationManager`, `Environment`, `Grid`, and `Agent`.
    2.  **Evolution Core:** Encapsulates the NEAT algorithm. `EvolutionManager` handles population pools, crossover, mutation, and speciation.
    3.  **Visualization Manager:** Handles real-time rendering of the 2D grid and agents using SFML.
    4.  **Data Management:** Responsible for loading JSON configurations and logging generation statistics to CSV/JSON.

### Hardware/Software Mapping
-   **Platform:** Linux Workstation.
-   **Compute Distribution:**
    -   **CPU:** Handles the main simulation loop and physics logic.
    -   **GPU:** Uses NVIDIA CUDA for massively parallel neural inference and fitness evaluation.
-   **Software Stack:** C++ (Core), SFML (Visualization), CUDA Toolkit (Acceleration), Python (Offline Analysis).

## Poster Summary

### Overview
-   **Motivation:** Scalability issues with manually designed training scenarios and real-world data for AI.
-   **Objective:** Create a research environment for evolutionary algorithms independent of real-world data.
-   **Approach:** Use a Predator-Prey simulation to generate evolutionary and genetic data for analysis.
-   **Key Algorithm:** NEAT (NeuroEvolution of Augmenting Topologies) chosen because it evolves both neural network weights and topologies, enabling complex behaviors to emerge autonomously.

### System Architecture & Implementation
-   **Modules:**
    -   *Simulation Engine:* 2D World, Deterministic Time Steps, Physics.
    -   *Evolution Core:* Population Management, Genome Mutation & Crossover.
    -   *Visualization:* Real-time Rendering, Debug Overlays.
    -   *Data:* CSV/JSON Logging, Experiment Metadata.
-   **Engineering Strategy:** Heterogeneous computing architecture.
    -   **CPU:** Control logic, Physics, UI.
    -   **GPU (CUDA):** Neural inference, Fitness evaluation, Parallel computation.
-   **Technologies:** C++17, SFML, CUDA, Python.
