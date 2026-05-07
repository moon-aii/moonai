use anyhow::{Context as _, Result, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{RenderSnapshotReadback, UiStatsReadback};
use crate::tick::checks::InvariantCheckReadback;
use crate::tick::compaction::FreeListStateReadback;
use crate::tick::evolution::{EvolutionManager, GpuEvolutionConfig};
use crate::tick::genome::{PopulationKind, SeededAgentSnapshot};
use crate::types::{OUTPUT_COUNT, SENSOR_COUNT};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSimulationConfig {
    pub food_capacity: u32,
    pub world_size: f32,
    pub predator_speed: f32,
    pub prey_speed: f32,
    pub interaction_range: f32,
    pub energy_drain_per_tick: f32,
    pub energy_gain_from_kill: f32,
    pub energy_gain_from_food: f32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub max_age: u32,
    pub seed: u64,
}

impl GpuSimulationConfig {
    pub fn for_simulation(config: &SimulationConfig) -> Result<Self> {
        if config.grid_size <= 0 {
            bail!("grid_size must be positive for GPU simulation, got {}", config.grid_size);
        }
        if config.predator_speed <= 0.0 {
            bail!("predator_speed must be positive for GPU simulation, got {}", config.predator_speed);
        }
        if config.prey_speed <= 0.0 {
            bail!("prey_speed must be positive for GPU simulation, got {}", config.prey_speed);
        }
        if config.interaction_range < 0.0 {
            bail!("interaction_range must be non-negative for GPU simulation, got {}", config.interaction_range);
        }
        if config.energy_drain_per_tick < 0.0 {
            bail!(
                "energy_drain_per_tick must be non-negative for GPU simulation, got {}",
                config.energy_drain_per_tick
            );
        }
        if config.energy_gain_from_kill < 0.0 {
            bail!(
                "energy_gain_from_kill must be non-negative for GPU simulation, got {}",
                config.energy_gain_from_kill
            );
        }
        if config.energy_gain_from_food < 0.0 {
            bail!(
                "energy_gain_from_food must be non-negative for GPU simulation, got {}",
                config.energy_gain_from_food
            );
        }
        if config.initial_energy <= 0.0 {
            bail!("initial_energy must be positive for GPU simulation, got {}", config.initial_energy);
        }
        if config.max_energy <= 0.0 {
            bail!("max_energy must be positive for GPU simulation, got {}", config.max_energy);
        }
        if config.max_age <= 0 {
            bail!("max_age must be positive for GPU simulation, got {}", config.max_age);
        }

        Ok(Self {
            food_capacity: as_non_negative_u32(config.food_count, "food_count")?,
            world_size: config.grid_size as f32,
            predator_speed: config.predator_speed,
            prey_speed: config.prey_speed,
            interaction_range: config.interaction_range,
            energy_drain_per_tick: config.energy_drain_per_tick,
            energy_gain_from_kill: config.energy_gain_from_kill,
            energy_gain_from_food: config.energy_gain_from_food,
            initial_energy: config.initial_energy,
            max_energy: config.max_energy,
            max_age: as_non_negative_u32(config.max_age, "max_age")?,
            seed: config.seed as i64 as u64,
        })
    }
}

pub struct SimulationState {
    evolution: EvolutionManager,
    config: GpuSimulationConfig,
}

impl SimulationState {
    pub fn init_from_config(config: &SimulationConfig) -> Result<Self> {
        let evolution_config = GpuEvolutionConfig::for_seed_stage(config, SENSOR_COUNT as u32, OUTPUT_COUNT as u32)?;
        let simulation_config = GpuSimulationConfig::for_simulation(config)?;

        let mut evolution = EvolutionManager::create(evolution_config)?;
        evolution.seed_initial_population()?;
        evolution.initialize_simulation(simulation_config)?;
        if evolution_config.predator_capacity > 0 {
            let _ = evolution.compile_population(PopulationKind::Predator, 0)?;
        }
        if evolution_config.prey_capacity > 0 {
            let _ = evolution.compile_population(PopulationKind::Prey, 0)?;
        }

        Ok(Self { evolution, config: simulation_config })
    }

    pub const fn config(&self) -> GpuSimulationConfig {
        self.config
    }

    pub fn tick(&mut self) -> Result<UiStatsReadback> {
        self.evolution.simulation_step()
    }

    pub fn ui_stats(&self) -> Result<UiStatsReadback> {
        self.evolution.simulation_ui_stats()
    }

    pub fn free_list_state(&self) -> Result<FreeListStateReadback> {
        self.evolution.simulation_free_list_state()
    }

    pub fn render_snapshot(&self, max_predators: u32, max_prey: u32, max_food: u32) -> Result<RenderSnapshotReadback> {
        self.evolution.render_snapshot(max_predators, max_prey, max_food)
    }

    pub fn seeded_agent_snapshot(&self, population_kind: PopulationKind, slot: u32) -> Result<SeededAgentSnapshot> {
        self.evolution.seeded_agent_snapshot(population_kind, slot)
    }

    pub fn check_invariants(&self) -> Result<InvariantCheckReadback> {
        self.evolution.check_invariants()
    }
}

fn as_non_negative_u32(value: i32, field_name: &str) -> Result<u32> {
    if value < 0 {
        bail!("{field_name} must be non-negative, got {value}");
    }
    u32::try_from(value).with_context(|| format!("{field_name} could not be converted to u32: {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulation_config(seed: i32) -> SimulationConfig {
        SimulationConfig {
            grid_size: 64,
            predator_count: 4,
            prey_count: 6,
            food_count: 10,
            predator_speed: 1.0,
            prey_speed: 1.0,
            interaction_range: 128.0,
            energy_drain_per_tick: 0.02,
            energy_gain_from_kill: 0.25,
            energy_gain_from_food: 0.15,
            initial_energy: 0.5,
            max_energy: 1.0,
            max_age: 64,
            max_hidden_nodes: 6,
            seed,
            ..SimulationConfig::default()
        }
    }

    fn runtime_ready() -> bool {
        EvolutionManager::runtime_status().is_success()
    }

    #[test]
    fn simulation_initializes_food_and_ui_state() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let state = SimulationState::init_from_config(&simulation_config(61))?;
        let stats = state.ui_stats()?;
        let free_list = state.free_list_state()?;
        let snapshot = state.render_snapshot(16, 16, 16)?;

        assert_eq!(stats.tick, 0);
        assert_eq!(stats.predator_count, 4);
        assert_eq!(stats.prey_count, 6);
        assert_eq!(free_list.tick, 0);
        assert_eq!(free_list.predator_free_slots, 0);
        assert_eq!(free_list.prey_free_slots, 0);
        assert_eq!(free_list.active_food_count, 10);
        assert_eq!(snapshot.header.total_predators, 4);
        assert_eq!(snapshot.header.total_prey, 6);
        assert_eq!(snapshot.header.total_food, 10);
        assert_eq!(snapshot.predators.len(), 4);
        assert_eq!(snapshot.prey.len(), 6);
        assert_eq!(snapshot.food.len(), 10);

        Ok(())
    }

    #[test]
    fn simulation_ticks_are_deterministic_for_same_seed() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let mut state_a = SimulationState::init_from_config(&simulation_config(62))?;
        let mut state_b = SimulationState::init_from_config(&simulation_config(62))?;

        for _ in 0..3 {
            assert_eq!(state_a.tick()?, state_b.tick()?);
        }

        assert_eq!(state_a.ui_stats()?, state_b.ui_stats()?);
        assert_eq!(state_a.free_list_state()?, state_b.free_list_state()?);
        assert_eq!(
            state_a.seeded_agent_snapshot(PopulationKind::Predator, 0)?,
            state_b.seeded_agent_snapshot(PopulationKind::Predator, 0)?
        );
        assert_eq!(state_a.render_snapshot(16, 16, 16)?, state_b.render_snapshot(16, 16, 16)?);
        assert_eq!(state_a.check_invariants()?, state_b.check_invariants()?);

        Ok(())
    }

    #[test]
    fn simulation_tick_advances_positions_and_counters() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let mut state = SimulationState::init_from_config(&simulation_config(63))?;
        let before = state.seeded_agent_snapshot(PopulationKind::Predator, 0)?;
        let stats = state.tick()?;
        let after = state.seeded_agent_snapshot(PopulationKind::Predator, 0)?;
        let free_list = state.free_list_state()?;

        assert_eq!(stats.tick, 1);
        assert_eq!(free_list.tick, 1);
        assert!(after.pos_x != before.pos_x || after.pos_y != before.pos_y);
        assert!(
            stats.food_eaten > 0
                || stats.kills > 0
                || free_list.predator_free_slots > 0
                || free_list.prey_free_slots > 0
        );

        Ok(())
    }
}
