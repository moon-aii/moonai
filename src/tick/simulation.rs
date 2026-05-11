use anyhow::Result;

use crate::experiment::SimulationConfig;
use crate::profile_scope;
use crate::tick::buffers::{FreeListStateReadback, MetricsSummaryReadback, RenderSnapshotReadback, UiStatsReadback};
use crate::tick::evolution::{EvolutionManager, GpuEvolutionConfig};
use crate::tick::species::{
    RepresentativeGenomeHeader, RepresentativeGenomeReadback, SpeciesBatchReadbackHeader, SpeciesSummaryReadback,
};
use serde::{Deserialize, Serialize};

pub const SENSOR_COUNT: u32 = 35;
pub const OUTPUT_COUNT: u32 = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNetworkReadbackHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub node_count: u16,
    pub eval_node_count: u16,
    pub output_count: u16,
    pub connection_count: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSnapshotReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub input_count: u16,
    pub reserved: u16,
    pub inputs: [f32; 35],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectedAgentNetworkReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub node_count: u16,
    pub output_count: u16,
    pub activation_count: u16,
    pub reserved: u16,
    pub output_0: f32,
    pub output_1: f32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PopulationKind {
    Predator = 0,
    Prey = 1,
}

pub struct SimulationState {
    evolution: EvolutionManager,
    config: SimulationConfig,
}

impl SimulationState {
    pub fn init_from_config(simulation_config: &SimulationConfig) -> Result<Self> {
        let evolution_config = GpuEvolutionConfig::for_seed_stage(simulation_config, SENSOR_COUNT, OUTPUT_COUNT)?;

        let mut evolution = EvolutionManager::create(evolution_config)?;
        evolution.seed_initial_population()?;
        evolution.set_simulation_config(*simulation_config)?;
        let grid_cell_size = simulation_config.vision_range.max(1.0);
        let grid_cols = ((simulation_config.grid_size / grid_cell_size).ceil() as u32).max(1);
        let grid_rows = ((simulation_config.grid_size / grid_cell_size).ceil() as u32).max(1);
        evolution.set_spatial_grid(grid_cell_size, grid_cols, grid_rows)?;
        evolution.ensure_food_buffer()?;
        evolution.ensure_counter_buffer()?;
        evolution.ensure_free_lists()?;
        evolution.ensure_reproduction_buffers()?;
        evolution.ensure_metrics_buffer()?;
        evolution.ensure_spatial_grid_buffers()?;
        evolution.reset_counters()?;
        evolution.initialize_free_lists()?;
        evolution.seed_food()?;
        evolution.reset_reproduction_state()?;
        if evolution_config.predator_capacity > 0 {
            let _ = evolution.compile_population(PopulationKind::Predator, 0)?;
        }
        if evolution_config.prey_capacity > 0 {
            let _ = evolution.compile_population(PopulationKind::Prey, 0)?;
        }
        evolution.build_spatial_grid()?;
        evolution.compute_sensor_inputs()?;
        evolution.simulation_refresh_reports()?;

        Ok(Self { evolution, config: *simulation_config })
    }

    pub const fn config(&self) -> SimulationConfig {
        self.config
    }

    pub fn tick(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("tick");

        self.evolution.build_spatial_grid()?;
        self.evolution.compute_sensor_inputs()?;
        self.evolution.infer_population(PopulationKind::Predator)?;
        self.evolution.infer_population(PopulationKind::Prey)?;
        self.evolution.update_vitals(PopulationKind::Predator)?;
        self.evolution.update_vitals(PopulationKind::Prey)?;
        self.evolution.apply_movement(PopulationKind::Predator)?;
        self.evolution.apply_movement(PopulationKind::Prey)?;
        self.evolution.build_spatial_grid()?;
        self.evolution.resolve_food()?;
        self.evolution.resolve_combat()?;
        let predator_births = self.evolution.reproduction_candidate_count(PopulationKind::Predator)?;
        self.ensure_birth_capacity(PopulationKind::Predator, predator_births)?;
        self.evolution.run_reproduction(PopulationKind::Predator)?;
        let prey_births = self.evolution.reproduction_candidate_count(PopulationKind::Prey)?;
        self.ensure_birth_capacity(PopulationKind::Prey, prey_births)?;
        self.evolution.run_reproduction(PopulationKind::Prey)?;
        self.evolution.advance_tick()?;

        let ui_stats = self.evolution.simulation_ui_stats()?;
        if self.config.report_interval_ticks > 0 && ui_stats.tick % self.config.report_interval_ticks == 0 {
            self.evolution.simulation_refresh_reports()?;
        }
        Ok(ui_stats)
    }

    pub fn ui_stats(&self) -> Result<UiStatsReadback> {
        self.evolution.simulation_ui_stats()
    }

    pub fn free_list_state(&self) -> Result<FreeListStateReadback> {
        self.evolution.simulation_free_list_state()
    }

    pub fn metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        self.evolution.simulation_metrics_summary()
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        self.evolution.simulation_refresh_reports()
    }

    pub fn species_summaries(
        &mut self,
        population_kind: PopulationKind,
        max_species: u32,
    ) -> Result<(SpeciesBatchReadbackHeader, Vec<SpeciesSummaryReadback>, Vec<RepresentativeGenomeHeader>)> {
        self.evolution.species_summaries(population_kind, max_species)
    }

    pub fn representative_genome(
        &self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        self.evolution.representative_genome(population_kind, slot)
    }

    pub fn sensor_snapshot(&self, population_kind: PopulationKind, slot: u32) -> Result<SensorSnapshotReadback> {
        self.evolution.sensor_snapshot(population_kind, slot)
    }

    pub fn selected_agent_network(
        &self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        self.evolution.selected_agent_network(population_kind, slot)
    }

    pub fn render_snapshot(&self, max_predators: u32, max_prey: u32, max_food: u32) -> Result<RenderSnapshotReadback> {
        self.evolution.render_snapshot(max_predators, max_prey, max_food)
    }

    fn ensure_birth_capacity(&mut self, population_kind: PopulationKind, births_pending: u32) -> Result<()> {
        profile_scope!("birth_cap");

        if births_pending == 0 {
            return Ok(());
        }

        let live_count = self.evolution.population_live_count(population_kind)?;
        let free_list_state = self.evolution.simulation_free_list_state()?;
        let free_slots = match population_kind {
            PopulationKind::Predator => free_list_state.predator_free_slots,
            PopulationKind::Prey => free_list_state.prey_free_slots,
        };
        let capacity = self.evolution.population_capacity(population_kind);
        let required_live = live_count.saturating_add(births_pending);
        if free_slots >= births_pending && required_live <= ((capacity * 9) / 10) {
            return Ok(());
        }

        let mut new_capacity = if capacity == 0 { 1 } else { capacity };
        while new_capacity.saturating_sub(live_count) < births_pending || required_live > ((new_capacity * 9) / 10) {
            new_capacity = if new_capacity == 0 { 1 } else { new_capacity.saturating_mul(2) };
        }
        self.evolution.expand_population(population_kind, new_capacity)?;
        self.evolution.build_spatial_grid()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulation_config(seed: u64) -> SimulationConfig {
        SimulationConfig {
            grid_size: 64.0,
            predator_count: 4,
            prey_count: 6,
            food_count: 10,
            predator_speed: 1.0,
            prey_speed: 1.0,
            vision_range: 128.0,
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

    #[test]
    fn simulation_initializes_food_and_ui_state() -> Result<()> {
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
        let mut state_a = SimulationState::init_from_config(&simulation_config(62))?;
        let mut state_b = SimulationState::init_from_config(&simulation_config(62))?;

        for _ in 0..3 {
            assert_eq!(state_a.tick()?, state_b.tick()?);
        }

        assert_eq!(state_a.ui_stats()?, state_b.ui_stats()?);
        assert_eq!(state_a.free_list_state()?, state_b.free_list_state()?);
        assert_eq!(state_a.metrics_summary()?, state_b.metrics_summary()?);
        assert_eq!(
            state_a.sensor_snapshot(PopulationKind::Predator, 0)?,
            state_b.sensor_snapshot(PopulationKind::Predator, 0)?
        );
        assert_eq!(state_a.render_snapshot(16, 16, 16)?, state_b.render_snapshot(16, 16, 16)?);

        Ok(())
    }

    #[test]
    fn simulation_tick_advances_positions_and_counters() -> Result<()> {
        let mut state = SimulationState::init_from_config(&simulation_config(63))?;
        let stats = state.tick()?;
        let free_list = state.free_list_state()?;

        assert_eq!(stats.tick, 1);
        assert_eq!(free_list.tick, 1);
        assert!(
            stats.food_eaten > 0
                || stats.kills > 0
                || free_list.predator_free_slots > 0
                || free_list.prey_free_slots > 0
        );

        Ok(())
    }

    #[test]
    fn simulation_food_deactivates_when_respawn_disabled() -> Result<()> {
        let mut config = simulation_config(68);
        config.food_respawn_rate = 0.0;
        let mut state = SimulationState::init_from_config(&config)?;

        let before = state.free_list_state()?;
        let stats = state.tick()?;
        let after = state.free_list_state()?;
        let snapshot = state.render_snapshot(16, 16, config.food_count)?;

        assert!(stats.food_eaten > 0);
        assert_eq!(before.active_food_count, config.food_count);
        assert_eq!(after.active_food_count, before.active_food_count - stats.food_eaten);
        assert_eq!(snapshot.food.len() as u32, after.active_food_count);
        assert_eq!(snapshot.header.total_food, after.active_food_count);

        Ok(())
    }

    #[test]
    fn simulation_food_respawns_when_respawn_rate_is_one() -> Result<()> {
        let mut config = simulation_config(69);
        config.food_respawn_rate = 1.0;
        let mut state = SimulationState::init_from_config(&config)?;

        let before = state.free_list_state()?;
        let stats = state.tick()?;
        let after = state.free_list_state()?;
        let snapshot = state.render_snapshot(16, 16, config.food_count)?;

        assert!(stats.food_eaten > 0);
        assert_eq!(before.active_food_count, config.food_count);
        assert_eq!(after.active_food_count, config.food_count);
        assert_eq!(snapshot.food.len() as u32, config.food_count);
        assert_eq!(snapshot.header.total_food, config.food_count);

        Ok(())
    }

    #[test]
    fn simulation_sensor_snapshot_encodes_targets_and_walls() -> Result<()> {
        let state = SimulationState::init_from_config(&simulation_config(64))?;
        let sensors = state.sensor_snapshot(PopulationKind::Predator, 0)?;

        assert_eq!(sensors.population_kind, PopulationKind::Predator);
        assert_eq!(sensors.slot, 0);
        assert_eq!(sensors.input_count as u32, SENSOR_COUNT);
        assert!(sensors.inputs[..10].iter().any(|value| value.abs() > f32::EPSILON));
        assert!(sensors.inputs[10..20].iter().any(|value| value.abs() > f32::EPSILON));
        assert!(sensors.inputs[20..30].iter().any(|value| value.abs() > f32::EPSILON));
        assert!((sensors.inputs[30] - 0.5).abs() < 1e-6);
        assert_eq!(sensors.inputs[31], 0.0);
        assert_eq!(sensors.inputs[32], 0.0);

        Ok(())
    }

    #[test]
    fn simulation_metrics_summary_refreshes_on_report_interval() -> Result<()> {
        let mut config = simulation_config(67);
        config.report_interval_ticks = 1;
        let mut state = SimulationState::init_from_config(&config)?;

        let stats = state.tick()?;
        let metrics = state.metrics_summary()?;

        assert_eq!(metrics.tick, stats.tick);
        assert_eq!(metrics.predator_count, stats.predator_count);
        assert_eq!(metrics.prey_count, stats.prey_count);
        assert!(metrics.predator_species > 0 || metrics.predator_count == 0);
        assert!(metrics.prey_species > 0 || metrics.prey_count == 0);

        Ok(())
    }
}
