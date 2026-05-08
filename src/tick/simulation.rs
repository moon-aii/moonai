use anyhow::{Context as _, Result, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{RenderSnapshotReadback, SpatialGridReadback, UiStatsReadback};
use crate::tick::checks::InvariantCheckReadback;
use crate::tick::compaction::{CompactionSummaryReadback, FreeListStateReadback};
use crate::tick::evolution::{EvolutionManager, GpuEvolutionConfig};
use crate::tick::genome::{PopulationKind, SeededAgentSnapshot};
use crate::tick::inference::SensorSnapshotReadback;
use crate::tick::metrics_reduce::MetricsSummaryReadback;
use crate::tick::mutation::PHASE3_MAX_CONNECTION_ATTEMPTS;
use crate::tick::reproduction::ReproductionSummaryReadback;
use crate::tick::species::{
    RepresentativeGenomeHeader, RepresentativeGenomeReadback, SpeciesBatchReadbackHeader, SpeciesSummaryReadback,
};
use crate::types::{OUTPUT_COUNT, SENSOR_COUNT};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSimulationConfig {
    pub food_capacity: u32,
    pub world_size: f32,
    pub predator_speed: f32,
    pub prey_speed: f32,
    pub vision_range: f32,
    pub interaction_range: f32,
    pub mate_range: f32,
    pub energy_drain_per_tick: f32,
    pub energy_gain_from_kill: f32,
    pub energy_gain_from_food: f32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub reproduction_energy_threshold: f32,
    pub reproduction_energy_cost: f32,
    pub offspring_initial_energy: f32,
    pub mutation_rate: f32,
    pub weight_mutation_power: f32,
    pub add_node_rate: f32,
    pub add_connection_rate: f32,
    pub delete_connection_rate: f32,
    pub max_connection_attempts: u32,
    pub max_age: u32,
    pub report_interval_ticks: u32,
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
        if config.vision_range <= 0.0 {
            bail!("vision_range must be positive for GPU simulation, got {}", config.vision_range);
        }
        if config.interaction_range < 0.0 {
            bail!("interaction_range must be non-negative for GPU simulation, got {}", config.interaction_range);
        }
        if config.mate_range < 0.0 {
            bail!("mate_range must be non-negative for GPU simulation, got {}", config.mate_range);
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
        if config.reproduction_energy_threshold <= 0.0 {
            bail!(
                "reproduction_energy_threshold must be positive for GPU simulation, got {}",
                config.reproduction_energy_threshold
            );
        }
        if config.reproduction_energy_cost < 0.0 {
            bail!(
                "reproduction_energy_cost must be non-negative for GPU simulation, got {}",
                config.reproduction_energy_cost
            );
        }
        if config.offspring_initial_energy <= 0.0 {
            bail!(
                "offspring_initial_energy must be positive for GPU simulation, got {}",
                config.offspring_initial_energy
            );
        }
        for (name, value) in [
            ("mutation_rate", config.mutation_rate),
            ("add_node_rate", config.add_node_rate),
            ("add_connection_rate", config.add_connection_rate),
            ("delete_connection_rate", config.delete_connection_rate),
        ] {
            if !(0.0..=1.0).contains(&value) {
                bail!("{name} must be in [0, 1] for GPU simulation, got {value}");
            }
        }
        if config.weight_mutation_power <= 0.0 {
            bail!("weight_mutation_power must be positive for GPU simulation, got {}", config.weight_mutation_power);
        }
        if config.max_age <= 0 {
            bail!("max_age must be positive for GPU simulation, got {}", config.max_age);
        }
        if config.report_interval_ticks <= 0 {
            bail!("report_interval_ticks must be positive for GPU simulation, got {}", config.report_interval_ticks);
        }

        Ok(Self {
            food_capacity: as_non_negative_u32(config.food_count, "food_count")?,
            world_size: config.grid_size as f32,
            predator_speed: config.predator_speed,
            prey_speed: config.prey_speed,
            vision_range: config.vision_range,
            interaction_range: config.interaction_range,
            mate_range: config.mate_range,
            energy_drain_per_tick: config.energy_drain_per_tick,
            energy_gain_from_kill: config.energy_gain_from_kill,
            energy_gain_from_food: config.energy_gain_from_food,
            initial_energy: config.initial_energy,
            max_energy: config.max_energy,
            reproduction_energy_threshold: config.reproduction_energy_threshold,
            reproduction_energy_cost: config.reproduction_energy_cost,
            offspring_initial_energy: config.offspring_initial_energy,
            mutation_rate: config.mutation_rate,
            weight_mutation_power: config.weight_mutation_power,
            add_node_rate: config.add_node_rate,
            add_connection_rate: config.add_connection_rate,
            delete_connection_rate: config.delete_connection_rate,
            max_connection_attempts: PHASE3_MAX_CONNECTION_ATTEMPTS,
            max_age: as_non_negative_u32(config.max_age, "max_age")?,
            report_interval_ticks: as_non_negative_u32(config.report_interval_ticks, "report_interval_ticks")?,
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

    pub fn spatial_grid_state(&self) -> Result<SpatialGridReadback> {
        self.evolution.simulation_spatial_grid_state()
    }

    pub fn reproduction_summary(&self, population_kind: PopulationKind) -> Result<ReproductionSummaryReadback> {
        self.evolution.simulation_reproduction_summary(population_kind)
    }

    pub fn metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        self.evolution.simulation_metrics_summary()
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        self.evolution.simulation_refresh_reports()
    }

    pub fn compact_population(&mut self, population_kind: PopulationKind) -> Result<CompactionSummaryReadback> {
        self.evolution.simulation_compact_population(population_kind)
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

    pub fn render_snapshot(&self, max_predators: u32, max_prey: u32, max_food: u32) -> Result<RenderSnapshotReadback> {
        self.evolution.render_snapshot(max_predators, max_prey, max_food)
    }

    pub fn seeded_agent_snapshot(&self, population_kind: PopulationKind, slot: u32) -> Result<SeededAgentSnapshot> {
        self.evolution.seeded_agent_snapshot(population_kind, slot)
    }

    pub fn population_summary(
        &self,
        population_kind: PopulationKind,
    ) -> Result<crate::tick::buffers::PopulationSummaryReadback> {
        self.evolution.population_summary(population_kind)
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

    use crate::tick::evolution::GpuEvolutionConfig;
    use crate::types::{OUTPUT_COUNT, SENSOR_COUNT};

    fn simulation_config(seed: i32) -> SimulationConfig {
        SimulationConfig {
            grid_size: 64,
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

    fn runtime_ready() -> bool {
        EvolutionManager::runtime_status().is_success()
    }

    fn expected_wall_sensor(negative_side_dist: f32, positive_side_dist: f32, vision_range: f32) -> f32 {
        let negative_in_range = negative_side_dist < vision_range;
        let positive_in_range = positive_side_dist < vision_range;

        if !negative_in_range && !positive_in_range {
            return 0.0;
        }

        if negative_in_range && (!positive_in_range || negative_side_dist <= positive_side_dist) {
            return -(1.0 - (negative_side_dist / vision_range));
        }

        1.0 - (positive_side_dist / vision_range)
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
        assert_eq!(state.spatial_grid_state()?.predator_entries, 4);
        assert_eq!(snapshot.header.total_predators, 4);
        assert_eq!(snapshot.header.total_prey, 6);
        assert_eq!(snapshot.header.total_food, 10);
        assert_eq!(snapshot.predators.len(), 4);
        assert_eq!(snapshot.prey.len(), 6);
        assert_eq!(snapshot.food.len(), 10);

        Ok(())
    }

    #[test]
    fn simulation_config_ffi_roundtrips() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let config = simulation_config(60);
        let evolution_config = GpuEvolutionConfig::for_seed_stage(&config, SENSOR_COUNT as u32, OUTPUT_COUNT as u32)?;
        let simulation_config = GpuSimulationConfig::for_simulation(&config)?;
        let manager = EvolutionManager::create(evolution_config)?;
        let roundtrip = manager.debug_roundtrip_simulation_config(simulation_config)?;

        assert_eq!(roundtrip, simulation_config);
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
        assert_eq!(state_a.spatial_grid_state()?, state_b.spatial_grid_state()?);
        assert_eq!(state_a.metrics_summary()?, state_b.metrics_summary()?);
        assert_eq!(
            state_a.reproduction_summary(PopulationKind::Predator)?,
            state_b.reproduction_summary(PopulationKind::Predator)?
        );
        assert_eq!(
            state_a.reproduction_summary(PopulationKind::Prey)?,
            state_b.reproduction_summary(PopulationKind::Prey)?
        );
        assert_eq!(
            state_a.sensor_snapshot(PopulationKind::Predator, 0)?,
            state_b.sensor_snapshot(PopulationKind::Predator, 0)?
        );
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

    #[test]
    fn simulation_sensor_snapshot_encodes_targets_and_walls() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let state = SimulationState::init_from_config(&simulation_config(64))?;
        let predator = state.seeded_agent_snapshot(PopulationKind::Predator, 0)?;
        let sensors = state.sensor_snapshot(PopulationKind::Predator, 0)?;

        assert_eq!(sensors.population_kind, PopulationKind::Predator);
        assert_eq!(sensors.slot, 0);
        assert_eq!(usize::from(sensors.input_count), SENSOR_COUNT);
        assert!(sensors.inputs[..10].iter().any(|value| value.abs() > f32::EPSILON));
        assert!(sensors.inputs[10..20].iter().any(|value| value.abs() > f32::EPSILON));
        assert!(sensors.inputs[20..30].iter().any(|value| value.abs() > f32::EPSILON));
        assert!((sensors.inputs[30] - 0.5).abs() < 1e-6);
        assert_eq!(sensors.inputs[31], 0.0);
        assert_eq!(sensors.inputs[32], 0.0);
        assert!((sensors.inputs[33] - expected_wall_sensor(predator.pos_x, 64.0 - predator.pos_x, 128.0)).abs() < 1e-6);
        assert!((sensors.inputs[34] - expected_wall_sensor(predator.pos_y, 64.0 - predator.pos_y, 128.0)).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn simulation_builds_spatial_grid_from_vision_range() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let mut config = simulation_config(65);
        config.vision_range = 16.0;
        let state = SimulationState::init_from_config(&config)?;
        let grid = state.spatial_grid_state()?;

        assert_eq!(grid.grid_cols, 4);
        assert_eq!(grid.grid_rows, 4);
        assert_eq!(grid.cell_count, 16);
        assert!((grid.cell_size - 16.0).abs() < 1e-6);
        assert_eq!(grid.predator_entries, 4);
        assert_eq!(grid.prey_entries, 6);
        assert_eq!(grid.food_entries, 10);

        Ok(())
    }

    #[test]
    fn simulation_reproduction_expands_capacity_and_updates_metrics() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let mut config = simulation_config(66);
        config.predator_count = 0;
        config.prey_count = 2;
        config.food_count = 0;
        config.interaction_range = 0.0;
        config.mate_range = 128.0;
        config.energy_drain_per_tick = 0.0;
        config.initial_energy = 1.0;
        config.max_energy = 2.0;
        config.reproduction_energy_threshold = 0.5;
        config.reproduction_energy_cost = 0.1;
        config.offspring_initial_energy = 0.25;
        config.mutation_rate = 0.0;
        config.add_node_rate = 0.0;
        config.add_connection_rate = 0.0;
        config.delete_connection_rate = 0.0;
        config.report_interval_ticks = 1;

        let mut state = SimulationState::init_from_config(&config)?;
        let before = state.population_summary(PopulationKind::Prey)?;
        let stats = state.tick()?;
        let after = state.population_summary(PopulationKind::Prey)?;
        let reproduction = state.reproduction_summary(PopulationKind::Prey)?;
        let metrics = state.metrics_summary()?;

        assert_eq!(before.capacity, 2);
        assert!(after.capacity >= 4);
        assert!(after.live_count >= 3);
        assert_eq!(stats.prey_births, reproduction.births);
        assert!(reproduction.births > 0);
        assert_eq!(metrics.tick, 1);
        assert_eq!(metrics.prey_births, reproduction.births);
        assert_eq!(metrics.prey_count, after.live_count);

        Ok(())
    }

    #[test]
    fn simulation_metrics_summary_refreshes_on_report_interval() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

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

    #[test]
    fn simulation_compaction_rebuilds_dense_population_layout() -> Result<()> {
        if !runtime_ready() {
            return Ok(());
        }

        let mut config = simulation_config(68);
        config.predator_count = 1;
        config.prey_count = 2;
        config.food_count = 0;
        config.interaction_range = 128.0;
        config.mate_range = 0.0;
        config.reproduction_energy_threshold = 2.0;
        config.report_interval_ticks = 1;

        let mut state = SimulationState::init_from_config(&config)?;
        let _ = state.tick()?;
        let before = state.population_summary(PopulationKind::Prey)?;
        let summary = state.compact_population(PopulationKind::Prey)?;
        let after = state.population_summary(PopulationKind::Prey)?;
        let first_slot = state.seeded_agent_snapshot(PopulationKind::Prey, 0)?;

        assert!(before.live_count < before.capacity);
        assert_eq!(summary.previous_capacity, before.capacity);
        assert_eq!(summary.live_count, after.live_count);
        assert_eq!(summary.free_slots_after, after.capacity - after.live_count);
        assert_eq!(summary.compacted, 1);
        assert_eq!(first_slot.alive, 1);

        Ok(())
    }
}
