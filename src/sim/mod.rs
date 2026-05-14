mod helpers;
use crate::sim::helpers::*;

use crate::experiment::SimulationConfig;
use crate::profile_scope;
use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::ptr::{self};

pub const MAX_SPECIES_SUMMARIES: u32 = 64;
pub const SENSOR_COUNT: u32 = 35;
pub const OUTPUT_COUNT: u32 = 2;
const PHASE3_CONNECTION_GROWTH_BUDGET_CAP: u32 = 16;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceGenomeBuffers {
    connection_from: *mut i32,
    connection_to: *mut i32,
    connection_weight: *mut f32,
    connection_innovation: *mut u32,
    connection_enabled: *mut u8,
    node_types: *mut u8,
    num_connections: *mut u16,
    num_nodes: *mut u16,
    connection_stride: u32,
    node_stride: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceCompiledNetworkBuffers {
    eval_order: *mut u16,
    connection_offsets: *mut u32,
    output_indices: *mut u16,
    connection_sources: *mut u16,
    connection_weights: *mut f32,
    node_counts: *mut u16,
    eval_counts: *mut u16,
    connection_counts: *mut u16,
    node_stride: u32,
    connection_stride: u32,
    output_stride: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DevicePopulationBuffers {
    pos_x: *mut f32,
    pos_y: *mut f32,
    vel_x: *mut f32,
    vel_y: *mut f32,
    energy: *mut f32,
    age: *mut f32,
    alive: *mut u8,
    species_id: *mut u32,
    entity_id: *mut u32,
    generation: *mut u32,
    rng_state: *mut u64,
    sensor_inputs: *mut f32,
    genome: DeviceGenomeBuffers,
    compiled: DeviceCompiledNetworkBuffers,
    capacity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceInnovationState {
    next_innovation: u32,
    next_node_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationCounters {
    tick: u32,
    predator_births: u32,
    prey_births: u32,
    predator_deaths: u32,
    prey_deaths: u32,
    kills: u32,
    food_eaten: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationGridEntry {
    slot: u32,
    pos_x: f32,
    pos_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodGridEntry {
    slot: u32,
    pos_x: f32,
    pos_y: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricsReduceScratch {
    predator_energy_sum: f32,
    prey_energy_sum: f32,
    predator_complexity_sum: f32,
    prey_complexity_sum: f32,
    predator_generation_sum: f32,
    prey_generation_sum: f32,
    predator_count: u32,
    prey_count: u32,
    max_predator_generation: u32,
    max_prey_generation: u32,
    predator_species_mask: u64,
    prey_species_mask: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct FoodBuffer {
    pos_x: *mut f32,
    pos_y: *mut f32,
    active: *mut u8,
    capacity: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct DeviceState {
    config: GpuEvolutionConfig,
    simulation: SimulationConfig,
    predator: DevicePopulationBuffers,
    prey: DevicePopulationBuffers,
    food: FoodBuffer,
    innovation: *mut DeviceInnovationState,
    next_entity_id: *mut u32,
    counters: *mut SimulationCounters,
    predator_free_list: *mut u32,
    prey_free_list: *mut u32,
    predator_free_len: *mut u32,
    prey_free_len: *mut u32,
    predator_mate_claims: *mut u32,
    prey_mate_claims: *mut u32,
    predator_reproduction_pairs: *mut ReproductionPairReadback,
    prey_reproduction_pairs: *mut ReproductionPairReadback,
    predator_pair_count: *mut u32,
    prey_pair_count: *mut u32,
    population_live_count_scratch: *mut u32,
    ui_stats_scratch: *mut UiStatsReadback,
    free_list_state_scratch: *mut FreeListStateReadback,
    sensor_snapshot_scratch: *mut SensorSnapshotReadback,
    compiled_header_scratch: *mut CompiledNetworkReadbackHeader,
    selected_network_scratch: *mut SelectedAgentNetworkReadback,
    metrics_summary: *mut MetricsSummaryReadback,
    metrics_reduce_scratch: *mut MetricsReduceScratch,
    species_summaries_scratch: *mut SpeciesSummaryReadback,
    representative_headers_scratch: *mut RepresentativeGenomeHeader,
    species_count_scratch: *mut u32,
    render_header_scratch: *mut RenderSnapshotHeader,
    render_predators_scratch: *mut RenderAgentReadback,
    render_prey_scratch: *mut RenderAgentReadback,
    render_food_scratch: *mut RenderFoodReadback,
    predator_cell_counts: *mut u32,
    predator_cell_offsets: *mut u32,
    predator_cell_write_offsets: *mut u32,
    predator_grid_entries: *mut PopulationGridEntry,
    prey_cell_counts: *mut u32,
    prey_cell_offsets: *mut u32,
    prey_cell_write_offsets: *mut u32,
    prey_grid_entries: *mut PopulationGridEntry,
    food_cell_counts: *mut u32,
    food_cell_offsets: *mut u32,
    food_cell_write_offsets: *mut u32,
    food_grid_entries: *mut FoodGridEntry,
    food_claimed_by: *mut u32,
    prey_claimed_by: *mut u32,
    grid_cols: u32,
    grid_rows: u32,
    grid_cell_capacity: u32,
    grid_cell_size: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSummaryReadback {
    pub population_kind: PopulationKind,
    pub species_id: u32,
    pub size: u32,
    pub representative_slot: u32,
    pub avg_complexity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepresentativeGenomeHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub generation: u32,
    pub species_id: u32,
    pub num_nodes: u16,
    pub num_connections: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenomeNodeReadback {
    pub id: u32,
    pub node_type: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GenomeConnectionReadback {
    pub from_node: i32,
    pub to_node: i32,
    pub weight: f32,
    pub innovation: u32,
    pub enabled: u8,
    pub reserved0: u8,
    pub reserved1: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentativeGenomeReadback {
    pub header: RepresentativeGenomeHeader,
    pub nodes: Vec<GenomeNodeReadback>,
    pub connections: Vec<GenomeConnectionReadback>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeciesBatchReadbackHeader {
    pub population_kind: PopulationKind,
    pub species_count: u32,
    pub returned_species_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreeListStateReadback {
    pub tick: u32,
    pub predator_free_slots: u32,
    pub prey_free_slots: u32,
    pub active_food_count: u32,
    pub food_capacity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricsSummaryReadback {
    pub tick: u32,
    pub predator_count: u32,
    pub prey_count: u32,
    pub predator_births: u32,
    pub prey_births: u32,
    pub predator_deaths: u32,
    pub prey_deaths: u32,
    pub predator_species: u32,
    pub prey_species: u32,
    pub avg_predator_complexity: f32,
    pub avg_prey_complexity: f32,
    pub avg_predator_energy: f32,
    pub avg_prey_energy: f32,
    pub max_predator_generation: u32,
    pub avg_predator_generation: f32,
    pub max_prey_generation: u32,
    pub avg_prey_generation: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UiStatsReadback {
    pub tick: u32,
    pub predator_count: u32,
    pub prey_count: u32,
    pub predator_births: u32,
    pub prey_births: u32,
    pub predator_deaths: u32,
    pub prey_deaths: u32,
    pub kills: u32,
    pub food_eaten: u32,
    pub avg_predator_energy: f32,
    pub avg_prey_energy: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderSnapshotHeader {
    pub tick: u32,
    pub total_predators: u32,
    pub total_prey: u32,
    pub total_food: u32,
    pub returned_predators: u32,
    pub returned_prey: u32,
    pub returned_food: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderAgentReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub species_id: u32,
    pub generation: u32,
    pub age: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub dir_x: f32,
    pub dir_y: f32,
    pub energy: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderFoodReadback {
    pub slot: u32,
    pub active: u8,
    pub reserved0: u8,
    pub reserved1: u16,
    pub pos_x: f32,
    pub pos_y: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderSnapshotReadback {
    pub header: RenderSnapshotHeader,
    pub predators: Vec<RenderAgentReadback>,
    pub prey: Vec<RenderAgentReadback>,
    pub food: Vec<RenderFoodReadback>,
}

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

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct GpuEvolutionConfig {
    pub predator_capacity: u32,
    pub prey_capacity: u32,
    pub initial_predator_count: u32,
    pub initial_prey_count: u32,
    pub world_size: f32,
    pub initial_energy: f32,
    pub max_energy: f32,
    pub seed: u64,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub node_stride: u32,
    pub connection_stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMutationConfig {
    pub mutation_rate: f32,
    pub weight_mutation_power: f32,
    pub add_node_rate: f32,
    pub add_connection_rate: f32,
    pub delete_connection_rate: f32,
    pub max_connection_attempts: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReproductionPairReadback {
    pub parent_a_slot: u32,
    pub parent_b_slot: u32,
}

fn readback<T>(context: &str, mut call: impl FnMut(*mut T) -> i32) -> Result<T> {
    let mut out = MaybeUninit::<T>::uninit();
    check_cuda_status(call(out.as_mut_ptr()), context)?;
    // SAFETY: A successful readback call fully initializes the output struct.
    Ok(unsafe { out.assume_init() })
}

pub struct Simulation {
    pub config: SimulationConfig,
    device_state: DeviceState,
    predator_capacity: u32,
    prey_capacity: u32,
}

impl Drop for Simulation {
    fn drop(&mut self) {
        match self.clean_buffers() {
            Ok(()) => (),
            _ => (),
        }
    }
}

impl Simulation {
    pub fn init(config: &SimulationConfig) -> Result<Self> {
        let mut simulation = Simulation { device_state: DeviceState::default(), predator_capacity: config.predator_count, prey_capacity: config.prey_count, config: *config };
        simulation.init_dev()?;
        Ok(simulation)
    }

    fn init_dev(&mut self) -> Result<()> {
        self.device_state.simulation = self.config;
        self.device_state.grid_cell_size = self.config.vision_range.max(1.0);
        self.device_state.grid_cols = ((self.config.grid_size / self.device_state.grid_cell_size).ceil() as u32).max(1);
        self.device_state.grid_rows = ((self.config.grid_size / self.device_state.grid_cell_size).ceil() as u32).max(1);

        let hidden_budget = self.config.max_hidden_nodes;

        let seeded_node_count = SENSOR_COUNT
            .checked_add(OUTPUT_COUNT)
            .and_then(|value| value.checked_add(1))
            .context("seed-stage node stride overflowed")?;
        let seeded_connection_count = SENSOR_COUNT
            .checked_add(1)
            .and_then(|value| value.checked_mul(OUTPUT_COUNT))
            .context("seed-stage connection stride overflowed")?;
        let extra_connection_capacity = hidden_budget.saturating_mul(2).clamp(4, PHASE3_CONNECTION_GROWTH_BUDGET_CAP);

        let node_stride = seeded_node_count.checked_add(hidden_budget).context("phase-3 node stride overflowed")?;
        let connection_stride = seeded_connection_count
            .checked_add(extra_connection_capacity)
            .context("phase-3 connection stride overflowed")?;

        self.device_state.config = GpuEvolutionConfig {
            predator_capacity: self.config.predator_count,
            prey_capacity: self.config.prey_count,
            initial_predator_count: self.config.predator_count,
            initial_prey_count: self.config.prey_count,
            world_size: self.config.grid_size,
            initial_energy: self.config.initial_energy,
            max_energy: self.config.max_energy,
            seed: self.config.seed,
            num_inputs: SENSOR_COUNT,
            num_outputs: OUTPUT_COUNT,
            node_stride,
            connection_stride,
        };

        check_cuda_status(unsafe { dev_create(self.get_dev_state()) } , "dev_evolution_create")?;
        check_cuda_status(unsafe { dev_seed_initial_population(self.get_dev_state()) }, "dev_seed_initial_population")?;
        check_cuda_status(unsafe { dev_ensure_food_buffer(self.get_dev_state()) }, "dev_ensure_food_buffer")?;
        check_cuda_status(unsafe { dev_ensure_counter_buffer(self.get_dev_state()) }, "dev_ensure_counter_buffer")?;
        check_cuda_status(unsafe { dev_ensure_free_lists(self.get_dev_state()) }, "dev_ensure_free_lists")?;
        check_cuda_status(unsafe { dev_ensure_reproduction_buffers(self.get_dev_state()) }, "dev_ensure_reproduction_buffers")?;
        check_cuda_status(unsafe { dev_ensure_metrics_buffer(self.get_dev_state()) }, "dev_ensure_metrics_buffer")?;
        check_cuda_status(unsafe { dev_ensure_spatial_grid_buffers(self.get_dev_state()) }, "dev_ensure_spatial_grid_buffers")?;
        check_cuda_status(unsafe { dev_reset_counters(self.get_dev_state()) }, "dev_reset_counters")?;
        check_cuda_status(unsafe { dev_initialize_free_lists(self.get_dev_state()) }, "dev_initialize_free_lists")?;
        check_cuda_status(unsafe { dev_seed_food(self.get_dev_state()) }, "dev_seed_food")?;
        check_cuda_status(unsafe { dev_reset_reproduction_state(self.get_dev_state()) }, "dev_reset_reproduction_state")?;

        if self.predator_capacity > 0 {
            let _ = self.compile_population(PopulationKind::Predator, 0)?;
        }
        if self.prey_capacity > 0 {
            let _ = self.compile_population(PopulationKind::Prey, 0)?;
        }

        self.refresh_reports()?;

        Ok(())
    }

    pub fn tick(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("tick");

        self.build_spatial_grid()?;
        self.compute_sensor_inputs()?;
        self.infer_population(PopulationKind::Predator)?;
        self.infer_population(PopulationKind::Prey)?;
        self.update_vitals(PopulationKind::Predator)?;
        self.update_vitals(PopulationKind::Prey)?;
        self.apply_movement(PopulationKind::Predator)?;
        self.apply_movement(PopulationKind::Prey)?;
        self.build_spatial_grid()?;
        self.resolve_food()?;
        self.resolve_combat()?;
        let predator_births = self.reproduction_candidate_count(PopulationKind::Predator)?;
        self.ensure_birth_capacity(PopulationKind::Predator, predator_births)?;
        self.run_reproduction(PopulationKind::Predator)?;
        let prey_births = self.reproduction_candidate_count(PopulationKind::Prey)?;
        self.ensure_birth_capacity(PopulationKind::Prey, prey_births)?;
        self.run_reproduction(PopulationKind::Prey)?;
        self.advance_tick()?;

        let ui_stats = self.ui_stats()?;
        if self.config.report_interval_ticks > 0 && ui_stats.tick % self.config.report_interval_ticks == 0 {
            self.refresh_reports()?;
        }
        Ok(ui_stats)
    }

    pub fn ui_stats(&mut self) -> Result<UiStatsReadback> {
        profile_scope!("ui_stats");
        readback("moonai_gpu_simulation_ui_stats", |out| { unsafe { dev_ui_stats(self.get_dev_state(), out) } })
    }

    pub fn free_list_state(&mut self) -> Result<FreeListStateReadback> {
        readback("moonai_gpu_simulation_free_list_state", |out| { unsafe { dev_free_list_state(self.get_dev_state(), out) } })
    }

    pub fn metrics_summary(&mut self) -> Result<MetricsSummaryReadback> {
        readback("moonai_gpu_simulation_metrics_summary", |out| { unsafe { dev_metrics_summary(self.get_dev_state(), out) } })
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        profile_scope!("refresh_reports");
        let status = unsafe { dev_refresh_reports(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_refresh_reports")
    }

    pub fn sensor_snapshot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<SensorSnapshotReadback> {
        readback("moonai_gpu_simulation_sensor_snapshot", |out| {
            unsafe { dev_sensor_snapshot(self.get_dev_state(), population_kind, slot, out) }
        })
    }

    pub fn render_snapshot(&mut self, max_predators: u32, max_prey: u32, max_food: u32) -> Result<RenderSnapshotReadback> {
        let predator_capacity = usize::try_from(max_predators).context("predator render capacity overflowed")?;
        let prey_capacity = usize::try_from(max_prey).context("prey render capacity overflowed")?;
        let food_capacity = usize::try_from(max_food).context("food render capacity overflowed")?;
        let mut header = MaybeUninit::<RenderSnapshotHeader>::uninit();
        let empty_predator = RenderAgentReadback {
            population_kind: PopulationKind::Predator,
            slot: 0,
            entity_id: 0,
            species_id: 0,
            generation: 0,
            age: 0.0,
            pos_x: 0.0,
            pos_y: 0.0,
            dir_x: 0.0,
            dir_y: 0.0,
            energy: 0.0,
        };
        let empty_prey = RenderAgentReadback { population_kind: PopulationKind::Prey, ..empty_predator };
        let empty_food = RenderFoodReadback { slot: 0, active: 0, reserved0: 0, reserved1: 0, pos_x: 0.0, pos_y: 0.0 };
        let mut predators = vec![empty_predator; predator_capacity];
        let mut prey = vec![empty_prey; prey_capacity];
        let mut food = vec![empty_food; food_capacity];
        let predators_ptr = if predators.is_empty() { ptr::null_mut() } else { predators.as_mut_ptr() };
        let prey_ptr = if prey.is_empty() { ptr::null_mut() } else { prey.as_mut_ptr() };
        let food_ptr = if food.is_empty() { ptr::null_mut() } else { food.as_mut_ptr() };
        // SAFETY: `self.raw` is valid and each non-null output pointer refers to a buffer with the requested capacity.
        let status = unsafe {
            dev_render_snapshot(
                self.get_dev_state(),
                max_predators,
                max_prey,
                max_food,
                header.as_mut_ptr(),
                predators_ptr,
                prey_ptr,
                food_ptr,
            )
        };
        check_cuda_status(status, "moonai_gpu_simulation_render_snapshot")?;
        // SAFETY: A successful render-snapshot call initializes the header.
        let header = unsafe { header.assume_init() };
        let returned_predators =
            usize::try_from(header.returned_predators).context("predator render length overflowed")?;
        let returned_prey = usize::try_from(header.returned_prey).context("prey render length overflowed")?;
        let returned_food = usize::try_from(header.returned_food).context("food render length overflowed")?;
        if returned_predators > predators.len() || returned_prey > prey.len() || returned_food > food.len() {
            bail!(
                "render snapshot returned more entries than allocated: predators {} / {}, prey {} / {}, food {} / {}",
                returned_predators,
                predators.len(),
                returned_prey,
                prey.len(),
                returned_food,
                food.len()
            );
        }
        predators.truncate(returned_predators);
        prey.truncate(returned_prey);
        food.truncate(returned_food);
        Ok(RenderSnapshotReadback { header, predators, prey, food })
    }

    pub fn selected_agent_network(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        readback("moonai_gpu_evolution_selected_agent_network", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact network readback.
            unsafe { dev_selected_agent_network(self.get_dev_state(), population_kind, slot, out) }
        })
    }

    pub fn species_summaries(
        &mut self,
        population_kind: PopulationKind,
        max_species: u32,
    ) -> Result<(SpeciesBatchReadbackHeader, Vec<SpeciesSummaryReadback>, Vec<RepresentativeGenomeHeader>)> {
        let species_capacity = usize::try_from(max_species).context("species summary capacity overflowed")?;
        let mut header = MaybeUninit::<SpeciesBatchReadbackHeader>::uninit();
        let empty_summary = SpeciesSummaryReadback {
            population_kind,
            species_id: 0,
            size: 0,
            representative_slot: 0,
            avg_complexity: 0.0,
        };
        let empty_representative = RepresentativeGenomeHeader {
            population_kind,
            slot: 0,
            entity_id: 0,
            generation: 0,
            species_id: 0,
            num_nodes: 0,
            num_connections: 0,
        };
        let mut summaries = vec![empty_summary; species_capacity];
        let mut representatives = vec![empty_representative; species_capacity];
        let summaries_ptr = if summaries.is_empty() { ptr::null_mut() } else { summaries.as_mut_ptr() };
        let representatives_ptr =
            if representatives.is_empty() { ptr::null_mut() } else { representatives.as_mut_ptr() };
        // SAFETY: `self.raw` is valid and the summary/representative buffers each provide `max_species` slots when non-null.
        let status = unsafe {
            dev_species_summaries(
                self.get_dev_state(),
                population_kind,
                max_species,
                header.as_mut_ptr(),
                summaries_ptr,
                representatives_ptr,
            )
        };
        check_cuda_status(status, "moonai_gpu_evolution_species_summaries")?;
        // SAFETY: A successful species-batch call initializes the header.
        let header = unsafe { header.assume_init() };
        let returned_len =
            usize::try_from(header.returned_species_count).context("species summary returned length overflowed")?;
        if returned_len > summaries.len() || returned_len > representatives.len() {
            bail!(
                "species summary returned {} entries but the host buffers only allocated {} summary slots and {} representative slots",
                returned_len,
                summaries.len(),
                representatives.len()
            );
        }
        summaries.truncate(returned_len);
        representatives.truncate(returned_len);
        Ok((header, summaries, representatives))
    }

    pub fn representative_genome(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        let header = readback("moonai_gpu_evolution_representative_genome_header", |out| unsafe {
            dev_representative_genome_header(self.get_dev_state(), population_kind, slot, out)
        })?;
        let returned_nodes = usize::from(header.num_nodes);
        let returned_connections = usize::from(header.num_connections);

        let mut node_types = vec![0_u8; returned_nodes];
        if !node_types.is_empty() {
            let status = unsafe {
                dev_representative_genome_node_types(
                    self.get_dev_state(),
                    population_kind,
                    slot,
                    header.num_nodes.into(),
                    node_types.as_mut_ptr(),
                )
            };
            check_cuda_status(status, "moonai_gpu_evolution_representative_genome_node_types")?;
        }

        let nodes = node_types
            .into_iter()
            .enumerate()
            .map(|(id, node_type)| GenomeNodeReadback { id: id as u32, node_type, reserved0: 0, reserved1: 0 })
            .collect();

        let mut from_nodes = vec![0_i32; returned_connections];
        let mut to_nodes = vec![0_i32; returned_connections];
        let mut weights = vec![0.0_f32; returned_connections];
        let mut innovations = vec![0_u32; returned_connections];
        let mut enabled_flags = vec![0_u8; returned_connections];
        if returned_connections > 0 {
            let status = unsafe {
                dev_representative_genome_connections(
                    self.get_dev_state(),
                    population_kind,
                    slot,
                    header.num_connections.into(),
                    from_nodes.as_mut_ptr(),
                    to_nodes.as_mut_ptr(),
                    weights.as_mut_ptr(),
                    innovations.as_mut_ptr(),
                    enabled_flags.as_mut_ptr(),
                )
            };
            check_cuda_status(status, "moonai_gpu_evolution_representative_genome_connections")?;
        }

        let connections = from_nodes
            .into_iter()
            .zip(to_nodes)
            .zip(weights)
            .zip(innovations)
            .zip(enabled_flags)
            .map(|((((from_node, to_node), weight), innovation), enabled)| GenomeConnectionReadback {
                from_node,
                to_node,
                weight,
                innovation,
                enabled,
                reserved0: 0,
                reserved1: 0,
            })
            .collect();

        Ok(RepresentativeGenomeReadback { header, nodes, connections })
    }

    fn ensure_birth_capacity(&mut self, population_kind: PopulationKind, births_pending: u32) -> Result<()> {
        profile_scope!("birth_cap");

        if births_pending == 0 {
            return Ok(());
        }

        let live_count = self.population_live_count(population_kind)?;
        let free_list_state = self.free_list_state()?;
        let free_slots = match population_kind {
            PopulationKind::Predator => free_list_state.predator_free_slots,
            PopulationKind::Prey => free_list_state.prey_free_slots,
        };
        let capacity = self.population_capacity(population_kind);
        let required_live = live_count.saturating_add(births_pending);
        if free_slots >= births_pending && required_live <= ((capacity * 9) / 10) {
            return Ok(());
        }

        let mut new_capacity = if capacity == 0 { 1 } else { capacity };
        while new_capacity.saturating_sub(live_count) < births_pending || required_live > ((new_capacity * 9) / 10) {
            new_capacity = if new_capacity == 0 { 1 } else { new_capacity.saturating_mul(2) };
        }
        self.expand_population(population_kind, new_capacity)?;
        self.build_spatial_grid()?;
        Ok(())
    }

    fn build_spatial_grid(&mut self) -> Result<()> {
        profile_scope!("spatial_grid");
        let status = unsafe { dev_build_spatial_grid(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_build_spatial_grid")
    }

    fn compute_sensor_inputs(&mut self) -> Result<()> {
        profile_scope!("sensor_inputs");
        let status = unsafe { dev_compute_sensor_inputs(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_compute_sensor_inputs")
    }

    fn infer_population(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("inference");
        let status = unsafe { dev_infer_population(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_infer_population")
    }

    fn update_vitals(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("update_vitals");
        let status = unsafe { dev_update_vitals(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_update_vitals")
    }

    fn resolve_food(&mut self) -> Result<()> {
        profile_scope!("resolve_food");
        let status = unsafe { dev_resolve_food(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_food")
    }

    fn resolve_combat(&mut self) -> Result<()> {
        profile_scope!("resolve_combat");
        let status = unsafe { dev_resolve_combat(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_combat")
    }

    fn apply_movement(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("apply_movement");
        let status = unsafe { dev_apply_movement(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_apply_movement")
    }

    fn reproduction_candidate_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        profile_scope!("reprod_candidate");
        readback("moonai_gpu_simulation_reproduction_candidate_count", |out| unsafe {
            dev_reproduction_candidate_count(self.get_dev_state(), population_kind, out)
        })
    }

    fn expand_population(&mut self, population_kind: PopulationKind, new_capacity: u32) -> Result<()> {
        let status =
            unsafe { dev_expand_population(self.get_dev_state(), population_kind, new_capacity) };
        check_cuda_status(status, "moonai_gpu_simulation_expand_population")?;
        match population_kind {
            PopulationKind::Predator => self.predator_capacity = new_capacity,
            PopulationKind::Prey => self.prey_capacity = new_capacity,
        }
        Ok(())
    }

    fn run_reproduction(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("run_reprod");
        let pair_count = self.reproduction_candidate_count(population_kind)?;
        if pair_count == 0 {
            return Ok(());
        }

        let free_slots = self.reproduction_free_slots(population_kind, pair_count)?;
        if free_slots.is_empty() {
            return Ok(());
        }

        let pairs = self.reproduction_pairs(population_kind, pair_count)?;
        let mutation_config = self.mutation_config()?;
        let births_applied = pairs.len().min(free_slots.len());
        for index in 0..births_applied {
            let pair = pairs[index];
            let offspring_slot = free_slots[index];
            self.crossover_slot(population_kind, pair.parent_a_slot, pair.parent_b_slot, offspring_slot)?;
            self.mutate_slot(population_kind, offspring_slot, mutation_config)?;
            let _ = self.compile_slot(population_kind, offspring_slot)?;
        }

        self.apply_reproduction_energy(population_kind, births_applied as u32)
    }

    fn advance_tick(&mut self) -> Result<()> {
        profile_scope!("advance_tick");
        let status = unsafe { dev_advance_tick(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_advance_tick")
    }

    fn population_live_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        readback("moonai_gpu_evolution_population_live_count", |out| unsafe {
            dev_population_live_count(self.get_dev_state(), population_kind, out)
        })
    }

    const fn population_capacity(&self, population_kind: PopulationKind) -> u32 {
        match population_kind {
            PopulationKind::Predator => self.predator_capacity,
            PopulationKind::Prey => self.prey_capacity,
        }
    }

    fn compile_population(
        &mut self,
        population_kind: PopulationKind,
        inspected_slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        readback("moonai_gpu_evolution_compile_population", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compile readback header.
            unsafe { dev_compile_population(self.get_dev_state(), population_kind, inspected_slot, out) }
        })
    }

    fn compile_slot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<CompiledNetworkReadbackHeader> {
        readback("moonai_gpu_evolution_compile_slot", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compile readback header.
            unsafe { dev_compile_slot(self.get_dev_state(), population_kind, slot, out) }
        })
    }

    fn reproduction_pairs(
        &mut self,
        population_kind: PopulationKind,
        pair_count: u32,
    ) -> Result<Vec<ReproductionPairReadback>> {
        let capacity = usize::try_from(pair_count).context("reproduction pair capacity overflowed")?;
        let mut pairs = vec![ReproductionPairReadback { parent_a_slot: 0, parent_b_slot: 0 }; capacity];
        let mut returned_pairs = 0_u32;
        let pairs_ptr = if pairs.is_empty() { ptr::null_mut() } else { pairs.as_mut_ptr() };
        let status = unsafe {
            dev_read_reproduction_pairs(
                self.get_dev_state(),
                population_kind,
                pair_count,
                pairs_ptr,
                &mut returned_pairs,
            )
        };
        check_cuda_status(status, "moonai_gpu_simulation_read_reproduction_pairs")?;
        let returned_pairs = usize::try_from(returned_pairs).context("reproduction pair length overflowed")?;
        if returned_pairs > pairs.len() {
            bail!(
                "reproduction pair readback returned {} entries but the host buffer only allocated {}",
                returned_pairs,
                pairs.len()
            );
        }
        pairs.truncate(returned_pairs);
        Ok(pairs)
    }

    fn reproduction_free_slots(&mut self, population_kind: PopulationKind, slot_count: u32) -> Result<Vec<u32>> {
        let capacity = usize::try_from(slot_count).context("free-slot capacity overflowed")?;
        let mut slots = vec![0_u32; capacity];
        let mut returned_slots = 0_u32;
        let slots_ptr = if slots.is_empty() { ptr::null_mut() } else { slots.as_mut_ptr() };
        let status = unsafe {
            dev_read_free_slots(
                self.get_dev_state(),
                population_kind,
                slot_count,
                slots_ptr,
                &mut returned_slots,
            )
        };
        check_cuda_status(status, "moonai_gpu_simulation_read_free_slots")?;
        let returned_slots = usize::try_from(returned_slots).context("free-slot length overflowed")?;
        if returned_slots > slots.len() {
            bail!(
                "free-slot readback returned {} entries but the host buffer only allocated {}",
                returned_slots,
                slots.len()
            );
        }
        slots.truncate(returned_slots);
        Ok(slots)
    }

    const fn mutation_config(&self) -> Result<GpuMutationConfig> {
        Ok(GpuMutationConfig {
            mutation_rate: self.config.mutation_rate,
            weight_mutation_power: self.config.weight_mutation_power,
            add_node_rate: self.config.add_node_rate,
            add_connection_rate: self.config.add_connection_rate,
            delete_connection_rate: self.config.delete_connection_rate,
            max_connection_attempts: 16,
        })
    }

    fn crossover_slot(
        &mut self,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> Result<()> {
        let status = unsafe {
            dev_crossover( self.get_dev_state(), population_kind, parent_a_slot, parent_b_slot, offspring_slot)
        };
        check_cuda_status(status, "moonai_gpu_evolution_crossover")
    }

    fn mutate_slot(&mut self, population_kind: PopulationKind, slot: u32, config: GpuMutationConfig) -> Result<()> {
        let status = unsafe { dev_mutate_slot(self.get_dev_state(), population_kind, slot, &config) };
        check_cuda_status(status, "moonai_gpu_evolution_mutate_slot")
    }

    fn apply_reproduction_energy(&mut self, population_kind: PopulationKind, births_applied: u32) -> Result<()> {
        let status = unsafe {
            dev_apply_reproduction_energy(self.get_dev_state(), population_kind, births_applied)
        };
        check_cuda_status(status, "moonai_gpu_simulation_apply_reproduction_energy")
    }

    const unsafe fn get_dev_state(&mut self) -> *mut DeviceState {
        &mut self.device_state as *mut DeviceState
    }

    fn clean_buffers(&mut self) -> Result<()> {
        cuda_free(&mut self.device_state.food.pos_x)?;
        cuda_free(&mut self.device_state.food.pos_y)?;
        cuda_free(&mut self.device_state.food.active)?;
        self.device_state.food.capacity = 0;

        cuda_free(&mut self.device_state.innovation)?;
        cuda_free(&mut self.device_state.next_entity_id)?;
        cuda_free(&mut self.device_state.counters)?;
        cuda_free(&mut self.device_state.predator_free_list)?;
        cuda_free(&mut self.device_state.prey_free_list)?;
        cuda_free(&mut self.device_state.predator_free_len)?;
        cuda_free(&mut self.device_state.prey_free_len)?;

        cuda_free(&mut self.device_state.population_live_count_scratch)?;
        cuda_free(&mut self.device_state.ui_stats_scratch)?;
        cuda_free(&mut self.device_state.free_list_state_scratch)?;
        cuda_free(&mut self.device_state.sensor_snapshot_scratch)?;
        cuda_free(&mut self.device_state.compiled_header_scratch)?;
        cuda_free(&mut self.device_state.selected_network_scratch)?;
        cuda_free(&mut self.device_state.metrics_summary)?;
        cuda_free(&mut self.device_state.metrics_reduce_scratch)?;
        cuda_free(&mut self.device_state.species_summaries_scratch)?;
        cuda_free(&mut self.device_state.representative_headers_scratch)?;
        cuda_free(&mut self.device_state.species_count_scratch)?;
        cuda_free(&mut self.device_state.render_header_scratch)?;

        Ok(())
    }
}

unsafe extern "C" {
    fn dev_create(out_state: *mut DeviceState) -> i32;
    fn dev_seed_initial_population(state: *mut DeviceState) -> i32;
    fn dev_population_live_count( state: *mut DeviceState, population_kind: PopulationKind, out_live_count: *mut u32,) -> i32;
    fn dev_ensure_food_buffer(state: *mut DeviceState) -> i32;
    fn dev_ensure_counter_buffer(state: *mut DeviceState) -> i32;
    fn dev_ensure_free_lists(state: *mut DeviceState) -> i32;
    fn dev_ensure_reproduction_buffers(state: *mut DeviceState) -> i32;
    fn dev_ensure_metrics_buffer(state: *mut DeviceState) -> i32;
    fn dev_ensure_spatial_grid_buffers(state: *mut DeviceState) -> i32;
    fn dev_reset_counters(state: *mut DeviceState) -> i32;
    fn dev_initialize_free_lists(state: *mut DeviceState) -> i32;
    fn dev_seed_food(state: *mut DeviceState) -> i32;
    fn dev_reset_reproduction_state(state: *mut DeviceState) -> i32;
    fn dev_build_spatial_grid(state: *mut DeviceState) -> i32;
    fn dev_compute_sensor_inputs(state: *mut DeviceState) -> i32;
    fn dev_infer_population(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_update_vitals(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_resolve_food(state: *mut DeviceState) -> i32;
    fn dev_resolve_combat(state: *mut DeviceState) -> i32;
    fn dev_apply_movement(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_reproduction_candidate_count( state: *mut DeviceState, population_kind: PopulationKind, out_pair_count: *mut u32,) -> i32;
    fn dev_read_reproduction_pairs( state: *mut DeviceState, population_kind: PopulationKind, max_pairs: u32, out_pairs: *mut ReproductionPairReadback, out_returned_pairs: *mut u32,) -> i32;
    fn dev_read_free_slots( state: *mut DeviceState, population_kind: PopulationKind, max_slots: u32, out_slots: *mut u32, out_returned_slots: *mut u32,) -> i32;
    fn dev_apply_reproduction_energy( state: *mut DeviceState, population_kind: PopulationKind, births_applied: u32,) -> i32;
    fn dev_expand_population( state: *mut DeviceState, population_kind: PopulationKind, new_capacity: u32,) -> i32;
    fn dev_advance_tick(state: *mut DeviceState) -> i32;
    fn dev_ui_stats(state: *mut DeviceState, out_stats: *mut UiStatsReadback) -> i32;
    fn dev_free_list_state( state: *mut DeviceState, out_state: *mut FreeListStateReadback,) -> i32;
    fn dev_metrics_summary( state: *mut DeviceState, out_summary: *mut MetricsSummaryReadback,) -> i32;
    fn dev_refresh_reports(state: *mut DeviceState) -> i32;
    fn dev_sensor_snapshot( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, out_snapshot: *mut SensorSnapshotReadback,) -> i32;
    fn dev_render_snapshot( state: *mut DeviceState, max_predators: u32, max_prey: u32, max_food: u32, out_header: *mut RenderSnapshotHeader, out_predators: *mut RenderAgentReadback, out_prey: *mut RenderAgentReadback, out_food: *mut RenderFoodReadback,) -> i32;
    fn dev_compile_population( state: *mut DeviceState, population_kind: PopulationKind, inspected_slot: u32, out_header: *mut CompiledNetworkReadbackHeader,) -> i32;
    fn dev_compile_slot( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, out_header: *mut CompiledNetworkReadbackHeader,) -> i32;
    fn dev_crossover( state: *mut DeviceState, population_kind: PopulationKind, parent_a_slot: u32, parent_b_slot: u32, offspring_slot: u32,) -> i32;
    fn dev_mutate_slot( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, config: *const GpuMutationConfig,) -> i32;
    fn dev_selected_agent_network( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, out_network: *mut SelectedAgentNetworkReadback,) -> i32;
    fn dev_species_summaries( state: *mut DeviceState, population_kind: PopulationKind, max_species: u32, out_header: *mut SpeciesBatchReadbackHeader, out_summaries: *mut SpeciesSummaryReadback, out_representatives: *mut RepresentativeGenomeHeader,) -> i32;
    fn dev_representative_genome_header( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, out_header: *mut RepresentativeGenomeHeader,) -> i32;
    fn dev_representative_genome_node_types( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, max_nodes: u32, out_node_types: *mut u8,) -> i32;
    fn dev_representative_genome_connections( state: *mut DeviceState, population_kind: PopulationKind, slot: u32, max_connections: u32, out_from_nodes: *mut i32, out_to_nodes: *mut i32, out_weights: *mut f32, out_innovations: *mut u32, out_enabled_flags: *mut u8,) -> i32;
}
