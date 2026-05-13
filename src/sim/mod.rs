use crate::experiment::SimulationConfig;
use crate::profile_scope;
use anyhow::{Context as _, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

pub const MAX_SPECIES_SUMMARIES: u32 = 64;
pub const SENSOR_COUNT: u32 = 35;
pub const OUTPUT_COUNT: u32 = 2;
const PHASE3_CONNECTION_GROWTH_BUDGET_CAP: u32 = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodBuffer {
  pos_x: *mut f32,
  pos_y: *mut f32,
  active: *mut u8,
  capacity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuEvolutionState {
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

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStatus {
    Success = 0,
    InvalidArgument = 1,
    AllocationFailed = 2,
    KernelLaunchFailed = 3,
    DeviceCopyFailed = 4,
    RuntimeUnavailable = 5,
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[repr(C)]
struct GpuEvolutionStateHandle {
    _private: [u8; 0],
}

impl GpuEvolutionConfig {
    pub fn for_seed_stage(simulation: &SimulationConfig, num_inputs: u32, num_outputs: u32) -> Result<Self> {
        let predator_capacity = simulation.predator_count;
        let prey_capacity = simulation.prey_count;
        let hidden_budget = simulation.max_hidden_nodes;

        let seeded_node_count = num_inputs
            .checked_add(num_outputs)
            .and_then(|value| value.checked_add(1))
            .context("seed-stage node stride overflowed")?;
        let seeded_connection_count = num_inputs
            .checked_add(1)
            .and_then(|value| value.checked_mul(num_outputs))
            .context("seed-stage connection stride overflowed")?;
        let extra_connection_capacity = hidden_budget.saturating_mul(2).clamp(4, PHASE3_CONNECTION_GROWTH_BUDGET_CAP);

        let node_stride = seeded_node_count.checked_add(hidden_budget).context("phase-3 node stride overflowed")?;
        let connection_stride = seeded_connection_count
            .checked_add(extra_connection_capacity)
            .context("phase-3 connection stride overflowed")?;

        Ok(Self {
            predator_capacity,
            prey_capacity,
            initial_predator_count: predator_capacity,
            initial_prey_count: prey_capacity,
            world_size: simulation.grid_size,
            initial_energy: simulation.initial_energy,
            max_energy: simulation.max_energy,
            seed: simulation.seed,
            num_inputs,
            num_outputs,
            node_stride,
            connection_stride,
        })
    }
}

fn readback<T>(context: &str, mut call: impl FnMut(*mut T) -> CudaStatus) -> Result<T> {
    let mut out = MaybeUninit::<T>::uninit();
    check_cuda_status(call(out.as_mut_ptr()), context)?;
    // SAFETY: A successful readback call fully initializes the output struct.
    Ok(unsafe { out.assume_init() })
}

pub fn runtime_status() -> CudaStatus {
    // SAFETY: This entrypoint performs an internal CUDA runtime probe and returns a POD status code.
    unsafe { moonai_gpu_runtime_available() }
}

fn check_cuda_status(status: CudaStatus, context: &str) -> Result<()> {
    match status {
        CudaStatus::Success => Ok(()),
        _ => {
            // SAFETY: This debug entrypoint returns the last CUDA runtime error code captured by the FFI layer.
            let cuda_error_code = unsafe { moonai_gpu_last_cuda_error_code() };
            Err(anyhow!("{context} failed with status {status:?} (raw CUDA error code {cuda_error_code})"))
        }
    }
}

pub struct Simulation {
    raw: NonNull<GpuEvolutionStateHandle>,
    predator_capacity: u32,
    prey_capacity: u32,
    config: SimulationConfig,
}

impl Drop for Simulation {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was allocated by the matching create entrypoint and is dropped exactly once here.
        unsafe { moonai_gpu_evolution_destroy(self.raw.as_ptr()) };
    }
}

impl Simulation {
    pub fn init(simulation_config: &SimulationConfig) -> Result<Self> {
        let evolution_config = GpuEvolutionConfig::for_seed_stage(simulation_config, SENSOR_COUNT, OUTPUT_COUNT)?;

        let predator_capacity = evolution_config.predator_capacity;
        let prey_capacity = evolution_config.prey_capacity;
        let mut raw = ptr::null_mut();
        // SAFETY: The CUDA entrypoint reads a plain-old-data config and writes a newly allocated opaque handle.
        let status = unsafe { moonai_gpu_evolution_create(&evolution_config, &mut raw) };
        check_cuda_status(status, "moonai_gpu_evolution_create")?;
        let raw = NonNull::new(raw).ok_or_else(|| anyhow!("moonai_gpu_evolution_create returned a null state"))?;

        let mut simulation = Simulation { raw, predator_capacity, prey_capacity, config: *simulation_config };

        simulation.seed_initial_population()?;
        simulation.set_simulation_config()?;
        let grid_cell_size = simulation_config.vision_range.max(1.0);
        let grid_cols = ((simulation_config.grid_size / grid_cell_size).ceil() as u32).max(1);
        let grid_rows = ((simulation_config.grid_size / grid_cell_size).ceil() as u32).max(1);
        simulation.set_spatial_grid(grid_cell_size, grid_cols, grid_rows)?;
        simulation.ensure_food_buffer()?;
        simulation.ensure_counter_buffer()?;
        simulation.ensure_free_lists()?;
        simulation.ensure_reproduction_buffers()?;
        simulation.ensure_metrics_buffer()?;
        simulation.ensure_spatial_grid_buffers()?;
        simulation.reset_counters()?;
        simulation.initialize_free_lists()?;
        simulation.seed_food()?;
        simulation.reset_reproduction_state()?;
        if evolution_config.predator_capacity > 0 {
            let _ = simulation.compile_population(PopulationKind::Predator, 0)?;
        }
        if evolution_config.prey_capacity > 0 {
            let _ = simulation.compile_population(PopulationKind::Prey, 0)?;
        }
        simulation.build_spatial_grid()?;
        simulation.compute_sensor_inputs()?;
        simulation.simulation_refresh_reports()?;

        Ok(simulation)
    }

    pub const fn config(&self) -> SimulationConfig {
        self.config
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

        let ui_stats = self.simulation_ui_stats()?;
        if self.config.report_interval_ticks > 0 && ui_stats.tick % self.config.report_interval_ticks == 0 {
            self.simulation_refresh_reports()?;
        }
        Ok(ui_stats)
    }

    pub fn ui_stats(&self) -> Result<UiStatsReadback> {
        self.simulation_ui_stats()
    }

    pub fn free_list_state(&self) -> Result<FreeListStateReadback> {
        self.simulation_free_list_state()
    }

    pub fn metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        self.simulation_metrics_summary()
    }

    pub fn refresh_reports(&mut self) -> Result<()> {
        self.simulation_refresh_reports()
    }

    pub fn sensor_snapshot(&self, population_kind: PopulationKind, slot: u32) -> Result<SensorSnapshotReadback> {
        readback("moonai_gpu_simulation_sensor_snapshot", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact sensor readback.
            unsafe { moonai_gpu_simulation_sensor_snapshot(self.raw.as_ptr(), population_kind, slot, out) }
        })
    }

    pub fn render_snapshot(&self, max_predators: u32, max_prey: u32, max_food: u32) -> Result<RenderSnapshotReadback> {
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
            moonai_gpu_simulation_render_snapshot(
                self.raw.as_ptr(),
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
        &self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SelectedAgentNetworkReadback> {
        readback("moonai_gpu_evolution_selected_agent_network", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact network readback.
            unsafe { moonai_gpu_evolution_selected_agent_network(self.raw.as_ptr(), population_kind, slot, out) }
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
            moonai_gpu_evolution_species_summaries(
                self.raw.as_ptr(),
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
        &self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<RepresentativeGenomeReadback> {
        let header = readback("moonai_gpu_evolution_representative_genome_header", |out| unsafe {
            moonai_gpu_evolution_representative_genome_header(self.raw.as_ptr(), population_kind, slot, out)
        })?;
        let returned_nodes = usize::from(header.num_nodes);
        let returned_connections = usize::from(header.num_connections);

        let mut node_types = vec![0_u8; returned_nodes];
        if !node_types.is_empty() {
            let status = unsafe {
                moonai_gpu_evolution_representative_genome_node_types(
                    self.raw.as_ptr(),
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
                moonai_gpu_evolution_representative_genome_connections(
                    self.raw.as_ptr(),
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
        let free_list_state = self.simulation_free_list_state()?;
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

    fn seed_initial_population(&mut self) -> Result<()> {
        // SAFETY: `self.raw` is a valid opaque state allocated by `moonai_gpu_evolution_create`.
        let status = unsafe { moonai_gpu_evolution_seed_initial_population(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_evolution_seed_initial_population")
    }

    fn set_simulation_config(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_set_config(self.raw.as_ptr(), &self.config) };
        check_cuda_status(status, "moonai_gpu_simulation_set_config")?;
        Ok(())
    }

    fn set_spatial_grid(&mut self, cell_size: f32, grid_cols: u32, grid_rows: u32) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_set_grid(self.raw.as_ptr(), cell_size, grid_cols, grid_rows) };
        check_cuda_status(status, "moonai_gpu_simulation_set_grid")
    }

    fn ensure_food_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_food_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_food_buffer")
    }

    fn ensure_counter_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_counter_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_counter_buffer")
    }

    fn ensure_free_lists(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_free_lists(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_free_lists")
    }

    fn ensure_reproduction_buffers(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_reproduction_buffers(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_reproduction_buffers")
    }

    fn ensure_metrics_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_metrics_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_metrics_buffer")
    }

    fn ensure_spatial_grid_buffers(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_spatial_grid_buffers(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_spatial_grid_buffers")
    }

    fn reset_counters(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_reset_counters(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_reset_counters")
    }

    fn initialize_free_lists(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_initialize_free_lists(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_initialize_free_lists")
    }

    fn seed_food(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_seed_food(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_seed_food")
    }

    fn reset_reproduction_state(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_reset_reproduction_state(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_reset_reproduction_state")
    }

    fn build_spatial_grid(&mut self) -> Result<()> {
        profile_scope!("spatial_grid");
        let status = unsafe { moonai_gpu_simulation_build_spatial_grid(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_build_spatial_grid")
    }

    fn compute_sensor_inputs(&mut self) -> Result<()> {
        profile_scope!("sensor_inputs");
        let status = unsafe { moonai_gpu_simulation_compute_sensor_inputs(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_compute_sensor_inputs")
    }

    fn infer_population(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("inference");
        let status = unsafe { moonai_gpu_simulation_infer_population(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_infer_population")
    }

    fn update_vitals(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("update_vitals");
        let status = unsafe { moonai_gpu_simulation_update_vitals(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_update_vitals")
    }

    fn resolve_food(&mut self) -> Result<()> {
        profile_scope!("resolve_food");
        let status = unsafe { moonai_gpu_simulation_resolve_food(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_food")
    }

    fn resolve_combat(&mut self) -> Result<()> {
        profile_scope!("resolve_combat");
        let status = unsafe { moonai_gpu_simulation_resolve_combat(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_combat")
    }

    fn apply_movement(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("apply_movement");
        let status = unsafe { moonai_gpu_simulation_apply_movement(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_apply_movement")
    }

    fn reproduction_candidate_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        profile_scope!("reprod_candidate");
        readback("moonai_gpu_simulation_reproduction_candidate_count", |out| unsafe {
            moonai_gpu_simulation_reproduction_candidate_count(self.raw.as_ptr(), population_kind, out)
        })
    }

    fn expand_population(&mut self, population_kind: PopulationKind, new_capacity: u32) -> Result<()> {
        let status =
            unsafe { moonai_gpu_simulation_expand_population(self.raw.as_ptr(), population_kind, new_capacity) };
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
        let status = unsafe { moonai_gpu_simulation_advance_tick(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_advance_tick")
    }

    fn simulation_ui_stats(&self) -> Result<UiStatsReadback> {
        profile_scope!("ui_stats");
        readback("moonai_gpu_simulation_ui_stats", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact UI stats readback.
            unsafe { moonai_gpu_simulation_ui_stats(self.raw.as_ptr(), out) }
        })
    }

    fn simulation_free_list_state(&self) -> Result<FreeListStateReadback> {
        readback("moonai_gpu_simulation_free_list_state", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact free-list readback.
            unsafe { moonai_gpu_simulation_free_list_state(self.raw.as_ptr(), out) }
        })
    }

    fn simulation_metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        readback("moonai_gpu_simulation_metrics_summary", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact metrics readback.
            unsafe { moonai_gpu_simulation_metrics_summary(self.raw.as_ptr(), out) }
        })
    }

    fn simulation_refresh_reports(&mut self) -> Result<()> {
        profile_scope!("refresh_reports");
        let status = unsafe { moonai_gpu_simulation_refresh_reports(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_refresh_reports")
    }

    fn population_live_count(&self, population_kind: PopulationKind) -> Result<u32> {
        readback("moonai_gpu_evolution_population_live_count", |out| unsafe {
            moonai_gpu_evolution_population_live_count(self.raw.as_ptr(), population_kind, out)
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
            unsafe { moonai_gpu_evolution_compile_population(self.raw.as_ptr(), population_kind, inspected_slot, out) }
        })
    }

    fn compile_slot(&mut self, population_kind: PopulationKind, slot: u32) -> Result<CompiledNetworkReadbackHeader> {
        readback("moonai_gpu_evolution_compile_slot", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compile readback header.
            unsafe { moonai_gpu_evolution_compile_slot(self.raw.as_ptr(), population_kind, slot, out) }
        })
    }

    fn reproduction_pairs(
        &self,
        population_kind: PopulationKind,
        pair_count: u32,
    ) -> Result<Vec<ReproductionPairReadback>> {
        let capacity = usize::try_from(pair_count).context("reproduction pair capacity overflowed")?;
        let mut pairs = vec![ReproductionPairReadback { parent_a_slot: 0, parent_b_slot: 0 }; capacity];
        let mut returned_pairs = 0_u32;
        let pairs_ptr = if pairs.is_empty() { ptr::null_mut() } else { pairs.as_mut_ptr() };
        let status = unsafe {
            moonai_gpu_simulation_read_reproduction_pairs(
                self.raw.as_ptr(),
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

    fn reproduction_free_slots(&self, population_kind: PopulationKind, slot_count: u32) -> Result<Vec<u32>> {
        let capacity = usize::try_from(slot_count).context("free-slot capacity overflowed")?;
        let mut slots = vec![0_u32; capacity];
        let mut returned_slots = 0_u32;
        let slots_ptr = if slots.is_empty() { ptr::null_mut() } else { slots.as_mut_ptr() };
        let status = unsafe {
            moonai_gpu_simulation_read_free_slots(
                self.raw.as_ptr(),
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
            moonai_gpu_evolution_crossover(
                self.raw.as_ptr(),
                population_kind,
                parent_a_slot,
                parent_b_slot,
                offspring_slot,
            )
        };
        check_cuda_status(status, "moonai_gpu_evolution_crossover")
    }

    fn mutate_slot(&mut self, population_kind: PopulationKind, slot: u32, config: GpuMutationConfig) -> Result<()> {
        let status = unsafe { moonai_gpu_evolution_mutate_slot(self.raw.as_ptr(), population_kind, slot, &config) };
        check_cuda_status(status, "moonai_gpu_evolution_mutate_slot")
    }

    fn apply_reproduction_energy(&mut self, population_kind: PopulationKind, births_applied: u32) -> Result<()> {
        let status = unsafe {
            moonai_gpu_simulation_apply_reproduction_energy(self.raw.as_ptr(), population_kind, births_applied)
        };
        check_cuda_status(status, "moonai_gpu_simulation_apply_reproduction_energy")
    }
}

unsafe extern "C" {
    fn moonai_gpu_runtime_available() -> CudaStatus;
    fn moonai_gpu_evolution_create(
        config: *const GpuEvolutionConfig,
        out_state: *mut *mut GpuEvolutionStateHandle,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_destroy(state: *mut GpuEvolutionStateHandle);
    fn moonai_gpu_evolution_seed_initial_population(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_evolution_population_live_count(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        out_live_count: *mut u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_set_config(
        state: *mut GpuEvolutionStateHandle,
        config: *const SimulationConfig,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_set_grid(
        state: *mut GpuEvolutionStateHandle,
        cell_size: f32,
        grid_cols: u32,
        grid_rows: u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_food_buffer(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_counter_buffer(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_free_lists(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_reproduction_buffers(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_metrics_buffer(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ensure_spatial_grid_buffers(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_reset_counters(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_initialize_free_lists(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_seed_food(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_reset_reproduction_state(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_build_spatial_grid(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_compute_sensor_inputs(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_infer_population(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_update_vitals(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_resolve_food(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_resolve_combat(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_apply_movement(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_reproduction_candidate_count(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        out_pair_count: *mut u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_read_reproduction_pairs(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        max_pairs: u32,
        out_pairs: *mut ReproductionPairReadback,
        out_returned_pairs: *mut u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_read_free_slots(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        max_slots: u32,
        out_slots: *mut u32,
        out_returned_slots: *mut u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_apply_reproduction_energy(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        births_applied: u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_expand_population(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        new_capacity: u32,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_advance_tick(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_ui_stats(
        state: *const GpuEvolutionStateHandle,
        out_stats: *mut UiStatsReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_free_list_state(
        state: *const GpuEvolutionStateHandle,
        out_state: *mut FreeListStateReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_metrics_summary(
        state: *const GpuEvolutionStateHandle,
        out_summary: *mut MetricsSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_refresh_reports(state: *mut GpuEvolutionStateHandle) -> CudaStatus;
    fn moonai_gpu_simulation_sensor_snapshot(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        out_snapshot: *mut SensorSnapshotReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_render_snapshot(
        state: *const GpuEvolutionStateHandle,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
        out_header: *mut RenderSnapshotHeader,
        out_predators: *mut RenderAgentReadback,
        out_prey: *mut RenderAgentReadback,
        out_food: *mut RenderFoodReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_compile_population(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        inspected_slot: u32,
        out_header: *mut CompiledNetworkReadbackHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_compile_slot(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        out_header: *mut CompiledNetworkReadbackHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_crossover(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_mutate_slot(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        config: *const GpuMutationConfig,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_selected_agent_network(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        out_network: *mut SelectedAgentNetworkReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_species_summaries(
        state: *mut GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        max_species: u32,
        out_header: *mut SpeciesBatchReadbackHeader,
        out_summaries: *mut SpeciesSummaryReadback,
        out_representatives: *mut RepresentativeGenomeHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_representative_genome_header(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        out_header: *mut RepresentativeGenomeHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_representative_genome_node_types(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        max_nodes: u32,
        out_node_types: *mut u8,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_representative_genome_connections(
        state: *const GpuEvolutionStateHandle,
        population_kind: PopulationKind,
        slot: u32,
        max_connections: u32,
        out_from_nodes: *mut i32,
        out_to_nodes: *mut i32,
        out_weights: *mut f32,
        out_innovations: *mut u32,
        out_enabled_flags: *mut u8,
    ) -> CudaStatus;
    fn moonai_gpu_last_cuda_error_code() -> i32;
}
