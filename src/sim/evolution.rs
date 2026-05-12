use crate::profile_scope;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::experiment::SimulationConfig;
use crate::sim::buffers::{
    FreeListStateReadback, MetricsSummaryReadback, RenderAgentReadback, RenderFoodReadback, RenderSnapshotHeader,
    RenderSnapshotReadback, UiStatsReadback,
};
use crate::sim::simulation::PopulationKind;
use crate::sim::simulation::{CompiledNetworkReadbackHeader, SelectedAgentNetworkReadback, SensorSnapshotReadback};

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

impl CudaStatus {
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

pub fn check_cuda(status: CudaStatus, context: &str) -> anyhow::Result<()> {
    if status.is_success() { Ok(()) } else { Err(anyhow::anyhow!("{context} failed with status {status:?}")) }
}
use crate::sim::species::{
    GenomeConnectionReadback, GenomeNodeReadback, RepresentativeGenomeHeader, RepresentativeGenomeReadback,
    SpeciesBatchReadbackHeader, SpeciesSummaryReadback,
};

const PHASE3_CONNECTION_GROWTH_BUDGET_CAP: u32 = 16;

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

pub struct EvolutionManager {
    raw: NonNull<GpuEvolutionStateHandle>,
    simulation: Option<SimulationConfig>,
    predator_capacity: u32,
    prey_capacity: u32,
}

impl EvolutionManager {
    pub fn runtime_status() -> CudaStatus {
        // SAFETY: This entrypoint performs an internal CUDA runtime probe and returns a POD status code.
        unsafe { moonai_gpu_runtime_available() }
    }

    pub fn create(config: GpuEvolutionConfig) -> Result<Self> {
        let predator_capacity = config.predator_capacity;
        let prey_capacity = config.prey_capacity;
        let mut raw = ptr::null_mut();
        // SAFETY: The CUDA entrypoint reads a plain-old-data config and writes a newly allocated opaque handle.
        let status = unsafe { moonai_gpu_evolution_create(&config, &mut raw) };
        check_cuda_status(status, "moonai_gpu_evolution_create")?;
        let raw = NonNull::new(raw).ok_or_else(|| anyhow!("moonai_gpu_evolution_create returned a null state"))?;
        Ok(Self { raw, simulation: None, predator_capacity, prey_capacity })
    }

    pub fn seed_initial_population(&mut self) -> Result<()> {
        // SAFETY: `self.raw` is a valid opaque state allocated by `moonai_gpu_evolution_create`.
        let status = unsafe { moonai_gpu_evolution_seed_initial_population(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_evolution_seed_initial_population")
    }

    pub fn set_simulation_config(&mut self, config: SimulationConfig) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_set_config(self.raw.as_ptr(), &config) };
        check_cuda_status(status, "moonai_gpu_simulation_set_config")?;
        self.simulation = Some(config);
        Ok(())
    }

    pub fn set_spatial_grid(&mut self, cell_size: f32, grid_cols: u32, grid_rows: u32) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_set_grid(self.raw.as_ptr(), cell_size, grid_cols, grid_rows) };
        check_cuda_status(status, "moonai_gpu_simulation_set_grid")
    }

    pub fn ensure_food_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_food_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_food_buffer")
    }

    pub fn ensure_counter_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_counter_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_counter_buffer")
    }

    pub fn ensure_free_lists(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_free_lists(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_free_lists")
    }

    pub fn ensure_reproduction_buffers(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_reproduction_buffers(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_reproduction_buffers")
    }

    pub fn ensure_metrics_buffer(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_metrics_buffer(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_metrics_buffer")
    }

    pub fn ensure_spatial_grid_buffers(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_ensure_spatial_grid_buffers(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_ensure_spatial_grid_buffers")
    }

    pub fn reset_counters(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_reset_counters(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_reset_counters")
    }

    pub fn initialize_free_lists(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_initialize_free_lists(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_initialize_free_lists")
    }

    pub fn seed_food(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_seed_food(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_seed_food")
    }

    pub fn reset_reproduction_state(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_reset_reproduction_state(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_reset_reproduction_state")
    }

    pub fn build_spatial_grid(&mut self) -> Result<()> {
        profile_scope!("spatial_grid");
        let status = unsafe { moonai_gpu_simulation_build_spatial_grid(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_build_spatial_grid")
    }

    pub fn compute_sensor_inputs(&mut self) -> Result<()> {
        profile_scope!("sensor_inputs");
        let status = unsafe { moonai_gpu_simulation_compute_sensor_inputs(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_compute_sensor_inputs")
    }

    pub fn infer_population(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("inference");
        let status = unsafe { moonai_gpu_simulation_infer_population(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_infer_population")
    }

    pub fn update_vitals(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("update_vitals");
        let status = unsafe { moonai_gpu_simulation_update_vitals(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_update_vitals")
    }

    pub fn resolve_food(&mut self) -> Result<()> {
        profile_scope!("resolve_food");
        let status = unsafe { moonai_gpu_simulation_resolve_food(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_food")
    }

    pub fn resolve_combat(&mut self) -> Result<()> {
        profile_scope!("resolve_combat");
        let status = unsafe { moonai_gpu_simulation_resolve_combat(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_resolve_combat")
    }

    pub fn apply_movement(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("apply_movement");
        let status = unsafe { moonai_gpu_simulation_apply_movement(self.raw.as_ptr(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_apply_movement")
    }

    pub fn reproduction_candidate_count(&mut self, population_kind: PopulationKind) -> Result<u32> {
        profile_scope!("reprod_candidate");
        readback("moonai_gpu_simulation_reproduction_candidate_count", |out| unsafe {
            moonai_gpu_simulation_reproduction_candidate_count(self.raw.as_ptr(), population_kind, out)
        })
    }

    pub fn expand_population(&mut self, population_kind: PopulationKind, new_capacity: u32) -> Result<()> {
        let status =
            unsafe { moonai_gpu_simulation_expand_population(self.raw.as_ptr(), population_kind, new_capacity) };
        check_cuda_status(status, "moonai_gpu_simulation_expand_population")?;
        match population_kind {
            PopulationKind::Predator => self.predator_capacity = new_capacity,
            PopulationKind::Prey => self.prey_capacity = new_capacity,
        }
        Ok(())
    }

    pub fn run_reproduction(&mut self, population_kind: PopulationKind) -> Result<()> {
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

    pub fn advance_tick(&mut self) -> Result<()> {
        profile_scope!("advance_tick");
        let status = unsafe { moonai_gpu_simulation_advance_tick(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_advance_tick")
    }

    pub fn simulation_ui_stats(&self) -> Result<UiStatsReadback> {
        profile_scope!("ui_stats");
        readback("moonai_gpu_simulation_ui_stats", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact UI stats readback.
            unsafe { moonai_gpu_simulation_ui_stats(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_free_list_state(&self) -> Result<FreeListStateReadback> {
        readback("moonai_gpu_simulation_free_list_state", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact free-list readback.
            unsafe { moonai_gpu_simulation_free_list_state(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        readback("moonai_gpu_simulation_metrics_summary", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact metrics readback.
            unsafe { moonai_gpu_simulation_metrics_summary(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_refresh_reports(&mut self) -> Result<()> {
        profile_scope!("refresh_reports");
        let status = unsafe { moonai_gpu_simulation_refresh_reports(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_refresh_reports")
    }

    pub fn population_live_count(&self, population_kind: PopulationKind) -> Result<u32> {
        readback("moonai_gpu_evolution_population_live_count", |out| unsafe {
            moonai_gpu_evolution_population_live_count(self.raw.as_ptr(), population_kind, out)
        })
    }

    pub const fn population_capacity(&self, population_kind: PopulationKind) -> u32 {
        match population_kind {
            PopulationKind::Predator => self.predator_capacity,
            PopulationKind::Prey => self.prey_capacity,
        }
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

    pub fn compile_population(
        &mut self,
        population_kind: PopulationKind,
        inspected_slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        readback("moonai_gpu_evolution_compile_population", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compile readback header.
            unsafe { moonai_gpu_evolution_compile_population(self.raw.as_ptr(), population_kind, inspected_slot, out) }
        })
    }

    pub fn compile_slot(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<CompiledNetworkReadbackHeader> {
        readback("moonai_gpu_evolution_compile_slot", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compile readback header.
            unsafe { moonai_gpu_evolution_compile_slot(self.raw.as_ptr(), population_kind, slot, out) }
        })
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

    fn mutation_config(&self) -> Result<GpuMutationConfig> {
        let simulation = self.simulation.ok_or_else(|| anyhow!("simulation config not set"))?;
        Ok(GpuMutationConfig {
            mutation_rate: simulation.mutation_rate,
            weight_mutation_power: simulation.weight_mutation_power,
            add_node_rate: simulation.add_node_rate,
            add_connection_rate: simulation.add_connection_rate,
            delete_connection_rate: simulation.delete_connection_rate,
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

impl Drop for EvolutionManager {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was allocated by the matching create entrypoint and is dropped exactly once here.
        unsafe { moonai_gpu_evolution_destroy(self.raw.as_ptr()) };
    }
}

fn readback<T>(context: &str, mut call: impl FnMut(*mut T) -> CudaStatus) -> Result<T> {
    let mut out = MaybeUninit::<T>::uninit();
    check_cuda_status(call(out.as_mut_ptr()), context)?;
    // SAFETY: A successful readback call fully initializes the output struct.
    Ok(unsafe { out.assume_init() })
}

fn check_cuda_status(status: CudaStatus, context: &str) -> Result<()> {
    match check_cuda(status, context) {
        Ok(()) => Ok(()),
        Err(_) => {
            // SAFETY: This debug entrypoint returns the last CUDA runtime error code captured by the FFI layer.
            let cuda_error_code = unsafe { moonai_gpu_last_cuda_error_code() };
            Err(anyhow!("{context} failed with status {status:?} (raw CUDA error code {cuda_error_code})"))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::simulation::PopulationKind;
    use crate::sim::simulation::{OUTPUT_COUNT, SENSOR_COUNT};

    fn smoke_simulation_config(seed: u64) -> SimulationConfig {
        SimulationConfig {
            grid_size: 256.0,
            predator_count: 8,
            prey_count: 12,
            initial_energy: 0.36,
            max_energy: 2.0,
            max_hidden_nodes: 6,
            mutation_rate: 1.0,
            add_node_rate: 1.0,
            add_connection_rate: 1.0,
            delete_connection_rate: 0.25,
            seed,
            ..SimulationConfig::default()
        }
    }

    fn seed_manager(seed: u64) -> Result<EvolutionManager> {
        let config = GpuEvolutionConfig::for_seed_stage(&smoke_simulation_config(seed), SENSOR_COUNT, OUTPUT_COUNT)?;
        let mut manager = EvolutionManager::create(config)?;
        manager.seed_initial_population()?;
        Ok(manager)
    }

    #[test]
    fn gpu_evolution_is_deterministic_for_same_seed() -> Result<()> {
        let mut manager_a = seed_manager(44)?;
        let mut manager_b = seed_manager(44)?;

        let predator_compile_a = manager_a.compile_population(PopulationKind::Predator, 2)?;
        let predator_compile_b = manager_b.compile_population(PopulationKind::Predator, 2)?;
        let prey_compile_a = manager_a.compile_population(PopulationKind::Prey, 0)?;
        let prey_compile_b = manager_b.compile_population(PopulationKind::Prey, 0)?;
        let inspect_a = manager_a.selected_agent_network(PopulationKind::Predator, 2)?;
        let inspect_b = manager_b.selected_agent_network(PopulationKind::Predator, 2)?;
        let species_batch_a = manager_a.species_summaries(PopulationKind::Predator, 64)?;
        let species_batch_b = manager_b.species_summaries(PopulationKind::Predator, 64)?;

        assert_eq!(predator_compile_a, predator_compile_b);
        assert_eq!(prey_compile_a, prey_compile_b);
        assert_eq!(inspect_a, inspect_b);
        assert_eq!(species_batch_a, species_batch_b);

        Ok(())
    }

    #[test]
    fn gpu_end_to_end_seed_mutate_compile_inspect_smoke() -> Result<()> {
        let mut manager = seed_manager(55)?;

        let predator_compile = manager.compile_population(PopulationKind::Predator, 2)?;
        let prey_compile = manager.compile_population(PopulationKind::Prey, 0)?;
        let inspected_network = manager.selected_agent_network(PopulationKind::Predator, 2)?;
        let (species_batch_header, species_batch, representatives) =
            manager.species_summaries(PopulationKind::Predator, 64)?;

        assert!(predator_compile.node_count > 0);
        assert!(predator_compile.eval_node_count > 0);
        assert!(prey_compile.node_count > 0);
        assert_eq!(inspected_network.output_count, OUTPUT_COUNT as u16);
        assert!(species_batch_header.species_count > 0);
        assert_eq!(species_batch_header.returned_species_count as usize, species_batch.len());
        assert_eq!(species_batch.len(), representatives.len());
        assert!(species_batch.iter().any(|summary| summary.size > 0));

        Ok(())
    }
}
