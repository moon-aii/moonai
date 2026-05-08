use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{
    PopulationSummaryReadback, RenderAgentReadback, RenderFoodReadback, RenderSnapshotHeader, RenderSnapshotReadback,
    SpatialGridReadback, UiStatsReadback,
};
use crate::tick::checks::{CudaStatus, InvariantCheckReadback, check_cuda};
use crate::tick::compaction::{CompactionSummaryReadback, FreeListStateReadback};
use crate::tick::compiled::CompiledNetworkReadbackHeader;
use crate::tick::crossover::CrossoverSummaryReadback;
use crate::tick::genome::{PopulationKind, SeededAgentSnapshot};
use crate::tick::inference::SensorSnapshotReadback;
use crate::tick::innovation::{DeviceInnovationState, InnovationLogReadbackHeader, InnovationRecord};
use crate::tick::metrics_reduce::MetricsSummaryReadback;
use crate::tick::mutation::{GpuMutationConfig, MutationSummaryReadback};
use crate::tick::network::SelectedAgentNetworkReadback;
use crate::tick::reproduction::ReproductionSummaryReadback;
use crate::tick::simulation::GpuSimulationConfig;
use crate::tick::species::{
    GenomeConnectionReadback, GenomeNodeReadback, RepresentativeGenomeHeader, RepresentativeGenomeReadback,
    SpeciesBatchReadbackHeader, SpeciesSummaryReadback,
};

const PHASE3_HIDDEN_NODE_BUDGET_CAP: u32 = 32;
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

impl GpuEvolutionConfig {
    pub fn for_seed_stage(simulation: &SimulationConfig, num_inputs: u32, num_outputs: u32) -> Result<Self> {
        let predator_capacity = as_non_negative_u32(simulation.predator_count, "predator_count")?;
        let prey_capacity = as_non_negative_u32(simulation.prey_count, "prey_count")?;
        let hidden_budget =
            as_non_negative_u32(simulation.max_hidden_nodes, "max_hidden_nodes")?.min(PHASE3_HIDDEN_NODE_BUDGET_CAP);
        if simulation.grid_size <= 0 {
            bail!("grid_size must be positive for GPU evolution, got {}", simulation.grid_size);
        }
        if simulation.initial_energy <= 0.0 {
            bail!("initial_energy must be positive for GPU evolution, got {}", simulation.initial_energy);
        }
        if simulation.max_energy <= 0.0 {
            bail!("max_energy must be positive for GPU evolution, got {}", simulation.max_energy);
        }

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
            world_size: simulation.grid_size as f32,
            initial_energy: simulation.initial_energy,
            max_energy: simulation.max_energy,
            seed: simulation.seed as i64 as u64,
            num_inputs,
            num_outputs,
            node_stride,
            connection_stride,
        })
    }
}

pub struct EvolutionManager {
    raw: NonNull<c_void>,
    config: GpuEvolutionConfig,
}

impl EvolutionManager {
    pub fn runtime_status() -> CudaStatus {
        // SAFETY: This entrypoint performs an internal CUDA runtime probe and returns a POD status code.
        unsafe { moonai_gpu_runtime_available() }
    }

    pub fn create(config: GpuEvolutionConfig) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: The CUDA entrypoint reads a plain-old-data config and writes a newly allocated opaque handle.
        let status = unsafe { moonai_gpu_evolution_create(&config, &mut raw) };
        check_cuda_status(status, "moonai_gpu_evolution_create")?;
        let raw = NonNull::new(raw).ok_or_else(|| anyhow!("moonai_gpu_evolution_create returned a null state"))?;
        Ok(Self { raw, config })
    }

    pub fn seed_initial_population(&mut self) -> Result<()> {
        // SAFETY: `self.raw` is a valid opaque state allocated by `moonai_gpu_evolution_create`.
        let status = unsafe { moonai_gpu_evolution_seed_initial_population(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_evolution_seed_initial_population")
    }

    pub fn population_summary(&self, population_kind: PopulationKind) -> Result<PopulationSummaryReadback> {
        readback("moonai_gpu_evolution_population_summary", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact readback.
            unsafe { moonai_gpu_evolution_population_summary(self.raw.as_ptr(), population_kind, out) }
        })
    }

    pub fn seeded_agent_snapshot(&self, population_kind: PopulationKind, slot: u32) -> Result<SeededAgentSnapshot> {
        readback("moonai_gpu_evolution_seeded_agent_snapshot", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact readback.
            unsafe { moonai_gpu_evolution_seeded_agent_snapshot(self.raw.as_ptr(), population_kind, slot, out) }
        })
    }

    pub fn ui_stats(&self) -> Result<UiStatsReadback> {
        readback("moonai_gpu_evolution_ui_stats", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact readback.
            unsafe { moonai_gpu_evolution_ui_stats(self.raw.as_ptr(), out) }
        })
    }

    pub fn innovation_state(&self) -> Result<DeviceInnovationState> {
        readback("moonai_gpu_evolution_innovation_state", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact readback.
            unsafe { moonai_gpu_evolution_innovation_state(self.raw.as_ptr(), out) }
        })
    }

    pub fn initialize_simulation(&mut self, config: GpuSimulationConfig) -> Result<()> {
        // SAFETY: `self.raw` is valid and `config` is copied into the CUDA runtime surface.
        let status = unsafe { moonai_gpu_simulation_initialize(self.raw.as_ptr(), &config) };
        check_cuda_status(status, "moonai_gpu_simulation_initialize")
    }

    pub fn debug_roundtrip_simulation_config(&self, config: GpuSimulationConfig) -> Result<GpuSimulationConfig> {
        readback("moonai_gpu_debug_roundtrip_simulation_config", |out| {
            // SAFETY: `config` is POD input and `out` points to writable storage for the same POD layout.
            unsafe { moonai_gpu_debug_roundtrip_simulation_config(&config, out) }
        })
    }

    pub fn simulation_step(&mut self) -> Result<UiStatsReadback> {
        readback("moonai_gpu_simulation_step", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the mapped UI stats snapshot.
            unsafe { moonai_gpu_simulation_step(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_ui_stats(&self) -> Result<UiStatsReadback> {
        readback("moonai_gpu_simulation_ui_stats", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the mapped UI stats snapshot.
            unsafe { moonai_gpu_simulation_ui_stats(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_free_list_state(&self) -> Result<FreeListStateReadback> {
        readback("moonai_gpu_simulation_free_list_state", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact free-list readback.
            unsafe { moonai_gpu_simulation_free_list_state(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_spatial_grid_state(&self) -> Result<SpatialGridReadback> {
        readback("moonai_gpu_simulation_spatial_grid_state", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact spatial-grid readback.
            unsafe { moonai_gpu_simulation_spatial_grid_state(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_reproduction_summary(
        &self,
        population_kind: PopulationKind,
    ) -> Result<ReproductionSummaryReadback> {
        readback("moonai_gpu_simulation_reproduction_summary", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact reproduction readback.
            unsafe { moonai_gpu_simulation_reproduction_summary(self.raw.as_ptr(), population_kind, out) }
        })
    }

    pub fn simulation_metrics_summary(&self) -> Result<MetricsSummaryReadback> {
        readback("moonai_gpu_simulation_metrics_summary", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact metrics readback.
            unsafe { moonai_gpu_simulation_metrics_summary(self.raw.as_ptr(), out) }
        })
    }

    pub fn simulation_refresh_reports(&mut self) -> Result<()> {
        let status = unsafe { moonai_gpu_simulation_refresh_reports(self.raw.as_ptr()) };
        check_cuda_status(status, "moonai_gpu_simulation_refresh_reports")
    }

    pub fn simulation_compact_population(
        &mut self,
        population_kind: PopulationKind,
    ) -> Result<CompactionSummaryReadback> {
        readback("moonai_gpu_simulation_compact_population", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the compact compaction readback.
            unsafe { moonai_gpu_simulation_compact_population(self.raw.as_ptr(), population_kind, out) }
        })
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

    pub fn innovation_log(&self, max_records: u32) -> Result<(InnovationLogReadbackHeader, Vec<InnovationRecord>)> {
        let record_capacity = usize::try_from(max_records).context("innovation log record capacity overflowed")?;
        let mut header = MaybeUninit::<InnovationLogReadbackHeader>::uninit();
        let mut records =
            vec![InnovationRecord { from_node: 0, to_node: 0, innovation: 0, record_kind: 0 }; record_capacity];
        let records_ptr = if records.is_empty() { ptr::null_mut() } else { records.as_mut_ptr() };
        // SAFETY: `self.raw` is valid, `header` points to writable storage, and `records_ptr` points to a buffer with
        // `max_records` slots when non-null.
        let status = unsafe {
            moonai_gpu_evolution_innovation_log(self.raw.as_ptr(), max_records, header.as_mut_ptr(), records_ptr)
        };
        check_cuda_status(status, "moonai_gpu_evolution_innovation_log")?;
        // SAFETY: A successful innovation-log call initializes the header.
        let header = unsafe { header.assume_init() };
        let returned_len = usize::try_from(header.returned_len).context("innovation log returned length overflowed")?;
        if returned_len > records.len() {
            bail!(
                "innovation log returned {} records but the host buffer only allocated {} slots",
                returned_len,
                records.len()
            );
        }
        records.truncate(returned_len);
        Ok((header, records))
    }

    pub fn mutate_population(
        &mut self,
        population_kind: PopulationKind,
        config: GpuMutationConfig,
    ) -> Result<MutationSummaryReadback> {
        readback("moonai_gpu_evolution_mutate_population", |out| {
            // SAFETY: `self.raw` is valid, `config` is POD, and `out` points to writable readback storage.
            unsafe { moonai_gpu_evolution_mutate_population(self.raw.as_ptr(), population_kind, &config, out) }
        })
    }

    pub fn crossover(
        &mut self,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> Result<CrossoverSummaryReadback> {
        readback("moonai_gpu_evolution_crossover", |out| {
            // SAFETY: `self.raw` is valid and all slot indices are copied by value into the CUDA entrypoint.
            unsafe {
                moonai_gpu_evolution_crossover(
                    self.raw.as_ptr(),
                    population_kind,
                    parent_a_slot,
                    parent_b_slot,
                    offspring_slot,
                    out,
                )
            }
        })
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

    pub fn classify_species(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<(SpeciesSummaryReadback, RepresentativeGenomeHeader)> {
        let mut summary = MaybeUninit::<SpeciesSummaryReadback>::uninit();
        let mut header = MaybeUninit::<RepresentativeGenomeHeader>::uninit();
        // SAFETY: `self.raw` is valid and both pointers refer to writable compact readback storage.
        let status = unsafe {
            moonai_gpu_evolution_classify_species(
                self.raw.as_ptr(),
                population_kind,
                slot,
                summary.as_mut_ptr(),
                header.as_mut_ptr(),
            )
        };
        check_cuda_status(status, "moonai_gpu_evolution_classify_species")?;
        // SAFETY: A successful classification call initializes both outputs.
        Ok(unsafe { (summary.assume_init(), header.assume_init()) })
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
        let node_capacity =
            usize::try_from(self.config.node_stride).context("representative genome node capacity overflowed")?;
        let connection_capacity = usize::try_from(self.config.connection_stride)
            .context("representative genome connection capacity overflowed")?;
        let mut header = MaybeUninit::<RepresentativeGenomeHeader>::uninit();
        let empty_node = GenomeNodeReadback { id: 0, node_type: 0, reserved0: 0, reserved1: 0 };
        let empty_connection = GenomeConnectionReadback {
            from_node: 0,
            to_node: 0,
            weight: 0.0,
            innovation: 0,
            enabled: 0,
            reserved0: 0,
            reserved1: 0,
        };
        let mut nodes = vec![empty_node; node_capacity];
        let mut connections = vec![empty_connection; connection_capacity];
        let nodes_ptr = if nodes.is_empty() { ptr::null_mut() } else { nodes.as_mut_ptr() };
        let connections_ptr = if connections.is_empty() { ptr::null_mut() } else { connections.as_mut_ptr() };

        let status = unsafe {
            moonai_gpu_evolution_representative_genome(
                self.raw.as_ptr(),
                population_kind,
                slot,
                header.as_mut_ptr(),
                self.config.node_stride,
                nodes_ptr,
                self.config.connection_stride,
                connections_ptr,
            )
        };
        check_cuda_status(status, "moonai_gpu_evolution_representative_genome")?;
        let header = unsafe { header.assume_init() };

        let returned_nodes = usize::from(header.num_nodes);
        let returned_connections = usize::from(header.num_connections);
        if returned_nodes > nodes.len() || returned_connections > connections.len() {
            bail!(
                "representative genome returned {} nodes and {} connections but the host buffers only allocated {} nodes and {} connections",
                returned_nodes,
                returned_connections,
                nodes.len(),
                connections.len()
            );
        }
        nodes.truncate(returned_nodes);
        connections.truncate(returned_connections);
        Ok(RepresentativeGenomeReadback { header, nodes, connections })
    }

    pub fn check_invariants(&self) -> Result<InvariantCheckReadback> {
        readback("moonai_gpu_evolution_check_invariants", |out| {
            // SAFETY: `self.raw` is valid and `out` points to writable storage for the invariant summary.
            unsafe { moonai_gpu_evolution_check_invariants(self.raw.as_ptr(), out) }
        })
    }
}

impl Drop for EvolutionManager {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was allocated by the matching create entrypoint and is dropped exactly once here.
        unsafe { moonai_gpu_evolution_destroy(self.raw.as_ptr()) };
    }
}

fn as_non_negative_u32(value: i32, field_name: &str) -> Result<u32> {
    if value < 0 {
        bail!("{field_name} must be non-negative, got {value}");
    }
    u32::try_from(value).map_err(|_| anyhow!("{field_name} could not be converted to u32: {value}"))
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
    fn moonai_gpu_evolution_create(config: *const GpuEvolutionConfig, out_state: *mut *mut c_void) -> CudaStatus;
    fn moonai_gpu_evolution_destroy(state: *mut c_void);
    fn moonai_gpu_evolution_seed_initial_population(state: *mut c_void) -> CudaStatus;
    fn moonai_gpu_evolution_population_summary(
        state: *const c_void,
        population_kind: PopulationKind,
        out_summary: *mut PopulationSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_seeded_agent_snapshot(
        state: *const c_void,
        population_kind: PopulationKind,
        slot: u32,
        out_snapshot: *mut SeededAgentSnapshot,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_ui_stats(state: *const c_void, out_stats: *mut UiStatsReadback) -> CudaStatus;
    fn moonai_gpu_evolution_innovation_state(
        state: *const c_void,
        out_innovation: *mut DeviceInnovationState,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_innovation_log(
        state: *const c_void,
        max_records: u32,
        out_header: *mut InnovationLogReadbackHeader,
        out_records: *mut InnovationRecord,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_initialize(state: *mut c_void, config: *const GpuSimulationConfig) -> CudaStatus;
    fn moonai_gpu_debug_roundtrip_simulation_config(
        config: *const GpuSimulationConfig,
        out_config: *mut GpuSimulationConfig,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_step(state: *mut c_void, out_stats: *mut UiStatsReadback) -> CudaStatus;
    fn moonai_gpu_simulation_ui_stats(state: *const c_void, out_stats: *mut UiStatsReadback) -> CudaStatus;
    fn moonai_gpu_simulation_free_list_state(state: *const c_void, out_state: *mut FreeListStateReadback)
    -> CudaStatus;
    fn moonai_gpu_simulation_spatial_grid_state(
        state: *const c_void,
        out_state: *mut SpatialGridReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_reproduction_summary(
        state: *const c_void,
        population_kind: PopulationKind,
        out_summary: *mut ReproductionSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_metrics_summary(
        state: *const c_void,
        out_summary: *mut MetricsSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_refresh_reports(state: *mut c_void) -> CudaStatus;
    fn moonai_gpu_simulation_compact_population(
        state: *mut c_void,
        population_kind: PopulationKind,
        out_summary: *mut CompactionSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_sensor_snapshot(
        state: *const c_void,
        population_kind: PopulationKind,
        slot: u32,
        out_snapshot: *mut SensorSnapshotReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_render_snapshot(
        state: *const c_void,
        max_predators: u32,
        max_prey: u32,
        max_food: u32,
        out_header: *mut RenderSnapshotHeader,
        out_predators: *mut RenderAgentReadback,
        out_prey: *mut RenderAgentReadback,
        out_food: *mut RenderFoodReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_mutate_population(
        state: *mut c_void,
        population_kind: PopulationKind,
        config: *const GpuMutationConfig,
        out_summary: *mut MutationSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_crossover(
        state: *mut c_void,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
        out_summary: *mut CrossoverSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_compile_population(
        state: *mut c_void,
        population_kind: PopulationKind,
        inspected_slot: u32,
        out_header: *mut CompiledNetworkReadbackHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_selected_agent_network(
        state: *const c_void,
        population_kind: PopulationKind,
        slot: u32,
        out_network: *mut SelectedAgentNetworkReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_classify_species(
        state: *mut c_void,
        population_kind: PopulationKind,
        slot: u32,
        out_summary: *mut SpeciesSummaryReadback,
        out_header: *mut RepresentativeGenomeHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_species_summaries(
        state: *mut c_void,
        population_kind: PopulationKind,
        max_species: u32,
        out_header: *mut SpeciesBatchReadbackHeader,
        out_summaries: *mut SpeciesSummaryReadback,
        out_representatives: *mut RepresentativeGenomeHeader,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_representative_genome(
        state: *const c_void,
        population_kind: PopulationKind,
        slot: u32,
        out_header: *mut RepresentativeGenomeHeader,
        max_nodes: u32,
        out_nodes: *mut GenomeNodeReadback,
        max_connections: u32,
        out_connections: *mut GenomeConnectionReadback,
    ) -> CudaStatus;
    fn moonai_gpu_evolution_check_invariants(
        state: *const c_void,
        out_checks: *mut InvariantCheckReadback,
    ) -> CudaStatus;
    fn moonai_gpu_last_cuda_error_code() -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::genome::PopulationKind;
    use crate::tick::inference::{OUTPUT_COUNT, SENSOR_COUNT};

    fn smoke_simulation_config(seed: i32) -> SimulationConfig {
        SimulationConfig {
            grid_size: 256,
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

    fn seed_manager(seed: i32) -> Result<EvolutionManager> {
        let config = GpuEvolutionConfig::for_seed_stage(
            &smoke_simulation_config(seed),
            SENSOR_COUNT as u32,
            OUTPUT_COUNT as u32,
        )?;
        let mut manager = EvolutionManager::create(config)?;
        manager.seed_initial_population()?;
        Ok(manager)
    }

    #[test]
    fn population_summary_readback_serializes() -> Result<()> {
        let summary = PopulationSummaryReadback {
            population_kind: PopulationKind::Predator,
            live_count: 8,
            capacity: 8,
            next_entity_id: 21,
            innovation_counter: 72,
            next_node_id: 38,
            avg_energy: 0.5,
            avg_connections: 72.0,
        };
        let json = serde_json::to_string(&summary)?;
        assert!(json.contains("Predator"));
        assert!(json.contains("avg_connections"));
        Ok(())
    }

    #[test]
    fn gpu_seeding_is_deterministic_for_same_seed() -> Result<()> {
        let manager_a = seed_manager(42)?;
        let manager_b = seed_manager(42)?;

        assert_eq!(manager_a.innovation_state()?, manager_b.innovation_state()?);
        assert_eq!(
            manager_a.population_summary(PopulationKind::Predator)?,
            manager_b.population_summary(PopulationKind::Predator)?
        );
        assert_eq!(
            manager_a.population_summary(PopulationKind::Prey)?,
            manager_b.population_summary(PopulationKind::Prey)?
        );
        assert_eq!(
            manager_a.seeded_agent_snapshot(PopulationKind::Predator, 0)?,
            manager_b.seeded_agent_snapshot(PopulationKind::Predator, 0)?
        );
        assert_eq!(
            manager_a.seeded_agent_snapshot(PopulationKind::Prey, 3)?,
            manager_b.seeded_agent_snapshot(PopulationKind::Prey, 3)?
        );
        assert_eq!(manager_a.ui_stats()?, manager_b.ui_stats()?);

        Ok(())
    }

    #[test]
    fn gpu_seeding_changes_when_seed_changes() -> Result<()> {
        let manager_a = seed_manager(42)?;
        let manager_b = seed_manager(43)?;

        let predator_a = manager_a.seeded_agent_snapshot(PopulationKind::Predator, 0)?;
        let predator_b = manager_b.seeded_agent_snapshot(PopulationKind::Predator, 0)?;
        assert_ne!(predator_a.genome_hash, predator_b.genome_hash);
        assert_ne!(predator_a.pos_x, predator_b.pos_x);
        assert_ne!(predator_a.pos_y, predator_b.pos_y);

        Ok(())
    }

    #[test]
    fn gpu_evolution_is_deterministic_for_same_seed() -> Result<()> {
        let simulation = smoke_simulation_config(44);
        let mutation_config = GpuMutationConfig::for_simulation(&simulation)?;
        let mut manager_a = seed_manager(44)?;
        let mut manager_b = seed_manager(44)?;

        let predator_mutation_a = manager_a.mutate_population(PopulationKind::Predator, mutation_config)?;
        let predator_mutation_b = manager_b.mutate_population(PopulationKind::Predator, mutation_config)?;
        let prey_mutation_a = manager_a.mutate_population(PopulationKind::Prey, mutation_config)?;
        let prey_mutation_b = manager_b.mutate_population(PopulationKind::Prey, mutation_config)?;
        let crossover_a = manager_a.crossover(PopulationKind::Predator, 0, 1, 2)?;
        let crossover_b = manager_b.crossover(PopulationKind::Predator, 0, 1, 2)?;
        let predator_compile_a = manager_a.compile_population(PopulationKind::Predator, 2)?;
        let predator_compile_b = manager_b.compile_population(PopulationKind::Predator, 2)?;
        let prey_compile_a = manager_a.compile_population(PopulationKind::Prey, 0)?;
        let prey_compile_b = manager_b.compile_population(PopulationKind::Prey, 0)?;
        let inspect_a = manager_a.selected_agent_network(PopulationKind::Predator, 2)?;
        let inspect_b = manager_b.selected_agent_network(PopulationKind::Predator, 2)?;
        let species_a = manager_a.classify_species(PopulationKind::Predator, 2)?;
        let species_b = manager_b.classify_species(PopulationKind::Predator, 2)?;
        let innovation_log_a = manager_a.innovation_log(64)?;
        let innovation_log_b = manager_b.innovation_log(64)?;
        let species_batch_a = manager_a.species_summaries(PopulationKind::Predator, 64)?;
        let species_batch_b = manager_b.species_summaries(PopulationKind::Predator, 64)?;
        let invariants_a = manager_a.check_invariants()?;
        let invariants_b = manager_b.check_invariants()?;

        assert_eq!(predator_mutation_a, predator_mutation_b);
        assert_eq!(prey_mutation_a, prey_mutation_b);
        assert_eq!(crossover_a, crossover_b);
        assert_eq!(predator_compile_a, predator_compile_b);
        assert_eq!(prey_compile_a, prey_compile_b);
        assert_eq!(inspect_a, inspect_b);
        assert_eq!(species_a, species_b);
        assert_eq!(innovation_log_a, innovation_log_b);
        assert_eq!(species_batch_a, species_batch_b);
        assert_eq!(manager_a.innovation_state()?, manager_b.innovation_state()?);
        assert_eq!(invariants_a, invariants_b);

        Ok(())
    }

    #[test]
    fn gpu_end_to_end_seed_mutate_compile_inspect_smoke() -> Result<()> {
        let simulation = smoke_simulation_config(55);
        let mutation_config = GpuMutationConfig::for_simulation(&simulation)?;
        let mut manager = seed_manager(55)?;

        let predator_mutation = manager.mutate_population(PopulationKind::Predator, mutation_config)?;
        let prey_mutation = manager.mutate_population(PopulationKind::Prey, mutation_config)?;
        let crossover = manager.crossover(PopulationKind::Predator, 0, 1, 2)?;
        let predator_compile = manager.compile_population(PopulationKind::Predator, 2)?;
        let prey_compile = manager.compile_population(PopulationKind::Prey, 0)?;
        let inspected_network = manager.selected_agent_network(PopulationKind::Predator, 2)?;
        let (species_summary, representative) = manager.classify_species(PopulationKind::Predator, 2)?;
        let (innovation_log_header, innovation_log) = manager.innovation_log(64)?;
        let (species_batch_header, species_batch, representatives) =
            manager.species_summaries(PopulationKind::Predator, 64)?;
        let invariants = manager.check_invariants()?;
        let innovation_state = manager.innovation_state()?;

        assert!(predator_mutation.agents_mutated > 0);
        assert!(prey_mutation.agents_mutated > 0);
        assert!(crossover.inherited_connections > 0);
        assert!(predator_compile.node_count > 0);
        assert!(predator_compile.eval_node_count > 0);
        assert!(prey_compile.node_count > 0);
        assert_eq!(inspected_network.output_count, OUTPUT_COUNT as u16);
        assert!(species_summary.size > 0);
        assert_eq!(representative.species_id, species_summary.species_id);
        assert!(innovation_log_header.total_len > 0);
        assert_eq!(innovation_log_header.returned_len as usize, innovation_log.len());
        assert!(innovation_log.iter().any(|record| record.innovation > 0));
        assert!(species_batch_header.species_count > 0);
        assert_eq!(species_batch_header.returned_species_count as usize, species_batch.len());
        assert_eq!(species_batch.len(), representatives.len());
        assert!(species_batch.iter().any(|summary| summary.size > 0));
        assert_eq!(invariants.invalid_node_counts, 0);
        assert_eq!(invariants.invalid_connection_counts, 0);
        assert_eq!(invariants.invalid_connection_bounds, 0);
        assert_eq!(invariants.invalid_compiled_offsets, 0);
        assert_eq!(invariants.invalid_eval_nodes, 0);
        assert_eq!(invariants.invalid_output_indices, 0);
        assert_eq!(invariants.invalid_species_assignments, 0);
        assert_eq!(invariants.innovation_log_overflow, 0);
        assert!(innovation_state.log_len > 0);

        Ok(())
    }
}
