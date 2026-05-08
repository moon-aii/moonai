use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{
    FreeListStateReadback, MetricsSummaryReadback, RenderAgentReadback, RenderFoodReadback, RenderSnapshotHeader,
    RenderSnapshotReadback, UiStatsReadback,
};
use crate::tick::network::{CompiledNetworkReadbackHeader, SelectedAgentNetworkReadback, SensorSnapshotReadback};
use crate::tick::simulation::PopulationKind;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStatus {
    Success = 0,
}

impl CudaStatus {
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

pub fn check_cuda(status: CudaStatus, context: &str) -> anyhow::Result<()> {
    if status.is_success() { Ok(()) } else { Err(anyhow::anyhow!("{context} failed with status {status:?}")) }
}
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

    pub fn initialize_simulation(&mut self, config: GpuSimulationConfig) -> Result<()> {
        // SAFETY: `self.raw` is valid and `config` is copied into the CUDA runtime surface.
        let status = unsafe { moonai_gpu_simulation_initialize(self.raw.as_ptr(), &config) };
        check_cuda_status(status, "moonai_gpu_simulation_initialize")
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
    fn moonai_gpu_simulation_initialize(state: *mut c_void, config: *const GpuSimulationConfig) -> CudaStatus;
    fn moonai_gpu_simulation_step(state: *mut c_void, out_stats: *mut UiStatsReadback) -> CudaStatus;
    fn moonai_gpu_simulation_ui_stats(state: *const c_void, out_stats: *mut UiStatsReadback) -> CudaStatus;
    fn moonai_gpu_simulation_free_list_state(state: *const c_void, out_state: *mut FreeListStateReadback)
    -> CudaStatus;
    fn moonai_gpu_simulation_metrics_summary(
        state: *const c_void,
        out_summary: *mut MetricsSummaryReadback,
    ) -> CudaStatus;
    fn moonai_gpu_simulation_refresh_reports(state: *mut c_void) -> CudaStatus;
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
    fn moonai_gpu_last_cuda_error_code() -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tick::network::{OUTPUT_COUNT, SENSOR_COUNT};
    use crate::tick::simulation::PopulationKind;

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
