use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{PopulationSummaryReadback, UiStatsReadback};
use crate::tick::checks::{CudaStatus, InvariantCheckReadback, check_cuda};
use crate::tick::compiled::CompiledNetworkReadbackHeader;
use crate::tick::crossover::CrossoverSummaryReadback;
use crate::tick::genome::{PopulationKind, SeededAgentSnapshot};
use crate::tick::innovation::DeviceInnovationState;
use crate::tick::mutation::{GpuMutationConfig, MutationSummaryReadback};
use crate::tick::network::SelectedAgentNetworkReadback;
use crate::tick::species::{RepresentativeGenomeHeader, SpeciesSummaryReadback};

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

    pub const fn seeded_node_count(self) -> u32 {
        self.num_inputs + self.num_outputs + 1
    }

    pub const fn seeded_connection_count(self) -> u32 {
        (self.num_inputs + 1) * self.num_outputs
    }

    pub const fn hidden_node_budget(self) -> u32 {
        self.node_stride - self.seeded_node_count()
    }

    pub const fn extra_connection_capacity(self) -> u32 {
        self.connection_stride - self.seeded_connection_count()
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

    pub const fn config(&self) -> GpuEvolutionConfig {
        self.config
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
    use crate::types::{OUTPUT_COUNT, SENSOR_COUNT};

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

    fn gpu_runtime_ready() -> bool {
        EvolutionManager::runtime_status().is_success()
    }

    #[test]
    fn gpu_seed_stage_config_reserves_phase3_growth_budget() -> Result<()> {
        let config =
            GpuEvolutionConfig::for_seed_stage(&smoke_simulation_config(17), SENSOR_COUNT as u32, OUTPUT_COUNT as u32)?;
        assert_eq!(config.seeded_node_count(), (SENSOR_COUNT + OUTPUT_COUNT + 1) as u32);
        assert_eq!(config.seeded_connection_count(), ((SENSOR_COUNT + 1) * OUTPUT_COUNT) as u32);
        assert_eq!(config.hidden_node_budget(), 6);
        assert_eq!(config.extra_connection_capacity(), 12);
        assert_eq!(config.node_stride, config.seeded_node_count() + 6);
        assert_eq!(config.connection_stride, config.seeded_connection_count() + 12);
        Ok(())
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
        if !gpu_runtime_ready() {
            return Ok(());
        }

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
        if !gpu_runtime_ready() {
            return Ok(());
        }

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
        if !gpu_runtime_ready() {
            return Ok(());
        }

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
        let invariants_a = manager_a.check_invariants()?;
        let invariants_b = manager_b.check_invariants()?;

        assert_eq!(predator_mutation_a, predator_mutation_b);
        assert_eq!(prey_mutation_a, prey_mutation_b);
        assert_eq!(crossover_a, crossover_b);
        assert_eq!(predator_compile_a, predator_compile_b);
        assert_eq!(prey_compile_a, prey_compile_b);
        assert_eq!(inspect_a, inspect_b);
        assert_eq!(species_a, species_b);
        assert_eq!(manager_a.innovation_state()?, manager_b.innovation_state()?);
        assert_eq!(invariants_a, invariants_b);

        Ok(())
    }

    #[test]
    fn gpu_end_to_end_seed_mutate_compile_inspect_smoke() -> Result<()> {
        if !gpu_runtime_ready() {
            return Ok(());
        }

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
