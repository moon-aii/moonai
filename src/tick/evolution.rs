use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use anyhow::{Context as _, Result, anyhow, bail};

use crate::config::SimulationConfig;
use crate::tick::buffers::{PopulationSummaryReadback, UiStatsReadback};
use crate::tick::checks::{CudaStatus, check_cuda};
use crate::tick::genome::{PopulationKind, SeededAgentSnapshot};
use crate::tick::innovation::DeviceInnovationState;

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
        if simulation.grid_size <= 0 {
            bail!("grid_size must be positive for GPU evolution, got {}", simulation.grid_size);
        }
        if simulation.initial_energy <= 0.0 {
            bail!("initial_energy must be positive for GPU evolution, got {}", simulation.initial_energy);
        }
        if simulation.max_energy <= 0.0 {
            bail!("max_energy must be positive for GPU evolution, got {}", simulation.max_energy);
        }

        let node_stride = num_inputs
            .checked_add(num_outputs)
            .and_then(|value| value.checked_add(1))
            .context("seed-stage node stride overflowed")?;
        let connection_stride = num_inputs
            .checked_add(1)
            .and_then(|value| value.checked_mul(num_outputs))
            .context("seed-stage connection stride overflowed")?;

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
    fn gpu_seed_stage_config_uses_initial_genome_strides() -> Result<()> {
        let config =
            GpuEvolutionConfig::for_seed_stage(&smoke_simulation_config(17), SENSOR_COUNT as u32, OUTPUT_COUNT as u32)?;
        assert_eq!(config.seeded_node_count(), (SENSOR_COUNT + OUTPUT_COUNT + 1) as u32);
        assert_eq!(config.seeded_connection_count(), ((SENSOR_COUNT + 1) * OUTPUT_COUNT) as u32);
        assert_eq!(config.node_stride, config.seeded_node_count());
        assert_eq!(config.connection_stride, config.seeded_connection_count());
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
}
