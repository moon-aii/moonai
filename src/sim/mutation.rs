use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn mutate_batch(
        &mut self,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
        config: GpuMutationConfig,
    ) -> Result<()> {
        let status =
            unsafe { dev_mutate_batch(self.get_dev_state(), population_kind, births_applied, free_slot_base, &config) };
        check_cuda_status(status, "moonai_gpu_evolution_mutate_batch")
    }
}

unsafe extern "C" {
    fn dev_mutate_batch(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
        config: *const GpuMutationConfig,
    ) -> i32;
}
