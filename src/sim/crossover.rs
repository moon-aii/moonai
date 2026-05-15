use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn crossover_batch(
        &mut self,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
    ) -> Result<()> {
        let status =
            unsafe { dev_crossover_batch(self.get_dev_state(), population_kind, births_applied, free_slot_base) };
        check_cuda_status(status, "moonai_gpu_evolution_crossover_batch")
    }
}

unsafe extern "C" {
    fn dev_crossover_batch(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        births_applied: u32,
        free_slot_base: u32,
    ) -> i32;
}
