use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn crossover_slot(
        &mut self,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> Result<()> {
        let status = unsafe {
            dev_crossover(self.get_dev_state(), population_kind, parent_a_slot, parent_b_slot, offspring_slot)
        };
        check_cuda_status(status, "moonai_gpu_evolution_crossover")
    }
}

unsafe extern "C" {
    fn dev_crossover(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        parent_a_slot: u32,
        parent_b_slot: u32,
        offspring_slot: u32,
    ) -> i32;
}
