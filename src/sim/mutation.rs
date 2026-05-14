use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn mutate_slot(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
        config: GpuMutationConfig,
    ) -> Result<()> {
        let status = unsafe { dev_mutate_slot(self.get_dev_state(), population_kind, slot, &config) };
        check_cuda_status(status, "moonai_gpu_evolution_mutate_slot")
    }
}

unsafe extern "C" {
    fn dev_mutate_slot(
        state: *mut DeviceState,
        population_kind: PopulationKind,
        slot: u32,
        config: *const GpuMutationConfig,
    ) -> i32;
}
