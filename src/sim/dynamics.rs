use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn infer_population(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("inference");
        let status = unsafe { dev_infer_population(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_infer_population")
    }

    pub(super) fn update_vitals(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("update_vitals");
        let status = unsafe { dev_update_vitals(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_update_vitals")
    }

    pub(super) fn apply_movement(&mut self, population_kind: PopulationKind) -> Result<()> {
        profile_scope!("apply_movement");
        let status = unsafe { dev_apply_movement(self.get_dev_state(), population_kind) };
        check_cuda_status(status, "moonai_gpu_simulation_apply_movement")
    }

    pub(super) fn advance_tick(&mut self) -> Result<()> {
        profile_scope!("advance_tick");
        let status = unsafe { dev_advance_tick(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_advance_tick")
    }
}

unsafe extern "C" {
    fn dev_infer_population(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_update_vitals(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_apply_movement(state: *mut DeviceState, population_kind: PopulationKind) -> i32;
    fn dev_advance_tick(state: *mut DeviceState) -> i32;
}
