use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn infer_populations(&mut self) -> Result<()> {
        profile_scope!("inference");
        check_cuda_status(
            unsafe { dev_infer_population(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_infer_predator_population",
        )?;
        check_cuda_status(
            unsafe { dev_infer_population(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_infer_prey_population",
        )?;
        cuda_synchronize("moonai_gpu_simulation_infer_populations")
    }

    pub(super) fn update_population_vitals(&mut self) -> Result<()> {
        profile_scope!("update_vitals");
        check_cuda_status(
            unsafe { dev_update_vitals(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_update_predator_vitals",
        )?;
        check_cuda_status(
            unsafe { dev_update_vitals(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_update_prey_vitals",
        )?;
        cuda_synchronize("moonai_gpu_simulation_update_population_vitals")
    }

    pub(super) fn apply_population_movement(&mut self) -> Result<()> {
        profile_scope!("apply_movement");
        check_cuda_status(
            unsafe { dev_apply_movement(self.get_dev_state(), PopulationKind::Predator) },
            "moonai_gpu_simulation_apply_predator_movement",
        )?;
        check_cuda_status(
            unsafe { dev_apply_movement(self.get_dev_state(), PopulationKind::Prey) },
            "moonai_gpu_simulation_apply_prey_movement",
        )?;
        cuda_synchronize("moonai_gpu_simulation_apply_population_movement")
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
