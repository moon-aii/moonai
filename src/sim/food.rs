use super::*;

use anyhow::{Context as _, Result};

impl Simulation {
    pub(super) fn resolve_food(&mut self) -> Result<()> {
        profile_scope!("resolve_food");
        if self.device_state.prey.capacity == 0 || self.device_state.food.capacity == 0 {
            return Ok(());
        }

        cuda_memset_byte(self.device_state.food_claimed_by, 0xFF, self.device_state.food.capacity as usize)
            .context("moonai_gpu_simulation_reset_food_claims")?;
        check_cuda_status(
            unsafe { dev_resolve_food_claims(self.get_dev_state()) },
            "moonai_gpu_simulation_resolve_food_claims",
        )?;
        check_cuda_status(unsafe { dev_finalize_food(self.get_dev_state()) }, "moonai_gpu_simulation_finalize_food")?;
        let status = unsafe { dev_respawn_food(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_respawn_food")
    }
}

unsafe extern "C" {
    fn dev_resolve_food_claims(state: *mut DeviceState) -> i32;
    fn dev_finalize_food(state: *mut DeviceState) -> i32;
    fn dev_respawn_food(state: *mut DeviceState) -> i32;
}
