use super::*;

use anyhow::{Context as _, Result};

impl Simulation {
    pub(super) fn resolve_combat(&mut self) -> Result<()> {
        profile_scope!("resolve_combat");
        if self.device_state.predator.capacity == 0 || self.device_state.prey.capacity == 0 {
            return Ok(());
        }

        cuda_memset_byte(self.device_state.prey_claimed_by, 0xFF, self.device_state.prey.capacity as usize)
            .context("moonai_gpu_simulation_reset_prey_claims")?;
        check_cuda_status(
            unsafe { dev_resolve_combat_claims(self.get_dev_state()) },
            "moonai_gpu_simulation_resolve_combat_claims",
        )?;
        let status = unsafe { dev_finalize_combat(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_finalize_combat")?;
        cuda_synchronize("moonai_gpu_simulation_resolve_combat")
    }
}

unsafe extern "C" {
    fn dev_resolve_combat_claims(state: *mut DeviceState) -> i32;
    fn dev_finalize_combat(state: *mut DeviceState) -> i32;
}
