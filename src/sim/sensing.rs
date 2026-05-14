use super::*;

use anyhow::Result;

impl Simulation {
    pub(super) fn compute_sensor_inputs(&mut self) -> Result<()> {
        profile_scope!("sensor_inputs");
        let status = unsafe { dev_compute_sensor_inputs(self.get_dev_state()) };
        check_cuda_status(status, "moonai_gpu_simulation_compute_sensor_inputs")
    }

    pub(super) fn read_sensor_snapshot(
        &mut self,
        population_kind: PopulationKind,
        slot: u32,
    ) -> Result<SensorSnapshotReadback> {
        let status = unsafe { dev_write_sensor_snapshot(self.get_dev_state(), population_kind, slot) };
        check_cuda_status(status, "moonai_gpu_simulation_sensor_snapshot")?;
        device_read("moonai_gpu_simulation_sensor_snapshot_readback", self.device_state.sensor_snapshot_scratch)
    }
}

unsafe extern "C" {
    fn dev_compute_sensor_inputs(state: *mut DeviceState) -> i32;
    fn dev_write_sensor_snapshot(state: *mut DeviceState, population_kind: PopulationKind, slot: u32) -> i32;
}
