use crate::tick::genome::PopulationKind;
use crate::types::SENSOR_COUNT;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSnapshotReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub input_count: u16,
    pub reserved: u16,
    pub inputs: [f32; SENSOR_COUNT],
}
