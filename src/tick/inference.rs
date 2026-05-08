use crate::tick::genome::PopulationKind;

pub const SENSOR_COUNT: usize = 35;
pub const OUTPUT_COUNT: usize = 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorSnapshotReadback {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub input_count: u16,
    pub reserved: u16,
    pub inputs: [f32; SENSOR_COUNT],
}
