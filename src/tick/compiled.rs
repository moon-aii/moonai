use serde::{Deserialize, Serialize};

use crate::tick::genome::PopulationKind;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceCompiledNetworkBuffers {
    pub eval_order: *mut u16,
    pub connection_offsets: *mut u32,
    pub output_indices: *mut u16,
    pub connection_sources: *mut u16,
    pub connection_weights: *mut f32,
    pub node_counts: *mut u16,
    pub eval_counts: *mut u16,
    pub connection_counts: *mut u16,
    pub node_stride: u32,
    pub connection_stride: u32,
    pub output_stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNetworkReadbackHeader {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub node_count: u16,
    pub eval_node_count: u16,
    pub output_count: u16,
    pub connection_count: u16,
}
