use serde::{Deserialize, Serialize};

pub const INPUT_NODE_TYPE: u8 = 0;
pub const HIDDEN_NODE_TYPE: u8 = 1;
pub const OUTPUT_NODE_TYPE: u8 = 2;
pub const BIAS_NODE_TYPE: u8 = 3;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PopulationKind {
    Predator = 0,
    Prey = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceGenomeBuffers {
    pub connection_from: *mut i32,
    pub connection_to: *mut i32,
    pub connection_weight: *mut f32,
    pub connection_innovation: *mut u32,
    pub connection_enabled: *mut u8,
    pub node_types: *mut u8,
    pub num_connections: *mut u16,
    pub num_nodes: *mut u16,
    pub connection_stride: u32,
    pub node_stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeededAgentSnapshot {
    pub population_kind: PopulationKind,
    pub slot: u32,
    pub entity_id: u32,
    pub generation: u32,
    pub species_id: u32,
    pub alive: u8,
    pub reserved0: u8,
    pub reserved1: u16,
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub energy: f32,
    pub age: f32,
    pub num_nodes: u16,
    pub num_connections: u16,
    pub genome_hash: u64,
}
