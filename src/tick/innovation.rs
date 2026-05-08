use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInnovationState {
    pub next_innovation: u32,
    pub next_node_id: u32,
    pub log_capacity: u32,
    pub log_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InnovationRecord {
    pub from_node: u32,
    pub to_node: u32,
    pub innovation: u32,
    pub record_kind: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InnovationLogReadbackHeader {
    pub total_len: u32,
    pub stored_len: u32,
    pub returned_len: u32,
    pub dropped_len: u32,
}
