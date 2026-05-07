use serde::{Deserialize, Serialize};

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStatus {
    Success = 0,
    InvalidArgument = 1,
    AllocationFailed = 2,
    KernelLaunchFailed = 3,
    DeviceCopyFailed = 4,
    RuntimeUnavailable = 5,
}

impl CudaStatus {
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

pub fn check_cuda(status: CudaStatus, context: &str) -> anyhow::Result<()> {
    if status.is_success() { Ok(()) } else { Err(anyhow::anyhow!("{context} failed with status {status:?}")) }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantCheckReadback {
    pub predator_agents_checked: u32,
    pub prey_agents_checked: u32,
    pub invalid_node_counts: u32,
    pub invalid_connection_counts: u32,
    pub invalid_connection_bounds: u32,
    pub invalid_compiled_offsets: u32,
    pub invalid_eval_nodes: u32,
    pub invalid_output_indices: u32,
    pub invalid_species_assignments: u32,
    pub innovation_log_overflow: u32,
}
