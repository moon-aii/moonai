#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaStatus {
    Success = 0,
}

impl CudaStatus {
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

pub fn check_cuda(status: CudaStatus, context: &str) -> anyhow::Result<()> {
    if status.is_success() { Ok(()) } else { Err(anyhow::anyhow!("{context} failed with status {status:?}")) }
}
