macro_rules! cuda_check {
    ($result:expr) => {
        if $result != cudaError_t::cudaSuccess { Err(anyhow::anyhow!("CUDA error: {:?}", $result)) } else { Ok(()) }
    };
}
