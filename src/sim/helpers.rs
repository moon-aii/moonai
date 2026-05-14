use anyhow::{Result, anyhow};
use std::os::raw::{c_void};

pub fn check_cuda_status(status: i32, context: &str) -> Result<()> {
    match status {
        0 => Ok(()),
        _ => Err(anyhow!("{context} failed (raw CUDA error code {status})")),
    }
}

pub fn cuda_malloc<T>(ptr: &mut *mut T, count: usize) -> Result<()> {
    check_cuda_status( unsafe { c_cuda_malloc(ptr as *mut *mut T as *mut *mut c_void, count * std::mem::size_of::<T>()) }, "cuda_malloc")
}

pub fn cuda_free<T>(ptr: &mut *mut T) -> Result<()> {
    check_cuda_status( unsafe { c_cuda_free(ptr as *mut *mut T as *mut *mut c_void) }, "cuda_free")
}

pub fn cuda_memset_zero<T>(ptr: &mut *mut T, size: usize) -> Result<()> {
    check_cuda_status( unsafe { c_cuda_memset_zero(ptr as *mut *mut T as *mut *mut c_void, size) }, "cuda_memset_zero")
}

pub fn cuda_dev_to_host<T>(dev_ptr: &mut *mut T, host_ptr: &mut *mut T, size: usize) -> Result<()> {
    check_cuda_status( unsafe { c_cuda_dev_to_host(dev_ptr as *mut *mut T as *mut *mut c_void, host_ptr as *mut *mut T as *mut *mut c_void, size) }, "cuda_dev_to_host")
}

pub fn cuda_host_to_dev<T>(dev_ptr: &mut *mut T, host_ptr: &mut *mut T, size: usize) -> Result<()> {
    check_cuda_status( unsafe { c_cuda_host_to_dev(dev_ptr as *mut *mut T as *mut *mut c_void, host_ptr as *mut *mut T as *mut *mut c_void, size) }, "cuda_host_to_dev")
}

pub fn cuda_sync() -> Result<()> {
    check_cuda_status( unsafe { c_cuda_sync() }, "cuda_sync")
}

unsafe extern "C" {
    fn c_cuda_malloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn c_cuda_free(ptr: *mut *mut c_void) -> i32;
    fn c_cuda_memset_zero(ptr: *mut *mut c_void, size: usize) -> i32;
    fn c_cuda_dev_to_host(dev_ptr: *mut *mut c_void, host_ptr: *mut *mut c_void, size: usize) -> i32;
    fn c_cuda_host_to_dev(dev_ptr: *mut *mut c_void, host_ptr: *mut *mut c_void, size: usize) -> i32;
    fn c_cuda_sync() -> i32;
}
