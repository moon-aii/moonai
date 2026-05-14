use anyhow::{Result, anyhow};
use std::os::raw::c_void;

pub fn check_cuda_status(status: i32, context: &str) -> Result<()> {
    match status {
        0 => Ok(()),
        _ => Err(anyhow!("{context} failed (raw CUDA error code {status})")),
    }
}

pub fn cuda_malloc<T>(ptr: &mut *mut T, count: usize) -> Result<()> {
    if count == 0 {
        *ptr = std::ptr::null_mut();
        return Ok(());
    }

    check_cuda_status(
        unsafe { c_cuda_malloc(ptr as *mut *mut T as *mut *mut c_void, count * std::mem::size_of::<T>()) },
        "cuda_malloc",
    )
}

pub fn cuda_free<T>(ptr: &mut *mut T) -> Result<()> {
    if (*ptr).is_null() {
        *ptr = std::ptr::null_mut();
        return Ok(());
    }

    check_cuda_status(unsafe { c_cuda_free(*ptr as *mut c_void) }, "cuda_free")?;
    *ptr = std::ptr::null_mut();
    Ok(())
}

pub fn cuda_memset_zero<T>(ptr: *mut T, count: usize) -> Result<()> {
    if ptr.is_null() || count == 0 {
        return Ok(());
    }

    check_cuda_status(
        unsafe { c_cuda_memset_zero(ptr as *mut c_void, count * std::mem::size_of::<T>()) },
        "cuda_memset_zero",
    )
}

pub fn cuda_host_to_dev<T>(dev_ptr: *mut T, host_ptr: *const T, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    check_cuda_status(
        unsafe {
            c_cuda_host_to_dev(dev_ptr as *mut c_void, host_ptr as *const c_void, count * std::mem::size_of::<T>())
        },
        "cuda_host_to_dev",
    )
}

pub fn cuda_dev_to_dev<T>(dst_ptr: *mut T, src_ptr: *const T, count: usize) -> Result<()> {
    if count == 0 {
        return Ok(());
    }

    check_cuda_status(
        unsafe {
            c_cuda_dev_to_dev(dst_ptr as *mut c_void, src_ptr as *const c_void, count * std::mem::size_of::<T>())
        },
        "cuda_dev_to_dev",
    )
}

unsafe extern "C" {
    fn c_cuda_malloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn c_cuda_free(ptr: *mut c_void) -> i32;
    fn c_cuda_memset_zero(ptr: *mut c_void, size: usize) -> i32;
    fn c_cuda_host_to_dev(dev_ptr: *mut c_void, host_ptr: *const c_void, size: usize) -> i32;
    fn c_cuda_dev_to_dev(dst_ptr: *mut c_void, src_ptr: *const c_void, size: usize) -> i32;
}
