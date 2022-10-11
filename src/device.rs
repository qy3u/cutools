use anyhow::{Result, bail};

use crate::ffi;

pub fn device_count() -> usize {
    unsafe { ffi::get_device_count() as usize }
}

pub fn sm_count() -> usize {
    unsafe { ffi::get_sm_count() as usize }
}

pub fn cuda_core_count() -> Result<usize> {
    let count = unsafe { ffi::get_device_cuda_core_count() };
    if count == 0 {
        bail!("Unknown device type");
    }


    Ok(count as usize)
}