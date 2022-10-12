use anyhow::{Result, bail};

use crate::ffi;


// cudaDeviceScheduleAuto: u32 = 0;
// cudaDeviceScheduleSpin: u32 = 1;
// cudaDeviceScheduleYield: u32 = 2;
// cudaDeviceScheduleBlockingSync: u32 = 4;
#[derive(Debug, Copy, Clone)]
pub enum SyncMode {
    Auto = 0,
    Spin = 1,
    Yield = 2,
    Blocking = 4,
}

pub fn device_count() -> usize {
    unsafe { ffi::get_device_count() as usize }
}

pub fn set_sync_mode(mode: SyncMode) {
    unsafe { ffi::set_device_flags(mode as u32)}
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