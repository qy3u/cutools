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

pub fn count() -> usize {
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

pub fn set_device(index: usize) {
    assert!(index < count(), "invalid index for set device: {}", index);

    unsafe {
        ffi::set_device(index);
    }
}

pub fn sync() {
    unsafe { ffi::sync_device() }
}

pub fn check_and_sync() {
    unsafe { ffi::check_and_sync() }
}

pub fn get_last_error() -> Result<(), String> {
    let code = unsafe { crate::ffi::get_last_error() };
    if code != 0 {
        let err_msg = unsafe {
            std::ffi::CStr::from_ptr(crate::ffi::get_error_string(code))
        };
        Err(err_msg.to_str().unwrap().to_owned())
    } else {
        Ok(())
    }
}

pub fn reset() {
    unsafe { ffi::reset_device() }
}
