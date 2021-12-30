use crate::ffi;

pub fn get_device_count() -> u32 {
    unsafe { ffi::get_device_count() }
}

pub fn get_device_cuda_core_count() -> u32 {
    let count = unsafe { ffi::get_device_cuda_core_count() };
    if count == -1 {
        panic!("Unknown device type");
    }

    count as u32
}
