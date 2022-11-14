use std::ffi::{c_void, c_char};

extern "C" {
    // Stream
    pub fn create_stream() -> *const c_void;
    pub fn destory_stream(stream: *const c_void);
    pub fn wait_stream(stream: *const c_void);
    pub fn wait_default_stream();
    pub fn get_inner_stream(stream: *const c_void) -> *const c_void;

    // Buffer
    pub fn alloc_gpu_buffer(bytes: usize) -> *const u8;
    pub fn free_gpu_buffer(buf: *const u8);

    pub fn alloc_locked_buffer(bytes: usize) -> *mut u8;
    pub fn free_locked_buffer(buffer: *mut u8);

    pub fn cu_memset(dev_ptr: *const u8, value: u8, count: usize);

    // Transport
    pub fn host_to_device(host: *const u8, device: *const u8, bytes: usize);
    pub fn device_to_host(device: *const u8, host: *const u8, bytes: usize);
    pub fn host_to_device_2d(host: *const u8, device: *const u8,
        host_pitch: usize, dev_pitch: usize, width: usize, height: usize);
    pub fn device_to_device(dst: *mut u8, src: *const u8, bytes: usize);

    // Device
    pub fn set_device(index: usize);
    pub fn set_device_flags(flags: u32);
    pub fn get_device_count() -> u32;
    pub fn get_sm_count() -> u32;
    pub fn get_device_cuda_core_count() -> u32;
    pub fn sync_device();
    pub fn check_and_sync();

    pub fn get_last_error() -> u32;
    pub fn get_error_string(code: u32) -> *const c_char;
    pub fn reset_device();
}
