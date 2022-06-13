use std::ffi::c_void;

extern "C" {
    // Stream
    pub fn create_stream() -> *const c_void;
    pub fn destory_stream(stream: *const c_void);
    pub fn wait_stream(stream: *const c_void);
    pub fn wait_default_stream();
    pub fn get_inner_stream(stream: *const c_void) -> *const c_void;

    // Buffer
    pub fn alloc_gpu_buffer(bytes: usize) -> *const u8;
    #[cfg(not(feature = "cache-buffer"))]
    pub fn free_gpu_buffer(buf: *const u8);

    pub fn alloc_locked_buffer(bytes: usize) -> *mut u8;
    #[cfg(not(feature = "cache-buffer"))]
    pub fn free_locked_buffer(buffer: *mut u8);

    #[cfg(feature = "cache-buffer")]
    pub fn cu_memset(dev_ptr: *const u8, value: u8, count: usize);

    // Transport
    pub fn host_to_device(host: *const u8, device: *const u8, bytes: usize);
    pub fn device_to_host(device: *const u8, host: *const u8, bytes: usize);
    pub fn host_to_device_with_stream(
        host: *const u8,
        device: *const u8,
        bytes: usize,
        stream: *const c_void,
    );
    pub fn device_to_host_with_stream(
        device: *const u8,
        host: *const u8,
        bytes: usize,
        stream: *const c_void,
    );

    // Utils
    pub fn set_device(index: usize);
    pub fn check_and_sync();
    pub fn get_device_count() -> u32;
    pub fn get_device_cuda_core_count() -> i32;
}
