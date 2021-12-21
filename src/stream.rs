use std::ffi::c_void;

use crate::ffi::{create_stream, destory_stream, get_inner_stream, wait_stream};

#[cfg(feature = "global-stream")]
use lazy_static::lazy_static;

#[cfg(feature = "global-stream")]
use waitpool::Pooled;

#[cfg(feature = "global-stream")]
lazy_static! {
    pub static ref DEFAULT_STREAM: Stream = Stream::new();
    static ref STREAM_POOL: waitpool::Pool<Stream> = {
        let mut pool = waitpool::Pool::new();
        for _ in 0..14 {
            pool.pool(Stream::new());
        }
        pool
    };
}

pub struct Stream(*const c_void);

impl Stream {
    pub fn new() -> Self {
        unsafe { Self(create_stream()) }
    }

    pub fn wait(&self) {
        unsafe { wait_stream(self.ptr()) };
    }

    pub fn ptr(&self) -> *const c_void {
        unsafe { get_inner_stream(self.0) }
    }

    #[cfg(feature = "global-stream")]
    pub fn random() -> Pooled<Self> {
        STREAM_POOL.get()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { destory_stream(self.ptr()) };
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}
