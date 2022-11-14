use std::ffi::c_void;

use crate::ffi::{
    create_stream, destory_stream, get_inner_stream, wait_default_stream, wait_stream,
};

pub struct Stream(*const c_void);

impl Stream {
    pub fn new() -> Self {
        unsafe { Self(create_stream()) }
    }

    pub fn wait(&self) {
        unsafe { wait_stream(self.ptr()) };
    }

    pub fn wait_default() {
        unsafe { wait_default_stream() };
    }

    pub fn ptr(&self) -> *const c_void {
        unsafe { get_inner_stream(self.0) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { destory_stream(self.ptr()) };
    }
}

unsafe impl Send for Stream {}
