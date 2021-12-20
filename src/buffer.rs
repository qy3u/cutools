use std::slice;

use anyhow::{ensure, Result};
use log::debug;

use crate::ffi;
use crate::stream::Stream;

pub struct DeviceBuffer {
    inner: DevicePtr,
}

impl DeviceBuffer {
    pub fn new(len: usize) -> Self {
        let mut inner = DevicePtr {
            ptr: unsafe { ffi::alloc_gpu_buffer(len) },
            len,
        };

        inner.write_from(&[0u8]).unwrap();

        Self { inner }
    }

    pub fn from_slice<T>(s: &[T]) -> Self {
        assert!(s.len() > 0);

        let len = std::mem::size_of::<T>() * s.len();
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, len) };

        let inner = DevicePtr {
            ptr: unsafe { ffi::alloc_gpu_buffer(len) },
            len,
        };

        let mut buf = Self { inner };
        buf.write_from(s.into()).unwrap();

        buf
    }

    pub fn from_slice_with_stream<T>(s: &[T], stream: &Stream) -> Self {
        assert!(s.len() > 0);

        let len = std::mem::size_of::<T>() * s.len();
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, len) };

        let inner = DevicePtr {
            ptr: unsafe { ffi::alloc_gpu_buffer(len) },
            len,
        };

        let mut buf = Self { inner };
        buf.write_from_with_stream(s.into(), &stream).unwrap();

        buf
    }

    pub fn ptr(&self) -> DevicePtr {
        self.inner
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn ptr2(&self, offset: isize, len: usize) -> DevicePtr {
        self.inner.ptr2(offset, len)
    }

    pub fn load(&self) -> Vec<u8> {
        let mut res = vec![0u8; self.len()];
        self.read_into((&mut res[..]).into()).unwrap();
        return res;
    }

    pub fn load_with_stream(&self, stream: &Stream) -> Vec<u8> {
        let mut res = vec![0u8; self.len()];
        self.read_into_with_stream((&mut res[..]).into(), stream)
            .unwrap();
        return res;
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        debug!("free gpu buffer");
        unsafe { ffi::free_gpu_buffer(self.ptr().ptr()) }
    }
}

impl DeviceBuffer {
    pub fn read_into(&self, mut dst: HostPtr) -> Result<HostPtr> {
        self.ptr().read_into(dst.as_mut())?;
        Ok(dst)
    }

    pub fn write_from(&mut self, src: HostPtr) -> Result<DevicePtr> {
        self.ptr().write_from(src.as_ref())?;
        Ok(self.ptr())
    }

    pub fn read_into_with_stream(&self, mut dst: HostPtr, stream: &Stream) -> Result<HostPtr> {
        self.ptr().read_into_with_stream(dst.as_mut(), stream)?;
        Ok(dst)
    }

    pub fn write_from_with_stream(&mut self, src: HostPtr, stream: &Stream) -> Result<DevicePtr> {
        self.ptr().write_from_with_stream(src.as_ref(), stream)?;
        Ok(self.ptr())
    }
}

pub struct CudaLockedMemBuffer {
    inner: *mut u8,
    size: usize,
}

impl CudaLockedMemBuffer {
    pub fn new(size: usize) -> Self {
        unsafe {
            Self {
                inner: ffi::alloc_locked_buffer(size),
                size,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

impl Drop for CudaLockedMemBuffer {
    fn drop(&mut self) {
        debug!("dropping CudaLockedMemBuffer");
        unsafe {
            ffi::free_locked_buffer(self.inner);
        }
    }
}

impl AsMut<[u8]> for CudaLockedMemBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.inner, self.size) }
    }
}

impl AsRef<[u8]> for CudaLockedMemBuffer {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.inner, self.size) }
    }
}

unsafe impl Send for CudaLockedMemBuffer {}

#[derive(Clone, Copy)]
pub struct DevicePtr {
    ptr: *const u8,
    len: usize,
}

impl DevicePtr {
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn ptr_mut(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn ptr2(&self, offset: isize, len: usize) -> DevicePtr {
        DevicePtr {
            ptr: unsafe { self.ptr.offset(offset) },
            len,
        }
    }

    pub fn write_from(&mut self, src: &[u8]) -> Result<()> {
        ensure!(
            src.len() <= self.len(),
            "length of src must less than device ptr"
        );
        unsafe {
            ffi::host_to_device(src.as_ptr(), self.ptr, src.len());
        }
        Ok(())
    }

    pub fn write_from_with_stream(&mut self, src: &[u8], stream: &Stream) -> Result<()> {
        ensure!(
            src.len() <= self.len(),
            "length of src must less than device ptr"
        );

        unsafe {
            ffi::host_to_device_with_stream(src.as_ptr(), self.ptr, src.len(), stream.ptr());
        }
        Ok(())
    }

    pub fn read_into(&mut self, dst: &mut [u8]) -> Result<()> {
        ensure!(
            self.len() <= dst.len(),
            "length of device ptr must less than dst"
        );
        unsafe {
            ffi::device_to_host(self.ptr, dst.as_ptr(), self.len());
        }
        Ok(())
    }

    pub fn read_into_with_stream(&mut self, dst: &mut [u8], stream: &Stream) -> Result<()> {
        ensure!(
            self.len() <= dst.len(),
            "length of device ptr must less than dst"
        );
        unsafe {
            ffi::device_to_host_with_stream(self.ptr, dst.as_ptr(), self.len(), stream.ptr());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct HostPtr {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for DevicePtr {}
unsafe impl Send for HostPtr {}

unsafe impl Sync for DevicePtr {}
unsafe impl Sync for HostPtr {}

impl HostPtr {
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl AsRef<[u8]> for HostPtr {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for HostPtr {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

impl From<&[u8]> for HostPtr {
    fn from(slice: &[u8]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
}

impl From<&mut [u8]> for HostPtr {
    fn from(slice: &mut [u8]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
}

impl From<&Vec<u8>> for HostPtr {
    fn from(data: &Vec<u8>) -> Self {
        (&data[..]).into()
    }
}

// impl<T> From<&[T]> for HostPtr {
//     fn from(slice: &[T]) -> Self {
//         let ptr = slice.as_ptr() as *const u8;
//         let len = std::mem::size_of::<T>() * slice.len();
//         Self { ptr, len }
//     }
// }
