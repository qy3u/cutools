use std::slice;

use anyhow::{ensure, Result};

use crate::ffi;

pub struct DeviceBuffer {
    inner: DevicePtr,
}

impl DeviceBuffer {
    pub fn new(len: usize) -> Self {
        let inner = DevicePtr {
            ptr: unsafe { ffi::alloc_gpu_buffer(len) },
            len,
        };

        Self { inner }
    }

    pub const fn null() -> Self {
        Self {
            inner: DevicePtr::null(),
        }
    }

    pub fn reset(&mut self) {
        unsafe {
            ffi::cu_memset(self.inner.ptr, 0, self.len());
        }
    }

    pub fn from_slice<T>(s: &[T]) -> Self {
        assert!(s.len() > 0);

        let len = std::mem::size_of::<T>() * s.len();
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, len) };

        let mut buf = Self::new(len);
        buf.write_from(s.into()).unwrap();

        buf
    }

    pub fn from_slice_padded<T>(s: &[T], dev_pitch: usize) -> Self {
        assert!(s.len() > 0);
        let src_pitch = std::mem::size_of::<T>();
        assert!(dev_pitch >= src_pitch, "dev_pitch <= src_pitch");

        let len = dev_pitch * s.len();

        let width = src_pitch;
        let height = s.len();

        let buf = Self::new(len);
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, src_pitch * s.len()) };

        buf.ptr()
            .write_from_2d(s, src_pitch, dev_pitch, width, height)
            .unwrap();
        buf
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn ptr(&self) -> DevicePtr {
        self.inner
    }

    pub fn ptr2(&self, offset: isize, len: usize) -> DevicePtr {
        self.inner.ptr2(offset, len)
    }


    pub fn rptr(&self) -> *const u8 {
        self.inner.ptr()
    }

    pub fn rptr_mut(&mut self) -> *mut u8 {
        self.inner.ptr_mut()
    }

    pub fn load(&self) -> Vec<u8> {
        let mut res = vec![0u8; self.len()];
        self.read_into((&mut res[..]).into()).unwrap();
        res
    }

    pub fn load_to_pinned_mem(&self) -> CudaPinnedMemBuffer {
        let res = CudaPinnedMemBuffer::new(self.len());
        self.read_into(res.ptr()).unwrap();
        res
    }

    pub fn load_as<T: Copy + Default>(&self) -> Vec<T> {
        assert_eq!(self.len() % std::mem::size_of::<T>(), 0);
        let mut res = vec![T::default(); self.len() / std::mem::size_of::<T>()];
        self.read_into((&mut res[..]).into()).unwrap();
        res
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }
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
}

pub struct CudaPinnedMemBuffer {
    inner: *mut u8,
    size: usize,
}

impl CudaPinnedMemBuffer {
    pub fn new(size: usize) -> Self {
        let res = unsafe {
            Self {
                inner: ffi::alloc_locked_buffer(size),
                size,
            }
        };
        res
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn ptr(&self) -> HostPtr {
        HostPtr {
            ptr: self.inner,
            len: self.len(),
        }
    }

    pub fn reset(&mut self) {
        self.as_mut().fill(0);
    }
}

impl Drop for CudaPinnedMemBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::free_locked_buffer(self.inner);
        }
    }
}

impl AsMut<[u8]> for CudaPinnedMemBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.inner, self.size) }
    }
}

impl AsRef<[u8]> for CudaPinnedMemBuffer {
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.inner, self.size) }
    }
}

impl<T> From<&[T]> for CudaPinnedMemBuffer {
    fn from(v: &[T]) -> Self {
        let len = std::mem::size_of::<T>() * v.len();
        let mut buf = Self::new(len);

        let src = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, len) };

        buf.as_mut().copy_from_slice(src);

        buf
    }
}

impl<T> From<Vec<T>> for CudaPinnedMemBuffer {
    fn from(v: Vec<T>) -> Self {
        let len = std::mem::size_of::<T>() * v.len();
        let mut buf = Self::new(len);

        let src = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, len) };

        buf.as_mut().copy_from_slice(src);

        buf
    }
}

unsafe impl Send for CudaPinnedMemBuffer {}

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

    pub const fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
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

    pub fn write_from_2d(
        &mut self,
        src: &[u8],
        src_pitch: usize,
        dev_pitch: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        ensure!(src.len() % src_pitch == 0, "invalid src and src_pitch");

        ensure!(
            src.len() / src_pitch * dev_pitch <= self.len(),
            "expected length must less than device ptr"
        );

        unsafe {
            ffi::host_to_device_2d(src.as_ptr(), self.ptr, src_pitch, dev_pitch, width, height);
        }

        Ok(())
    }

    pub fn read_into(&self, dst: &mut [u8]) -> Result<()> {
        ensure!(
            self.len() <= dst.len(),
            "length of device ptr must less than dst"
        );
        unsafe {
            ffi::device_to_host(self.ptr, dst.as_ptr(), self.len());
        }
        Ok(())
    }

    pub fn copy_from(&mut self, src: DevicePtr) -> Result<()> {
        ensure!(
            src.len() == self.len(),
            "DevicePtr copy_from: length of src and dst must be euqal"
        );

        unsafe { ffi::device_to_device(self.ptr_mut(), src.ptr(), src.len()) }

        Ok(())
    }
}

unsafe impl Send for DevicePtr {}

#[derive(Clone, Copy)]
pub struct HostPtr {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for HostPtr {}

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

impl<T: Copy> From<&Vec<T>> for HostPtr {
    fn from(data: &Vec<T>) -> Self {
        (&data[..]).into()
    }
}

impl<T: Copy> From<&[T]> for HostPtr {
    fn from(data: &[T]) -> Self {
        let len = data.len();
        assert_ne!(len, 0, "convert an empty slice into HostPtr");

        let width = std::mem::size_of::<T>();
        let slice =
            unsafe { std::slice::from_raw_parts(&data[0] as *const T as *const u8, len * width) };

        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
}

impl<T: Copy> From<&mut [T]> for HostPtr {
    fn from(data: &mut [T]) -> Self {
        let len = data.len();
        assert_ne!(len, 0, "convert an empty slice into HostPtr");

        let width = std::mem::size_of::<T>();
        let slice =
            unsafe { std::slice::from_raw_parts(&data[0] as *const T as *const u8, len * width) };

        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
}
