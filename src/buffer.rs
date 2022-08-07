use std::slice;

use anyhow::{ensure, Result};

use crate::ffi;
use crate::stream::Stream;

#[cfg(feature = "cache-buffer")]
use {
    lazy_static::lazy_static,
    std::{collections::HashMap, sync::Mutex},
};

#[cfg(feature = "trace")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "trace")]
static HIT: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "trace")]
static MISS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "cache-buffer")]
lazy_static! {
    static ref DEV_BUF_CACHE: Mutex<HashMap<usize, Vec<DeviceBuffer>>> = Mutex::new(HashMap::new());
    static ref LOCKED_CACHE: Mutex<HashMap<usize, Vec<CudaLockedMemBuffer>>> =
        Mutex::new(HashMap::new());
}

pub struct DeviceBuffer {
    inner: DevicePtr,
}

impl DeviceBuffer {
    pub fn new(len: usize) -> Self {
        #[cfg(feature = "cache-buffer")]
        {
            let mut guard = DEV_BUF_CACHE.lock().unwrap();

            let bufs = guard.entry(len).or_insert(Vec::new());
            return match bufs.pop() {
                Some(buf) => {
                    #[cfg(feature = "trace")]
                    {
                        let hit = HIT.fetch_add(1, Ordering::Relaxed) + 1;
                        if hit % 1000 == 0 {
                            let miss = MISS.load(Ordering::Relaxed);
                            log::trace!("device buffer cache hit: {}, miss: {}", hit, miss);
                        }
                    }

                    buf
                }
                None => {
                    #[cfg(feature = "trace")]
                    {
                        MISS.fetch_add(1, Ordering::Relaxed);
                        log::trace!("miss device buffer for size: {}bytes", len);
                        let stat = guard
                            .iter()
                            .map(|(k, v)| format!("{} * {}bytes", v.len(), k))
                            .collect::<Vec<String>>()
                            .join(",");
                        log::trace!("current cached buffers: {}", stat);
                    }

                    Self::_new(len)
                }
            };
        }

        #[cfg(not(feature = "cache-buffer"))]
        Self::_new(len)
    }

    fn _new(len: usize) -> Self {
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
            ffi::cu_memset(self.inner.ptr , 0, self.len());
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
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, len) };

        buf.ptr().write_from_2d(s, src_pitch, dev_pitch, width, height).unwrap();
        buf
    }

    pub fn from_slice_with_stream<T>(s: &[T], stream: &Stream) -> Self {
        assert!(s.len() > 0);

        let len = std::mem::size_of::<T>() * s.len();
        let s = unsafe { std::slice::from_raw_parts(&s[0] as *const T as *const u8, len) };

        let mut buf = Self::new(len);
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
        #[cfg(feature = "cache-buffer")]
        {
            self.reset();

            let inner = self.inner;
            let mut lock = DEV_BUF_CACHE.lock().unwrap();
            lock.entry(self.len())
                .or_insert(Vec::new())
                .push(Self { inner });
        }

        #[cfg(not(feature = "cache-buffer"))]
        {
            if self.inner.is_null() {
                return;
            }
            unsafe { ffi::free_gpu_buffer(self.ptr().ptr()) }
        }
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
    pub fn new(len: usize) -> Self {
        #[cfg(feature = "cache-buffer")]
        {
            let mut lock = LOCKED_CACHE.lock().unwrap();

            let bufs = lock.entry(len).or_insert(Vec::new());
            return bufs.pop().unwrap_or_else(|| Self::_new(len));
        }

        #[cfg(not(feature = "cache-buffer"))]
        Self::_new(len)
    }

    pub fn _new(size: usize) -> Self {
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

impl Drop for CudaLockedMemBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cache-buffer")]
        {
            self.reset();

            let mut lock = LOCKED_CACHE.lock().unwrap();
            lock.entry(self.len()).or_insert(Vec::new()).push(Self {
                inner: self.inner,
                size: self.size,
            });
        }

        #[cfg(not(feature = "cache-buffer"))]
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

impl<T> From<&[T]> for CudaLockedMemBuffer {
    fn from(v: &[T]) -> Self {
        let len = std::mem::size_of::<T>() * v.len();
        let mut buf = Self::new(len);

        let src = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, len) };

        buf.as_mut().copy_from_slice(src);

        buf
    }
}

impl<T> From<Vec<T>> for CudaLockedMemBuffer {
    fn from(v: Vec<T>) -> Self {
        let len = std::mem::size_of::<T>() * v.len();
        let mut buf = Self::new(len);

        let src = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, len) };

        buf.as_mut().copy_from_slice(src);

        buf
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
        height: usize
    ) -> Result<()> {
        ensure!(
            src.len() % src_pitch == 0,
            "invalid src and src_pitch"
        );

        ensure!(
            src.len() / src_pitch * dev_pitch <= self.len(),
            "expected length must less than device ptr"
        );

        unsafe {
            ffi::host_to_device_2d(src.as_ptr(), self.ptr, src_pitch, dev_pitch, width, height);
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

    pub fn write_from_2d_with_stream(
        &mut self,
        src: &[u8],
        src_pitch: usize,
        dev_pitch: usize,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            src.len() % src_pitch == 0,
            "invalid src and src_pitch"
        );

        ensure!(
            src.len() / src_pitch * dev_pitch <= self.len(),
            "expected length must less than device ptr"
        );

        unsafe {
            ffi::host_to_device_2d_with_stream(src.as_ptr(), self.ptr, src_pitch, dev_pitch, width, height, stream.ptr());
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

    pub unsafe fn to_owned(&self) -> Vec<u8> {
        let slice = std::slice::from_raw_parts(self.ptr, self.len);

        let mut v = vec![0; self.len];
        v.copy_from_slice(slice);

        v
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
