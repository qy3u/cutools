pub mod buffer;
pub mod device;
mod ffi;
pub mod stream;
pub mod utils;

pub fn gws(n: usize, lws: usize) -> usize {
    (n + lws - 1) / lws + 1
}
