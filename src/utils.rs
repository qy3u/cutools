use std::cell::RefCell;

use crate::ffi;

pub fn set_device_for_thread() {
    thread_local!(static INIT: RefCell<bool> = RefCell::new(false));

    INIT.with(|init| {
        if !*init.borrow() {
            unsafe {
                ffi::set_device(0); // always set to the first device
            }
        }
        *init.borrow_mut() = true;
    })
}

pub fn check_and_sync() {
    unsafe {
        ffi::check_and_sync();
    }
}
