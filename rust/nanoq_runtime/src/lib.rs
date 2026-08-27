mod archive;
mod manifest;
mod tokenizer;
mod validate;

use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn nanoq_archive_validate(data: *const u8, len: usize) -> i32 {
    if data.is_null() || len == 0 {
        return -1;
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    validate::validate_buffer(slice)
}

#[no_mangle]
pub extern "C" fn nanoq_archive_validate_path(path: *const c_char) -> i32 {
    if path.is_null() {
        return -1;
    }
    let s = unsafe { CStr::from_ptr(path) }.to_string_lossy();
    validate::validate_path(&s)
}

pub use tokenizer::*;
