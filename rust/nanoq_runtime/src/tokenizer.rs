use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use tokenizers::Tokenizer;

pub struct TokenizerHandle {
    inner: Tokenizer,
}

impl TokenizerHandle {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let inner = Tokenizer::from_bytes(data).map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, max_ids: usize) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().iter().take(max_ids).copied().collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, false).unwrap_or_default()
    }

    pub fn vocab_size(&self) -> i32 {
        self.inner.get_vocab_size(true) as i32
    }
}

#[no_mangle]
pub extern "C" fn nanoq_tokenizer_create(data: *const u8, len: usize) -> *mut TokenizerHandle {
    if data.is_null() || len == 0 {
        return ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    match TokenizerHandle::from_bytes(slice) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn nanoq_tokenizer_destroy(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[no_mangle]
pub extern "C" fn nanoq_tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    out_ids: *mut u32,
    max_ids: usize,
) -> i32 {
    if handle.is_null() || text.is_null() || out_ids.is_null() || max_ids == 0 {
        return -1;
    }
    let h = unsafe { &*handle };
    let s = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(valid_str) => valid_str,
        Err(_) => return -2,
    };
    let ids = h.encode(s, max_ids);
    let n = ids.len().min(max_ids);
    unsafe {
        ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, n);
    }
    n as i32
}

#[no_mangle]
pub extern "C" fn nanoq_tokenizer_decode(
    handle: *mut TokenizerHandle,
    ids: *const u32,
    num_ids: usize,
) -> *mut c_char {
    if handle.is_null() || ids.is_null() || num_ids == 0 {
        return ptr::null_mut();
    }
    let h = unsafe { &*handle };
    let slice = unsafe { std::slice::from_raw_parts(ids, num_ids) };
    match CString::new(h.decode(slice)) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn nanoq_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

#[no_mangle]
pub extern "C" fn nanoq_tokenizer_vocab_size(handle: *mut TokenizerHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).vocab_size() }
}
