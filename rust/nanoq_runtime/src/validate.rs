use crate::archive::{parse_v3_header, verify_blake3_footer, NANOQ_V3_MAGIC};
use std::fs;
use std::path::Path;

pub fn validate_buffer(data: &[u8]) -> i32 {
    if data.len() < 4 {
        return -1;
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != NANOQ_V3_MAGIC {
        return -2;
    }
    if verify_blake3_footer(data).is_err() {
        return -3;
    }
    if parse_v3_header(data).is_err() {
        return -4;
    }
    0
}

pub fn validate_path(path: &str) -> i32 {
    let p = Path::new(path);
    if !p.exists() {
        return -1;
    }
    match fs::read(p) {
        Ok(data) => validate_buffer(&data),
        Err(_) => -1,
    }
}
