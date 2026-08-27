use serde::Deserialize;
use std::collections::HashMap;

pub const NANOQ_V3_MAGIC: u32 = 0x4E515033;
pub const FOOTER_SIZE: usize = 32;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub offset: u64,
    pub size: u64,
    #[serde(default)]
    pub scale_offset: u64,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub block_size: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NanoqConfig {
    #[serde(default = "default_arch")]
    pub arch: String,
    #[serde(default = "default_vocab")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden")]
    pub hidden_size: i32,
    #[serde(default = "default_layers")]
    pub n_layers: i32,
    #[serde(default = "default_heads")]
    pub n_heads: i32,
    #[serde(default = "default_kv_heads")]
    pub n_kv_heads: i32,
    #[serde(default = "default_max_seq")]
    pub max_seq_len: i32,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    #[serde(default = "default_act")]
    pub act_fn: String,
}

fn default_arch() -> String { "gpt2".into() }
fn default_vocab() -> i32 { 50257 }
fn default_hidden() -> i32 { 768 }
fn default_layers() -> i32 { 6 }
fn default_heads() -> i32 { 12 }
fn default_kv_heads() -> i32 { 12 }
fn default_max_seq() -> i32 { 2048 }
fn default_norm_eps() -> f32 { 1e-5 }
fn default_rope() -> f32 { 10000.0 }
fn default_act() -> String { "gelu".into() }

#[derive(Debug)]
pub struct ParsedArchive {
    pub config: NanoqConfig,
    pub tensors: Vec<TensorEntry>,
    pub tokenizer_len: u32,
    pub payload_start: usize,
    pub payload_end: usize,
}

pub fn parse_v3_header(data: &[u8]) -> Result<ParsedArchive, String> {
    if data.len() < 16 {
        return Err("buffer too small".into());
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != NANOQ_V3_MAGIC {
        return Err(format!("invalid magic {magic:#x}"));
    }
    let index_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let index_start = 8;
    let index_end = index_start + index_len;
    if index_end > data.len() {
        return Err("truncated index".into());
    }
    let index_json = std::str::from_utf8(&data[index_start..index_end])
        .map_err(|e| e.to_string())?;
    let tensors: Vec<TensorEntry> = serde_json::from_str(index_json)
        .map_err(|e| format!("index parse error: {e}"))?;

    let config_len = u32::from_le_bytes(data[index_end..index_end + 4].try_into().unwrap()) as usize;
    let config_start = index_end + 4;
    let config_end = config_start + config_len;
    if config_end > data.len() {
        return Err("truncated config".into());
    }
    let config: NanoqConfig = serde_json::from_str(
        std::str::from_utf8(&data[config_start..config_end]).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("config parse error: {e}"))?;

    let tokenizer_len =
        u32::from_le_bytes(data[config_end..config_end + 4].try_into().unwrap());
    let tokenizer_start = config_end + 4;
    let tokenizer_end = tokenizer_start + tokenizer_len as usize;
    if tokenizer_end > data.len() {
        return Err("truncated tokenizer".into());
    }

    let payload_start = align_up(tokenizer_end, 64);
    if data.len() < payload_start + FOOTER_SIZE {
        return Err("missing payload/footer".into());
    }
    let payload_end = data.len() - FOOTER_SIZE;

    validate_tensor_bounds(&tensors, payload_start, payload_end)?;

    Ok(ParsedArchive {
        config,
        tensors,
        tokenizer_len,
        payload_start,
        payload_end,
    })
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) / align * align
}

fn validate_tensor_bounds(
    tensors: &[TensorEntry],
    payload_start: usize,
    payload_end: usize,
) -> Result<(), String> {
    for t in tensors {
        let end = payload_start as u64 + t.offset + t.size;
        if end > payload_end as u64 {
            return Err(format!("tensor {} out of bounds", t.name));
        }
        if t.scale_offset > 0 {
            let scale_end = payload_start as u64 + t.scale_offset + 4;
            if scale_end > payload_end as u64 {
                return Err(format!("tensor {} scale out of bounds", t.name));
            }
        }
    }
    Ok(())
}

pub fn verify_blake3_footer(data: &[u8]) -> Result<(), String> {
    if data.len() < FOOTER_SIZE {
        return Err("buffer too small for footer".into());
    }
    let body = &data[..data.len() - FOOTER_SIZE];
    let expected = &data[data.len() - FOOTER_SIZE..];
    let hash = blake3::hash(body);
    if hash.as_bytes() != expected {
        return Err("Blake3 footer mismatch".into());
    }
    Ok(())
}

pub fn tensor_map(tensors: &[TensorEntry]) -> HashMap<String, usize> {
    tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect()
}
