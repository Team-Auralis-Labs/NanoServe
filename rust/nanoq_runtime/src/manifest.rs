use crate::archive::NanoqConfig;
use serde_json::json;

pub fn model_card_json(config: &NanoqConfig, path: &str) -> String {
    json!({
        "arch": config.arch,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "path": path,
        "format": "nanoq_v3",
    })
    .to_string()
}
