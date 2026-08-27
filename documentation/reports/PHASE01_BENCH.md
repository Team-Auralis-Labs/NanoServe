# Phase 01 Benchmark

- Date: 2026-08-27
- Fixture: `tests/fixtures/distilgpt2-int8.nanoq` (fp32 weights, v3 archive)
- Device: CPU

## 5-run inference (prompt="Hello benchmark", max_tokens=16)

| Run | wall (s) | maxrss (KB) |
|-----|----------|-------------|
| 1-3 aggregate | 4.76 | 993020 |

## Notes

- Coherent greedy decode for prompt `Hello`: top token 383 (` The`), matching HuggingFace distilgpt2 reference.
- Native forward golden: `test_transformer_gpt2` best_token=383, logit=-31.64.
