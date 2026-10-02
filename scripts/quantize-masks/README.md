# Quantize tensor-type masks

Text files consumed by `llama-quantize --tensor-type-file`. Each non-empty line is `regex=ggml_type` (tensor name regex is lower-cased by quantize; type names are case-insensitive).

## Qwen 3.6 UDT

| File | Purpose |
|------|---------|
| `qwen36-ud-base.txt` | Baseline dynamic recipe + MoE router input |
| `qwen36-ud-v1-nextn.txt` | Preserve NextN / MTP head weights |
| `qwen36-ud-v2-turbo3.txt` | Lift Q/K for TurboQuant3 KV stacks |
| `qwen36-ud-v3-combined.txt` | Default release (v1 ∪ v2) |

See [docs/qwen-udt/RUNBOOK.md](../../docs/qwen-udt/RUNBOOK.md).
