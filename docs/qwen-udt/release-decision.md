# Qwen 3.6 UDT — release decision log

**Status:** pending benchmark data.

After running:

1. `scripts/bench-qwen-udt-matrix-local.sh` (throughput + acceptance)
2. `scripts/bench-qwen-udt-quality.sh` (WikiText-2 PPL)

fill the table below. Per `(model, bit-width)` publish **`v3`** by default. Publish **`v1`** or **`v2`** separately only if they beat `v3` on their target metric without regressing PPL / unrelated modes.

## Matrix

| Model | Bit (UDT) | PPL (wiki) v3 | PPL v1 | PPL v2 | Notes |
|-------|-----------|---------------|--------|--------|-------|
| 27B | Q3_K_XL | | | | |
| 27B | Q4_K_XL | | | | |
| 27B | Q5_K_XL | | | | |
| 27B | Q6_K_XL | | | | |
| 35B-A3B | Q3_K_XL | | | | |
| 35B-A3B | Q4_K_XL | | | | |
| 35B-A3B | Q5_K_XL | | | | |
| 35B-A3B | Q6_K_XL | | | | |

## Ship list

- [ ] `AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF` — files:
- [ ] `AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF` — files:

## Sign-off

- Date:
- Reviewer:
