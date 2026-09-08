# AMD ROCm / HIP: status, benchmark plan, roadmap

What this fork ships for AMD GPUs today, what we measure to compare ROCm (HIP)
against Vulkan with the TurboQuant KV cache, and what is still open. Written
for the team and for the AMD partner conversation; numbers go into
`docs/amd/results/` and a dated `benchmarks-<date>.md` next to this file.

## 1. What ships

| Archive | Targets | Runtime |
|---|---|---|
| `llama-turboquant-linux-x64-rocm.tar.gz` | RDNA2-RDNA4: gfx1030, gfx1100/1101/1102/1103, gfx1150/1151 (Ryzen AI 300 / Strix Halo), gfx1200/1201; CDNA: gfx90a, gfx942, gfx950 (MI200/MI300/MI350) | needs the ROCm runtime installed (rocBLAS/hipBLASLt for every gfx do not fit a release asset) |
| `llama-turboquant-windows-x64-rocm.zip` | the RDNA targets above | self-contained: HIP runtime + rocBLAS/hipBLAS + Tensile kernels for those targets bundled, only an Adrenalin driver is needed |
| `llama-turboquant-*-vulkan.*` | any Vulkan 1.2+ GPU | driver only |

Both ROCm archives are built with `--offload-compress` and `GGML_BACKEND_DL=ON`
(`libggml-hip.so` / `ggml-hip.dll` is dlopen'd). The Windows one follows the
upstream `windows-rocm` release job (ROCm 10.0 wheels, clang from the wheels,
`amdhip64_7.dll` next to the binaries because the loader prefers the exe
directory over the driver copy in System32, ggml-org/llama.cpp#26929) and adds
rocBLAS, which upstream stopped bundling and whose zip therefore no longer
loads without a ROCm SDK (ggml-org/llama.cpp#26996).

TurboQuant on HIP: the CUDA kernels compile unchanged (no `GGML_USE_HIP`
guards in any turbo kernel), all 21 turbo fattn-vec instances are listed in
`ggml/src/ggml-hip/CMakeLists.txt`. HIP-specific dispatch:

- `ggml/src/ggml-cuda/fattn.cu`, `#ifdef GGML_USE_HIP` block: decode with a
  quantized KV (batch <= 8) is forced onto the VEC kernel (inline dequant, no
  f16 scratch, HIP-graph safe); prefill falls through to TILE/MMA.
- `ggml/src/ggml-cuda/fattn.cu`, RDNA4 fast path: VEC preferred for quantized
  K/V at <= 2 query columns.
- `ggml/src/ggml-cuda/mmvq-tq.cu`: TQ4_1S/TQ3_1S weights use the scalar path on
  AMD (dp4a was measured slower on RDNA4).
- `ggml/src/ggml-cuda/vendors/hip.h`: variadic `__shfl_*_sync` shims and a
  32-bit `__ballot_sync`.

MTP (Gemma 4) and NextN (Qwen 3.6) speculative decoding live in `src/` and
have no backend code, so they run on HIP and Vulkan as-is.

Validated so far: MI300X and MI355X throughput (`docs/rocm-mi300x-test-results.md`,
April 2026: turbo4 decode at 89% / 84% of f16, prefill +4% / 98%), RDNA4 by the
community on Windows (RX 9070, 256k context in 16 GB with a 4-bit KV cache).

## 2. Benchmark plan: ROCm vs Vulkan with TurboQuant

### Hardware

- Phase 1, now: Hot Aisle MI300X VM (gfx942, ROCm preinstalled,
  https://hotaisle.xyz/quick-start). Gives the CDNA correctness check and the
  Instinct row. Vulkan on MI300X goes through Mesa RADV headless; RADV knows
  gfx942 but nobody tunes it for Instinct, so that comparison shows how much
  HIP is ahead on Instinct, not the consumer picture. Say so in the report.
- Phase 2, when a box is available: RDNA3/RDNA4, where Vulkan vs ROCm is a real
  contest (Hostkey rents RX 7900 XTX and Radeon AI PRO R9700, Vast.ai rents RX
  7900 XTX). Strix Halo is not rentable; ask AMD for a sample.

### Correctness first (CDNA only, once)

Build from source with tests on the MI300X and run

```bash
./build/bin/test-backend-ops -o SET_ROWS
./build/bin/test-backend-ops -o FLASH_ATTN_EXT
```

Reason: the turbo `set_rows` kernels in `ggml/src/ggml-cuda/set-rows.cu`
compute the lane as `j % WARP_SIZE` (32) while the HIP `__shfl_sync` shim
shuffles with `warpSize` (64 on gfx9) and `__ballot_sync` truncates the mask
to 32 bits. On wave64 hardware the upper half-wave may read the lower half's
bits. The MI300X numbers so far only checked speed and the WHT roundtrip, not
the KV cache contents. RDNA (wave32) is not affected. If the tests fail, fix
before publishing numbers.

### Matrix

Models (from the AtomicChat Hugging Face collections, NextN/MTP heads included):

| Model | Size | Speculative |
|---|---|---|
| Qwen3.6-35B-A3B UDT Q4_K_XL `_MTP` | 20.7 GiB, MoE | NextN |
| Qwen3.6-27B UDT Q4_K_XL `_MTP` | 17.7 GiB, dense | NextN |
| Gemma 4 26B-A4B + assistant Q4_K_M | MoE | MTP |
| Qwen3.5-9B Q4_K_M | small | none; the "long context in small memory" row |

| Axis | Values |
|---|---|
| backend | fork HIP (`linux-x64-rocm`), fork Vulkan (`linux-x64-vulkan`), upstream HIP of the same day (control: what the fork adds) |
| KV cache | f16, q8_0, turbo3, turbo4 |
| speculative | off, nextn / mtp |
| depth (`-d`) | 0, 8192, 32768, 131072 |
| tests | pp512, pp4096, tg128 |

Metrics: tokens/s per cell, KV memory at a given context, largest context that
fits, and wikitext-2 perplexity for turbo3/turbo4 vs f16 so "compression
without loss" is measured, not claimed. Flags: `-fa 1 -ngl 99 -ub 2048 -r 3`;
on gfx1151 also `ROCBLAS_USE_HIPBLASLT=1` (about +10% prompt processing,
lemonade-sdk/llamacpp-rocm#7).

### How to run

```bash
# once per backend build on the same machine
BIN_DIR=./rocm/build/bin   LABEL=hip    MODELS="$M1 $M2" PPL_FILE=wiki.test.raw bash scripts/bench-amd.sh
BIN_DIR=./vulkan/build/bin LABEL=vulkan MODELS="$M1 $M2" PPL_FILE=wiki.test.raw bash scripts/bench-amd.sh
BIN_DIR=./upstream/bin     LABEL=upstream-hip KV_TYPES="f16 q8_0" MODELS="$M1 $M2" bash scripts/bench-amd.sh

# one table
python3 scripts/bench-amd-report.py docs/amd/results/*.json --baseline hip/f16
```

`bench-amd.sh` writes, per run, `<host>__<label>__<stamp>.env.txt` (ROCm and
Mesa versions, devices, flags), one `...__<kv>.json` per KV type from
`llama-bench -o json`, and `...ppl.txt` when `PPL_FILE` is set.

Speculative decoding is a server feature, not a `llama-bench` one: use
`scripts/bench-matrix-qwen.sh` (NextN, f16/turbo3 x off/nextn) and the Gemma
MTP scripts in `scripts/run-gemma4-*-mtp-server.sh` against the same backend
builds, and put those rows in the same report.

### Report format

`docs/amd/results-template.md`. One table per machine, environment block on
top, the `env.txt` and JSON files committed next to it.

## 3. Open items (roadmap)

- wave64 correctness of turbo `set_rows` on CDNA (see above).
- D=576/640 flash attention on AMD: the tile instances are excluded on HIP
  (`ggml/src/ggml-hip/CMakeLists.txt`, local memory limit) but the dispatcher
  still returns TILE, so DeepSeek-MLA and GLM-4.7-Flash with a turbo KV abort at
  runtime on every AMD GPU. Return `BEST_FATTN_KERNEL_NONE` there instead.
- `-funsafe-math-optimizations` is forced for HIP (`ggml/src/ggml-hip/CMakeLists.txt`);
  upstream made it opt-in (`GGML_HIP_UNSAFE_MATH`, ggml-org/llama.cpp#26696).
  Decide with perplexity numbers.
- No GPU smoke test for the ROCm archives in CI (build-only runners). Candidate:
  a manual workflow driven from a rented AMD box.
- Upstream sync: since b10269 upstream merged RDNA4 flash-attention tuning
  (#28102), MMA flash attention for head dim 256 on RDNA (#26419), MMQ tuning
  for RDNA3/3.5 (#26284, #28195), `v_perm` based Q2_0/Q1_0 paths on gfx1201
  (#26753, #28398), GDN RDNA4 work (#28447), hipCUB (#26592); fp32 accumulation
  for MFMA flash attention (#28576) is open. Cheapest performance win we have.
- Atomic Chat client: Windows provider ranks CUDA then Vulkan only; add the
  `windows-x64-rocm` id (manifest schema regex, backend pick order) and drop
  the ROCm auto-upgrade exclusion once the archive size is known. The
  `integratedGpuOnly` guard currently keeps APUs on CPU, which excludes Strix
  Halo / Ryzen AI Max; gfx1150/gfx1151 need an exception.
- hipBLASLt: not bundled on Windows yet; `ROCBLAS_USE_HIPBLASLT=1` needs
  `hipblaslt.dll` and its kernel library next to rocBLAS.

## 4. AMD ecosystem pointers

- Lemonade Server (AMD's local AI server, orchestrates llama.cpp):
  https://lemonade-server.ai , nightly ROCm builds at
  https://github.com/lemonade-sdk/llamacpp-rocm (per-gfx archives with the
  runtime bundled).
- ISV hub and consumer partner pages: https://www.amd.com/en/ecosystem/isv.html ,
  example listing https://www.amd.com/en/ecosystem/isv/consumer-partners/lm-studio.html
- AMD AI Developer Program: USD 100 of Developer Cloud credits (MI300X).
- Community numbers used above: ggml-org/llama.cpp discussions #15021 (HIP per
  GPU) and #21043 (RDNA4 R9700 tuning), jagsan-cyber/turboquant-rocm-llamacpp
  (RX 9070, Windows), lemmy.world/post/48144487 (Qwen3.6-27B MTP on RX 9070 XT:
  25.9 -> 46 tok/s).
