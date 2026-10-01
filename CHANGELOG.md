# Changelog

One section per release, keyed by the exact tag. `verify-version` refuses to cut
a release whose tag has no section here, so this gets written *before* the tag is
pushed — the release notes are generated from it verbatim.

Write it for the person who downloads the build: what they get, what changed for
them, what to watch out for. Not a commit dump — the notes already carry the full
commit list underneath.

Releases before `b10269-1.5.0` predate this file; see the git history.

## Unreleased

### Added

- **Decision models in `llama-server`** (`--decision`, see `DECISION.md`). A
  separate server mode for Jev-class decision models: no chat routes, one
  worker (on the CPU unless `--decision-device` says otherwise), calibrated
  probabilities instead of generated text.
  - `POST /v1/systemone`: TypeSafe-compatible typed questions (noul, choice,
    score) over a state.
  - `POST /v1/router/score`: executor cards in, an independent calibrated
    `p_success` per candidate out (Platt calibration from the model's spec; 501
    unless calibrated or `--decision-allow-uncalibrated`).
  - `/health`, `/props` (`decision` block with limits, plan and calibration),
    `/v1/models` capabilities, an error envelope with stable reason codes, and a
    bounded queue (429 + `Retry-After`).
  - New flags: `--decision`, `--decision-spec`, `--decision-plan`,
    `--decision-queue`, `--decision-max-items`, `--decision-allow-uncalibrated`,
    `--decision-debug`, `--decision-kernels`, `--decision-precision`,
    `--decision-device`, `--decision-gpu`, `--decision-strict-placement`.
  - `systemone` answers carry `action.act_probability` (the reference agent's
    action head, `round(softmax(act logits)[0], 4)`); with `--decision-debug`
    the raw act logits are in `debug.act_logits`. Router answers do not have it.
  - An internal error answers 500 with a fixed message; the exception text goes
    to the server log only.
  - `--decision-precision default|strict`: the matmul precision request of the
    laya graph. The CPU computes the same bits in both modes; on CUDA `strict`
    is the parity-audit mode (PEDANTIC on every matmul, slower), not a serving
    mode.
  - Experimental: `--decision-device gpu|auto` (and `--decision-gpu`,
    `--decision-strict-placement`) runs the laya graph on a GPU through the
    stock ggml backends; the default stays `cpu`. `/props.decision` reports the
    device and where the graph nodes run. Metal passes its parity tier
    (`f16-class`) against the CPU on the full parity corpus; CUDA (RTX 4090)
    passes its tiers, `--decision-precision strict` included, and Vulkan
    (NVIDIA) passes `f16-class` on the multilingual model. On CUDA the graph
    dequantizes Q8_0 weights to F32 (the quantized-activation kernels lose this
    model's outliers) and, in `strict`, F16 weights too, and `strict` computes
    the RoPE tables on the host.
  - `--decision-device auto` uses the GPU when there is one and falls back to
    the CPU, with a warning, when the device fails to load, initialize or warm
    up, or when CPU-only kernels are requested. A device whose weight buffer
    cannot be allocated now fails the load (`gpu`) or falls back (`auto`)
    instead of computing from host memory; `/props.decision.placement` counts
    weights the device cannot run (`host_weights`), and
    `--decision-strict-placement` refuses them. ROCm / MUSA builds, other
    backends and quantized types other than Q8_0 load with a warning: they have
    no parity run.
  - `llama-laya-cli` gets `--plan packed|sequential` (default `packed`;
    `sequential` runs one graph per question, as the server does),
    `--jsonl FILE` (one load, one output line per input line), `--device`,
    `--gpu`, `--strict-placement` and `--precision`.
  - `LAYA_TRACE_DIR=<dir>` dumps the per-layer outputs of every forward pass
    (`tools/laya/trace_diff.py` compares two dumps) for `llama-laya-cli` and
    `llama-decision-bench`; the server honours it only with `--decision-debug`,
    since the dump holds the activations of every request.
  - Measured on servers: an RTX 4090 answers 13-24x faster at p50 than 32
    threads of a 64-core EPYC on `laya-multilingual` F16 (systemone with one
    question: 7.1 vs 93.8 ms), an RTX 5060 Ti 9-13x faster than 28 threads of
    an EPYC 9334, and the decision process then uses almost no CPU. Use an F16
    GGUF on a GPU: Q8_0 is not faster there. `DECISION.md` ("GPU backends") has
    the tables and the recommendation. Macs are not measured yet, so the app
    default stays `cpu`.
  - `scripts/bench-decision-device.py`: paired CPU vs GPU bench (alternating
    blocks, load and GPU sampling, time to ready, optionally a chat model
    generating on the same GPU). `scripts/bench-decision-report.py --parity`
    prints a speed number only when a passing parity gate of the same build,
    GGUF and device covers it.
- **Laya decision model support** (`tools/laya`, `llama-laya-cli`), based on
  upstream PR #29363: converter for `laya-multilingual`, GGUF arch `laya`, a
  self-contained CPU graph, and a tokenizer that matches the HF tokenizer exactly.
  Inputs follow the laya 0.3.21 PyTorch reference. On a 2605-question parity set
  the F32 GGUF matches the reference to 6.6e-4 in scorer logits with identical
  answers; F16 changes 2 answers, both near-ties.
- **English Laya checkpoints** `convaiinnovations/laya` (512 ctx) and
  `laya-typed-decisions` (1024 ctx), ModernBERT-large encoder: the converter
  handles them, `tools/laya` gets a second tokenizer (ModernBERT / OLMo
  byte-level BPE with NFC, 0 mismatches against the HF tokenizer on 13.6k strings,
  10.7k of them English-heavy), the special ids come from the tokenizer (`[CLS]`
  50281, `[MASK]` 50284 with lstrip; a GGUF whose mask id is not
  `laya.marker_token_id` no longer loads), and the checkpoint's
  `temperature_by_options` buckets are stored in the GGUF and applied like the
  reference. On 1491 English questions the F32 GGUFs match the PyTorch reference
  with identical answers (max |dlogit| 4.2e-4 and 6.8e-5). F16 through the server
  keeps every answer with the `blas` kernels, the macOS default (1.3e-2 and
  4.6e-3); with the `default` ggml kernels, the Linux and Windows default, it
  reaches 0.24 and 3.8e-2 with one near-tie answer change on `laya`
  (`llama-laya-cli`, which packs the questions of an item: 0.44, two changes). About 2.2x the compute of `laya-multilingual` per token on an
  M4 Max (router N=1 221 vs 111 ms at `-t 12`) and 944 vs 519 MiB footprint in
  F16 (see `DECISION.md`, "English checkpoints").
  `laya-multilingual` results are unchanged bit for bit with the `default`
  kernels (`llama-laya-cli`, and the server with `--decision-kernels default`).
- **Model metadata `decision.spec`**, stamped into a GGUF without reconversion
  with `gguf-py/gguf/scripts/gguf_decision_spec.py`. A calibration marked
  required must cover every question type and option count, or the model does not
  load.
- **`llama-decision-bench`** and `scripts/bench-decision.sh`: in-process latency
  (p50/p95/p99 per phase), CPU seconds, memory and logits-hash determinism of
  decision requests, plus server time-to-ready and first request after idle, with
  the machine's power source and load recorded next to the numbers.
- **Router training and calibration tools** (`DECISION.md`, "Training and
  calibration tools"): `tools/decision/reference.py`, a stdlib Python reference of
  the router input (strict JSON, executor cards, `card-v1`, router state,
  `escape-control` from the GGUF vocabulary), checked against the engine byte for
  byte on 540 golden requests (`test-decision-reference`) and through the server's
  render route; the card as JSON Schema; `scripts/fit-router-calibration.py`
  (collects the server's raw router logits, fits the Platt `router.calibration`
  block with bootstrap-interval ECE / Brier before and after, and prints the
  `gguf_decision_spec.py` command instead of touching the GGUF); and
  `scripts/router-baselines.py` (pass-rate heuristic and logistic regression on
  card metrics, same data format and metrics).
- **Decision CPU levers** (laya engine): `token_embd` stays in a read-only file
  mapping and the GGUF metadata is freed after the load (Q8_0 with the `default`
  kernels: RSS 637 -> 262 MiB and footprint 764 -> 389 MiB after load; the macOS
  default `auto` adds the BLAS backend, 408 MiB; results bitwise unchanged); one persistent ggml
  threadpool; a warm-up pass at load (`--no-warmup` skips it); `--load-mode mlock`
  / `mmap+mlock` lock the model; the default thread count is the performance cores
  (Windows: highest `EfficiencyClass`); `KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=passive`
  unless set. New `--decision-kernels auto|default|repack|blas|repack+blas` (also
  spec `plan.kernels`, shown in `/props.decision.plan.kernels`): `auto`, the new
  default, uses the Accelerate BLAS backend on macOS, which is 2-2.3x faster, uses
  3-4x fewer CPU seconds and is closer to the PyTorch reference than the ggml kernels
  for F16 and Q8_0 (F16: 2604/2604 argmax agreement); `repack` enables
  the CPU repack buffers for quantized weights. The BLAS backend runs with
  min(`-t`, 8) threads (it starts new threads for every weight conversion; at
  `-t 12`, 12 threads were up to 10% slower than 4 or 8 on `laya-multilingual`, and
  4 was 3-5% slower than 8 on the larger English encoder). `llama-laya-cli` gets `--kernels`,
  `--no-mmap` and `--mlock` and keeps the `default` kernels. `/props.decision`
  also shows the BLAS threads and the weight bytes loaded / mapped / repacked.

- **Laya checkpoints without Python.** `llama-server --decision -m <dir>` and
  `llama-laya-cli -m <dir>` take a laya Hugging Face checkpoint directory (an HF
  snapshot works as it is). The first start converts it into a GGUF cache,
  later starts reuse it ("converted" / "cache hit" in the log), and the cached
  GGUF is loaded exactly like `-m FILE`. The cache lives in
  `$LLAMA_CACHE/laya/gguf-cache` or the user cache (macOS
  `~/Library/Caches/llama.cpp/laya/gguf-cache`, Linux `~/.cache/...`, Windows
  `%LOCALAPPDATA%\...`), never in the checkpoint; new flags
  `--decision-convert-cache DIR` and `--decision-convert-type f16|f32`
  (default f16; the converter's q8_0 is not the precision-protected Q8_0 recipe,
  use `tests/laya/quantize.sh` on the f16 GGUF). The cache key covers the
  converter version, the outtype and the files the converter can read (root,
  `encoder/`, `tokenizer/`; files up to 8 MiB by content, larger ones such as the
  weights by size + mtime + symlink target). Writes are atomic and synced to disk.
  `/props.decision` shows `source` (`gguf` / `checkpoint-dir`) and `cache_path`.
  The conversion is the new `llama-laya-convert` (C++, no Python; also
  usable on its own), byte-identical to `convert_hf_to_gguf.py` for all three
  published checkpoints in f32, f16 and q8_0, with and without `--model-name`
  (18 of 18 files), and 6-19x faster (0.2-1.8 s per f16 file). A first start of
  `laya-multilingual` takes about 1.2-1.3 s instead of 0.3 s and peaks at about
  515 MiB instead of 372 MiB RSS.

### Notes

- Decision mode runs on the CPU by default (`--decision-device cpu`); next to a
  GPU chat server keep it there and start it with `--device none`.
- **Stricter laya GGUF checks**: a GGUF without `laya.attention.sliding_window`,
  `laya.attention.sliding_window_pattern`, `laya.rope.freq_base` or
  `laya.rope.freq_base_swa`, or with `laya.act_classes` below 1, no longer
  loads. Every GGUF made by this converter has them; hand-made or older
  third-party files may need reconversion.
- A laya GGUF started without `--decision` fails fast with a hint.
- **Decision server numerics on macOS change**: the default kernels there are now
  `cpu+blas` (Accelerate), so macOS server logits differ slightly from Phase 1 and
  from Linux and Windows, where `auto` stays on the ggml kernels and nothing
  changes. `--decision-kernels default` restores the Phase 1 numerics bit for bit on
  every platform. An explicit `--decision-kernels blas` fails at load on a
  build without a BLAS backend.
- Recommended laya precision is F16: Q8_0 is about 1.5x faster but changes about
  2% of answers on the parity set.
- Windows: decision model, spec and input paths may contain non-ASCII characters
  (for example a Cyrillic user folder), and `llama-laya-cli` prints LF line ends on
  every platform, so its output compares byte for byte with the golden files.
- A router calibration fitted with `scripts/fit-router-calibration.py` pins the
  kernels its logits came from (`plan.kernels`): one fitted on a Mac pins `blas`
  and needs a BLAS build elsewhere. Collect on the target platform, or start the
  collecting server with `--decision-kernels default`.

### Changed

- **Qwen3.5 / Qwen3-Next / Kimi-Linear / Kimi-K3 / BailingMoeV3 numerics:** the
  Gated DeltaNet q/k normalization now uses upstream's `rsqrt(sum(x^2) + eps)`
  form (upstream 5fdfa6282). Logits of these models shift slightly; results
  measured on earlier builds are not bit-identical.

## b10269-1.6.0

### Added

- **Windows AMD ROCm archive** `llama-turboquant-windows-x64-rocm.zip`. HIP
  backend for RDNA2 through RDNA4 and Ryzen AI 300 / Ryzen AI Max (gfx1030,
  gfx1100/1101/1102/1103, gfx1150/1151, gfx1200/1201), self-contained: the HIP
  runtime and the BLAS DLLs it links are bundled (about 100 MB), only a current
  Adrenalin driver is needed, no ROCm SDK install. Until now Windows + Radeon
  meant the Vulkan build; HIP is the faster prompt-processing path on these
  GPUs. Everything in the archive is Authenticode-signed, the AMD DLLs with
  AMD's signature where they ship one.
- **Linux ROCm archive** now also targets gfx950 (Instinct MI350/MI355X),
  gfx1150 (Ryzen AI 300) and gfx1103 (Radeon 780M/760M).
- `docs/amd/`: what ships for AMD, the ROCm vs Vulkan benchmark plan for the
  TurboQuant KV cache, and the open items. `scripts/bench-amd.sh` runs that
  matrix per backend build, `scripts/bench-amd-report.py` renders one table.

### Changed

- ROCm builds no longer pass `GGML_HIP_ROCWMMA_FATTN`; upstream removed the
  rocWMMA flash-attention path and the flag was a no-op.

### Notes

- **The Windows ROCm archive is beta.** It is built, signed and checked for
  completeness in CI, but has not been run on a Radeon yet. `llama-server
  --list-devices` must show a HIP device; if it does not, or if loading fails,
  report the GPU, driver version and the error, and use the Vulkan archive in
  the meantime.
- Atomic Chat does not select the Windows ROCm backend yet; the client change
  follows separately. Until then it is a manual download.

## b10269-1.5.1

### Fixed

- **Ling-3.0-flash (BailingMoeV3) no longer emits garbage token bursts.** The
  model is trained with clamped SwiGLU activations in its late layers, and the
  per-layer limits live in `config.json` under `expert_swiglu_limit_list` and
  `share_expert_swiglu_limit_list`. The public HF modeling code ignores those
  keys and so did this port, which caused deterministic transient logit
  collapse - output like `count += 1eville` dropped into otherwise fine
  generations. Measured at roughly -20 pass@1 on HumanEval (72.6% -> 93%+ with
  the fix); the garbage-token repro is eliminated.

### Notes

- **Re-convert your Ling-3.0-flash GGUF to get the fix.** The clamp limits are
  written by the converter into two new KVs (`{arch}.swiglu_clamp_exp` and
  `{arch}.swiglu_clamp_shexp`); a GGUF produced before this release does not
  carry them, and the runtime then defaults to no clamping. Re-download the
  quant or re-run `conversion/bailingmoe.py`.
- Both KVs are optional and default to zero, so existing GGUFs and every other
  architecture are unaffected. The graph needed no change - the SwiGLU clamp
  branches in `build_ffn` / `build_moe_ffn` already trigger on a nonzero
  per-layer limit, matching the vLLM `SwigluStepAndMul` semantics.

## b10269-1.5.0

### Added

- **NVIDIA DGX Spark (GB10) support.** New archive
  `llama-turboquant-linux-arm64-cuda-13.3`, built natively for aarch64 with
  CUDA 13.3 and sm_121 SASS. Other arm64 NVIDIA machines (GH200, GB200, Jetson
  Thor) run it too, JITing the kernels from PTX on first launch. This is the
  first Linux arm64 build the fork ships — until now arm64 meant macOS only.
- **BailingMoeV3 (Ling 3.0) architecture support**, including the KDA gate
  handling.

### Changed

- **Linux CUDA archives are roughly half the size** — 1657 → 956 MB (12.4) and
  1879 → 1028 MB (13.3) measured across both the `.zip` and `.tar.gz`. The zips
  were storing `libcublas.so` → `.so.13` → `.so.13.5.1.27` as three full copies
  because `zip` followed the symlinks.
- **CUDA 13.3 builds ship Ampere PTX (`80-virtual`).** A100/H100/B200 were
  falling back to the Turing PTX floor, which silently disabled `cp.async` and
  the Ampere MMA path — both gated on `__CUDA_ARCH__ >= 800`. Those cards get
  Ampere-class kernels now. No architecture lost support in this release.
- Windows CUDA builds got their architecture lists pinned, all runner cores, a
  ccache that can actually hold a CUDA build, and 7-Zip instead of
  `Compress-Archive`. Release turnaround drops accordingly.

### Notes

- The DGX Spark archive has **not yet been validated on real GB10 hardware** —
  it is built and arch-checked in CI (`cuobjdump` asserts sm_121 SASS is
  present), but nobody has run it on a Spark yet. Treat this one as beta and
  report back.
- The CUDA 13.3 archives now use `-compress-mode=size`. Kernel SASS is
  unchanged and inference speed is unaffected; the fatbin is decompressed once
  at module load. It needs a driver from the CUDA 12.4 era or newer.
