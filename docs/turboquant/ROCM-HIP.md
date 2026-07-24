# ROCm & HIP for TurboQuant — an engineer's lecture

> Audience: you, a C++ inference/edge engineer who wants to *own* the AMD
> backend, not cherry-pick upstream PRs. This doc is meant to be read
> top-to-bottom once, then used as a reference. It is opinionated and
> specific to **this fork** — every file path is real, every claim was
> checked against the tree on 2026-07-24.

---

## 0. The one-paragraph orientation

`llama.cpp` runs models through **ggml**, a tensor library with pluggable
**backends**. There is no separate "AMD backend": AMD GPUs are served by the
**CUDA backend compiled through HIP**. HIP is AMD's near-clone of the CUDA
language; the same `.cu` kernels compile for NVIDIA (via `nvcc`) *and* for
AMD (via `hipcc`/`clang`). So "add AMD support" does **not** mean writing a
new backend — it means (a) building `ggml-hip`, (b) making sure our
TurboQuant kernels behave on AMD's execution model, and (c) proving it on
real silicon (Strix Halo). Most of (a) and (b) already exist in this tree.
Your real job is CI + validation, plus understanding enough of (b) to debug
it when a kernel misbehaves on RDNA/CDNA.

---

## 1. The vocabulary, disambiguated

You said the CUDA/HIP/ROCm/gfx/wavefront words blur together. Here is the
clean mental model.

| Term | What it actually is | NVIDIA analogue |
|---|---|---|
| **ROCm** | AMD's whole GPU-compute *platform*: driver (amdgpu/KFD), runtime (HIP runtime, `hsa`), math libraries (rocBLAS, hipBLASLt, rocWMMA), compiler (a fork of LLVM/clang), profiler. Think "CUDA Toolkit + driver", the entire stack. | CUDA Toolkit + driver |
| **HIP** | The *language and API* inside ROCm. C++ with `__global__`/`__device__` kernels, `hipMalloc`, `hipLaunchKernelGGL`. ~1:1 with CUDA — deliberately, so CUDA code ports. | CUDA C++ + runtime API |
| **hipcc / amdclang++** | The *compiler driver*. Compiles HIP (and thus hipified CUDA) to AMD GPU code. Under the hood it's clang targeting the `amdgcn` architecture. | nvcc |
| **HIPIFY** | A source translator that rewrites `cuda*`→`hip*`, `cudaMalloc`→`hipMalloc`, etc. ggml mostly avoids it by using a compat header (see §3). | — |
| **gfx____** | The *GPU ISA target*, like `gfx1151`. This is the exact instruction set of one GPU generation — you must compile for the ones you ship to. Strix Halo = **gfx1151**. | `sm_90`, compute capability |
| **RDNA / CDNA** | AMD's two microarchitecture families. **RDNA** = consumer/gaming/APU (RX 7000, Strix Halo). **CDNA** = datacenter (MI300). They differ in wavefront width and matrix hardware — this matters for kernels (§5). | (NVIDIA has one line: Ada/Hopper/Blackwell) |
| **wavefront** | AMD's SIMT execution group. **64 lanes on CDNA, 32 on RDNA.** The NVIDIA "warp" is always 32. This 32-vs-64 split is *the* portability landmine. | warp (always 32) |
| **HSA / KFD** | The kernel driver + userspace runtime that actually dispatches to the GPU. You mostly don't touch it, but `rocminfo` talks to it, and Strix Halo needs a recent-enough KFD. | CUDA driver |
| **NPU (XDNA)** | The *separate* AI engine on Strix Halo (the "Ryzen AI" NPU). **Not** ROCm, not GPU, not our target — served by a different stack (Vitis/XDNA). We target the **iGPU** (RDNA 3.5, gfx1151), not the NPU. Flagged here because "AMD Halo AI" marketing conflates them. | (no direct analogue) |

**"AMD Halo"** in your ask = **Strix Halo** = Ryzen AI Max APU: Zen 5 CPU +
big RDNA 3.5 iGPU (**gfx1151**) + XDNA NPU, all sharing one pool of unified
LPDDR5X. For us it is "a laptop/mini-PC with a fat iGPU and up to 128 GB of
GPU-addressable RAM" — which is *exactly* the machine where quantized KV
cache (TurboQuant's whole point) pays off, because you can hold big
contexts.

---

## 2. How ggml backends fit together (the C++ picture)

```
llama.cpp / server
        │  builds a ggml_cgraph (the compute graph: nodes = ops, edges = tensors)
        ▼
   ggml-backend.cpp   ← the scheduler. splits the graph across backends,
        │               allocates buffers, orders execution. (This is the
        │               file whose GGML_SCHED_MAX_SPLIT_INPUTS you just bumped.)
        ├── CPU backend
        ├── Vulkan backend        (ggml-vulkan/)
        └── "CUDA" backend        (ggml-cuda/)   ← ONE source tree,
              ├── compiled by nvcc  → libggml-cuda.so   (NVIDIA)
              └── compiled by hipcc → libggml-hip.so    (AMD)   ← ggml-hip/
```

Key realization: **`ggml-hip/` is not a second implementation.** Look at
`ggml/src/ggml-hip/CMakeLists.txt` — it `file(GLOB ...)`s the `.cu`/`.cuh`
files out of `../ggml-cuda/` and compiles *those* with the HIP language
enabled. So a kernel you write in `ggml-cuda/` is automatically an AMD kernel
too. The compat layer that makes `cuda*` symbols mean `hip*` lives in
`ggml-cuda/vendors/hip.h` (included when `GGML_USE_HIP` is defined).

A ggml **op** (e.g. `GGML_OP_MUL_MAT`, `GGML_OP_FLASH_ATTN_EXT`) is dispatched
by `ggml-cuda.cu`'s big switch to a kernel. Each backend advertises which ops
it `supports_op`, and the scheduler routes accordingly. When you add a new
quant type, you are adding: a data layout (`ggml-common.h`), a
dequant/`get_rows` path, a matrix-vector kernel (`mmvq`), and — for KV cache
— flash-attention instantiations. TurboQuant already did all of this for
CUDA; the point of this doc is that AMD gets it *for free at the source
level*, and only needs care at the ISA level (§5).

---

## 3. What already exists in THIS fork (so you don't rewrite it)

Checked on 2026-07-24:

- **`ggml/src/ggml-hip/CMakeLists.txt`** — the HIP build. Globs the CUDA
  sources, sets `CMAKE_HIP_ARCHITECTURES`, links `hip::host`, `roc::rocblas`,
  `hip::hipblas`. Turbo fattn-vec instances are in the explicit source list.
- **`ggml/CMakeLists.txt`** — `option(GGML_HIP ...)`, plus
  `GGML_HIP_ROCWMMA_FATTN` (use rocWMMA matrix cores for flash-attention) and
  related knobs.
- **TurboQuant kernels are already AMD-aware.** `ggml-cuda/mmvq-tq.cu` has
  explicit RDNA paths, e.g. *"On RDNA4, sudot4 throughput differs from NVIDIA
  dp4a — this path is faster"* and a scalar-half path for `TQ4_1S` because
  *"dp4a regresses on RDNA4"*. Someone (TheTom lineage) already did the hard,
  ISA-specific tuning. This is the single most important fact for planning:
  **the kernels are ported, not just portable.**
- **Upstream has a working ROCm release job** we can copy from:
  `.github/workflows/release.yml` `ubuntu-22-rocm` — ROCm 7.2.1,
  `gpu_targets: gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201`.
  **gfx1151 (Strix Halo) is already in that list.**
- **`ggml/src/ggml-hip/` Windows support** exists too (`windows-setup-rocm`
  action is present under `.github/actions/`).

What does **not** exist yet: a TurboQuant *release/dev-build job* that ships
an AMD archive, and any *validation on real AMD hardware*. That is the gap.

---

## 4. The plan, concretely (what you actually write)

Four milestones, smallest-first. Milestone 1 is a day; the kernels are done.

**M1 — Linux ROCm build in CI (no GPU needed to compile).**
- Add a `linux-x64-rocm` job to `.github/workflows/dev-build.yml` (and later
  `release-turboquant.yml`), modeled on upstream's `ubuntu-22-rocm`.
- Steps: install ROCm via `repo.radeon.com` apt, `cmake -DGGML_HIP=ON
  -DAMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1151;gfx90a;gfx942"
  -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_BACKEND_DL=ON`, build, bundle
  `libggml-hip.so` + the rocBLAS/hipBLASLt runtime libs (they're big — the
  archive will be ~1–2 GB, like CUDA).
- `GGML_BACKEND_DL=ON` matters: it makes the HIP backend a dlopen'd plugin
  like our CUDA one, so one archive can carry CPU+HIP and fall back cleanly.
- Deliverable: `llama-turboquant-linux-x64-rocm.tar.gz` in dev-latest.

**M2 — Validate on real Strix Halo.** CI can't rent gfx1151 easily, so this
is manual (your machine or a borrowed one). The checklist is in §6. This is
where you confirm turbo3 KV actually works and measure it.

**M3 — Windows ROCm** (optional, after Linux proves out). The
`windows-setup-rocm` action exists; MSVC+hipcc has sharp edges. Lower
priority — most Halo mini-PCs run Linux for AI.

**M4 — Release integration.** Add the job to `release-turboquant.yml`, add
the backend id to the app's `atomic-chat-conf` manifest so Atomic Chat offers
an "AMD (ROCm)" backend.

There is **no kernel-writing milestone** unless validation (M2) surfaces a
correctness bug on gfx1151 specifically — see §5 for what that would look
like.

---

## 5. The part you actually need to understand: the execution model

This is the "SIMT / vector registers / wavefront" section you asked for,
grounded in what can bite a TurboQuant kernel on AMD.

### 5.1 SIMT and the wavefront

A GPU kernel is written as scalar code for one **thread (lane)**, but the
hardware runs a whole **wavefront** of lanes in lockstep on one SIMD unit —
one instruction, many lanes, different data (SIMT). On NVIDIA a warp is
**always 32** lanes. On AMD it's **64 on CDNA, 32 on RDNA**. Our target
(gfx1151, RDNA 3.5) is **32** — which is friendly, because most of ggml's
CUDA code assumes `WARP_SIZE == 32`.

Why you care: kernels do **warp-level reductions** — summing a value across
all lanes via `__shfl_xor`/`__reduce`. If a kernel hardcodes 32 but runs on a
64-wide wavefront, half the data is dropped → silent garbage, not a crash.
ggml handles this with a `WARP_SIZE` macro that the HIP compat layer sets per
target. In `mmvq-tq.cu` you can see `for (int ib = lane; ib < blocks_per_row;
ib += WARP_SIZE)` — that's warp-stride iteration; it's correct on both widths
*as long as `WARP_SIZE` is right*. **Rule: never hardcode 32 in a
TurboQuant kernel; always use `WARP_SIZE`.** This is the #1 source of
AMD-only correctness bugs.

### 5.2 Vector registers (VGPRs) and occupancy

Each lane has private **vector registers (VGPRs)**. A wavefront's register
need determines **occupancy** — how many wavefronts a compute unit can keep
in flight to hide memory latency. A kernel that uses too many VGPRs (big
unrolled loops, many live temporaries) runs at low occupancy and stalls on
memory. AMD gives 256 VGPRs/lane on RDNA; spilling past that goes to scratch
(slow). For quantized matvec (memory-bound, our bread and butter) you want
*low* register pressure and *high* occupancy. When you profile a slow turbo
kernel on Halo, "VGPR pressure → low occupancy" is the first hypothesis. Tool:
`rocprofv2` / `omniperf`.

### 5.3 The matrix hardware (WMMA / MFMA) and flash-attention

Modern GPUs have dedicated matrix units. NVIDIA: **Tensor Cores** (via WMMA /
MMA). AMD: **WMMA** on RDNA, **MFMA** on CDNA. `rocWMMA` is AMD's library
exposing them, and `GGML_HIP_ROCWMMA_FATTN=ON` routes flash-attention through
it. This is the path that makes attention fast on AMD. Caveat baked into the
tree: on some RDNA parts rocWMMA FA has quality/perf limits (there are notes
about it in `NEXTN.md`). So: build with it ON, but M2 must A/B it (FA on vs
off) for both speed and correctness. For the TurboQuant KV story, the
flash-attention `fattn-vec` instances are the ones typed on `turbo2/3/4` —
those are the kernels that read your quantized KV cache directly.

### 5.4 dp4a vs sudot4 — a concrete TurboQuant example

Quantized matvec multiplies 4×int8 lanes and accumulates: NVIDIA has `dp4a`,
AMD RDNA has `v_dot4_i32_i8`/`sudot4`. They exist on both but with *different
throughput characteristics*, so the fastest code differs. That's exactly why
`mmvq-tq.cu` already forks: `dp4a` path for NVIDIA, `sudot4`/scalar-half path
for RDNA4 where `dp4a` "regresses". **This is what platform-specific kernel
work looks like** — same math, different instruction selection guarded by
`#ifdef`/arch checks. You now have a worked example in-tree to imitate if a
new quant needs the same treatment.

### 5.5 Unified memory on Halo — the strategic bit

Strix Halo has **no discrete VRAM**: the iGPU addresses system LPDDR5X (up to
~96–128 GB usable as GPU memory). Consequences:
- Huge contexts/models fit — the machine's whole selling point for us.
- But bandwidth is shared with the CPU and lower than discrete VRAM (~256
  GB/s class, not 1 TB/s). So it's **bandwidth-bound**, which is precisely
  where quantized KV (turbo3) wins: fewer bytes per token read = more tokens/s.
  Expect TurboQuant to show a *bigger relative* gain on Halo than on a 4090.
- `GGML_HIP_NO_VMM` and how buffers are pinned matter more here; watch for it
  in M2.

---

## 6. How to test it — theory then practice

**Compile-time (CI, no GPU):** if it builds `libggml-hip.so` for the target
gfx list, the kernels are ISA-valid. That's M1 and catches most breakage.

**Correctness (needs gfx1151):**
1. `rocminfo` — confirm the runtime sees `gfx1151`.
2. `test-backend-ops` — ggml's own op-level test harness. Run it with the HIP
   backend selected; it compares every op (incl. turbo types) against the CPU
   reference at tolerance. **This is your primary correctness gate** and it
   directly catches the WARP_SIZE-class bugs from §5.1. Path:
   `tests/test-backend-ops.cpp`.
3. End-to-end: `llama-cli`/`llama-server` with a small model,
   `-ctk turbo3 -ctv turbo3 -fa on -ngl 99`, assert coherent output, then A/B
   `GGML_HIP_ROCWMMA_FATTN` on/off.
4. A silent CPU-fallback must be treated as failure (like our vast.ai smoke
   does for CUDA): assert the GPU actually did the work.

**Performance (needs gfx1151):** `llama-bench` for pp512/tg128, compare
turbo3-KV vs f16-KV. The win should be *larger* than on discrete NVIDIA
because Halo is bandwidth-bound (§5.5).

**Where CI stops and you start:** GitHub has no gfx1151 runners and vast.ai's
AMD supply is thin, so M2/M3 GPU validation is manual or on a self-hosted
Halo runner. Building is automated; *proving it runs* is hands-on — that's
the edge-AI-engineer part, not the PR-monkey part.

---

## 7. Reading list inside this repo

- `ggml/src/ggml-hip/CMakeLists.txt` — the build you'll extend.
- `ggml/src/ggml-cuda/vendors/hip.h` — the cuda→hip compat shim; read it once
  to see how the "one source tree" trick works.
- `ggml/src/ggml-cuda/mmvq-tq.cu` — the AMD-aware TurboQuant matvec; your
  template for arch-specific tuning.
- `ggml/src/ggml-cuda/fattn*.cu*` + `template-instances/fattn-vec-instance-turbo*`
  — the quantized-KV flash-attention kernels.
- `ggml/src/ggml-backend.cpp` — the scheduler (ops, splits, supports_op).
- upstream `.github/workflows/release.yml` `ubuntu-22-rocm` — the CI recipe to
  adapt.
- `docs/build.md` (HIP section) — the canonical local build invocation.

---

## 8. TL;DR for planning

- AMD support = the CUDA backend built through HIP. No new backend.
- The hard part (ISA-tuned TurboQuant kernels for RDNA) **is already done** in
  this fork. Verify, don't rewrite.
- Your deliverables are **CI jobs** (build `libggml-hip.so` for a gfx list
  incl. **gfx1151**) and **real-hardware validation** on Strix Halo.
- The one class of bug to fear is **wavefront-width assumptions** (§5.1);
  `test-backend-ops` on real gfx1151 is the gate that catches it.
- Halo is bandwidth-bound unified memory → TurboQuant KV should shine there
  *more* than on a 4090. That's the story worth measuring and publishing.
