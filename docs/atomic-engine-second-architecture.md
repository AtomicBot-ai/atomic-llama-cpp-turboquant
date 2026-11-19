# P10A: second MoE architecture on Metal

## L2 scope

The initial qualified profile is Qwen3.5 MoE / Ornith 35B on Apple M4 Max.
Local GGUF metadata identifies LFM2.5-24B-A2B-AD-Q4_K_M as `lfm2moe`:
40 layers, two leading dense layers, 64 experts with four selected, sigmoid
routing with selection bias, and short-convolution state mixed with attention.
The other available large checkpoints use `qwen35moe`; changing only their size
would not establish a second architecture. No download is needed.

Hypothesis: the existing original-GGUF expert projection bridge can preserve
LFM's upstream router, dense prefix and short-convolution state without a new
executor or model-specific numerical kernels.

1. Inspect metadata and probe first/last expert bytes through source and Metal
   store; identify actual quantization and slice shapes.
2. Extend only the explicit architecture capability gate and any confirmed
   format constraint. Generalize the diagnostic prefill resize trigger to the
   first MoE layer rather than assuming layer zero.
3. Compare a short resident/streamed request, then a useful multi-microbatch
   prompt. Use identical greedy sampling and full-logit/state captures.
4. Run one shrink/grow cycle with an impossible target and real cache misses;
   verify actual backend release, unchanged outputs and admission bounds.
5. Record the exact supported profile and remaining capabilities separately.

The store, shared ledger, original router and state implementations remain the
same. No resident fallback, changed quantization, state truncation, prefetch,
speculation or second backend is introduced. Checks are selected for this
architecture; no aggregate suite or performance matrix is part of P10A.

## Findings and implementation

Metadata inspection found four local checkpoints: Ornith 35B, Ornith 122B
(two shards), Qwen3.6 35B and LFM2.5 24B. The first three declare `qwen35moe`;
LFM declares `lfm2moe`. Its 15,359,770,880-byte GGUF contains 428 tensors,
including 114 routed projections across 38 MoE layers. Routed weights occupy
14,251,720,704 bytes. Their formats are 66 Q4_K, 43 Q5_K and five Q6_K tensors;
the largest expert slice is 2,580,480 bytes. Full-file identity was not hashed;
file size/mtime and first-MiB hash are identity hints in the evidence record.

The source and Metal store each sampled the first and last expert of every
routed tensor: 228 slices, 445,366,272 logical bytes and matching FNV-1a digest
`a1faf376f593ef11`. The 16 MiB store stayed below its cap (16,607,760-byte
high-water) and returned all managed memory on shrink to zero. Metal allocation
fell from 11,747,328 to 475,136 bytes.

Three small implementation changes were sufficient:

- `src/llama-model.cpp` admits `LLM_ARCH_LFM2MOE` through the existing explicit
  streaming gate. All existing backend, mapping, MTP and folded-weight exclusions
  remain enforced.
- `src/llama-expert-executor.cpp` admits original Q6_K slices. It still uses
  the existing bounded executor and upstream Metal indexed-matmul kernels.
  Real comparisons below exercise the checkpoint's mixed Q4_K/Q5_K/Q6_K layers.
- `tools/expert-source/inference.cpp` triggers prefill resize at the first
  `ffn_moe_down-*` node, including models with a dense prefix. Previously it
  required layer zero, which silently missed LFM's first MoE layer at index 2.

No LFM graph, router or state code changed. Sigmoid probabilities, selection
bias, selected-weight normalization and expert weighting retain their upstream
semantics. The two dense FFNs and 30 short-convolution / ten attention layers
retain upstream placement and state ownership. Unlike the first Qwen3.5 profile,
this checkpoint has no shared-expert branch. Allocation reuse and bounded
same-expert vector micrographs remain enabled and are included in the comparisons.

## Numerical and memory evidence

All four inference runs used the same Metal binary, one sequence, context 512,
microbatch 32, 99 requested GPU layers, greedy sampling and no warmup. Runs and
the store probe were sequential, with no overlapping builds or test processes.

The short pair used `The capital of France is` (five prompt tokens), eight
committed output tokens and a 4 GiB streamed cap. Complete logits and IDs matched
bitwise; serialized state matched digest `6b56ba30f7670731`. Streaming produced
4,875 misses and 9,515,630,592 logical read bytes. Its high-water was
4,294,963,968 bytes against 4,294,967,296. The short output is a diagnostic
prefix, not a claimed complete answer.

The useful pair used `docs/atomic-engine-lfm-prefill.txt`: 200 prompt tokens
across seven microbatches and a completed 52-token summary ending at EOG 124900.
The fixture supplies chat-style delimiters and a closed thinking prefix
explicitly; the runner does not apply a chat template. The output explains
pinning active buffers and releasing unused buffers before admission.
Full logits, all committed IDs, EOG and serialized state matched resident
execution exactly (state digest `84ac8e66a7115244`).

The streamed useful request exercised this sequence:

1. Start at 4096 MiB and request 3456 MiB from inside the first routed prefill
   graph. Receipt 1 was applied at a synchronized boundary during prefill.
2. After two committed tokens, grow to 4608 MiB. The immediate Metal allocation
   delta was zero; capacity growth does not eagerly allocate buffers.
3. After four tokens, shrink to 3456 MiB. Metal released 1,205,469,184 bytes
   (about 1.12 GiB) in 12.1935 ms; immediate physical footprint fell by
   1,216,037,536 bytes.
4. After six tokens, reject 1 MiB below the 3,563,967,392-byte live engine floor.
   Applied cap, used bytes and Metal allocation were unchanged.
5. After eight tokens, grow to 4096 MiB without immediate allocation and finish
   the same complete answer with the same state.

All 16 captured admission snapshots across the two streamed runs satisfied
used <= applied. The live run's lifetime high-water was 4,831,801,696 bytes
against a maximum applied cap of 4,831,838,208. Final used memory was
4,290,808,048 bytes under 4 GiB. Peak reported process footprint was
3,309,636,704 bytes; the shared admission contract still includes its fixed
2 GiB runtime envelope and is not an OS-wide RSS guarantee.

The useful streaming run recorded 51,078 misses, 87 hits and 99,412,180,992
logical read bytes. It took 20.2934 seconds versus 0.9253 seconds resident;
prefill took 11.6772 versus 0.5020 seconds. These single runs compare different
memory capacities and establish correctness/offload, not a throughput benefit
or controlled cold-SSD benchmark. The large repeated read volume and low hit
count remain performance limitations.

## Reproduction

Set `$MODEL` to the original LFM GGUF. Build only the relevant targets:

```sh
cmake --build build-metal --target llama-expert-source llama-expert-infer -j4
build-metal/bin/llama-expert-source --sample "$MODEL"
build-metal/bin/llama-expert-source --store-mib 16 --backend MTL0 "$MODEL"
```

For the short control, use `--prompt 'The capital of France is' --tokens 8`;
run once without a budget and once with `--budget-mib 4096`, keeping all other
arguments identical and separate output/logit paths. The useful controls are:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-engine-lfm-prefill.txt --tokens 64 \
  --output /tmp/atomic-engine-p10a-prefill-resident.json \
  --logits /tmp/atomic-engine-p10a-prefill-resident.f32

build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-engine-lfm-prefill.txt --tokens 64 \
  --budget-mib 4096 --prefill-budget-mib 3456 \
  --resize-at 2:4608 --resize-at 4:3456 --resize-at 6:1 --resize-at 8:4096 \
  --output /tmp/atomic-engine-p10a-prefill-live.json \
  --logits /tmp/atomic-engine-p10a-prefill-live.f32

cmp /tmp/atomic-engine-p10a-prefill-resident.f32 /tmp/atomic-engine-p10a-prefill-live.f32
```

Compare prompt/committed IDs, stop ID, `state_fnv1a64`, receipts and memory
snapshots in the JSON reports as well. Raw reports and artifact/source hashes
are recorded under `p10a_second_architecture` in `atomic-engine-evidence.json`.
No unrelated unit cases, full suite, aggregate verification or broad matrix ran.

## Qualified support profiles and limits

- **Qwen3.5 MoE / Ornith 35B:** prior Apple M4 Max Metal evidence, context 512,
  microbatches 32/128, 5 GiB initial budget, manual resize, LRU and selected
  ClassLRU controls; P05 injected-signal qualification is separately scoped.
- **LFM2 MoE / LFM2.5-24B-A2B-AD-Q4_K_M:** P10A Apple M4 Max Metal evidence,
  context 512, microbatch 32, LRU, 4 GiB initial budget, 3456/4608 MiB live targets,
  mixed Q4_K/Q5_K/Q6_K, useful completed prefill/decode and preserved hybrid state.

P10A is complete for these bounded compatibility checks. The later
[LFM profile](atomic-engine-lfm-performance.md) adds larger-buffer reuse and
ClassLRU/live-control evidence. LFM automatic pressure, CPU inference, larger
contexts and other LFM checkpoints remain unqualified. Fused expert storage
remains outside the real-checkpoint evidence.
Prefetch and speculative accelerators are not enabled; no speculation state
rollback or architecture-specific accelerator is inferred from basic inference
equivalence. P10B requires another backend/device, and P11 release qualification
remains separate. There is no claim that arbitrary MoE architectures work.
