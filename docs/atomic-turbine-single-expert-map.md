# Known-expert Metal dispatch reduction

## Decision and mechanism

Enable prepared single-expert maps by default for the qualified Qwen3.5 MoE and
LFM2 MoE Metal streaming profiles. Retain `--expert-single-map off` in the diagnostic
runner and the idle-only `llama_expert_single_map_set` control for comparisons.
CPU, vector waves and other architectures retain their previous defaults. Preview
remains independently off by default.

The executor already converts each routed projection into bounded one-expert
operations. For its 32-row matrix waves, local route IDs are immutable zeros.
Previously every operation still dispatched the generic GPU route-map prepass,
the invalid-row zeroing kernel, a map-dependency barrier, and the matrix kernel.
The first two kernels and their internal barrier are now skipped for explicitly
prepared operations. The matrix kernel, pipeline selection, 32-row padding,
quantized expert bytes, arithmetic and scatter order are unchanged. Top-level
graph submission counts do not decrease.

A private Metal preparation hook validates a bound, contiguous F32 output,
a single expert and input slot, 32 rows, immutable leaf I32 IDs, shared host-visible
storage and buffer bounds. Every ID must be zero. It writes the exact descriptor
`[1, 0, 0, 32]` and row list `0..31` into the output's existing 148-byte map tail,
then marks the operation. Rejection clears the mark and retains generic behavior;
a one-expert tensor alone never implies that arbitrary GPU IDs are valid zeros.

The caller must own the output and IDs exclusively with no GPU work in flight.
IDs, tensor layouts/bindings and the prepared tail remain unchanged through
completion; activation values may be populated before submission. Executor
construction and flush boundaries already provide this lifetime. No buffer,
thread, graph dependency or model payload is added. Existing workspace admission,
lease retention, failure synchronization and the fixed runtime envelope remain
in force. The raw matrix-kernel source hash is unchanged.

## Paired runtime evidence

All comparisons run sequentially on the same candidate binary with profiling,
greedy sampling, context 512, microbatch 32, LRU, default known-route overlap and
preview off. Each architecture uses off/on, on/off, off/on order. The first off
run has substantially slower source transfer and is not used to attribute the
steady comparison's request-level gain.

For Qwen, the two subsequent pairs are:

- Generic 7.3209 s, prepared 4.9937 s: 31.8% lower request time.
- Generic 7.3609 s, prepared 4.7601 s: 35.3% lower request time.

Prefill changes from 5.9000/5.9359 s to 3.3470/3.3432 s. Source transfer is
2.196/2.201 s versus 2.212/2.071 s; GPU command service is 2.825/2.874 s versus
0.874/0.880 s. Preparing 44,568 maps takes 1.68–1.69 ms and skips 89,136 kernel
dispatches plus their map barriers. Request submissions remain 17,367. All six
runs read 38,824,181,760 logical bytes and preserve all saved resident logits,
16 committed token IDs and state `9755c384a23111fe` bitwise.

For LFM, the two subsequent pairs are:

- Generic 8.9769 s, prepared 8.6756 s: 3.4% lower request time.
- Generic 8.9098 s, prepared 8.7162 s: 2.2% lower request time.

Prefill changes from 4.4794/4.4534 s to 3.9432/4.2961 s. Source transfer is
5.446/5.428 s versus 5.446/5.472 s; GPU command service is 2.107/2.127 s versus
1.079/1.074 s. Preparing 25,629 maps takes 1.23–1.42 ms and skips 51,258 kernel
dispatches plus map barriers. Request submissions remain 13,077. All six runs
read 99,582,345,216 logical bytes and preserve the complete 52-token answer,
EOG, all logits and state `84ac8e66a7115244` bitwise. LFM's request gain is modest:
source reads dominate much of its remaining work.

GPU service includes command-buffer intervals and barriers; it is not pure
arithmetic time and must not be added to overlapping CPU phase totals. These
pairs establish a repeatable gain for the measured profiles, not a confidence
interval, cold-SSD result or a broad hardware/checkpoint performance guarantee.
The first off runs took 10.1997 s (Qwen) and 11.6881 s (LFM), with source-transfer
times 6.760/8.271 s; all remain recorded, but the larger first-pair gains are not
the advertised result.

## Memory and targeted qualification

No managed allocation size or minimum changes between off and on:

- Qwen: floor 4,933,600,320 bytes; high-water 5,368,572,192 under a
  5,368,709,120-byte cap.
- LFM: floor 3,567,199,296 bytes; high-water 4,294,966,576 under a
  4,294,967,296-byte cap.

Two new Metal test cases pass 25 assertions. They compare CPU-prepared map bytes
with the completed GPU prepass, compare outputs bitwise, reject skipped/nonzero
and GPU-produced IDs or insufficient output storage, and verify the generic
skip-row zeroing fallback. Executor coverage includes shared and strided inputs,
multiple grouped waves, shrink to the unchanged single-wave floor, grow, released
pins, and the vector fallback.

After default activation, the Qwen live probe shrinks during prefill from 5120
to 4706 MiB, exercises latest-request-wins, grows to 6144 MiB, shrinks, rejects
1 MiB without losing the applied cap, and grows/shrinks again. All snapshots fit
the applied cap, all saved logits/tokens/state match, and 44,568 maps are prepared
without an explicit enable flag. Lifetime high-water is 6,442,131,440 bytes under
the maximum applied cap of 6,442,450,944 bytes.

A separate default-on Qwen microbatch-128 completion checks the existing resident
reference, including token 23, EOG 248046 and state `d8bd6c5d1cf32b92`.
Inference, executor-test and server targets build. No full test suite, aggregate
verify, server request, native OS-pressure test or unrelated workload matrix ran.
The scope is 12 fixed-cap comparison requests and two default-activation probes.

## Reproduction and evidence

Compare `--expert-single-map off` and `--expert-single-map on` with otherwise
identical runner options. Profile fields `prepared_map_ops` and `map_prepare`
report successful host preparation count and its nested graph-setup time.

The selected unit command is:

```sh
build-metal/bin/test-expert-executor --backend MTL0 'prepared .*'
```

Local reports use `/tmp/atomic-turbine-qwen-overlap-single-map-*` and
`/tmp/atomic-turbine-p10a-single-map-*`. Build/test logs use
`/tmp/atomic-turbine-single-map-*`. Results, exact-output checks, raw artifact
hashes and previous source/binary snapshots are recorded under `single_expert_map`
in `atomic-turbine-evidence.json`. The private map layout intentionally follows
the in-tree Metal kernel; requalify it if upstream changes that layout.
