# Qwen3.5 MoE demand-read overlap qualification

## Decision and scope

The [bounded LFM overlap](atomic-engine-lfm-overlap.md) implementation is now
enabled by default for `qwen35moe` on Metal as well. The measured model is
Ornith-1.5-35B-A3B Q4_K/IQ4_XS on Apple M4 Max: 40 blocks, 256 experts and eight
active experts. Other Qwen architecture IDs remain opt-in. This is evidence for
that checkpoint/configuration, not qualification of all models sharing its
architecture ID, the available 122B checkpoint or Qwen3.6 weights.

Only the architecture default changes. The actual-router demand order, original
quantized bytes, numerical kernels, four-wave GPU groups, maximum four future
leases and existing admission formula are unchanged. No payload pool, thread,
workspace or minimum-budget increase is introduced. CPU and non-matrix waves
retain sequential behavior; insufficient headroom disables optional read-ahead.
The diagnostic `--expert-read-overlap on|off` override remains available.

## Matched microbatch-128 comparison

The existing 199-token completion fixture returns `8` (token 23) followed by
EOG 248046. Controls use context 512, microbatch 128, a 5 GiB managed cap, LRU
and the same pre-change binary with explicit on/off switches.

The first off run took 10.5455 seconds and incurred 17.059 GB of OS-charged
reads. It is retained as an initial observation but excluded from the matched
speed comparison. The following on run and subsequent repeated off control
incurred similar OS-charged reads, 7.372 and 7.274 GB. These are partially cached
observations, not fully warm file-cache or controlled cold-SSD measurements.

- Request time: **8.3058 seconds off, 6.1438 seconds on**, 26.03% lower.
- Prefill: **8.1589 seconds off, 5.9929 seconds on**, 26.55% lower.
- Completion wait: **2.6889 seconds off, 0.3536 seconds on**.
- Source reads: **4.8432 seconds off, 4.9859 seconds on**. The gain comes from
  moving read work into the GPU interval, not from faster source reads.

The on request acquires 29,109 future demand experts between submit and wait,
reading 16,533,291,008 bytes there. Host acquisition in that interval totals
4.8163 seconds. This nested phase is included in both `compute` and `acquire`;
it is not an exact CPU/GPU interval intersection and must not be added to those
parent counters. Summed GPU command intervals remain similar, 1.8008 / 1.7910
seconds off/on; no kernel speedup is inferred.

The complete store counters match exactly: 30,999 misses, zero hits,
17,606,770,688 logical read bytes, 553 allocations, 30,446 reused buffers,
31,899 matmuls and 8,055 submissions. High-water remains 5,368,307,504 bytes
below the unchanged 5,368,709,120-byte cap. The minimum remains
5,042,521,152 bytes for this microbatch-128 LRU configuration.

Full logits, committed IDs, EOG and state `d8bd6c5d1cf32b92` match the saved
native resident control in every completion-fixture request.

## All-hot control and observed variability

An explicit 24 GiB cap and one warmup retain the measured expert working set.
Every measured request then has zero source reads and 30,999 hits. The first
off/on pair took 3.3687 / 3.5154 seconds, a possible 4.35% regression. The
profile showed variation in submit, graph setup and gather as well as the
future-expert lookup work, so it was not dismissed without another control.

The unchanged binary then ran three measured requests after one warmup for
each mode, in on-then-off order:

- On: 3.3907, 3.3233 and 3.2424 seconds; median **3.3233**.
- Off: 3.3176, 3.3622 and 3.5693 seconds; median **3.3622**.

The ranges overlap and the initial regression is not persistent in this small
control. No all-hot speedup or statistical non-regression guarantee is claimed.
Every full logit sequence and state matches resident; no extra reads occur.
The substantial fixed-cap gain justifies enabling the bounded path for the
qualified architecture while retaining an explicit off switch.

## Final default, live control and trace

The final binary uses the new default without a command-line override. Its
microbatch-32 live case uses the existing 190-token prompt and 16-token prefix.
Prefill shrinks from 5120 to 4800 MiB. At token 2, a pending 6144 MiB request is
superseded by 4706 MiB; subsequent targets are 6144 at token 4, 4800 at token 8,
rejected 1 at token 10, 6144 at token 12, 4706 at token 14 and 5120 at token 15.
The near-floor target is updated from the older 4704 MiB fixture because the
previous GPU-grouping work raised the floor. This change does not raise it again.

The ClassLRU minimum is 4,934,103,568 bytes, below 4706 MiB (4,934,598,656).
Between tokens 2 and 4, 1,920 matmuls use 1,920 submissions, preserving the
single-wave fallback. The largest shrink frees 1,407,909,888 Metal bytes in
51.419 ms and reduces footprint by 1,447,149,760 bytes. A later near-floor
shrink frees 1,062,862,848 Metal bytes in 30.261 ms. Grow allocates no immediate
Metal storage; below-floor rejection preserves applied cap and used bytes.
Lifetime high-water is 6,442,044,496 bytes below the maximum applied
6,442,450,944-byte cap. Prefill records 42,228 overlap acquisitions. Full logits
match native resident and all 16 IDs/state `9755c384a23111fe` match the saved
qualified control. This fixture is a prefix, not a completed answer.

A final default-enabled microbatch-128 ATET3 trace exactly replays runtime LRU
hits and physical allocations. All 30,999 acquisitions have matching releases;
at most eight distinct experts are pinned and none remain at the end. ClassLRU
predicts zero saved read bytes on this completion trace, consistent with the
earlier cache evaluation. The default remains LRU. The trace request again
matches resident full logits, IDs, EOG and state; its timing is excluded from
the matched performance comparison.

## Checks, provenance and limits

Only directly relevant model controls ran: nine invocations containing thirteen
measured requests and four warmups. The additional off control addressed the
initial cold-cache mismatch; the repeated all-hot controls addressed the observed
possible regression. Unchanged executor/store fixture tests were not rerun for
the architecture-default extension. No full suite, aggregate verification,
unrelated model matrix or native pressure-stress run was performed.

Inference and server targets build. The final binary used for live/trace differs
from the comparison binary only in the default architecture selector and
comments; the measured override path is unchanged. Server request behavior is
not separately requalified. GPU runs and builds were sequential. Results do not
qualify tail latency, power, larger contexts, all Qwen checkpoints or full P07
preview/shared-first acceleration. Raw reports, controls, hashes and acceptance
checks are stored under `p07_qwen_overlap` in `atomic-engine-evidence.json`.

## Reproduction

Set `$MODEL` to the original Ornith 35B GGUF:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 128 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-engine-completion.txt --tokens 32 \
  --budget-mib 5120 --profile-experts --expert-read-overlap on \
  --output /tmp/qwen-overlap-on.json --logits /tmp/qwen-overlap-on.f32
```

Use `off` and distinct output paths for the matched control. For all-hot use
`--budget-mib 24576 --warmup 1`; subtract before/after cumulative counters.
The repeated diagnostic additionally uses `--repeat 3`. For a default trace,
omit the override and add `--expert-trace /tmp/qwen-overlap.atet`, then replay
that file with `tools/expert-source/replay-cache.py`.

The live case uses `--ubatch 32 --prompt-file docs/atomic-engine-prefill.txt
--tokens 16 --cache-policy class-lru --prefill-budget-mib 4800 --resize-at 2:6144
--resize-at 2:4706 --resize-at 4:6144 --resize-at 8:4800 --resize-at 10:1
--resize-at 12:6144 --resize-at 14:4706 --resize-at 15:5120` without an overlap
override. Budget targets belong to this exact configuration.
