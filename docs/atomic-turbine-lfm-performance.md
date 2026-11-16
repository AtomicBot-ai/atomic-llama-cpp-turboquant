# LFM streaming profile and allocation reuse

## Scope and decision

P10A established resident equivalence for LFM2.5 24B on M4 Max Metal. This
follow-up profiles the completed 200-token prompt / 52-token answer at a fixed
4 GiB engine cap, context 512 and microbatch 32. Original router, kernels,
quantization and active state remain unchanged.

The integrated steps are a fixed-cap phase profile, removal of confirmed
allocation churn, precise trace/replay accounting for retained capacity, and
targeted logits/state/budget and live-release comparisons. No full suite or
unrelated backend qualification is part of this work.

The baseline spends 8.7866 seconds in compute/submission and 9.9662 seconds in
expert acquisition, including 2.8547 seconds in allocation/preparation and
5.2147 seconds in source reads. Its 51,165 misses perform 99,582,345,216 logical
read bytes; there are no hits. Mixed Q4_K/Q5_K/Q6_K expert sizes cause 21,009
physical buffer allocations despite 30,156 exact-size reuses.

The selected change allows the policy-selected completed victim to serve a
smaller expert when its physical allocation is large enough. Its full charge
remains held, including excess capacity, throughout the new read. The tensor
arena is rebuilt for the new shape, the entire buffer is cleared, and the entry
is published only after successful complete read/upload/synchronization. Pinned
or retiring victims remain excluded. If staging cannot fit with the retained
buffer, normal free-and-admit logic applies. No detached pool retains memory
after shrink. The minimum working floor is unchanged.

Larger retained buffers can reduce effective cache capacity; this is a measured
tradeoff, not free memory. ATET2 therefore records both minimum required and
actual physical allocation. Replay simulates victim reuse separately for LRU
and upstream-derived ClassLRU, and validates runtime hits and physical sizes.
Historical ATET1 traces remain supported with their original exact-size rules.

## Accepted result

With phase timers enabled on both fixed-cap LRU runs, the request fell from
20.2817 to 17.2264 seconds (15.1%); prefill fell from 11.7141 to 9.8459 seconds
(15.9%). New buffer allocations fell from 21,009 to 974 and reuse increased from
30,156 to 50,191. Allocation/preparation fell from 2.8547 to 0.7416 seconds and
admission from 0.6884 to 0.1434 seconds. Compute remained approximately 8.77
seconds, with unchanged 51,165 submissions and 99,582,345,216 logical read bytes.
This isolates allocation churn as the accepted improvement. Source reads and
GPU submission/computation remain substantial; no asynchronous I/O or different
numerical kernel is introduced without further evidence.

Full logits remain bitwise equal to the resident P10A control, with the same
52 committed tokens, EOG 124900 and serialized state `84ac8e66a7115244`.
The optimized fixed-cap LRU high-water is 4,294,947,408 bytes below the
4,294,967,296-byte cap. The engine floor remains 3,563,967,392 bytes. Larger
retained allocations reduce final resident entry count from 372 to 280; there
were no LRU hits in either control, so this did not increase reads here. It may
affect locality on other workloads; no general speedup is claimed.

The ATET2 LRU trace exactly reproduced all 51,165 runtime accesses and actual
allocation sizes. ClassLRU predicted 3,356 hits and 47,809 misses, saving
6,560,587,776 logical read bytes (6.59%). A live ClassLRU control matched the
predicted hits, misses, evictions and 93,021,757,440 read bytes exactly. Its
123,584-byte history is included in the same total cap, raising the floor to
3,564,090,976 bytes. It also preserves full logits and state. Request time was
17.1949 seconds against LRU's 17.2264: this single-run difference does not
establish a latency win. Keep LRU as default and ClassLRU opt-in. Replay phase
labels classify one-token projections as decode; total counters are the reliable
comparison when prefill's final output projection contains one token.

A final ClassLRU live-control request covered prefill shrink to 3456 MiB,
grow to 4608 MiB, shrink back to 3456 MiB, rejection of 1 MiB and recovery to
4096 MiB. Shrink released 1,204,355,072 Metal bytes in 10.3984 ms, with a
1,214,775,920-byte immediate footprint drop. Grow had zero immediate Metal
allocation; impossible admission left cap and used bytes unchanged. All recorded
used/applied bounds held, and lifetime high-water 4,831,833,696 remained below
the maximum 4,831,838,208-byte cap. Full logits, answer and state still match.

## Focused checks and reproduction

The new store fixture proves oversized reuse retains full physical accounting,
clears the old payload beyond the new shape, excludes pinned entries, reports
required/actual sizes separately and returns all allocation on shrink. The
existing failed-read reuse case verifies that a failed reread cannot publish a
recycled entry. These two cases passed 18 assertions. After trace metadata was
extended, only the oversized and demand-trace cases were checked (16 assertions).
The directly affected replay file passed five cases, including ATET1 compatibility,
ATET2 physical mismatch rejection and a retained victim that cannot fit staging.

Five real-model runs cover baseline profile, optimized profile, trace capture,
ClassLRU control and live resize. Builds and GPU runs were sequential. Timers
are enabled only on the three fixed-cap profile/control runs; tracing is enabled
only on its dedicated capture. OS page-cache state is uncontrolled. This is
single-workload causal evidence, not a statistical latency/tail qualification,
physical SSD-byte measurement or cross-architecture performance claim. No full
suite, aggregate verification or unrelated tests ran.

With `$MODEL` set to the original LFM GGUF, the current fixed-cap profile is:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-turbine-lfm-prefill.txt --tokens 64 \
  --budget-mib 4096 --profile-experts \
  --output /tmp/lfm-profile.json --logits /tmp/lfm-profile.f32
```

For a trace, replace `--profile-experts` with `--expert-trace /tmp/lfm.atet`,
using distinct report/logit paths, then run:

```sh
python3 tools/expert-source/replay-cache.py /tmp/lfm.atet --output /tmp/lfm-replay.json
```

The candidate control adds `--cache-policy class-lru` to the fixed-cap command.
The live-control command uses that policy and adds
`--prefill-budget-mib 3456 --resize-at 2:4608 --resize-at 4:3456 --resize-at 6:1 --resize-at 8:4096`.
Compare full `.f32` output and state/receipt fields with the P10A resident control.
Raw reports, trace, hashes and decisions are preserved under `p06_lfm` in the
evidence record. The earlier P10A and Qwen measurements remain historical controls.

The subsequent [GPU submission profile](atomic-turbine-lfm-gpu.md) extends the
executor to bounded multi-expert groups and ATET3 lease events. The measurements
above retain the single-wave implementation as their historical control.
