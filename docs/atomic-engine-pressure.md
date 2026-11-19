# P05: automatic memory pressure control

## L2 scope and decision

The controller proposes budgets; P04 remains the sole owner of applying them
after synchronized inference graphs or an explicit idle poll. No background
thread frees weights or active KV/recurrent state. Optional prefetch/draft pools
do not exist in this profile; completed expert allocations are the only victims.

1. Implement a deterministic controller with supplied monotonic time and replay
   warning/critical/normal, stale/missing signals, headroom, flapping, manual
   priority and changing floor. Run only that test file before runtime wiring.
2. Integrate configuration, signal submission and status with the existing
   request/apply API. Retain critical events across signal coalescing.
3. Add a macOS dispatch memory-pressure source. Callbacks only record signals;
   sample conservative free-page headroom at safe polling points.
4. Expose shared common/server configuration and native diagnostic controls.
5. Run a bounded real-model signal scenario with a small neighboring allocation,
   compare logits/state and physical release, and record platform limitations.

Fixed mode retains its cap. Manual requests immediately suspend automatic mode,
including a request later rejected below the floor; explicit reconfiguration
resumes automatic control. Automatic mode starts from the current cap, clamps
to the preferred ceiling, and never silently raises that ceiling to fit state.

Warning shrinks by 256 MiB at most once per second. Critical requests the larger
of the caller's lower bound and current engine floor immediately at the next
safe point. Recovery requires continuously fresh normal samples for 30 seconds,
then grows by at most 64 MiB per second. Growth needs advisory headroom covering
the proposed cap's unused capacity plus a 1 GiB system reserve. Unknown/expired
normal or unknown headroom inhibits growth. Last-known elevated pressure remains
conservative until an explicit lower signal arrives. Pressure following growth
doubles the normal recovery delay, capped at five minutes. These defaults are
explicit policy choices, not OS-derived safe-memory guarantees.

The macOS source uses dispatch NORMAL/WARN/CRITICAL notifications. Initial level
is unknown until observed; absence of a warning is not interpreted as normal.
The OS signal is system-wide, not a byte budget. `host_statistics64` free pages
are advisory conservative headroom; neither inactive/file cache nor Metal's
recommended working-set size is treated as guaranteed free RAM.

Primary API references: [Apple pressure source](https://developer.apple.com/documentation/Dispatch/DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)
and [pressure flags](https://developer.apple.com/documentation/dispatch/dispatch_source_memorypressure_flags_t).
The local SDK marks `os_proc_available_memory` unavailable on macOS, so it is
not used. No global `memory_pressure -S` broadcast or unbounded stress allocation
is part of the checks.

## Configuration and status

The shared CLI exposes `--engine-budget-mib`, `--engine-memory-auto`,
`--engine-memory-min-mib` and `--engine-memory-reserve-mib`, with corresponding
`LLAMA_ARG_ENGINE_*` environment variables. Omit `--engine-memory-auto` for a
fixed budget. Automatic mode requires an explicit total budget and a native
source; unsupported platforms reject activation rather than silently polling
unknown signals. The embedding API can use an external signal provider.

For the qualified single-sequence profile:

```sh
build-metal/bin/llama-server -m "$MODEL" \
  -c 512 -b 512 -ub 32 -np 1 -t 8 -tb 8 -ngl 99 -fit off \
  --engine-budget-mib 5120 --engine-memory-auto \
  --engine-memory-min-mib 4800 --engine-memory-reserve-mib 1024 \
  --cache-ram 0 --ctx-checkpoints 0 --no-warmup --metrics \
  --host 127.0.0.1 --port 8080
```

Upstream auto-fit, speculation and multimodal projectors are rejected with a
total engine budget. The server also rejects nonzero prompt-cache/checkpoint
settings because those external state copies do not participate in admission.
The active KV/recurrent state remains available. Idle polling runs every second
on the inference thread outside the queue mutex; active inference polls at graph
boundaries. A sleeping server has already unloaded its model.

The experimental C++ API in `src/llama-atomic.h` provides:

- `llama_memory_policy_configure`: configure fixed/manual/automatic mode while
  idle. `system_source=true` selects macOS; false selects externally submitted
  samples. Invalid configuration, an unavailable source or a pending manual
  request returns false and preserves the previous configuration. Compare the
  preferred cap with `llama_memory_budget_get().minimum` before configuration.
- `llama_memory_pressure_submit`: thread-safe external sample submission;
  rejected outside automatic mode or when a native source owns the signals.
- `llama_memory_policy_get`: idle status including mode, pressure, reason,
  effective lower bound, preferred cap, reserve, sample age, advisory headroom,
  recovery delay and decision count. `llama_memory_policy_reason_name` formats
  the reason. Budget request/completion receipts retain P04 semantics.
- `llama_memory_budget_apply`: idle control poll; embedding hosts must call it
  periodically when no inference is running. It can return a retained prior
  budget rejection; inspect policy reason and receipt IDs separately.

`/metrics` exports integer `llamacpp:engine_memory_*` gauges for requested,
applied, engine minimum, used and released bytes; policy mode, reason, decisions;
pressure level and advisory headroom/validity. The engine minimum excludes the
optional policy lower bound. Mode is fixed=0/manual=1/automatic=2. Pressure is
unknown=0/normal=1/warning=2/critical=3. Reason values follow the declared enum
in `src/llama-memory-policy.h` (an experimental interface). Engine gauges are
omitted during concurrent yielded tasks to avoid racing the inference context.
Lifetime releases include ordinary cache evictions; use before/after snapshots
and Metal allocation deltas to attribute a particular shrink.

## Targeted evidence, 2026-11-13

The deterministic replay ran before runtime wiring: six cases / 58 assertions
covered critical coalescing, recovery/headroom/freshness, pressure after growth,
manual/fixed priority, invalid/changing floors, time/overflow and flapping.
Adding headroom observability reran only its affected case (12 assertions),
bringing the unique policy assertion count to 60. Two new server cases passed
seven assertions for idle thread/mutex ownership and common arguments/metrics.
Only the five-assertion argument/metric case was repeated after fixing integer
precision. No full suite or aggregate verification ran.

Real Metal replay used Ornith 35B, context 512, microbatch 32, the 190-token
prefill fixture, greedy 16-token output and LRU under a 5 GiB preferred cap.
A neighboring process held a touched 256 MiB allocation throughout the request;
pressure signals were injected, not inferred from that allocation. Diagnostic
recovery/interval were shortened to 200 ms to exercise transitions in one short
session; production defaults remain 30 seconds / one second.

Critical followed immediately by normal during prefill still shrank to 4800 MiB.
Decode recovered in 64 MiB increments. A second critical signal doubled recovery
to 400 ms and released 66,060,288 Metal bytes in 1.03 ms. At token 10 a manual
5120 MiB request suspended automatic mode; token 11's critical submission was
rejected. Explicit resume at token 12 accepted critical pressure and released
334,430,208 Metal bytes (319 MiB) in 6.44 ms, with a 343,785,568-byte immediate
physical-footprint decrease. These are individual observations, not latency
percentiles or a physical-footprint SLA.

All 37 recorded snapshots satisfied used <= applied <= 5 GiB; the lifetime
ledger peak was 5,368,291,248 bytes against 5,368,709,120. Full logits were
bitwise equal to the fixed-budget optimized control, committed IDs matched,
and serialized state retained digest `9755c384a23111fe`. The controller only
changed retained expert capacity. The run is correctness/release evidence,
not a speed comparison with an intentionally concurrent neighboring process.

Replay reproduction, with `$MODEL` set to the original GGUF:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --output /tmp/atomic-engine-p05-replay.json \
  --logits /tmp/atomic-engine-p05-replay.f32 \
  --prompt-file docs/atomic-engine-prefill.txt \
  --tokens 16 --ctx 512 --ubatch 32 --gpu-layers 99 --budget-mib 5120 \
  --state-digest --memory-policy replay --memory-min-mib 4800 \
  --memory-recovery-ms 200 --pressure-prefill \
  --pressure-at 1:normal:4096 --pressure-at 4:critical:0 \
  --pressure-at 5:normal:4096 --pressure-at 8:normal:4096 \
  --resize-at 10:5120 --pressure-at 11:critical:0 \
  --auto-resume-at 12 --pressure-at 12:critical:0 \
  --pressure-at 13:normal:4096
```

`--pressure-at` is `committed_count:level:advisory_headroom_MiB`.
`--pressure-prefill` injects critical then normal inside the first expert graph;
application waits for its synchronized boundary. The optional neighbor used
256 MiB of touched memory and was terminated after the request. Native diagnostic
mode is `--memory-policy system`; it cannot combine with replay injection.

The native server check rejected unaccounted state-cache settings before model
load, then started with those caches disabled, sampled valid free-page headroom
on its idle poll, exported exact integer budget counters and completed one
two-token greedy request. Graceful shutdown returned zero after cancelling the
dispatch source. Observed mode was automatic, level unknown, decisions zero;
headroom was 597,147,648 bytes and applied cap 5,368,709,120. This checks source
creation/poll/destruction and shared server wiring, not delivery of an elevated
native pressure event. The server admission check also corrected an empty,
sentinel-only upstream buffer-override list being mistaken for real overrides.

The macOS source and bounded replay are separate evidence: normal startup does
not prove native warning/critical delivery or prolonged competition with another
application. P05 is opt-in on this profile. Native pressure-event stress
qualification, additional architectures/platforms and release-matrix P11 remain
separate work; no global OS pressure was synthesized.
