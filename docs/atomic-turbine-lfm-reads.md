# LFM synchronous expert-read profile

## Measurement and decision

After grouped GPU submissions, source acquisition is the remaining dominant
host phase. Opt-in counters now separate source lookup/locking, file identity
validation, seek and transfer. These are nested inside `read`, not additional
terms to add to it. Transfer includes buffered I/O, kernel copying and page
faults; it is not a measurement of SSD time alone.

The initial 4 GiB LRU run took 13.6504 seconds, including 6.1117 seconds in
transfer. The OS charged about 8.47 GB of process reads during the request.
A second baseline was necessary to distinguish file-cache effects: it took
12.1952 seconds with 0.503 GB OS-charged reads. In this warm baseline, source
transfer occupied 4.8777 seconds, both identity checks together 40.56 ms and
seek 12.12 ms. Changing seek APIs or weakening validation is not justified by
this profile. The upload copy cost 1.1181 seconds and buffer preparation 0.6453.

The accepted change reads original GGUF bytes directly into an exclusively
owned CPU/shared Metal expert slot. It removes the per-miss heap payload and
subsequent tensor upload copy. A private Metal capability returns a host pointer
only for recognized shared buffers; private storage returns null. Other buffers
retain staged reads and tensor upload. Generic Metal `is_host` remains unchanged.

A fresh slot or completed, unpinned victim is unpublished during the read.
The payload is completely overwritten; only the physical tail beyond the new
payload is zeroed, including excess space retained from a larger victim. Both
file identity checks, exact-length reads, poisoning on failure and synchronous
publication remain intact. Failure destroys the unpublished slot and returns
its charges. No asynchronous reads, prefetch, changed router or quantization is
introduced.

The same full-payload I/O headroom is reserved before admission on both paths.
Direct reads leave this ticket uncommitted because no staging memory is
allocated. This preserves cache capacity, worst-case grouped-lease admission,
the working floor and ATET3 replay semantics. Physical-release counters exclude
cancelled reservations, so their totals correctly fall without implying that
less cache memory was reclaimed. The managed ledger still excludes kernel file
cache and is not an OS RSS limit.

## Result and safety evidence

On the same 200-token prompt and completed 52-token answer, direct reads took
**11.3910 seconds**, 6.59% below the warm baseline. Prefill fell from **7.1148 to
6.7682 seconds** (4.87%). Upload fell to 3.25 ms and preparation to 195.36 ms.
Transfer into shared buffers increased to 5.5374 seconds; this offsets part of
the savings. Compute also varied from 3.8558 to 4.0001 seconds. The observation
supports removal of redundant memory work, not a claim that storage became
faster. OS-charged reads were similar at 0.464 GB. These are individual warm
observations, not a statistical latency or controlled cold-SSD qualification.

All 99,582,345,216 logical bytes used the direct path. Misses (51,165), physical
allocations (937), reused slots (50,228), matmuls (52,905) and submissions
(13,077) exactly match baseline. Fixed-cap ledger high-water remains
4,294,966,576 bytes below the 4,294,967,296-byte cap; the floor remains
3,567,199,296 bytes. Full logits, all committed IDs, EOG 124900 and serialized
state `84ac8e66a7115244` match the saved resident control bitwise in both
baselines and the optimized run.

The ClassLRU live case starts at 4096 MiB, shrinks during prefill to 3456 MiB,
grows at token 2 to 4608 MiB, shrinks at token 4 to 3403 MiB, rejects 1 MiB at
token 6 and grows at token 8 to 4096 MiB. The near-floor shrink frees
1,261,568,000 Metal bytes in 11.45 ms; observed footprint drops by
1,272,480,536 bytes. The next two tokens perform 912 matmuls in 912 submissions,
preserving the single-wave fallback. Growth allocates no immediate Metal memory;
rejection leaves the applied cap and used bytes unchanged. Lifetime high-water
is 4,831,813,008 bytes below the maximum applied 4,831,838,208-byte cap. The
complete answer, full logits and state again match resident bitwise.

Targeted checks only:

- Four CPU store cases, 27 assertions: direct bytes and reserved-only staging,
  staged fallback, pinned/recycled publication and oversized-tail clearing.
- Three Metal store cases, 24 assertions: direct bytes/accounting, protected
  recycled slots, failed reread rollback and oversized-tail clearing.
- One Metal executor case, seven assertions: a source failure after the first
  group acquisition releases pending leases without submitting partial data.
- The four LFM requests above. No full suite, aggregate verification, new Qwen
  qualification, private-Metal inference or native OS-pressure stress run.

Builds and GPU runs were sequential. The inference and server targets rebuild;
server request behavior is not requalified by this package. Source checks still
require immutable files and do not protect against malicious concurrent writes.
Remaining transfer time and cache misses are measured costs; their reduction
needs separate evidence before changing I/O scheduling or cache policy.

## Reproduction

Use the original LFM GGUF as `$MODEL`:

```sh
cmake --build build-metal --target llama-expert-infer -j4
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-turbine-lfm-prefill.txt --tokens 64 \
  --budget-mib 4096 --profile-experts \
  --output /tmp/lfm-reads.json --logits /tmp/lfm-reads.f32
```

The live case adds `--cache-policy class-lru --prefill-budget-mib 3456
--resize-at 2:4608 --resize-at 4:3403 --resize-at 6:1 --resize-at 8:4096`.
The warm baseline precedes direct writes; its binary hash and raw reports are
recorded separately. `p06_lfm_reads` in `atomic-turbine-evidence.json` retains
reports, artifact hashes, source provenance and exact acceptance checks.
