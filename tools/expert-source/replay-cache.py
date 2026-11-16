#!/usr/bin/env python3
"""Compare LRU, ClassLRU and optional prefill frequency retention on ATET1/2/3.

No model access or inference. The trace's LRU observations are checked exactly;
no result is published on a mismatch. Capacity includes transient miss staging.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import struct


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class Cache:
    def __init__(self, policy):
        self.policy = policy
        self.entries = OrderedDict()
        self.classes = [OrderedDict() for _ in range(8)]
        self.history = {}  # Deliberately survives eviction, as in upstream.
        self.clocks = {}
        self.used = 0
        self.pins = {}
        self.peak_with_staging = 0
        self.counts = {phase: dict(hits=0, misses=0, read_bytes=0, evictions=0)
                       for phase in ("prefill", "decode")}

    def reset(self):
        if self.pins:
            raise ValueError("cannot replay trim with active leases")
        self.entries.clear()
        for queue in self.classes:
            queue.clear()
        self.used = 0

    def evict(self):
        if self.policy == "lru":
            key = next((key for key in self.entries if key not in self.pins), None)
        elif self.policy == "prefill_frequency":
            candidates = []
            for cls, queue in enumerate(self.classes):
                first = next((key for key in queue if key not in self.pins), None)
                last = next((key for key in reversed(queue) if key not in self.pins), None)
                for key in (first, last):
                    if key is not None:
                        crf, previous = self.history[key]
                        age = self.clocks.get(key[0], 0) - previous
                        score = age / 128.0 - math.log2(crf) if crf > 0 else math.inf
                        candidates.append((score, -cls, key))
            key = max(candidates)[2] if candidates else None
        else:
            candidates = []
            for cls, queue in enumerate(self.classes):
                key = next((key for key in queue if key not in self.pins), None)
                if key is not None:
                    age = self.clocks.get(key[0], 0) - self.history[key][1]
                    candidates.append((math.ldexp(age, -cls), -cls, key))
            key = max(candidates)[2] if candidates else None
        if key is None:
            raise ValueError("no unpinned eviction victim")
        allocation, cls = self.entries.pop(key)
        self.classes[cls].pop(key, None)
        self.used -= allocation
        return allocation

    def access(self, key, payload, allocation, capacity, phase, reuse_larger=False):
        counters = self.counts[phase]
        hit = key in self.entries
        cls = 0
        if self.policy != "lru":
            now = self.clocks.get(key[0], 0)
            crf, last = self.history.get(key, (0.0, now))
            crf = f32(f32(crf * f32(2 ** (-(now - last) / 128.0))) + 1.0)
            self.history[key] = (crf, now)
            cls = min(7, int(math.log2(crf)))
        if hit:
            old_allocation, old_cls = self.entries.pop(key)
            if not reuse_larger and old_allocation != allocation:
                raise ValueError("allocation changed for a cached expert")
            allocation = old_allocation
            self.classes[old_cls].pop(key, None)
            counters["hits"] += 1
        else:
            if allocation + payload > capacity:
                raise ValueError("trace cannot fit one expert and read staging")
            while self.used + allocation + payload > capacity:
                victim = self.evict()
                counters["evictions"] += 1
                # Runtime retains only the policy-selected victim, and only if
                # its full allocation plus transient staging still fits.
                if reuse_larger and victim >= allocation and self.used + victim + payload <= capacity:
                    allocation = victim
                    break
            self.peak_with_staging = max(self.peak_with_staging,
                                         self.used + allocation + payload)
            self.used += allocation
            counters["misses"] += 1
            counters["read_bytes"] += payload
        if self.used > capacity:
            raise ValueError("unobserved capacity reduction; unsupported trace")
        self.entries[key] = (allocation, cls)
        self.classes[cls][key] = None
        return hit

    def result(self):
        total = {name: sum(c[name] for c in self.counts.values())
                 for name in ("hits", "misses", "read_bytes", "evictions")}
        return dict(total=total, phases=self.counts, resident_bytes=self.used,
                    peak_cache_and_staging=self.peak_with_staging)


def replay(path, class_lru_overhead=0, prefill_frequency=False):
    caches = [Cache("lru"), Cache("class_lru")]
    if prefill_frequency:
        caches.append(Cache("prefill_frequency"))
    active = None
    tokens = 0
    accesses = 0
    projections = 0
    recorded_overhead = 0
    observed_policy = "lru"
    with path.open() as stream:
        version = stream.readline().strip()
        if version not in ("ATET1", "ATET2", "ATET3"):
            raise ValueError("unsupported trace version")
        for line_number, line in enumerate(stream, 2):
            kind, *fields = line.split()
            if len(fields) != (7 if version == "ATET1" else 8):
                raise ValueError(f"invalid event at line {line_number}")
            tensor, expert, n_tokens, payload, allocation, capacity, observed_hit = map(int, fields[:7])
            required = allocation if version == "ATET1" else int(fields[7])
            if kind == "C":
                if accesses or projections or allocation not in (0, payload):
                    raise ValueError("invalid policy configuration")
                class_lru_overhead = payload
                recorded_overhead = allocation
                observed_policy = "class_lru" if allocation else "lru"
            elif kind == "P":
                if active is not None:
                    for cache in caches:
                        cache.clocks[active] = cache.clocks.get(active, 0) + tokens
                active, tokens = tensor, n_tokens
                if tokens < 1:
                    raise ValueError("invalid projection token count")
                projections += 1
            elif kind == "X":
                for cache in caches:
                    cache.reset()
            elif kind == "R" and version == "ATET3":
                key = (tensor, expert)
                for cache in caches:
                    if key not in cache.pins:
                        raise ValueError("release without an active lease")
                    cache.pins[key] -= 1
                    if not cache.pins[key]:
                        del cache.pins[key]
            elif kind == "A":
                if tensor != active or min(payload, allocation, capacity, required) <= 0 or required > allocation or observed_hit not in (0, 1):
                    raise ValueError("invalid access event")
                phase = "decode" if tokens == 1 else "prefill"
                for cache in caches:
                    available = capacity + recorded_overhead - (class_lru_overhead if cache.policy != "lru" else 0)
                    hit = cache.access((tensor, expert), payload, required, available, phase, version != "ATET1")
                    if version == "ATET3":
                        cache.pins[(tensor, expert)] = cache.pins.get((tensor, expert), 0) + 1
                    if cache.policy == observed_policy and hit != bool(observed_hit):
                        raise ValueError(f"{observed_policy.upper()} diverged from runtime at line {line_number}")
                    if cache.policy == observed_policy and cache.entries[(tensor, expert)][0] != allocation:
                        raise ValueError(f"physical allocation diverged from runtime at line {line_number}")
                accesses += 1
            else:
                raise ValueError(f"unknown event {kind}")
    if not accesses:
        raise ValueError("empty trace")
    if any(cache.pins for cache in caches):
        raise ValueError("trace ended with active leases")
    baseline, candidate = [cache.result() for cache in caches[:2]]
    saved = baseline["total"]["read_bytes"] - candidate["total"]["read_bytes"]
    result = dict(trace=path.name, version=version, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                accesses=accesses, projections=projections, observed_policy=observed_policy,
                observed_policy_matches_runtime=True, class_lru_history_bytes=class_lru_overhead,
                lru=baseline, class_lru=candidate, saved_read_bytes=saved,
                saved_read_fraction=saved / baseline["total"]["read_bytes"],
                limitations="Synchronous UMA, fixed total cap, one request. One-token projections are classified as decode. Replay does not predict latency or physical SSD traffic.")
    if prefill_frequency:
        result["prefill_frequency"] = caches[2].result()
        before = candidate["phases"]["prefill"]["read_bytes"]
        after = result["prefill_frequency"]["phases"]["prefill"]["read_bytes"]
        result["frequency_saved_prefill_bytes"] = before - after
        result["frequency_saved_prefill_fraction"] = (before - after) / before if before else 0
        result["limitations"] += " Frequency mode is simulated for the whole trace; only prefill counts are a candidate prediction. Runtime restores LRU for decode."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-lru-overhead", type=int, default=0, help="History bytes for older traces without a configuration record")
    parser.add_argument("--prefill-frequency", action="store_true", help="Also evaluate bounded decayed-frequency candidates; compare prefill counts only")
    args = parser.parse_args()
    results = [replay(path, args.class_lru_overhead, args.prefill_frequency) for path in args.trace]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(f"{result['trace']}: exact runtime replay; ClassLRU saves "
              f"{result['saved_read_fraction']:.2%} logical bytes")
        if args.prefill_frequency:
            print(f"  prefill frequency candidate saves "
                  f"{result['frequency_saved_prefill_fraction']:.2%} prefill bytes versus ClassLRU")


if __name__ == "__main__":
    main()
