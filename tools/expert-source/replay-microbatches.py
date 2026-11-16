#!/usr/bin/env python3
"""Evaluate layer-major expert reuse without changing an inference graph.

Requires an ATET3 trace of exactly one complete prefill and its tensor manifest
(`index name` lines). Validate every observed cache decision before publishing
candidate results. No runtime, numerical-equivalence or latency claim is made.
"""

import argparse
from array import array
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location("expert_cache", Path(__file__).with_name("replay-cache.py"))
_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache)
Cache = _cache.Cache


@dataclass
class Projection:
    tensor: int
    tokens: int
    layer: int
    role: str
    events: array = field(default_factory=lambda: array("Q"))

    def records(self):
        # kind (0 acquire, 1 release), expert, payload, allocation, capacity,
        # observed hit, required allocation. Projection owns the tensor index.
        for offset in range(0, len(self.events), 7):
            yield self.events[offset:offset + 7]


def load_trace(path, manifest, prefill_tokens):
    names = {}
    for line in manifest.read_text().splitlines():
        index, name = line.split()
        if int(index) in names:
            raise ValueError("duplicate tensor index")
        names[int(index)] = name
    projections = []
    pins = {}
    config = None
    capacities = []
    with path.open() as stream:
        if stream.readline().strip() != "ATET3":
            raise ValueError("requires ATET3 leases")
        for number, line in enumerate(stream, 2):
            kind, *fields = line.split()
            if len(fields) != 8:
                raise ValueError(f"invalid event at line {number}")
            tensor, expert, tokens, payload, allocation, capacity, hit, required = map(int, fields)
            if min(tensor, expert, tokens, payload, allocation, capacity, hit, required) < 0:
                raise ValueError("negative trace field")
            if kind == "C":
                if config is not None or projections or allocation != payload:
                    raise ValueError("requires one active retention policy at trace start")
                config = payload
            elif kind == "P":
                if pins or config is None or tokens < 1:
                    raise ValueError("invalid projection boundary or outstanding lease")
                match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", names.get(tensor, ""))
                if not match:
                    raise ValueError("unmapped routed projection")
                projections.append(Projection(tensor, tokens, int(match[1]), match[2]))
            elif kind in ("A", "R"):
                if not projections or projections[-1].tensor != tensor:
                    raise ValueError("event outside its projection")
                key = (tensor, expert)
                if kind == "A":
                    if min(payload, allocation, capacity, required) <= 0 or required > allocation or hit not in (0, 1):
                        raise ValueError("invalid acquire")
                    pins[key] = pins.get(key, 0) + 1
                    capacities.append(capacity)
                else:
                    if key not in pins:
                        raise ValueError("release without lease")
                    pins[key] -= 1
                    if not pins[key]:
                        del pins[key]
                projections[-1].events.extend((int(kind == "R"), expert, payload, allocation, capacity, hit, required))
            else:
                raise ValueError("reset or unknown event: fixed prefill only")
    if pins or not projections or not capacities:
        raise ValueError("empty or incomplete trace")
    role_order = {"gate": 0, "up": 1, "down": 2}
    signature = sorted({(p.layer, p.role, p.tensor) for p in projections}, key=lambda x: (x[0], role_order[x[1]]))
    layers = sorted({x[0] for x in signature})
    if layers != list(range(len(layers))) or len(signature) != len(layers) * 3:
        raise ValueError("requires contiguous layers with gate/up/down projections")
    batches = []
    for p in projections:
        if p.tensor == signature[0][2]:
            batches.append([])
        if not batches:
            raise ValueError("trace starts inside a microbatch")
        batches[-1].append(p)
    for batch in batches:
        actual = [(p.layer, p.role, p.tensor) for p in batch]
        # The last layer's token-local FFN may be pruned for a non-output batch.
        if len(actual) not in (len(signature), len(signature) - 3) or actual != signature[:len(actual)]:
            raise ValueError("unexpected layer order or partial microbatch")
        if any(p.tokens != batch[0].tokens for p in batch if p.layer != layers[-1]):
            raise ValueError("inconsistent microbatch shape")
        if any(p.tokens > batch[0].tokens for p in batch):
            raise ValueError("last layer cannot expand the token count")
    if sum(batch[0].tokens for batch in batches) != prefill_tokens:
        raise ValueError("prefill token count mismatch: reject missing batches or decode suffix")
    return batches, dict(history_bytes=config, capacity_min=min(capacities), capacity_max=max(capacities),
                         layers=len(layers), projections=len(projections), acquisitions=len(capacities))


def schedule(batches, window, batch_tokens):
    if window < 1 or batch_tokens < 1:
        raise ValueError("positive window and server batch size required")
    pending = []
    within_batch = 0

    def emit(group):
        for layer in sorted({p.layer for batch in group for p in batch}):
            for batch in group:
                yield from (p for p in batch if p.layer == layer)

    for batch in batches:
        count = batch[0].tokens
        if within_batch + count > batch_tokens:
            raise ValueError("microbatch crosses a server batch boundary")
        pending.append(batch)
        within_batch += count
        if len(pending) == window or within_batch == batch_tokens:
            yield from emit(pending)
            pending = []
        if within_batch == batch_tokens:
            within_batch = 0
    yield from emit(pending)


def simulate(projections, policy, capacity=None, exact=False):
    cache = Cache(policy)
    for p in projections:
        if cache.pins:
            raise ValueError("outstanding lease at a scheduling boundary")
        for kind, expert, payload, allocation, recorded_capacity, observed_hit, required in p.records():
            key = p.tensor, expert
            if kind == 0:
                hit = cache.access(key, payload, required,
                                   recorded_capacity if capacity is None else capacity, "prefill", True)
                if exact and (hit != bool(observed_hit) or cache.entries[key][0] != allocation):
                    raise ValueError(f"observed cache diverged at tensor {p.tensor}, expert {expert}")
                cache.pins[key] = cache.pins.get(key, 0) + 1
            else:
                if key not in cache.pins:
                    raise ValueError("release without lease")
                cache.pins[key] -= 1
                if not cache.pins[key]:
                    del cache.pins[key]
        cache.clocks[p.tensor] = cache.clocks.get(p.tensor, 0) + p.tokens
    if cache.pins:
        raise ValueError("trace ended with outstanding leases")
    return cache.result()


def evaluate(path, manifest, prefill_tokens, observed_policy, windows, reserve, batch_tokens):
    batches, meta = load_trace(path, manifest, prefill_tokens)
    baseline = simulate((p for batch in batches for p in batch), observed_policy, exact=True)
    available = meta["capacity_min"] - reserve
    if reserve < 0 or available <= 0:
        raise ValueError("invalid hypothetical activation reserve")
    results = {}
    for window in windows:
        result = simulate(schedule(batches, window, batch_tokens), "lru", available)
        result["saved_read_fraction"] = 1 - result["total"]["read_bytes"] / baseline["total"]["read_bytes"]
        results[str(window)] = result
    with path.open("rb") as stream:
        trace_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(trace=path.name, trace_sha256=trace_digest,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                prefill_tokens=prefill_tokens, microbatches=len(batches), server_batch_tokens=batch_tokens,
                observed_policy=observed_policy, observed_decisions_and_allocations_match=True,
                **meta, hypothetical_activation_reserve=reserve, candidate_capacity=available,
                observed=baseline, layer_major_lru=results,
                limitations="Offline logical I/O only. Recorded routes assume unchanged arithmetic. "
                "Reserve is a sensitivity parameter, not measured graph workspace. "
                "No runtime schedule, bitwise equivalence, TTFT or physical SSD claim. "
                "All candidates preserve projection order inside a layer, microbatch chronology per layer "
                "and recorded leases; windows stop at server batch boundaries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, required=True)
    parser.add_argument("--observed-policy", choices=("class_lru", "prefill_frequency"), required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--reserve-mib", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.trace, args.manifest, args.prefill_tokens, args.observed_policy,
                      args.windows, args.reserve_mib * 1024**2, args.batch_tokens)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exact observed replay: {result['acquisitions']:,} acquisitions")
    for window, row in result["layer_major_lru"].items():
        print(f"Window {window}: {row['total']['read_bytes']:,} logical bytes, "
              f"{row['saved_read_fraction']:.2%} reduction (offline only)")


if __name__ == "__main__":
    main()
