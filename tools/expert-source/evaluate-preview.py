#!/usr/bin/env python3
"""Evaluate ATP2 next-layer preview against bound ATET3 demand; no inference.

The bounded replay assumes reads finish before demand. It measures logical I/O
and cache pollution, NOT GPU overlap, latency, or physical SSD traffic.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("cache_replay", Path(__file__).with_name("replay-cache.py"))
cache_replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_replay)
Cache = cache_replay.Cache


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(scores, bias=None):
    require(np.isfinite(scores).all(), "nonfinite preview scores")
    if bias is not None:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80, 80))) + bias
    return np.argsort(-scores, axis=1, kind="stable")[:, :4].tolist()


def candidates(rows):
    """Rank first, then row order; global deduplication, never four per row."""
    seen = set()
    for slot in range(min(4, len(rows[0]))):
        for row in rows:
            expert = row[slot]
            if expert not in seen:
                seen.add(expert)
                yield expert


def load_capture(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(records) > 1, "empty capture")
    header, records = records[0], records[1:]
    require(header["format"] == "ATP2" and header["architecture"] in ("lfm2moe", "qwen35moe")
            and header["sequence"] == 0 and header["output_policy"] == "last_row_only", "unsupported capture")
    binary = np.memmap(str(path) + ".bin", dtype=np.uint8, mode="r")
    end = 0

    def array(meta, shape, dtype):
        nonlocal end
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        require(meta["offset"] == end and meta["bytes"] == size and size > 0
                and end + size <= binary.size, "invalid or noncontiguous tensor range")
        result = np.ndarray(shape, dtype=dtype, buffer=binary, offset=end)
        require(np.isfinite(result).all(), "nonfinite captured tensor")
        end += size
        return result

    routers = {}
    first, last = header["first_layer"], header["last_layer"]
    require(first >= 0 and last > first, "invalid routed layer range")
    for router in header["routers"]:
        layer = router["layer"]
        require(layer not in routers, "duplicate router")
        w = router["weight"]
        require(w and w["type"] in ("f32", "f16"), "unsupported router weight")
        weight = array(w, (w["output"], w["input"]), "<f4" if w["type"] == "f32" else "<f2").astype(np.float32)
        bias = None
        if router["bias"] is not None:
            b = router["bias"]
            require(b["input"] == w["output"] and b["output"] == 1 and b["type"] in ("f32", "f16"), "invalid router bias")
            bias = array(b, (b["input"],), "<f4" if b["type"] == "f32" else "<f2").astype(np.float32)
        require((bias is not None) == (header["architecture"] == "lfm2moe"), "unexpected architecture bias")
        routers[layer] = (weight, bias)
    require(set(routers) == set(range(first, last + 1)), "missing router layer")
    tensors, bundles = {}, defaultdict(list)
    for t in header["tensors"]:
        require(t["index"] not in tensors and t["layer"] in routers and t["payload"] > 0
                and t["experts"] == routers[t["layer"]][0].shape[0], "invalid expert metadata")
        tensors[t["index"]] = t
        bundles[t["layer"]].append(t["index"])
    names = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
    require(set(bundles) == set(routers) and all(len(ts) == 3 and {tensors[t]["name"] for t in ts} == names
            for ts in bundles.values()), "incomplete expert bundle")
    keyed, groups = {}, defaultdict(list)
    previous_projection = -1
    for r in records:
        key = (r["call"], r["microbatch"], r["sequence"], r["layer"])
        require(key not in keyed and r["sequence"] == 0 and r["call"] >= 0 and r["microbatch"] >= 0
                and r["layer"] in routers and r["projection"] > previous_projection, "duplicate or unordered route binding")
        previous_projection = r["projection"]
        pos, ids = r["positions"], r["ids"]
        require(0 < len(pos) <= 256 and pos == sorted(set(pos)) and pos[0] >= 0 and len(ids) == len(pos), "invalid positions")
        n_experts, width = routers[r["layer"]][0].shape
        require(r["input"] == width and all(len(row) == len(ids[0]) and 0 < len(row) <= 32
                and len(row) == len(set(row)) and all(isinstance(e, int) and 0 <= e < n_experts for e in row)
                for row in ids), "invalid route IDs or width")
        r["activation"] = array(r, (len(pos), width), "<f4")
        keyed[key] = r
        groups[key[:3]].append(r)
    require(end == binary.size, "unbound binary tail")
    previous_group = None
    previous_end = 0
    for group, rows in groups.items():
        layers = [r["layer"] for r in rows]
        require(layers in (list(range(first, last)), list(range(first, last + 1))), "missing nonfinal route layer")
        pos = rows[0]["positions"]
        require(pos == list(range(pos[0], pos[-1] + 1)), "noncontiguous token positions")
        if previous_group is None:
            require(group == (0, 0, 0) and pos[0] == 0, "capture does not start at prefill")
        else:
            require((group == (previous_group[0], previous_group[1] + 1, 0)
                     or group == (previous_group[0] + 1, 0, 0)) and pos[0] == previous_end,
                    "stale call or microbatch position binding")
        previous_group, previous_end = group, pos[-1] + 1
        for r in rows:
            require(r["positions"] == pos or (r["layer"] == last and r["positions"] == pos[-1:]), "mismatched layer rows")
    return header, records, keyed, routers, tensors, bundles


def load_demand(path, records, tensors, bundles):
    lines = path.read_text().splitlines()
    require(lines[0] == "ATET3", "preview replay requires ATET3 ownership")
    events, projections, requirements = [], [], {}
    capacity = None
    for line in lines[1:]:
        fields = line.split()
        require(len(fields) == 9, "invalid ATET3 event")
        kind, f = fields[0], list(map(int, fields[1:]))
        t, e, tokens, payload, allocation, cap, hit, required = f
        require(kind in ("C", "P", "A", "R"), "unsupported trace event (fixed request only)")
        if kind == "C":
            require(not events and allocation == 0, "preview replay currently requires cold LRU")
        elif kind == "P":
            require(t in tensors and tokens > 0, "invalid projection")
            projections.append(dict(tensor=t, tokens=tokens, experts=[]))
        elif kind == "A":
            require(projections and t == projections[-1]["tensor"] and payload == tensors[t]["payload"]
                    and 0 <= e < tensors[t]["experts"] and hit in (0, 1)
                    and 0 < required <= allocation and cap > allocation, "invalid bound access")
            require(t not in requirements or requirements[t] == required, "varying allocation requirement")
            requirements[t] = required
            require(capacity is None or capacity == cap, "changing cache capacity")
            capacity = cap
            projections[-1]["experts"].append(e)
        events.append((kind, f))
    require(capacity and set(requirements) == set(tensors), "incomplete expert demand")
    cursor = 0
    for r in records:
        require(r["projection"] == cursor, "stale projection cursor")
        chunk = projections[cursor:cursor + 3]
        require(len(chunk) == 3 and {p["tensor"] for p in chunk} == set(bundles[r["layer"]])
                and tensors[chunk[0]["tensor"]]["name"] != "ffn_down_exps.weight", "wrong projection bundle")
        selected = {e for row in r["ids"] for e in row}
        require(all(p["tokens"] == len(r["positions"]) and len(p["experts"]) == len(selected)
                    and set(p["experts"]) == selected for p in chunk), "route IDs disagree with ATET demand")
        cursor += 3
    require(cursor == len(projections), "unbound demand projection")
    return events, requirements, capacity


def predictions(records, keyed, routers, last):
    pairs, accuracy = {}, {phase: dict(rows=0, predicted=0, actual=0, intersection=0) for phase in ("prefill", "decode")}
    for r in records:
        if r["layer"] == last:
            continue
        target = keyed.get((r["call"], r["microbatch"], r["sequence"], r["layer"] + 1))
        if target is None:
            require(r["layer"] + 1 == last, "missing next layer")
            continue  # Final layer is pruned in non-output prefill microbatches.
        row_index = {pos: i for i, pos in enumerate(r["positions"])}
        require(all(pos in row_index for pos in target["positions"]), "unmatched target row")
        x = r["activation"][[row_index[pos] for pos in target["positions"]]]
        weight, bias = routers[target["layer"]]
        require(x.shape[1] == weight.shape[1], "adjacent router width mismatch")
        predicted = rank(np.einsum("ri,ei->re", x, weight, optimize=False), bias)
        actual = target["ids"]
        phase = "prefill" if r["call"] == 0 else "decode"
        a = accuracy[phase]
        for pred, truth in zip(predicted, actual):
            a["rows"] += 1
            a["predicted"] += len(pred)
            a["actual"] += len(truth)
            a["intersection"] += len(set(pred) & set(truth))
        pairs[r["projection"]] = dict(layer=target["layer"], target=target["projection"],
                                      predicted=predicted, oracle=actual, phase=phase)
    for a in accuracy.values():
        a["precision"] = a["intersection"] / a["predicted"] if a["predicted"] else None
        a["recall"] = a["intersection"] / a["actual"] if a["actual"] else None
    return pairs, accuracy


def simulate(events, requirements, capacity, tensors, bundles, pairs, records, policy):
    cache = Cache("lru")
    phase_at = {r["projection"] + i: "prefill" if r["call"] == 0 else "decode" for r in records for i in range(3)}
    stats = {phase: dict(prefetched_bytes=0, useful_bytes=0, unused_bytes=0, admitted_experts=0,
                        skipped_resident=0, rejected_budget=0, generations=0) for phase in ("prefill", "decode")}
    # The existing executor holds at most eight demand leases. Reserve their
    # worst physical size plus one read reservation before optional admission.
    max_allocation = max(max(f[4], f[7]) for kind, f in events if kind == "A")
    reserve = 8 * max_allocation + max(t["payload"] for t in tensors.values())
    optional, pending = set(), {}
    release_at = None
    projection = -1
    peak_optional = 0
    peak_optional_bytes = 0
    peak_all = 0

    def unpin(key):
        require(key in cache.pins, "release without lease")
        cache.pins[key] -= 1
        if not cache.pins[key]:
            del cache.pins[key]

    def expire(cursor):
        for key, (payload, phase, deadline) in list(pending.items()):
            if cursor >= deadline:
                stats[phase]["unused_bytes"] += payload
                del pending[key]

    for kind, f in events:
        t, e, tokens, payload, allocation, cap, hit, required = f
        if kind == "P":
            projection += 1
            if release_at == projection:
                for key in optional:
                    unpin(key)
                optional.clear()
                release_at = None
            # At the target layer, old future leases become ordinary cache
            # residency. Its demand may now run beside one new future generation.
            # Pending below is only offline attribution, not another pinned queue.
            expire(projection)
            if policy != "baseline" and projection in pairs:
                require(not optional, "overlapping speculative generations")
                pair = pairs[projection]
                phase = pair["phase"]
                stats[phase]["generations"] += 1
                admitted = 0
                for expert in candidates(pair[policy]):
                    bundle = [(index, expert) for index in bundles[pair["layer"]]]
                    if all(key in cache.entries for key in bundle):
                        stats[phase]["skipped_resident"] += 1
                        continue
                    pinned = sum(cache.entries[key][0] for key in cache.pins)
                    needed = sum(cache.entries[key][0] if key in cache.entries else max_allocation
                                 for key in bundle if key not in cache.pins)
                    if pinned + needed + max(tensors[key[0]]["payload"] for key in bundle) > capacity - reserve:
                        stats[phase]["rejected_budget"] += 1
                        continue
                    for key in bundle:
                        index = key[0]
                        size = tensors[index]["payload"]
                        # Use the runtime LRU/reuse rules at the full cache cap.
                        # The admission above reserves mandatory working space;
                        # optional pins, unlike residency, cannot be evicted.
                        was_hit = cache.access(key, size, requirements[index], capacity, phase, True)
                        if not was_hit:
                            stats[phase]["prefetched_bytes"] += size
                            pending[key] = (size, phase, pair["target"] + 3)
                        cache.pins[key] = cache.pins.get(key, 0) + 1
                        optional.add(key)
                    admitted += 1
                    stats[phase]["admitted_experts"] += 1
                    if admitted == 4:
                        break
                if optional:
                    release_at = pair["target"]
                peak_optional = max(peak_optional, len(optional))
                peak_optional_bytes = max(peak_optional_bytes, sum(cache.entries[key][0] for key in optional))
        elif kind == "A":
            key = (t, e)
            phase = phase_at[projection]
            was_hit = cache.access(key, payload, required, cap, phase, True)
            if policy == "baseline":
                require(was_hit == bool(hit) and cache.entries[key][0] == allocation, "baseline differs from runtime")
            if key in pending:
                size, source_phase, _ = pending.pop(key)
                stats[source_phase]["useful_bytes" if was_hit else "unused_bytes"] += size
            cache.pins[key] = cache.pins.get(key, 0) + 1
        elif kind == "R":
            unpin((t, e))
        peak_all = max(peak_all, cache.used)
        require(cache.used <= capacity, "cache budget exceeded")
    require(not optional and not cache.pins, "unfinished lease generation")
    expire(float("inf"))
    result = cache.result()
    total_prefetch = sum(v["prefetched_bytes"] for v in stats.values())
    result.update(policy=policy, speculative=stats, demand_read_bytes=result["total"]["read_bytes"] - total_prefetch,
                  speculative_read_bytes=total_prefetch, mandatory_reserve=reserve,
                  max_optional_slices=peak_optional, max_optional_pinned_bytes=peak_optional_bytes,
                  capacity=capacity, peak_resident_bytes=peak_all)
    require(result["peak_cache_and_staging"] <= capacity and peak_optional <= 12, "bounded replay invariant failed")
    return result


def evaluate(capture, demand):
    header, records, keyed, routers, tensors, bundles = load_capture(capture)
    events, requirements, capacity = load_demand(demand, records, tensors, bundles)
    pairs, accuracy = predictions(records, keyed, routers, header["last_layer"])
    results = {policy: simulate(events, requirements, capacity, tensors, bundles, pairs, records, policy)
               for policy in ("baseline", "predicted", "oracle")}
    baseline = results["baseline"]["total"]["read_bytes"]
    for result in results.values():
        result["extra_total_read_bytes"] = result["total"]["read_bytes"] - baseline
        result["extra_total_read_fraction"] = result["extra_total_read_bytes"] / baseline
        result["demand_read_reduction_fraction"] = 1 - result["demand_read_bytes"] / baseline
    paths = (capture, Path(str(capture) + ".bin"), demand)
    return dict(architecture=header["architecture"], files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                records=len(records), pairs=len(pairs), row_accuracy=accuracy, replay=results,
                limitations="Offline float32 predictor, stable expert-ID ties; not backend-exact. Single sequence, cold fixed-cap LRU. Global Top4, whole three-slice bundles, one future generation; release optional pins at target entry. Perfect read completion, no predictor/scheduler cost or latency estimate. Oracle uses future IDs only as an optimistic control, not a realizable predictor. Diagnostic files and offline arrays are caller-owned, outside engine payload.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--demand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.capture, args.demand)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"architecture": result["architecture"], "row_accuracy": result["row_accuracy"],
                      "read_change": {k: v["extra_total_read_fraction"] for k, v in result["replay"].items()}}, indent=2))


if __name__ == "__main__":
    main()
