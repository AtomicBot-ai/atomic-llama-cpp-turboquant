#!/usr/bin/env python3
"""Get, set or verify the decision.spec contract of a GGUF without re-converting it.

    gguf_decision_spec.py get    model.gguf
    gguf_decision_spec.py set    model.gguf spec.json [-o out.gguf] [--force]
    gguf_decision_spec.py verify model.gguf [spec.json]

decision.spec is a UTF-8 JSON string (see DECISION.md). `set` stores the file bytes as-is,
so its sha256 is the one llama-server --decision reports, and writes the mirror keys
decision.spec_version / layout / model_id / model_version that the server checks against it.
`get` writes the stored bytes and one "\n", without newline translation on any OS.

`set` and `verify` apply the rules of the C++ loader (decision_spec_from_json in
tools/decision/decision-spec.cpp), so a spec that passes here also loads in the server.
Both run the cases of tests/decision/spec_cases.json (test-decision-calib and
tools/server/tests/unit/test_decision.py); change the two together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# gguf is imported by the commands only: validate_spec is stdlib-only for the tests

logger = logging.getLogger("gguf-decision-spec")

KEY_SPEC = "decision.spec"
KEY_VERSION = "decision.spec_version"
KEY_LAYOUT = "decision.layout"
KEY_MODEL_ID = "decision.model_id"
KEY_MODEL_VERSION = "decision.model_version"
MIRRORS = {KEY_LAYOUT: "layout", KEY_MODEL_ID: "model_id", KEY_MODEL_VERSION: "model_version"}
LAYOUTS = ("laya", "semif-letters")


INT32_MAX = 2**31 - 1
QTYPES = ("noul", "choice", "score")
KERNELS = ("auto", "default", "repack", "blas", "repack+blas")  # plan.kernels, as engine-laya.cpp reads it
LAYOUT_MAX_OPTIONS = {"laya": 20, "semif-letters": 16}
CONTRACTS = {  # layout -> (input contracts, special-token contracts)
    "laya": (("laya-v1", "laya-router-v1"), ("mask-to-space", "escape-control")),
    "semif-letters": (("semif-v1",), ("escape",)),
}
CARD_SCHEMA = "atomic.executor-card/1"


def _reject_constant(name: str):
    raise ValueError(f"{name} is not a JSON number")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _positive(v) -> bool:
    return _is_number(v) and v > 0


def _opt(obj: dict, key: str, check, what: str, default):
    """Value of an optional key (null = absent), like get_opt in decision-spec.cpp."""
    v = obj.get(key)
    if v is None:
        return default
    if not check(v):
        raise ValueError(f"{key} must be {what}")
    return v


def _opt_str(obj: dict, key: str, default):
    return _opt(obj, key, lambda v: isinstance(v, str), "a string", default)


def _opt_count(obj: dict, key: str, default):
    return _opt(obj, key, lambda v: _is_int(v) and 0 <= v <= INT32_MAX, "a non-negative integer", default)


def _is_uint_str(s: str) -> bool:
    return s != "" and all("0" <= c <= "9" for c in s)


def _valid_bucket(b: str) -> bool:
    if b == "*" or _is_uint_str(b):
        return True
    if len(b) > 1 and b.endswith("+"):
        return _is_uint_str(b[:-1])
    lo, dash, hi = b.partition("-")
    return dash != "" and _is_uint_str(lo) and _is_uint_str(hi)


def _bucket_rank(b: str, k: int) -> int:
    """0 exact "N", 1 range "A-B", 2 open "N+", 3 "*", -1 no match (decision-calib.cpp)."""
    if b == "*":
        return 3
    if b.endswith("+"):
        return 2 if k >= int(b[:-1]) else -1
    if "-" in b:
        lo, _, hi = b.partition("-")
        return 1 if int(lo) <= k <= int(hi) else -1
    return 0 if int(b) == k else -1


def _check_temperature(t) -> None:
    if not isinstance(t, dict):
        raise ValueError("calibration.temperature must be an object")
    for k, v in t.items():
        if k not in ("all",) + QTYPES:
            raise ValueError(f"calibration.temperature: unknown question type {k!r}")
        if isinstance(v, dict):
            for b, x in v.items():
                if not _valid_bucket(b):
                    raise ValueError(f"calibration.temperature.{k}: bad option-count bucket {b!r}")
                if not _positive(x):
                    raise ValueError(f"calibration.temperature.{k}.{b} must be a positive number")
        elif not _positive(v):
            raise ValueError(f"calibration.temperature.{k} must be a positive number or a bucket object")


def _covered(temperature: dict, qtype: str, k: int) -> bool:
    entry = temperature.get(qtype, temperature.get("all"))
    if entry is None:
        return False
    if not isinstance(entry, dict):
        return True
    return any(_bucket_rank(b, k) >= 0 for b in entry)


def _check_calibration(c, max_options: int) -> None:
    if not isinstance(c, dict):
        raise ValueError("calibration must be an object")
    method = _opt_str(c, "method", "none")
    required = _opt(c, "required", lambda v: isinstance(v, bool), "a boolean", False)
    _opt_str(c, "version", "")
    if method not in ("temperature", "none"):
        raise ValueError('calibration.method must be "temperature" or "none"')
    if "clamp" in c:
        cl = c["clamp"]
        if not isinstance(cl, list) or len(cl) != 2 or not _positive(cl[0]) or not _positive(cl[1]) or cl[0] > cl[1]:
            raise ValueError("calibration.clamp must be [min, max] with 0 < min <= max")
    temperature = c.get("temperature")
    if temperature is not None:
        _check_temperature(temperature)
    calibrated = method == "temperature" and bool(temperature)
    if required and not calibrated:
        raise ValueError("calibration is required but has no temperatures")
    if required:
        # a required calibration never falls back to T = 1
        for qtype in QTYPES:
            for k in range(2, (2 if qtype == "noul" else max_options) + 1):
                if not _covered(temperature, qtype, k):
                    raise ValueError(f"calibration is required but has no temperature for {qtype} with {k} options")


def _check_router(r) -> None:
    if not isinstance(r, dict):
        raise ValueError("router must be an object or null")
    try:
        schema = _opt_str(r, "card_schema", CARD_SCHEMA)
        renderer = _opt_str(r, "card_renderer", "card-v1")
        _opt_count(r, "max_card_tokens", -1)
    except ValueError as e:
        raise ValueError(f"router.{e}") from None
    if schema != CARD_SCHEMA:
        raise ValueError(f"router.card_schema must be {CARD_SCHEMA!r}")
    if renderer != "card-v1":
        raise ValueError('router.card_renderer must be "card-v1"')
    q = r.get("question")
    if q is not None:
        ins = q.get("instructions") if isinstance(q, dict) else None
        if not isinstance(q, dict) or q.get("type") != "noul" or not isinstance(ins, str) or ins.strip() == "":
            raise ValueError("router.question must be a noul question with instructions")
        if q.get("criteria") is not None and not isinstance(q["criteria"], dict):
            raise ValueError("router.question.criteria must be an object with true/false descriptions")
        # the laya request rules for a noul question (laya_question_parse), checked at load
        if isinstance(q.get("criteria"), dict) and not {str(k).lower() for k in q["criteria"]} <= {"true", "false"}:
            raise ValueError("router.question.criteria must be keyed only true/false")
        labels = q.get("labels")
        if labels is not None:
            ok = isinstance(labels, dict) and set(labels) == {"false", "true"} and all(isinstance(v, str) for v in labels.values())
            if not ok or not labels["false"].strip() or not labels["true"].strip() or labels["false"].strip() == labels["true"].strip():
                raise ValueError("router.question.labels must map exactly 'false' and 'true' to distinct non-empty strings")
    c = r.get("calibration")
    if c is not None:
        if not isinstance(c, dict) or c.get("method") != "platt":
            raise ValueError('router.calibration.method must be "platt"')
        if not _is_number(c.get("a")) or not _is_number(c.get("b")):
            raise ValueError("router.calibration needs finite numbers a and b")


def validate_spec(spec) -> None:
    """Raise ValueError on anything decision_spec_from_json rejects (full specs only)."""
    if not isinstance(spec, dict):
        raise ValueError("decision spec must be a JSON object")
    if not _is_int(spec.get("spec_version")) or spec["spec_version"] != 1:
        raise ValueError("spec_version must be 1")
    layout = spec.get("layout")
    if not isinstance(layout, str) or layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    inputs, specials = CONTRACTS[layout]
    _opt_str(spec, "model_id", "")
    _opt_str(spec, "model_version", "")
    input_contract = _opt_str(spec, "input_contract", inputs[0])
    special_tokens = _opt_str(spec, "special_tokens", specials[0])
    confidence = _opt_str(spec, "confidence", "laya" if layout == "laya" else "max_p")
    if confidence not in ("laya", "typesafe", "max_p"):
        raise ValueError('confidence must be "laya", "typesafe" or "max_p"')
    if input_contract not in inputs:
        raise ValueError(f"input_contract must be one of {inputs} for layout {layout}")
    if special_tokens not in specials:
        raise ValueError(f"special_tokens must be one of {specials} for layout {layout}")
    if input_contract == "laya-router-v1" and special_tokens != "escape-control":
        raise ValueError('input_contract "laya-router-v1" needs special_tokens "escape-control"')
    max_options = 0
    limits = spec.get("limits")
    if limits is not None:
        if not isinstance(limits, dict):
            raise ValueError("limits must be an object")
        try:
            max_options = _opt_count(limits, "max_options", 0)
            _opt_count(limits, "max_candidates", 16)
            _opt_count(limits, "max_prompt_tokens", 0)
        except ValueError as e:
            raise ValueError(f"limits.{e}") from None
    layout_max = LAYOUT_MAX_OPTIONS[layout]
    max_options = min(max_options, layout_max) if max_options > 0 else layout_max
    if spec.get("calibration") is not None:
        _check_calibration(spec["calibration"], max_options)
    plan = spec.get("plan")
    if plan is not None and (not isinstance(plan, dict) or ("name" in plan and not isinstance(plan["name"], str))):
        raise ValueError("plan must be an object with a string name")
    if plan is not None and "kernels" in plan and plan["kernels"] not in KERNELS:
        raise ValueError("plan.kernels must be auto, default, repack, blas or repack+blas")
    if spec.get("router") is not None:
        _check_router(spec["router"])


def parse_spec(text: str) -> dict:
    """Parse and validate a spec; NaN/Infinity are rejected like the C++ parser."""
    spec = json.loads(text, parse_constant=_reject_constant)
    validate_spec(spec)
    return spec


def mirror_values(spec: dict) -> dict[str, str]:
    return {key: spec[field] for key, field in MIRRORS.items() if isinstance(spec.get(field), str)}


def cmd_get(args) -> int:
    import gguf
    from gguf.scripts.gguf_new_metadata import get_field_data
    reader = gguf.GGUFReader(args.model, "r")
    text = get_field_data(reader, KEY_SPEC)
    if text is None:
        print(f"{args.model}: no {KEY_SPEC}", file=sys.stderr)
        return 1
    # the stored bytes plus one newline, on every OS: print() would turn each \n into \r\n on
    # Windows, and "get > spec.json" would no longer hash to the spec_sha256 the server reports
    sys.stdout.flush()
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.flush()
    return 0


def cmd_set(args) -> int:
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata, get_field_data
    text = Path(args.spec).read_bytes().decode("utf-8")
    spec = parse_spec(text)

    reader = gguf.GGUFReader(args.model, "r")
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    if (arch == "laya") != (spec["layout"] == "laya"):
        raise ValueError(f"layout {spec['layout']!r} does not fit a {arch!r} GGUF")

    out = Path(args.output) if args.output else Path(args.model)
    in_place = out.resolve() == Path(args.model).resolve()
    if out.exists() and not in_place and not args.force:
        raise FileExistsError(f"{out} exists; pass --force to overwrite")

    new_metadata = {
        KEY_SPEC: MetadataDetails(gguf.GGUFValueType.STRING, text),
        KEY_VERSION: MetadataDetails(gguf.GGUFValueType.UINT32, 1),
    }
    mirrors = mirror_values(spec)
    for key, value in mirrors.items():
        new_metadata[key] = MetadataDetails(gguf.GGUFValueType.STRING, value)
    # stale mirror keys from an older spec must not outlive it
    remove = [key for key in MIRRORS if key not in mirrors and reader.get_field(key) is not None]

    tmp = out.with_name(out.name + ".tmp")
    writer = gguf.GGUFWriter(tmp, arch=arch, endianess=reader.endianess)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    copy_with_new_metadata(reader, writer, new_metadata, remove)
    del reader
    os.replace(tmp, out)
    print(f"{out}: {KEY_SPEC} sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


def cmd_verify(args) -> int:
    import gguf
    from gguf.scripts.gguf_new_metadata import get_field_data
    reader = gguf.GGUFReader(args.model, "r")
    text = get_field_data(reader, KEY_SPEC)
    if text is None:
        print(f"{args.model}: no {KEY_SPEC}", file=sys.stderr)
        return 1
    spec = parse_spec(text)
    problems = []
    version = get_field_data(reader, KEY_VERSION)
    if version is not None and version != spec["spec_version"]:
        problems.append(f"{KEY_VERSION}={version} but the spec says {spec['spec_version']}")
    for key, field in MIRRORS.items():
        value = get_field_data(reader, key)
        if value is not None and value != spec.get(field):
            problems.append(f"{key}={value!r} but the spec says {spec.get(field)!r}")
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    if (arch == "laya") != (spec["layout"] == "laya"):
        problems.append(f"layout {spec['layout']!r} does not fit a {arch!r} GGUF")
    if args.spec is not None and Path(args.spec).read_bytes().decode("utf-8") != text:
        problems.append(f"{KEY_SPEC} differs from {args.spec}")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    print(f"{args.model}: {KEY_SPEC} sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()} {'FAIL' if problems else 'OK'}")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Get, set or verify the decision.spec metadata of a GGUF")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("get", help="print decision.spec")
    p.add_argument("model", type=Path)
    p = sub.add_parser("set", help="store a spec file as decision.spec (+ mirror keys)")
    p.add_argument("model", type=Path)
    p.add_argument("spec", type=Path)
    p.add_argument("-o", "--output", type=Path, help="write a copy instead of replacing the input")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    p = sub.add_parser("verify", help="check the mirror keys (and optionally the bytes of a spec file)")
    p.add_argument("model", type=Path)
    p.add_argument("spec", type=Path, nargs="?")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        rc = {"get": cmd_get, "set": cmd_set, "verify": cmd_verify}[args.command](args)
    except (ValueError, FileExistsError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
