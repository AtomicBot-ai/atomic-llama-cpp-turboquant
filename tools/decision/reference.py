#!/usr/bin/env python3
"""Python reference of the router input rendering in llama-server --decision (stdlib only).

Training code imports this module, so the text a router model is trained on is the text the
engine builds at inference time. Every function mirrors a C++ function in tools/decision:

    py_dumps / loads_strict        decision-json.cpp   (json.dumps(v, ensure_ascii=False), strict parse)
    parse_body                     decision-request.cpp decision_parse_body
    card_validate / card_render    decision-router.cpp  (atomic.executor-card/1, card-v1)
    parse_router                   decision-router.cpp  decision_parse_router
    router_state / state_prefix    decision-router.cpp  (card first, then criterion, then task)
    router_question / options      decision-spec.cpp + laya-decide.cpp (spec.router.question, noul options)
    escape_text                    laya-decide.cpp      laya_escape_text (mask-to-space, escape-control)
    control_strings_from_gguf      laya.cpp             (the escape-control set of a vocabulary)

Errors raise DecisionError(reason, message, param) with the reason and param of the engine's
400 answer. tests/decision/gen_router_golden.py writes golden/router_cases.jsonl with this
module; test-decision-reference (ctest) checks the engine against it byte for byte, and
test_decision.py checks the server's /v1/decision/render route on the same kind of cases.

CLI (text on stdout is UTF-8; --text prints one field with no trailing newline):

    reference.py router REQUEST.json [--spec SPEC.json] [--gguf MODEL.gguf | --control LIST.json]
                                     [--special-tokens mask-to-space|escape-control]
                                     [--candidate ID --text state|model|card]
    reference.py card CARD.json                  validate + card-v1 text
    reference.py dumps [FILE]                    json.dumps(ensure_ascii=False) of a strict parse
    reference.py control-tokens MODEL.gguf       {"mask": ..., "control": [...]} of a laya GGUF

"router" prints {"question", "warnings", "items": [{"id", "card_text", "state", "state_splits",
"model_text"}]}: state is what /v1/decision/render reports, model_text the escaped state the
tokenizer sees. special_tokens comes from the spec (default mask-to-space). escape-control needs
the vocabulary: --gguf reads it from the model, --control from a control-tokens JSON file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys

CARD_SCHEMA = "atomic.executor-card/1"
MAX_CANDIDATES = 16        # DECISION_MAX_CANDIDATES
MAX_CHECKS = 32            # DECISION_MAX_CHECKS
MAX_CARD_FIELD_BYTES = 4096
MAX_BODY_BYTES = 1024 * 1024
JSON_MAX_DEPTH = 128
INT64_MIN, INT64_MAX, UINT64_MAX = -(2**63), 2**63 - 1, 2**64 - 1

DEFAULT_MASK = "<mask>"
DEFAULT_QUESTION = {
    "type": "noul",
    "instructions": "Will the executor meet the success criterion on this task?",
    "criteria": {"true": "meets the criterion", "false": "does not meet the criterion"},
}

CARD_KEYS = ("schema", "name", "kind", "description", "checks")
CHECK_KEYS = ("skill", "status", "passed", "total", "criterion", "source", "version")
ID_RE = re.compile(r"[A-Za-z0-9._:/@+-]{1,128}")


class DecisionError(Exception):
    """An engine 4xx: reason is the envelope reason, param the envelope param ("" = none)."""

    def __init__(self, reason: str, message: str, param: str = ""):
        super().__init__(message)
        self.reason = reason
        self.param = param

    def as_dict(self) -> dict:
        return {"reason": self.reason, "message": str(self), "param": self.param}


#
# py-json
#

def py_dumps(v) -> str:
    """json.dumps(v, ensure_ascii=False): the text decision_py_dumps writes."""
    return json.dumps(v, ensure_ascii=False)


def _reject_constant(name):
    raise DecisionError("UNSUPPORTED_NUMBER", "NaN and Infinity are not JSON numbers")


def _parse_int(text):
    v = int(text)
    if v < INT64_MIN or v > UINT64_MAX:
        raise DecisionError("UNSUPPORTED_NUMBER", "integer outside the int64/uint64 range: " + text)
    return v


def _parse_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise DecisionError("UNSUPPORTED_NUMBER", "number out of range: " + text)
    return v


def _check_tree(v, depth: int) -> None:
    # nlohmann rejects lone surrogate escapes; decision_json_parse rejects more than 128 levels
    if isinstance(v, (dict, list)):
        if depth >= JSON_MAX_DEPTH:
            raise DecisionError("MALFORMED_JSON", "nesting deeper than %d" % JSON_MAX_DEPTH)
        items = v.items() if isinstance(v, dict) else enumerate(v)
        for k, x in items:
            if isinstance(k, str):
                _check_str(k)
            _check_tree(x, depth + 1)
    elif isinstance(v, str):
        _check_str(v)


def _check_str(s: str) -> None:
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            raise DecisionError("MALFORMED_JSON", "invalid string: surrogate U+%04X" % ord(ch))


# lexical tokens in document order: string, open, close, NaN/Infinity, number
_SCAN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|([\[{])|([\]}])|(-?(?:NaN|Infinity))|(-?[0-9][0-9.eE+-]*)', re.S)
_ESC_RE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|.)', re.S)


def _lone_surrogate(body: str) -> bool:
    """A \\uD800-\\uDBFF escape not directly followed by a \\uDC00-\\uDFFF one, or a low one alone."""
    high_end = -1  # end of the previous escape when it was a high surrogate
    for m in _ESC_RE.finditer(body):
        u = int(m.group(1), 16) if m.group(1) else -1
        if high_end >= 0:
            if m.start() != high_end or not 0xDC00 <= u <= 0xDFFF:
                return True
            high_end = -1
        elif 0xD800 <= u <= 0xDBFF:
            high_end = m.end()
        elif 0xDC00 <= u <= 0xDFFF:
            return True
    return high_end >= 0


def _first_scan_error(text: str) -> DecisionError | None:
    """The engine's SAX parser stops at the first failure in document order: a container past
    JSON_MAX_DEPTH, a lone surrogate escape, a bad number. json.loads rejects bad numbers while it
    scans but the other two only in _check_tree afterwards, so when it reports a bad number this
    finds which came first. Only called on text that is valid JSON up to the bad number."""
    depth = 0
    for m in _SCAN_RE.finditer(text):
        body, opn, cls, const, num = m.groups()
        if opn:
            depth += 1
            if depth > JSON_MAX_DEPTH:
                return DecisionError("MALFORMED_JSON", "nesting deeper than %d" % JSON_MAX_DEPTH)
        elif cls:
            depth -= 1
        elif body is not None:
            if _lone_surrogate(body):
                return DecisionError("MALFORMED_JSON", "invalid string: lone surrogate escape")
        elif const:
            return DecisionError("UNSUPPORTED_NUMBER", "NaN and Infinity are not JSON numbers")
        else:
            try:
                _parse_int(num) if re.fullmatch(r"-?[0-9]+", num) else _parse_float(num)
            except DecisionError as e:
                return e
    return None


def loads_strict(text):
    """decision_json_parse: RFC 8259 JSON with the engine's number policy.

    Duplicate keys keep the first position and the last value (a Python dict does this).
    Raises DecisionError MALFORMED_JSON or UNSUPPORTED_NUMBER."""
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as e:
            raise DecisionError("MALFORMED_JSON", "invalid UTF-8: %s" % e) from None
    if text.startswith("\ufeff"):
        text = text[1:]  # nlohmann skips one UTF-8 BOM at the start
    try:
        v = json.loads(text, parse_int=_parse_int, parse_float=_parse_float, parse_constant=_reject_constant)
    except DecisionError as e:
        # a bad number; the engine reports a deeper container or a lone surrogate before it first
        first = _first_scan_error(text)
        raise (first if first is not None and first.reason == "MALFORMED_JSON" else e) from None
    except json.JSONDecodeError as e:
        # the engine names NaN / Infinity where the parse stops, also after a minus sign
        if text.startswith(("NaN", "Infinity", "-NaN", "-Infinity"), e.pos):
            first = _first_scan_error(text[:e.pos])
            if first is not None and first.reason == "MALFORMED_JSON":
                raise first from None
            raise DecisionError("UNSUPPORTED_NUMBER", "NaN and Infinity are not JSON numbers") from None
        raise DecisionError("MALFORMED_JSON", str(e)) from None
    except (ValueError, RecursionError) as e:
        raise DecisionError("MALFORMED_JSON", str(e)) from None
    _check_tree(v, 0)
    return v


def parse_body(data) -> dict:
    """decision_parse_body: body bytes/text -> JSON object."""
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if len(raw) > MAX_BODY_BYTES:
        raise DecisionError("BODY_TOO_LARGE", "request body is larger than %d bytes" % MAX_BODY_BYTES)
    body = loads_strict(raw)
    if not isinstance(body, dict):
        raise DecisionError("BODY_NOT_OBJECT", "request body must be a JSON object")
    return body


def py_strip(s: str) -> str:
    """Python str.strip(): decision_py_strip."""
    return s.strip()


def is_blank(s: str) -> bool:
    return s.strip() == ""


def _is_int(v) -> bool:
    # a JSON integer as nlohmann sees it: Python bools are not, ints outside int64/uint64 would be floats
    return isinstance(v, int) and not isinstance(v, bool) and INT64_MIN <= v <= UINT64_MAX


#
# atomic.executor-card/1 and card-v1
#

def canon_line(s: str) -> str:
    """Card string made single-line: C0 controls, DEL and spaces are separators; runs collapse, ends trim."""
    out = []
    space = False
    for ch in s:
        o = ord(ch)
        if o < 0x20 or o == 0x7F or ch == " ":
            space = len(out) > 0
            continue
        if space:
            out.append(" ")
            space = False
        out.append(ch)
    return "".join(out)


def _card_fail(param: str, msg: str):
    raise DecisionError("INVALID_CARD", msg, param)


def _require_string(obj: dict, key: str, param: str, required: bool) -> None:
    if key not in obj:
        if required:
            _card_fail(param + "." + key, key + " is required")
        return
    v = obj[key]
    if not isinstance(v, str) or (required and canon_line(v) == ""):
        _card_fail(param + "." + key, key + (" must be a non-empty string" if required else " must be a string"))
    if len(v.encode("utf-8")) > MAX_CARD_FIELD_BYTES:
        _card_fail(param + "." + key, "%s is longer than %d bytes" % (key, MAX_CARD_FIELD_BYTES))


def card_validate(card, param: str = "card", warnings: list | None = None) -> list:
    """decision_card_validate: raises DecisionError INVALID_CARD; returns the warnings list
    (unknown fields, in the engine's order)."""
    if warnings is None:
        warnings = []
    if not isinstance(card, dict):
        _card_fail(param, "card must be an object")
    if card.get("schema") != CARD_SCHEMA or not isinstance(card.get("schema"), str):
        _card_fail(param + ".schema", 'card schema must be "%s"' % CARD_SCHEMA)
    _require_string(card, "name", param, True)
    _require_string(card, "kind", param, True)
    _require_string(card, "description", param, False)
    for k in card:
        if k not in CARD_KEYS:
            warnings.append(param + "." + k + ": unknown field ignored")
    if "checks" not in card:
        return warnings
    checks = card["checks"]
    if not isinstance(checks, list):
        _card_fail(param + ".checks", "checks must be an array")
    if len(checks) > MAX_CHECKS:
        _card_fail(param + ".checks", "at most %d checks per card" % MAX_CHECKS)
    for i, c in enumerate(checks):
        cp = "%s.checks[%d]" % (param, i)
        if not isinstance(c, dict):
            _card_fail(cp, "check must be an object")
        _require_string(c, "skill", cp, True)
        _require_string(c, "criterion", cp, False)
        _require_string(c, "source", cp, False)
        _require_string(c, "version", cp, False)
        status = c.get("status")
        if isinstance(status, str) and status == "measured":
            if "passed" not in c or "total" not in c or not _is_int(c["passed"]) or not _is_int(c["total"]):
                _card_fail(cp, "a measured check needs integer passed and total")
            if c["passed"] > INT64_MAX or c["total"] > INT64_MAX:
                _card_fail(cp, "passed and total must fit in int64")
            if c["total"] < 1 or c["passed"] < 0 or c["passed"] > c["total"]:
                _card_fail(cp, "a measured check needs 0 <= passed <= total and total >= 1")
        elif isinstance(status, str) and status == "missing":
            if "passed" in c or "total" in c:
                _card_fail(cp, "a missing check has no passed/total")
        else:
            _card_fail(cp + ".status", 'status must be "measured" or "missing"')
        for k in c:
            if k not in CHECK_KEYS:
                warnings.append(cp + "." + k + ": unknown field ignored")
    return warnings


def _opt(obj: dict, key: str) -> str:
    return canon_line(obj[key]) if key in obj else ""


def card_render(card: dict) -> str:
    """card-v1 text of a valid card (decision_card_render)."""
    lines = ["executor: " + _opt(card, "name"), "kind: " + _opt(card, "kind")]
    desc = _opt(card, "description")
    if desc:
        lines.append("description: " + desc)
    checks = card.get("checks", [])
    if not checks:
        return "\n".join(lines) + "\nchecks: none"
    out = "\n".join(lines) + "\nchecks:"
    for c in checks:
        out += "\n- " + _opt(c, "skill") + ": "
        if c["status"] == "measured":
            out += "passed %d of %d" % (c["passed"], c["total"])
            crit = _opt(c, "criterion")
            if crit:
                out += "; criterion: " + crit
        else:
            out += "not measured"
        src, ver = _opt(c, "source"), _opt(c, "version")
        if src or ver:
            out += "; source: " + src
            if ver:
                out += (" " if src else "") + ver
    return out


#
# /v1/router/score request
#

def candidate_id_valid(cid) -> bool:
    return isinstance(cid, str) and ID_RE.fullmatch(cid) is not None


def _required_text(body: dict, key: str) -> str:
    v = body.get(key)
    if not isinstance(v, str) or is_blank(v):
        raise DecisionError("INVALID_REQUEST", key + " must be a non-empty string", key)
    return v


def parse_truncation(body: dict) -> bool:
    """decision_parse_truncation: True for "error"."""
    t = body.get("truncation")
    if t is None or (isinstance(t, str) and t == "allow"):
        return False
    if isinstance(t, str) and t == "error":
        return True
    raise DecisionError("INVALID_REQUEST", 'truncation must be "allow" or "error"', "truncation")


def parse_router(body, max_candidates: int = MAX_CANDIDATES) -> dict:
    """decision_parse_router: {"model", "task", "criterion", "truncation_error", "warnings",
    "candidates": [{"id", "card", "card_text"}]}. Raises DecisionError."""
    if not isinstance(body, dict):
        raise DecisionError("BODY_NOT_OBJECT", "request body must be a JSON object")
    req = {"model": body["model"] if isinstance(body.get("model"), str) else ""}
    req["task"] = _required_text(body, "task")
    req["criterion"] = _required_text(body, "criterion")
    req["truncation_error"] = parse_truncation(body)
    cands = body.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise DecisionError("INVALID_REQUEST", "candidates must be a non-empty array", "candidates")
    if len(cands) > max_candidates:
        raise DecisionError("TOO_MANY_CANDIDATES", "at most %d candidates per request, got %d" % (max_candidates, len(cands)),
                            "candidates")
    warnings: list = []
    seen = set()
    out = []
    for i, c in enumerate(cands):
        param = "candidates[%d]" % i
        if not isinstance(c, dict):
            raise DecisionError("INVALID_CARD", "candidate must be an object", param)
        if not candidate_id_valid(c.get("id")):
            raise DecisionError("INVALID_CANDIDATE_ID", "candidate id must match [A-Za-z0-9._:/@+-]{1,128}", param + ".id")
        cid = c["id"]
        if cid in seen:
            raise DecisionError("DUPLICATE_CANDIDATE_ID", "duplicate candidate id '%s'" % cid, param + ".id")
        seen.add(cid)
        if "card" not in c:
            raise DecisionError("INVALID_CARD", "candidate has no card", param + ".card")
        card_validate(c["card"], param + ".card", warnings)
        for k in c:
            if k not in ("id", "card"):
                warnings.append(param + "." + k + ": unknown field ignored")
        out.append({"id": cid, "card": c["card"], "card_text": card_render(c["card"])})
    req["candidates"] = out
    req["warnings"] = warnings
    return req


def router_state_prefix(card_text: str, criterion: str) -> str:
    """The part of a router state that must never be cut."""
    return card_text + "\n\nsuccess criterion: " + criterion + "\n\ntask:"


def router_state(card_text: str, criterion: str, task: str) -> str:
    """Candidate state (laya-router-v1): card first, so a cut only hits the task tail."""
    return router_state_prefix(card_text, criterion) + " " + task


def state_splits(card_text: str, criterion: str) -> list:
    """Byte offsets where the engine may tokenize the state in pieces (after the card, after "task:")."""
    return [len(card_text.encode("utf-8")), len(router_state_prefix(card_text, criterion).encode("utf-8"))]


#
# spec.router.question and its laya options
#

def _ascii_lower(s: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def router_question(spec: dict | None) -> dict:
    """The noul question of spec.router (the built-in one when absent). A question without
    criteria gets the built-in criteria. The spec is assumed valid (gguf_decision_spec.py verify)."""
    router = (spec or {}).get("router")
    q = router.get("question") if isinstance(router, dict) else None
    if q is None:
        return json.loads(json.dumps(DEFAULT_QUESTION))
    q = json.loads(json.dumps(q))
    if q.get("criteria") is None:
        q["criteria"] = dict(DEFAULT_QUESTION["criteria"])
    return q


def render_criterion(v) -> str:
    return v if isinstance(v, str) else py_dumps(v)


def noul_options(q: dict) -> list:
    """Laya option texts of a noul question (laya_question_parse): "<label>: <description>"."""
    lowered: dict = {}
    for k, v in (q.get("criteria") or {}).items():
        lowered[_ascii_lower(k)] = v  # a later duplicate wins, at the first position
    labels = q.get("labels")
    fl, tl = ("false", "true") if labels is None else (py_strip(labels["false"]), py_strip(labels["true"]))

    def desc(v, default):
        return default if v is None or v == "" else render_criterion(v)

    return [fl + ": " + desc(lowered.get("false"), "no, the statement does not hold"),
            tl + ": " + desc(lowered.get("true"), "yes, the statement holds")]


#
# special tokens
#

class Escape:
    """laya_escape: the mask literal and the escape-control strings of one vocabulary."""

    def __init__(self, mask: str = DEFAULT_MASK, control=None):
        self.mask = mask
        self.control = sorted(set(s for s in list(control or []) + [mask] if s))
        self._by_first: dict = {}
        for s in self.control:
            if s:
                self._by_first.setdefault(s[0], []).append(s)
        for v in self._by_first.values():
            v.sort(key=len, reverse=True)

    def text(self, text: str, escape_control: bool) -> str:
        """laya_escape_text: mask-to-space, or with escape_control every control string -> one space
        (longest match, left to right; the strings cannot overlap, so this equals str.replace in any order)."""
        if not escape_control or not self._by_first:
            return text.replace(self.mask, " ") if self.mask else text
        out = []
        i, n = 0, len(text)
        while i < n:
            for s in self._by_first.get(text[i], ()):
                if text.startswith(s, i):
                    out.append(" ")
                    i += len(s)
                    break
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    def as_dict(self) -> dict:
        return {"mask": self.mask, "control": [s for s in self.control]}


def escape_text(text: str, escape_control: bool, esc: Escape | None = None) -> str:
    return (esc or Escape()).text(text, escape_control)


_UNUSED_RE = re.compile(r"(<unused[0-9]+>|\[unused[0-9]+\])")


def _is_unused_token(s: str) -> bool:
    return _UNUSED_RE.fullmatch(s) is not None


# GGUF value types
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_kv(path: str, keys: set) -> dict:
    """The metadata values of `keys` in a little-endian GGUF (v2/v3); tensors are not read."""
    out = {}
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated GGUF: " + path)
            return struct.unpack(fmt, b)[0]

        def rd_str():
            return f.read(rd("<Q")).decode("utf-8", errors="surrogateescape")

        def rd_val(t, keep):
            if t == 8:
                s = rd_str()
                return s if keep else None
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                if not keep and et in _GGUF_SCALARS:
                    f.seek(n * struct.calcsize(_GGUF_SCALARS[et]), 1)
                    return None
                return [rd_val(et, keep) for _ in range(n)]
            if t in _GGUF_SCALARS:
                return rd(_GGUF_SCALARS[t])
            raise ValueError("unknown GGUF value type %d in %s" % (t, path))

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: " + path)
        version = rd("<I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version %d: %s" % (version, path))
        rd("<Q")  # n_tensors
        n_kv = rd("<Q")
        for _ in range(n_kv):
            key = rd_str()
            t = rd("<I")
            v = rd_val(t, key in keys)
            if key in keys:
                out[key] = v
    return out


def control_strings_from_gguf(path: str) -> Escape:
    """The escape set laya_escape_init builds for a laya GGUF: every non-empty CONTROL (3) token,
    every USER_DEFINED (4) <unusedN> / [unusedN] token and the mask literal. metaspace-bpe
    vocabularies store USER_DEFINED spaces as ' ' and the loader restores U+2581, as here."""
    kv = read_gguf_kv(path, {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.mask_token_id",
                             "decision.laya.tokenizer"})
    tokens = kv.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError("missing tokenizer.ggml.tokens in " + path)
    types = kv.get("tokenizer.ggml.token_type") or []
    bytelevel = kv.get("decision.laya.tokenizer", "metaspace-bpe") == "bytelevel-bpe"
    mask_id = kv.get("tokenizer.ggml.mask_token_id", 4)
    control = []
    mask = None
    for i in range(min(len(tokens), len(types))):
        t, s = types[i], tokens[i]
        if t not in (3, 4) or s == "":
            continue
        if t == 4 and not bytelevel:
            s = s.replace(" ", "\u2581")
        if i == mask_id:
            mask = s
        if t == 3 or _is_unused_token(s):
            control.append(s)
    return Escape(mask if mask is not None else DEFAULT_MASK, control)


def load_escape(gguf: str | None = None, control_file: str | None = None) -> Escape:
    if gguf:
        return control_strings_from_gguf(gguf)
    if control_file:
        with open(control_file, encoding="utf-8") as f:
            d = json.load(f)
        return Escape(d.get("mask", DEFAULT_MASK), d.get("control", []))
    return Escape()


#
# one call for the training pipeline
#

def render_router(body, spec: dict | None = None, esc: Escape | None = None, max_candidates: int = MAX_CANDIDATES,
                  special_tokens: str | None = None) -> dict:
    """Everything the engine derives from a router request, without tokens:
    {"question": {type, instructions, criteria[, labels]}, "options": [false, true],
     "warnings": [...], "truncation_error": bool,
     "items": [{"id", "card_text", "state", "state_splits", "model_text"}]}.
    model_text is the state after special-token handling: the text the laya tokenizer reads.
    Raises DecisionError like the engine (400 reasons; token limits need the tokenizer)."""
    if isinstance(body, (str, bytes, bytearray)):
        body = parse_body(body)
    req = parse_router(body, max_candidates)
    st = special_tokens or (spec or {}).get("special_tokens") or "mask-to-space"
    escape_control = st == "escape-control"
    if escape_control and (esc is None or not esc.control or esc.control == [esc.mask]):
        raise ValueError("special_tokens escape-control needs the vocabulary's control strings (--gguf or --control)")
    esc = esc or Escape()
    q = router_question(spec)
    items = []
    for c in req["candidates"]:
        state = router_state(c["card_text"], req["criterion"], req["task"])
        items.append({
            "id": c["id"],
            "card_text": c["card_text"],
            "state": state,
            "state_splits": state_splits(c["card_text"], req["criterion"]),
            "model_text": esc.text(state, escape_control),
        })
    return {"question": q, "options": noul_options(q), "special_tokens": st, "warnings": req["warnings"],
            "truncation_error": req["truncation_error"], "items": items}


def laya_router_examples(body, spec: dict | None = None, esc: Escape | None = None) -> list:
    """One Laya noul example per candidate: {"id", "type", "instructions", "criteria", "state"}.
    state is the model text (escaped); the question is the spec router question."""
    r = render_router(body, spec, esc)
    q = r["question"]
    return [{"id": it["id"], "type": "noul", "instructions": q["instructions"], "criteria": q["criteria"],
             **({"labels": q["labels"]} if q.get("labels") is not None else {}), "state": it["model_text"]}
            for it in r["items"]]


#
# CLI
#

def _read_json(path: str):
    if path == "-":
        return loads_strict(sys.stdin.buffer.read())
    with open(path, "rb") as f:
        return loads_strict(f.read())


def _out(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python reference of the decision router rendering.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("router", help="render a /v1/router/score request")
    r.add_argument("request")
    r.add_argument("--spec", help="decision spec JSON (router question, special_tokens)")
    r.add_argument("--gguf", help="laya GGUF: control strings for escape-control and the mask literal")
    r.add_argument("--control", help="control-tokens JSON (output of the control-tokens command)")
    r.add_argument("--special-tokens", choices=("mask-to-space", "escape-control"))
    r.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    r.add_argument("--candidate", help="with --text: the candidate id")
    r.add_argument("--text", choices=("state", "model", "card"), help="print one text field raw")
    c = sub.add_parser("card", help="validate a card and print its card-v1 text")
    c.add_argument("card")
    d = sub.add_parser("dumps", help="py-json of a JSON document")
    d.add_argument("file", nargs="?", default="-")
    t = sub.add_parser("control-tokens", help="escape set of a laya GGUF")
    t.add_argument("gguf")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "router":
            spec = _read_json(a.spec) if a.spec else None
            esc = load_escape(a.gguf, a.control)
            with open(a.request, "rb") as f:
                res = render_router(f.read(), spec, esc, a.max_candidates, a.special_tokens)
            if a.text:
                items = [it for it in res["items"] if a.candidate is None or it["id"] == a.candidate]
                if not items:
                    print("no candidate %r" % a.candidate, file=sys.stderr, flush=True)
                    return 2
                _out({"state": items[0]["state"], "model": items[0]["model_text"], "card": items[0]["card_text"]}[a.text])
            else:
                _out(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        elif a.cmd == "card":
            card = _read_json(a.card)
            warnings = card_validate(card, "card")
            for w in warnings:
                print("warning: " + w, file=sys.stderr, flush=True)
            _out(card_render(card) + "\n")
        elif a.cmd == "dumps":
            _out(py_dumps(_read_json(a.file)) + "\n")
        elif a.cmd == "control-tokens":
            _out(json.dumps(control_strings_from_gguf(a.gguf).as_dict(), ensure_ascii=False) + "\n")
    except DecisionError as e:
        print(json.dumps({"error": e.as_dict()}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
