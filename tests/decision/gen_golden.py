#!/usr/bin/env python3
# regenerate the py-json golden files used by tests/test-decision-json.cpp
# usage: python3 tests/decision/gen_golden.py   (stdlib only, deterministic)
#
# golden/py_float.txt  one repr(float) per line: every power of two 2^-1074..2^1023 (+ negatives of
#                      a few), then random doubles (bit patterns, short decimals, boundary values)
# golden/py_json.jsonl one json.dumps(obj, ensure_ascii=False) per line: random nested objects
# golden/py_hashes.txt SHA-256 of repr() of 30000 doubles and of json.dumps() of 1000 nested objects,
#                      drawn from splitmix64; test-decision-json draws the same values and compares
import hashlib
import json
import math
import random
import struct
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden"
rng = random.Random(20260929)


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def floats():
    out = [math.ldexp(1.0, e) for e in range(-1074, 1024)]
    out += [-math.ldexp(1.0, e) for e in range(-1074, 1024, 97)]
    out += [0.0, -0.0, 1e-4, 9.999999999999999e-05, 1e16, 9999999999999998.0, 1e15, 0.1, 0.2, 0.3,
            1 / 3, 763.4, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 123456789012345680.0]
    while len(out) < 2098 + 23 + 16 + 600:
        b = rng.getrandbits(64)
        d = from_bits(b)
        if math.isfinite(d):
            out.append(d)
    for _ in range(400):
        out.append(round(rng.uniform(-1000, 1000), rng.randint(0, 6)))
    for _ in range(200):
        out.append(rng.uniform(1, 10) * 10.0 ** rng.randint(-8, 20))
    return out


def rand_str():
    alphabet = "abcXYZ 019_-\"\\/\n\t\r\b\f\x00\x01\x1f\x7f" + "\u00e9\u4e2d\u6587\u0438\u2028\u2029\U0001F600\u0301"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))


def rand_scalar():
    k = rng.randint(0, 9)
    if k == 0:
        return None
    if k == 1:
        return rng.random() < 0.5
    if k == 2:
        return rng.choice([0, -1, 2**53, 2**63 - 1, -(2**63), 2**63, 2**64 - 1, rng.randint(-10**6, 10**6)])
    if k in (3, 4):
        return rng.choice(floats_pool)
    return rand_str()


def rand_value(depth):
    if depth >= 4 or rng.random() < 0.4:
        return rand_scalar()
    if rng.random() < 0.5:
        return [rand_value(depth + 1) for _ in range(rng.randint(0, 4))]
    return {rand_str(): rand_value(depth + 1) for _ in range(rng.randint(0, 4))}


# ---- splitmix64 streams (same generator in tests/test-decision-json.cpp) ----

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n


SM_ALPHABET = [ord(c) for c in "abcXYZ 019_-\"\\/\n\t\r\b\f"] + \
    [0x00, 0x01, 0x1f, 0x7f, 0xe9, 0x4e2d, 0x6587, 0x438, 0x2028, 0x2029, 0x1F600, 0x301, 0xfeff, 0xffff, 0x10ffff]
SM_INTS = [0, -1, 2**53, 2**63 - 1, -(2**63), 2**63, 2**64 - 1]


def sm_double(r):
    # half raw bit patterns, half short decimals "[-]<m>e<e>" (correctly rounded by float() and strtod)
    while True:
        if r.below(2) == 0:
            d = from_bits(r.next())
        else:
            digits = 1 + r.below(17)
            m = r.next() % (10 ** digits)
            e = r.below(660) - 340
            neg = r.below(2) == 1
            d = float(("-" if neg else "") + str(m) + "e" + str(e))
        if math.isfinite(d):
            return d


def sm_str(r):
    n = r.below(13)
    return "".join(chr(SM_ALPHABET[r.below(len(SM_ALPHABET))]) for _ in range(n))


def sm_scalar(r):
    k = r.below(10)
    if k == 0:
        return None
    if k == 1:
        return r.below(2) == 1
    if k == 2:
        j = r.below(8)
        return SM_INTS[j] if j < 7 else r.below(2000001) - 1000000
    if k in (3, 4):
        return sm_double(r)
    return sm_str(r)


def sm_value(r, depth):
    if depth >= 4 or r.below(10) < 4:
        return sm_scalar(r)
    if r.below(2) == 0:
        n = r.below(5)
        return [sm_value(r, depth + 1) for _ in range(n)]
    n = r.below(5)
    d = {}
    for _ in range(n):
        k = sm_str(r)
        d[k] = sm_value(r, depth + 1)
    return d


def sm_object(r):
    n = 1 + r.below(5)
    d = {}
    for _ in range(n):
        k = sm_str(r)
        d[k] = sm_value(r, 1)
    return d


def sm_hashes():
    r = SplitMix64(1)
    floats_sha = hashlib.sha256("".join(repr(sm_double(r)) + "\n" for _ in range(30000)).encode("ascii")).hexdigest()
    r = SplitMix64(2)
    objects_sha = hashlib.sha256("".join(json.dumps(sm_object(r), ensure_ascii=False) + "\n" for _ in range(1000)).encode("utf-8")).hexdigest()
    return [("floats", 1, 30000, floats_sha), ("objects", 2, 1000, objects_sha)]


floats_pool = floats()

with open(GOLDEN / "py_float.txt", "w", encoding="ascii", newline="\n") as f:
    for d in floats_pool:
        f.write(repr(d) + "\n")

with open(GOLDEN / "py_json.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for _ in range(150):
        obj = {rand_str(): rand_value(1) for _ in range(rng.randint(1, 5))}
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with open(GOLDEN / "py_hashes.txt", "w", encoding="ascii", newline="\n") as f:
    f.write("# <stream> <splitmix64 seed> <count> <sha256>, see gen_golden.py\n")
    for name, seed, count, sha in sm_hashes():
        f.write(f"{name} {seed} {count} {sha}\n")

print("wrote", GOLDEN / "py_float.txt", len(floats_pool), "and", GOLDEN / "py_json.jsonl", 150, "and", GOLDEN / "py_hashes.txt")
