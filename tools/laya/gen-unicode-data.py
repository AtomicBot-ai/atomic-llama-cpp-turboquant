#!/usr/bin/env python3
"""Generate tools/laya/laya-unicode-data.inc: the Unicode tables of the laya byte-level BPE tokenizer.

    python tools/laya/gen-unicode-data.py tools/laya/laya-unicode-data.inc

Needs `tokenizers` (HF). The English laya checkpoints (ModernBERT / OLMo BPE) normalize with NFC and
pre-tokenize with the GPT-2 regex of the ByteLevel pre-tokenizer. HF tokenizers runs both on its own
Unicode tables: the NFC crate is older than Python's unicodedata (it lacks e.g. U+11938 and the
combining classes of marks added after Unicode 10), and the regex classes come from Oniguruma. So
every table is probed from HF tokenizers itself, with Python's unicodedata only as the candidate
list, and the script checks the tables against HF on every codepoint and on random mark sequences.

Tables:
  letter / number / space   \\p{L}, \\p{N}, \\s as the ByteLevel regex classifies each codepoint
  ccc                       canonical combining class (0 omitted)
  decomp                    full canonical decomposition (Hangul syllables are algorithmic)
  comp                      primary composites: (first, second) -> composite
"""

import random
import sys
import unicodedata

import tokenizers
from tokenizers import normalizers, pre_tokenizers

MAX_CP = 0x110000
NFC = normalizers.NFC()
NFD = normalizers.NFD()
BL = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT


def codepoints():
    return (c for c in range(MAX_CP) if not 0xD800 <= c <= 0xDFFF)


def classify(c: int) -> str:
    """'L', 'N', 'S' (whitespace) or 'O' from how the ByteLevel regex splits "a" + cc + "1"."""
    if c == 0x20:
        return "S"  # " ?" prefixes make the probe below ambiguous for U+0020 itself
    parts = [p for p, _ in BL.pre_tokenize_str("a" + chr(c) * 2 + "1")]
    if len(parts) == 2:
        return "N" if parts[0] == "a" else "L"
    if len(parts) == 3:
        return "O"
    if len(parts) == 4:
        return "S"
    raise RuntimeError("unexpected split of U+%04X: %r" % (c, parts))


def ranges(cps):
    out = []
    for c in sorted(cps):
        if out and out[-1][1] + 1 == c:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def hf_ccc() -> dict:
    """Combining classes HF knows: Python's value when HF reorders the mark, else 0."""
    out = {}
    for c in codepoints():
        cc = unicodedata.combining(chr(c))
        if cc == 0 or NFD.normalize_str(chr(c)) != chr(c):
            continue  # decomposed marks never reach reordering or composition
        if cc < 240:
            known = NFD.normalize_str("a\u0345" + chr(c)) == "a" + chr(c) + "\u0345"
        else:
            known = NFD.normalize_str("a" + chr(c) + "\u0301") == "a\u0301" + chr(c)
        if known:
            out[c] = cc
    return out


def hf_decomp() -> dict:
    out = {}
    for c in codepoints():
        if S_BASE <= c < S_BASE + S_COUNT:
            continue
        d = NFD.normalize_str(chr(c))
        if d != chr(c):
            out[c] = [ord(x) for x in d]
    return out


def hf_comp(decomp: dict) -> dict:
    out = {}
    for c in decomp:
        m = unicodedata.decomposition(chr(c))
        if not m or m.startswith("<"):
            continue
        parts = [int(x, 16) for x in m.split()]
        if len(parts) != 2:
            continue
        a, b = parts
        if NFC.normalize_str(chr(a) + chr(b)) == chr(c):
            out[(a, b)] = c
    return out


def nfc_py(s: str, ccc: dict, decomp: dict, comp: dict) -> str:
    """The C++ algorithm (laya-unicode.cpp) in Python, to check the tables against HF."""
    d = []
    for ch in s:
        c = ord(ch)
        if S_BASE <= c < S_BASE + S_COUNT:
            i = c - S_BASE
            d += [L_BASE + i // N_COUNT, V_BASE + (i % N_COUNT) // T_COUNT]
            if i % T_COUNT:
                d.append(T_BASE + i % T_COUNT)
        else:
            d += decomp.get(c, [c])
    i = 1
    while i < len(d):  # stable sort of every run of non-starters by ccc
        if 0 < ccc.get(d[i], 0) < ccc.get(d[i - 1], 0):
            d[i - 1], d[i] = d[i], d[i - 1]
            i = max(1, i - 1)
        else:
            i += 1
    if not d:
        return ""
    out = [d[0]]
    starter = 0 if ccc.get(d[0], 0) == 0 else -1
    last = 0 if starter == 0 else 256
    for c in d[1:]:
        cc = ccc.get(c, 0)
        comp_c = None
        if starter >= 0 and (last < cc or last == 0):
            a = out[starter]
            if L_BASE <= a < L_BASE + L_COUNT and V_BASE <= c < V_BASE + V_COUNT:
                comp_c = S_BASE + ((a - L_BASE) * V_COUNT + (c - V_BASE)) * T_COUNT
            elif S_BASE <= a < S_BASE + S_COUNT and (a - S_BASE) % T_COUNT == 0 and T_BASE < c < T_BASE + T_COUNT:
                comp_c = a + (c - T_BASE)
            else:
                comp_c = comp.get((a, c))
        if comp_c is not None:
            out[starter] = comp_c
            continue
        if cc == 0:
            starter = len(out)
        last = cc
        out.append(c)
    return "".join(chr(c) for c in out)


def check(ccc, decomp, comp) -> None:
    bad = 0
    for c in codepoints():
        s = chr(c)
        bad += nfc_py(s, ccc, decomp, comp) != NFC.normalize_str(s)
    marks = sorted(ccc) + [0x0301, 0x0308, 0x0323, 0x0345, 0x05B0, 0x0F71, 0x0F72, 0x0F74, 0x302A, 0x3099]
    starters = [ord(x) for x in "aeiouAEIOUnNcCsSzZyY"] + [0x1100, 0x1161, 0x11A8, 0xAC00, 0x0915, 0x09C7, 0x0B47, 0x0DD9, 0x1025, 0x3046, 0x0391, 0x03B1, 0x0415]
    comps = sorted(decomp)
    rng = random.Random(1)
    for _ in range(300000):
        s = "".join(chr(rng.choice([rng.choice(marks), rng.choice(starters), rng.choice(comps)])) for _ in range(rng.randint(1, 7)))
        if nfc_py(s, ccc, decomp, comp) != NFC.normalize_str(s):
            bad += 1
            if bad < 5:
                print("mismatch", [hex(ord(x)) for x in s], file=sys.stderr, flush=True)
    if bad:
        raise SystemExit("tables do not reproduce HF NFC: %d mismatches" % bad)
    print("NFC check: 0 mismatches", file=sys.stderr, flush=True)


def emit_ranges(name: str, rs, out) -> None:
    out.append("static const uint32_t %s[][2] = {" % name)
    row = []
    for lo, hi in rs:
        row.append("{0x%X,0x%X}," % (lo, hi))
        if len(row) == 8:
            out.append("    " + " ".join(row))
            row = []
    if row:
        out.append("    " + " ".join(row))
    out.append("};")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__, flush=True)
        sys.exit(1)

    cls = {"L": [], "N": [], "S": []}
    for c in codepoints():
        k = classify(c)
        if k in cls:
            cls[k].append(c)
    print("classes: L %d N %d S %d" % (len(cls["L"]), len(cls["N"]), len(cls["S"])), file=sys.stderr, flush=True)

    ccc = hf_ccc()
    decomp = hf_decomp()
    comp = hf_comp(decomp)
    print("ccc %d, decomp %d, comp %d" % (len(ccc), len(decomp), len(comp)), file=sys.stderr, flush=True)
    check(ccc, decomp, comp)

    out = [
        "// generated by tools/laya/gen-unicode-data.py from HF tokenizers %s, do not edit" % tokenizers.__version__,
        "// (Unicode tables of the HF NFC normalizer and ByteLevel regex classes)",
        "",
    ]
    emit_ranges("laya_ucd_letter", ranges(cls["L"]), out)
    emit_ranges("laya_ucd_number", ranges(cls["N"]), out)
    emit_ranges("laya_ucd_space", ranges(cls["S"]), out)

    ccc_rs = []
    for c in sorted(ccc):
        if ccc_rs and ccc_rs[-1][1] + 1 == c and ccc_rs[-1][2] == ccc[c]:
            ccc_rs[-1][1] = c
        else:
            ccc_rs.append([c, c, ccc[c]])
    out.append("static const uint32_t laya_ucd_ccc[][3] = {")
    for i in range(0, len(ccc_rs), 6):
        out.append("    " + " ".join("{0x%X,0x%X,%d}," % tuple(r) for r in ccc_rs[i:i + 6]))
    out.append("};")

    flat, rows = [], []
    for c in sorted(decomp):
        rows.append((c, len(flat), len(decomp[c])))
        flat += decomp[c]
    out.append("static const uint32_t laya_ucd_decomp[][3] = {")
    for i in range(0, len(rows), 6):
        out.append("    " + " ".join("{0x%X,%d,%d}," % r for r in rows[i:i + 6]))
    out.append("};")
    out.append("static const uint32_t laya_ucd_decomp_cps[] = {")
    for i in range(0, len(flat), 12):
        out.append("    " + " ".join("0x%X," % c for c in flat[i:i + 12]))
    out.append("};")

    out.append("static const uint32_t laya_ucd_comp[][3] = {")
    items = sorted(comp.items())
    for i in range(0, len(items), 5):
        out.append("    " + " ".join("{0x%X,0x%X,0x%X}," % (a, b, c) for (a, b), c in items[i:i + 5]))
    out.append("};")

    with open(sys.argv[1], "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("wrote " + sys.argv[1], file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
