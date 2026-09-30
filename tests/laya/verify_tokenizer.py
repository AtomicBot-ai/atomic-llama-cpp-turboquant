"""Tokenizer parity: llama-laya-cli --tokenize against the HF tokenizer of the checkpoint.

Builds a deterministic corpus (newline and tab runs, HTML tags, <unusedN> and
other control tokens, the mask literal after all kinds of whitespace, CJK,
emoji, NFC/NFD, rare codepoints that need byte fallback, long space-free words
such as base64, URLs and CJK runs, and random mixes), tokenizes it with
`tokenizer(text, add_special_tokens=False)` and with the C++ tokenizer, and
requires 0 mismatches. It also checks, on the HF tokenizer alone, that a router
state tokenizes as its head (card + criterion, ending in "task:") followed by
" " + task, which lets the engine tokenize the task once per request; and it
times 1 MB space-free inputs through the C++ tokenizer.

Usage:
    python tests/laya/verify_tokenizer.py <llama-laya-cli> <laya.gguf> <hf-tokenizer-dir> [n_random]

<hf-tokenizer-dir> holds tokenizer.json (e.g. the `tokenizer/` folder of
convaiinnovations/laya-multilingual). Needs `transformers`.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unicodedata

from transformers import AutoTokenizer


def corpus(n_random: int) -> list:
    rng = random.Random(1234)
    out = []

    # whitespace runs, alone and around words (runs longer than the 31-char added tokens too)
    for ch in ["\n", "\t", " ", "\r\n", "\u2581", "\u3000", "\u00a0", "\r", "\x0b", "\x0c"]:
        for k in [1, 2, 3, 5, 17, 30, 31, 32, 33, 62, 64, 100]:
            out += [ch * k, "a" + ch * k + "b", ch * k + "x", "x" + ch * k, " " + ch * k + " y"]
    out += ["", " ", "  ", "a", " a", "a ", " a ", "hello world", "  hello   world  "]

    # added tokens: HTML, control tokens, <unusedN>, and near misses
    html = ["<table>", "<tr>", "<td>", "</td>", "</tr>", "</table>", "<h1>", "</h1>", "<b>", "</b>",
            "<i>", "<u>", "<s>", "</s>", "<code>", "</code>", "<blockquote>", "<strong>", "<em>", "<sub>", "<sup>"]
    ctrl = ["<pad>", "<eos>", "<bos>", "<unk>", "<2mass>", "[@BOS@]", "<start_of_turn>", "<end_of_turn>",
            "[toxicity=0]"]
    unused = ["<unused%d>" % i for i in [0, 1, 9, 10, 42, 98, 99, 100, 999]]
    near = ["<unused>", "<unused-1>", "<unused1", "unused1>", "<eos", "eos>", "< eos>", "<EOS>", "<tab le>",
            "<<eos>>", "<unused1<eos>>", "<e<eos>os>", "<ma<mask>sk>", "<mask<mask>>", "<<mask>"]
    toks = html + ctrl + unused + near
    for t in toks:
        out += [t, "a" + t + "b", " " + t + " ", t + t, "x " + t + "y", t + "\n" + t, "\t" + t]
    for _ in range(300):
        out.append("".join(rng.choice(toks + ["a", " ", "\n", "b c", "\u4e2d\u6587"]) for _ in range(rng.randint(2, 12))))

    # the mask literal (lstrip) after every kind of whitespace
    for ws in ["", " ", "  ", "\t", "\n", "\n\n", " \n ", "\u3000", "\u00a0", "\u2028", "\u2581", " \t ", "\r\n"]:
        out += ["a" + ws + "<mask>", ws + "<mask>", "a" + ws + "<mask>b", "a" + ws + "<mask>" + ws + "<mask>",
                "<eos>" + ws + "<mask>", "<mask>" + ws, "x" + ws + "<mask> y"]

    # scripts
    texts = [
        "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u6563\u6b65\u5427\u3002",
        "\u6771\u4eac\u90fd\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002\u30ab\u30bf\u30ab\u30ca",
        "\ud55c\uad6d\uc5b4 \ubb38\uc7a5\uc744 \ud1a0\ud070\ud654\ud569\ub2c8\ub2e4.",
        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043a\u0430\u043a \u0434\u0435\u043b\u0430? \u0421\u0447\u0451\u0442 \u2116 42",
        "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
        "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
        "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
        "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
        "\u00c9l\u00e8ve na\u00efve caf\u00e9 \u00fcber Stra\u00dfe \u0142\u00f3d\u017a",
        "Invoice #A-1029, total $1,234.56 due 2026-09-30.",
        "{\"key\": [1, 2.5, null, true], \"nested\": {\"a\": \"b\"}}",
    ]
    emoji = ["\U0001F600", "\U0001F44D\U0001F3FD", "\U0001F468\u200d\U0001F469\u200d\U0001F467",
             "\U0001F1FA\U0001F1F8", "\u2764\ufe0f", "\U0001F9D1\u200d\U0001F4BB", "\U0001FAE0"]
    rare = ["\U00020000", "\U0002A6D6", "\U000E0001", "\U0010FFFD", "\ue000", "\uffff", "\u0378",
            "\x00", "\x01", "\x7f", "\x1f", "\u200b", "\u200d", "\ufeff", "\u2028", "\u2029", "\ufffd"]
    for t in texts:
        out += [t, t.replace(" ", ""), t * 3, " " + t, t + "\n" + t]
        out += [unicodedata.normalize("NFD", t), unicodedata.normalize("NFC", t),
                unicodedata.normalize("NFKD", t)]
    for e in emoji + rare:
        out += [e, "a" + e + "b", e * 5, " " + e + " x", e + "<mask>"]
    for base in ["e\u0301", "a\u0308", "\u1100\u1161\u11a8", "n\u0303o", "e\u0301\u0323"]:
        out += [base, base * 4, "caf" + base, unicodedata.normalize("NFC", base) * 3]

    # long space-free words
    b64 = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/") for _ in range(6000))
    cjk = "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(3000))
    for n in [16, 100, 513, 2000, 6000]:
        out += [b64[:n], "https://example.com/" + b64[:n] + "?q=" + b64[:n // 2] + "#frag",
                cjk[:n // 2], "a" * n, "ab" * (n // 2), "\u2581" * n, "0123456789" * (n // 10 + 1),
                "".join(rng.choice("0123456789abcdef") for _ in range(n))]

    # random mixes
    pool = list("abcdefghijklmnopqrstuvwxyz ABC.,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`0123456789\n\t") + \
        [t[0] for t in texts] + emoji + rare + toks + ["\u2581", "\u3000", " <mask>", "\n\n\n", "\t\t"]
    for _ in range(n_random):
        k = rng.randint(1, 80)
        s = "".join(rng.choice(pool) for _ in range(k))
        if rng.random() < 0.3:
            s = unicodedata.normalize(rng.choice(["NFC", "NFD", "NFKC", "NFKD"]), s)
        out.append(s)
    # random unicode (no surrogates)
    for _ in range(n_random // 4):
        s = "".join(chr(rng.choice([rng.randint(0x20, 0x7E), rng.randint(0xA0, 0x2FFF), rng.randint(0x3000, 0xD7FF),
                                    rng.randint(0xE000, 0xFFFD), rng.randint(0x10000, 0x10FFFF)]))
                    for _ in range(rng.randint(1, 40)))
        out.append(s)
    return out


def cli_tokenize(cli: str, model: str, texts: list) -> tuple:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(t, ensure_ascii=True) + "\n")
        path = f.name
    try:
        r = subprocess.run([cli, "-m", model, "--tokenize", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise RuntimeError("llama-laya-cli failed: " + r.stderr[-500:])
    ids = [json.loads(line) for line in r.stdout.splitlines()]
    stats = [line for line in r.stderr.splitlines() if line.startswith("laya tokenize:")]
    return ids, stats[-1] if stats else ""


def router_split_check(hf, rng_seed: int = 7, n: int = 1000) -> int:
    # state = card | "\n\nsuccess criterion: " + criterion + "\n\ntask:" | " " + task
    # (tools/decision/decision-router.cpp); card lines have no control characters
    rng = random.Random(rng_seed)
    field = ["pass", "rate", "\u4e2d\u6587", "<eos>", "<b>", "<mask>", ":", "task:", "\u2581", "\u3000", "e\u0301",
             "\U0001F600", "https://x.y/z", "<unused3>", "188/200", "-", "<"]
    words = field + ["\n", "\n\n", "  ", "\t", "\r\n"]
    bad = 0
    for _ in range(n):
        card = "executor: " + " ".join(rng.choice(field) for _ in range(rng.randint(1, 8))) + "\nkind: local\nchecks:" + \
            "".join("\n- " + rng.choice(field) + ": passed 3 of 4" for _ in range(rng.randint(0, 3)))
        crit = "".join(rng.choice(words + [" "]) for _ in range(rng.randint(1, 10)))
        task = "".join(rng.choice(words + [" ", "a", "b"]) for _ in range(rng.randint(0, 20)))
        pieces = [card, "\n\nsuccess criterion: " + crit + "\n\ntask:", " " + task]
        for esc in (False, True):
            ps = [p.replace("<mask>", " ") for p in pieces]
            if esc:
                ps = [p.replace("<eos>", " ").replace("<unused3>", " ") for p in ps]
            full = hf("".join(ps), add_special_tokens=False)["input_ids"]
            split = [i for p in ps for i in hf(p, add_special_tokens=False)["input_ids"]]
            bad += full != split
    return bad


def main():
    if len(sys.argv) < 4:
        print(__doc__, flush=True)
        sys.exit(1)
    cli, model, tok_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_random = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    for p in (cli, model, os.path.join(tok_dir, "tokenizer.json")):
        if not os.path.exists(p):
            print("missing: " + p, flush=True)
            sys.exit(1)

    hf = AutoTokenizer.from_pretrained(tok_dir)
    texts = corpus(n_random)
    print("corpus: %d strings, %d chars" % (len(texts), sum(len(t) for t in texts)), flush=True)

    t0 = time.time()
    want = [hf(t, add_special_tokens=False)["input_ids"] for t in texts]
    print("hf: %.2f s" % (time.time() - t0), flush=True)
    got, stats = cli_tokenize(cli, model, texts)
    print("cli: " + stats, flush=True)

    bad = [i for i, (a, b) in enumerate(zip(want, got)) if a != b]
    if len(got) != len(want):
        print("line count mismatch: %d vs %d" % (len(got), len(want)), flush=True)
        sys.exit(1)
    for i in bad[:10]:
        print("MISMATCH %r\n  hf  %s\n  cpp %s" % (texts[i][:200], want[i][:40], got[i][:40]), flush=True)
    print("tokenizer: %d / %d match" % (len(texts) - len(bad), len(texts)), flush=True)

    split_bad = router_split_check(hf)
    print("router head/tail split (hf): %d mismatches of 2000" % split_bad, flush=True)

    # 1 MB space-free inputs: time the C++ tokenizer, and check parity
    rng = random.Random(99)
    big = {
        "cjk": "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range((1 << 20) // 3)),
        "base64": "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/") for _ in range(1 << 20)),
        "a": "a" * (1 << 20),
        "url": ("https://example.com/path/" + "seg%d/" * 20 + "?q=") * 1 + "".join(rng.choice("abcdef0123456789%&=") for _ in range(1 << 20)),
        "newlines": "\n" * (1 << 20),
    }
    for name, text in big.items():
        ids, st = cli_tokenize(cli, model, [text])
        ok = ids[0] == hf(text, add_special_tokens=False)["input_ids"]
        print("1 MB %-8s %d bytes, %d tokens, %s, hf parity %s" % (name, len(text.encode()), len(ids[0]), st, ok), flush=True)
        bad += [] if ok else [-1]

    sys.exit(1 if bad or split_bad else 0)


if __name__ == "__main__":
    main()
