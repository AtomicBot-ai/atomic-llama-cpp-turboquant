"""Tokenizer parity: llama-laya-cli --tokenize against the HF tokenizer of the checkpoint.

Works for both laya tokenizers (metaspace-bpe: laya-multilingual; bytelevel-bpe: laya and
laya-typed-decisions). Builds a deterministic corpus (newline and tab runs, HTML tags, <unusedN> and
other control tokens, the added tokens of the checkpoint itself and near misses, the mask literal
after all kinds of whitespace, CJK, emoji, NFC/NFD, rare codepoints that need byte fallback, long
space-free words such as base64, URLs and CJK runs, an English-heavy section (prose, contractions,
numbers, code, JSON, markdown, whitespace runs, accents in NFC and NFD) and random mixes), tokenizes it with
`tokenizer(text, add_special_tokens=False)` and with the C++ tokenizer, and
requires 0 mismatches. It also checks, on the HF tokenizer alone, that a router
state tokenizes as its head (card + criterion, ending in "task:") followed by
" " + task, which lets the engine tokenize the task once per request; and it
times 1 MB space-free inputs through the C++ tokenizer.

Usage:
    python tests/laya/verify_tokenizer.py <llama-laya-cli> <laya.gguf> <hf-tokenizer-dir> [n_random]

<hf-tokenizer-dir> holds tokenizer.json (the `tokenizer/` folder of a laya checkpoint, e.g.
convaiinnovations/laya-multilingual or convaiinnovations/laya). Needs `transformers`.
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


WORDS = ("the quick brown fox jumps over lazy dog invoice refund customer support ticket billing account "
         "password reset shipping delayed order cancel subscription error timeout server request response "
         "model agent executor criterion passed failed measured results task summary extract field date total "
         "please thanks hello world yes no maybe it is was were be been being have has had do does did will "
         "would should could can may might must shall AI API JSON HTTP URL CPU GPU LLM Q4_K_M v1.2.3").split()


def english(rng: random.Random, n: int) -> list:
    """English-heavy strings: the text the English checkpoints are for."""
    out = []
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D",
                    "\u2019s", "\u2019t", "''s", "'", "' s", "'x"]
    for c in contractions:
        out += ["it" + c, "It" + c + " fine", "don" + c, " " + c, c, c + c, "a" + c + "b", "I" + c + "\n", "(" + c + ")"]
    nums = ["0", "7", "42", "123", "1234", "12345", "3.14159", "1,234,567.89", "-5", "+1", "1e-9", "0x1F", "2026-09-30",
            "10:30", "188/200", "99.5%", "$1,234.56", "\u20ac12", "1st", "2nd", "3rd", "4th", "1/2", "\u00bd", "\u0663\u0664"]
    for x in nums:
        out += [x, " " + x, "a" + x, x + "a", x + " items", "(" + x + ")", x + x]
    code = ["def f(x):\n    return x ** 2\n", "for (int i = 0; i < n; ++i) {\n\tsum += a[i];\n}",
            "SELECT * FROM users WHERE id = 42;", "if (a && b || !c) { return; }", "x = {'a': 1, \"b\": [2, 3]}",
            "<div class=\"row\"><span>Hi</span></div>", "#include <stdio.h>", "git commit -m \"fix: bug\"",
            "https://example.com/path/to/page?query=1&b=two#frag", "user.name+tag@example.co.uk", "C:\\Users\\me\\file.txt",
            "## Heading\n\n- item one\n- item two\n\n**bold** _it_ `code`", "| a | b |\n|---|---|\n| 1 | 2 |",
            "\"quoted\" 'single' `back` \u201ccurly\u201d \u2018curly\u2019", "... -- --- !!! ??? ;;; ::", "a\u2014b a\u2013b a\u2026b"]
    for c in code:
        out += [c, " " + c, c + "\n", c.upper(), c.replace(" ", "  ")]
    accents = ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9", "fa\u00e7ade", "co\u00f6perate", "Z\u00fcrich", "Ma\u00f1ana",
               "\u00c5ngstr\u00f6m", "\u212bngstr\u00f6m", "\u2126 ohm", "K\u212a kelvin", "\ufb01ne \ufb02ow"]
    for a in accents:
        for form in ("NFC", "NFD", "NFKC", "NFKD"):
            t = unicodedata.normalize(form, a)
            out += [t, "The " + t + " is here.", t.upper()]
    ws = [" ", "  ", "   ", "\t", "\n", "\n\n", " \n", "\n ", "\r\n", "\u00a0", "\u2009", "\u3000"]
    for _ in range(n):
        k = rng.randint(1, 30)
        words = []
        for _ in range(k):
            w = rng.choice(WORDS)
            r = rng.random()
            if r < 0.08:
                w = w.capitalize()
            elif r < 0.12:
                w = w.upper()
            elif r < 0.18:
                w += rng.choice(contractions[:7])
            elif r < 0.24:
                w += rng.choice([",", ".", "!", "?", ":", ";", ")", "\"", "'"])
            elif r < 0.28:
                w = rng.choice(nums)
            elif r < 0.30:
                w = rng.choice(accents)
            words.append(w)
        s = ""
        for w in words:
            s += w + (rng.choice(ws) if rng.random() < 0.15 else " ")
        if rng.random() < 0.3:
            s = s.strip()
        if rng.random() < 0.2:
            s = unicodedata.normalize(rng.choice(["NFD", "NFKD"]), s)
        out.append(s)
    return out


def vocab_tokens(hf, rng: random.Random) -> list:
    """The added tokens of this checkpoint (special or not), alone, glued, near misses, after whitespace."""
    out = []
    added = sorted(hf.get_added_vocab().keys())
    specials = sorted(set(hf.all_special_tokens))
    pick = specials + rng.sample(added, min(len(added), 60))
    for t in pick:
        if not t:
            continue
        out += [t, "a" + t + "b", " " + t + " ", t + t, "x " + t + "y", t + "\n" + t, "\t" + t, t[:-1], t[1:],
                t.lower(), t.upper(), t[:1] + " " + t[1:], "<" + t + ">", "[" + t + "]"]
    mask = hf.mask_token
    for ws in ["", " ", "  ", "\t", "\n", "\n\n", " \n ", "\u3000", "\u00a0", "\u2028", "\u2581", " \t ", "\r\n", "   "]:
        out += ["a" + ws + mask, ws + mask, "a" + ws + mask + "b", "a" + ws + mask + ws + mask, mask + ws,
                specials[0] + ws + mask, "x" + ws + mask + " y", "e\u0301" + ws + mask]
    for _ in range(400):
        out.append("".join(rng.choice(pick + WORDS + [" ", "  ", "\n", mask, "e\u0301", "\u00e9"]) for _ in range(rng.randint(2, 12))))
    return out


def corpus(n_random: int, hf=None) -> list:
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
    if hf is not None:
        out += vocab_tokens(hf, random.Random(4321))
    out += english(random.Random(5678), max(5000, n_random))

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
        r = subprocess.run([cli, "-m", model, "--tokenize", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    texts = corpus(n_random, hf)
    n_en = sum(1 for t in texts if t and sum(c.isascii() for c in t) >= 0.9 * len(t) and any(c.isalpha() for c in t))
    print("corpus: %d strings, %d chars, %d English-heavy (>= 90%% ASCII, has letters)" % (
        len(texts), sum(len(t) for t in texts), n_en), flush=True)

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
