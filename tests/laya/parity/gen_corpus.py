#!/usr/bin/env python3
"""Deterministic laya parity corpus (Phase 4b backend parity; DECISION.md "Backend parity tiers").

    python3 tests/laya/parity/gen_corpus.py -o build/parity/corpus/items.jsonl
    python3 tests/laya/parity/gen_corpus.py --check        # regenerate in memory, compare with corpus.json

Writes one {"id", "cat", "state", "questions"} object per line: one state and one or more typed
questions, i.e. one llama-laya-cli input file / one POST /v1/systemone body / one Agent batch.
The file is not committed; corpus.json records its sha256 and counts, and test_parity_corpus.py
checks that a fresh run reproduces it byte for byte.

Inputs are only the files next to this script: text/prose.txt (paragraphs written for this
corpus) and text/lexicon.json (short lines, labels, question sets written for this corpus).
No dataset rows, model cards or third-party text. Stdlib only; the random stream uses only
integer draws and random() / uniform() (no gauss, no pow, no libm), so the bytes do not depend
on the platform or the Python version.

Coverage (same shape as the Phase 4 laya-eval set): typed-decisions style 5-question workflows
on string / object / single-question / list states, preset question sets, mixed question types
on 12 state kinds (short and long strings, float-heavy objects, chat turns, string and number
lists), every option count 2..20 for choice and score, long options (48-token cap), many long
options (head budget), long instructions (head cut), long states that hit max_len (list states
are cut from the left), structured criterion values, tokenizer edge text, scalar states (a null
state is refused by the server), and compat shapes the reference accepts (choice list criteria,
noul labels, capitalised noul keys, non-string instructions). English items (all letters ASCII)
form the subset for the English checkpoints.
"""

import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260930


def read_text(name):
    with open(os.path.join(HERE, "text", name), "rb") as f:
        return f.read().decode("utf-8").replace("\r\n", "\n")


# tokenizer edge text; ASCII escapes only (control characters, markers, metaspace, bidi)
EDGE = [
    "Line one\nLine two\n\nParagraph after a blank line\n\n\n\nfour newlines",
    "\tindented with a tab\n\t\tdouble tab\n\t\t\ttriple tab\tinline tab",
    "Double  spaces   triple    quad     and trailing   ",
    "   leading spaces then text",
    "<table><tr><td>cell 1</td><td>cell 2</td></tr></table> and <b>bold</b> <i>it</i> <code>x=1</code>",
    "<h1>Title</h1><blockquote>quoted</blockquote><strong>strong</strong><sub>2</sub><sup>3</sup>",
    "special tokens in text: <eos> <bos> <pad> <unk> <start_of_turn>user hi<end_of_turn> [CLS] [SEP] [MASK] <2mass>",
    "literal metaspace \u2581 and \u2581\u2581 runs \u2581\u2581\u2581 mixed \u2581word",
    "mask token inside <mask> the text <mask><mask> and at end <mask>",
    "unused tokens <unused0> <unused12> literal",
    "CRLF line\r\nnext line\r\n\r\nand a form feed\x0c and vertical tab\x0b end",
    "NUL byte here:\x00 then text and a bell\x07 and escape\x1b[0m",
    "zero width\u200bspace\u200cjoiner\u200d and bidi \u202eRTL\u202c and nbsp\u00a0here and ideographic\u3000space",
    "private use \ue000\uf8ff and non-BMP \U00010348 \U0001d11e \U0001f701 and replacement \ufffd",
    "code: def f(x):\n    return {'a': [1, 2.5, None]}  # comment\n\nfor i in range(10):\n\tprint(i)",
    "url https://example.com/path?q=1&r=%20x#frag email a.b+c@example.org path C:\\Users\\x\\file.txt",
    "numbers 3.14159 -0.0 1e-5 6.02e23 0x1F 1,234,567.89 12,00,000 50% 12:45:07",
    "repeated " + "a" * 130,
    "long word Donaudampfschifffahrtsgesellschaftskapitaenswitwenpensionsversicherungsanstalt",
    "",
]

FLOAT_CONSTS = [0.1, 0.2, 0.3, 1.0, -0.0, 0.0, 1e-5, 1e-4, 9.999e-5, 1e16, 1e15, 123456789.123,
                5e-324, 1.7976931348623157e308, 2.5e-10, 3.0e21, 763.4, 100.0, 1e22, 0.1 + 0.2]

STATE_KINDS = ["str_short", "str_short", "str_edge", "str_long", "dict_float", "dict_float", "dict_long",
               "list_turns", "list_turns_long", "list_strings", "list_long_strings", "list_numbers"]

OBJ_KEYS = ["amount", "total", "currency", "customer", "status", "score", "ratio", "latency_ms", "items", "meta",
            "notes", "id", "ts", "flags", "name", "preference", "key", "a b", "x\"y", "tab\tkey", "nl\nkey"]


class Gen:
    def __init__(self):
        self.rng = random.Random(SEED)
        self.lex = json.loads(read_text("lexicon.json"))
        self.lex["prose"] = [p.strip("\n") for p in read_text("prose.txt").split("\n\n") if p.strip()]
        # English mode: text pools keep only entries whose letters are all ASCII (see is_english)
        self.lex_en = {k: [x for x in v if isinstance(x, str) and ascii_letters(x)]
                       for k, v in self.lex.items() if isinstance(v, list)}
        self.en = False
        self.items = []

    def lx(self, name):
        return self.lex_en[name] if self.en else self.lex[name]

    @property
    def multi(self):
        return self.lx("multi")

    @property
    def prose(self):
        return self.lx("prose")

    # ------------------------------------------------------------ scalars
    def label(self, i):
        lab = self.rng.choice(self.lx("labels"))
        return lab % i if "%d" in lab else lab

    def uniq_labels(self, k):
        out = []
        while len(out) < k:
            lab = self.label(len(out) + self.rng.randint(0, 999))
            if lab not in out:
                out.append(lab)
        return out

    def rand_float(self):
        r = self.rng
        pick = r.randint(0, 4)
        if pick == 0:
            return round(r.uniform(-1000, 1000), r.randint(0, 6))
        if pick == 1:
            return r.random()
        if pick == 2:
            return r.uniform(-1e6, 1e6)
        if pick == 3:
            return r.choice(FLOAT_CONSTS)
        return float("%de%d" % (r.randint(-999999, 999999), r.randint(-14, 9)))

    def rand_scalar(self):
        r = self.rng
        x = r.random()
        if x < 0.35:
            return self.rand_float()
        if x < 0.55:
            return r.randint(-10**6, 10**12)
        if x < 0.62:
            return r.choice([True, False, None])
        return r.choice(self.multi + EDGE[:16] + [r.choice(self.prose)[:120], "ok", "", "N/A"])

    def rand_obj(self, depth=0, width=None):
        r = self.rng
        w = width or r.randint(2, 8)
        o = {}
        for i in range(w):
            k = r.choice(OBJ_KEYS) + ("" if r.random() < 0.6 else "_%d" % i)
            x = r.random()
            if depth < 2 and x < 0.15:
                o[k] = self.rand_obj(depth + 1)
            elif depth < 2 and x < 0.3:
                o[k] = [self.rand_scalar() if r.random() < 0.7 else self.rand_obj(depth + 1, 2) for _ in range(r.randint(0, 5))]
            else:
                o[k] = self.rand_scalar()
        return o

    def long_text(self, min_chars):
        parts = []
        n = 0
        while n < min_chars:
            p = self.rng.choice(self.prose + self.multi)
            parts.append(p)
            n += len(p)
        return "\n\n".join(parts)

    def turns(self, n, long_turns=False):
        r = self.rng
        out = []
        for i in range(n):
            if long_turns:
                c = r.choice(self.multi + self.prose + EDGE[:14])
            else:
                c = r.choice(self.multi + [p[:200] for p in self.prose])
            out.append({"role": ["user", "assistant"][i % 2], "content": c, "turn": i, "sentiment": round(r.uniform(-1, 1), 3)})
        return out

    def gen_state(self, kind):
        r = self.rng
        if kind == "str_short":
            return r.choice(self.multi + [p[:600] for p in self.prose])
        if kind == "str_edge":
            return r.choice(EDGE) + " " + r.choice(self.multi)
        if kind == "str_long":
            return self.long_text(r.randint(6000, 14000))
        if kind == "dict_float":
            return self.rand_obj()
        if kind == "dict_long":
            return {"document": self.long_text(r.randint(6000, 12000)), "meta": self.rand_obj(), "amount": self.rand_float()}
        if kind == "list_turns":
            return self.turns(r.randint(1, 6))
        if kind == "list_turns_long":
            return self.turns(r.randint(25, 60), long_turns=True)
        if kind == "list_strings":
            return [r.choice(self.multi + EDGE[:10]) for _ in range(r.randint(1, 8))]
        if kind == "list_long_strings":
            return [r.choice(self.prose) for _ in range(r.randint(20, 40))]
        if kind == "list_numbers":
            return [self.rand_float() if r.random() < 0.7 else r.randint(-99, 99) for _ in range(r.randint(1, 30))]
        raise ValueError(kind)

    def line(self, n):
        # workflow text: mostly English prose, so that the English subset stays large
        r = self.rng
        return r.choice(self.multi) if r.random() < 0.3 else r.choice(self.prose)[:n]

    # ------------------------------------------------------------ workflow states
    def workflow_state(self, wf):
        r = self.rng
        lex = self.lex
        if wf == "agent_trace":
            steps = []
            for i in range(r.randint(1, 6)):
                tool = r.choice(lex["tools"])
                steps.append({"step": i + 1, "tool": tool, "args": {"target": r.choice(lex["tasks"])[:40], "limit": r.randint(1, 500)},
                              "ok": r.random() < 0.8, "ms": round(r.uniform(3, 4000), 1)})
            return {"task": r.choice(lex["tasks"]), "agent": "agent-v%d" % r.randint(1, 9), "steps": steps,
                    "final": r.choice(self.prose)[:r.randint(80, 300)],
                    "policy": ["no writes outside the sandbox", "ask before deleting data"][:r.randint(0, 2)],
                    "cost_usd": round(r.uniform(0, 3), 4)}
        if wf == "invoice_audit":
            lines = []
            total = 0
            for _ in range(r.randint(1, 7)):
                qty = r.randint(1, 40)
                cents = r.randint(99, 250000)
                lines.append({"desc": r.choice(lex["items"]), "qty": qty, "unit_price": cents / 100})
                total += qty * cents
            stated = total if r.random() < 0.7 else total + r.randint(-5000, 5000)
            st = {"vendor": r.choice(lex["vendors"]), "invoice_no": "INV-%06d" % r.randint(0, 999999),
                  "currency": r.choice(lex["currencies"]), "lines": lines, "total": stated / 100}
            if r.random() < 0.6:
                st["po"] = "PO-%05d" % r.randint(0, 99999)
            if r.random() < 0.4:
                st["notes"] = self.line(200)
            return st
        if wf == "support_ticket":
            return {"channel": r.choice(lex["channels"]), "tier": r.choice(["free", "plus", "business"]),
                    "customer": r.choice(lex["names"]), "subject": self.line(60), "body": self.line(500),
                    "history": [self.line(120) for _ in range(r.randint(0, 3))], "open_days": r.randint(0, 45)}
        if wf == "content_review":
            post = r.choice(EDGE[:12]) if r.random() < 0.2 else self.line(400)
            return {"platform": r.choice(lex["platforms"]), "post": post,
                    "author_age_days": r.randint(0, 3000), "reports": r.randint(0, 25), "prior_strikes": r.randint(0, 3)}
        raise ValueError(wf)

    # ------------------------------------------------------------ questions
    def gen_question(self, qt, k=None, long_opts=False, long_ins=False, structured=False):
        r = self.rng
        lex = self.lex
        ins = r.choice(self.lx("instr"))
        if long_ins:
            ins = ins + " " + self.long_text(r.randint(1200, 3000))
        if qt == "choice":
            k = k or r.randint(2, 20)
            crit = {}
            for lab in self.uniq_labels(k):
                x = r.random()
                if structured and x < 0.5:
                    crit[lab] = r.choice([None, "", 0, 1.5, False, True, {"desc": r.choice(self.lx("descs")), "w": self.rand_float()},
                                          [r.choice(self.lx("descs")), 2], -0.0, 1e-5])
                elif long_opts or x < 0.1:
                    crit[lab] = r.choice(self.lx("descs")[-1:] + [self.long_text(300)[:r.randint(200, 400)]])
                elif x < 0.2:
                    crit[lab] = None if r.random() < 0.5 else ""
                else:
                    crit[lab] = r.choice(self.lx("descs"))
            return {"type": "choice", "instructions": ins, "criteria": crit}
        if qt == "score":
            k = k or r.randint(2, 20)
            levels = []
            for i in range(k):
                if structured and r.random() < 0.4:
                    levels.append(r.choice([i, float(i) / 2, {"level": i, "desc": r.choice(self.lx("levels"))}, [i, r.choice(self.lx("levels"))], True]))
                elif long_opts:
                    levels.append(self.lx("descs")[-1] if r.random() < 0.5 else self.long_text(300)[:r.randint(200, 400)])
                else:
                    levels.append(r.choice(self.lx("levels")))
            return {"type": "score", "instructions": ins, "criteria": levels}
        q = {"type": "noul", "instructions": ins}
        x = r.random()
        if structured:
            q["criteria"] = r.choice([{"true": {"desc": "holds", "p": 0.5}}, {"false": 0, "true": 1}, {"true": ["a", "b"], "false": None},
                                      {"true": "", "false": ""}, {}, {"false": False, "true": True}, {"true": 1e-5}])
        elif long_opts:
            q["criteria"] = {"true": self.lx("descs")[-1], "false": self.long_text(300)[:300]}
        elif x < 0.35:
            pass
        elif x < 0.55:
            q["criteria"] = {"true": r.choice(self.lx("descs"))}
        elif x < 0.7:
            q["criteria"] = {"false": r.choice(self.lx("descs"))}
        else:
            q["criteria"] = {"false": r.choice(self.lx("descs")), "true": r.choice(self.lx("descs"))}
        return q

    def add(self, cat, state, questions):
        self.items.append({"id": "%s_%04d" % (cat, len(self.items)), "cat": cat, "state": state, "questions": questions})

    # ------------------------------------------------------------ corpus
    def build(self):
        r = self.rng
        wfs = self.lex["workflows"]
        wf_names = list(wfs.keys())

        # typed-decisions style: 5-question workflows on generated states
        for i in range(260):
            self.en = i % 4 != 3
            wf = wf_names[i % len(wf_names)]
            qs = json.loads(json.dumps(wfs[wf]))
            st = self.workflow_state(wf)
            if i < 100:
                self.add("td_str_multi", json.dumps(st, ensure_ascii=False), qs)
            elif i < 180:
                self.add("td_dict_multi", st, qs)
            elif i < 220:
                for qid, q in qs.items():
                    self.add("td_dict_single", st, {qid: q})
            else:
                self.add("td_list_multi", [{"field": k, "value": v} for k, v in st.items()], qs)

        # preset question sets on generated states
        fields = {"mail": "body", "safety": "prompt", "dispatch": "request", "forum": "post"}
        for name, qs in self.lex["presets"].items():
            for j in range(12):
                self.en = j % 2 == 0
                st = self.gen_state(r.choice(STATE_KINDS))
                if r.random() < 0.5:
                    st = {fields[name]: st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)[:3000]}
                self.add("preset_" + name, st, json.loads(json.dumps(qs)))

        # mixed question types
        for n in range(170):
            self.en = n % 2 == 0
            kind = STATE_KINDS[n % len(STATE_KINDS)]
            st = self.gen_state(kind)
            qs = {}
            for j in range(r.choice([1, 1, 2, 3, 4, 5, 6])):
                qt = r.choice(["choice", "score", "noul"])
                qs["q%d_%s" % (j, qt)] = self.gen_question(qt)
            self.add("gen_" + kind, st, qs)

        # every option count for choice and score
        for k in range(2, 21):
            for rep in range(4):
                self.en = rep < 2
                st = self.gen_state(r.choice(STATE_KINDS))
                self.add("sweep_k%02d" % k, st, {"c": self.gen_question("choice", k=k), "s": self.gen_question("score", k=k)})

        # head budget: long options, many long options, long instructions
        for j in range(30):
            self.en = j % 2 == 0
            st = self.gen_state(r.choice(STATE_KINDS))
            self.add("stress_longopt", st, {"c": self.gen_question("choice", k=r.randint(2, 5), long_opts=True),
                                            "n": self.gen_question("noul", long_opts=True)})
        for j in range(30):
            self.en = j % 2 == 0
            st = self.gen_state(r.choice(STATE_KINDS))
            self.add("stress_manylongopt", st, {"c": self.gen_question("choice", k=r.randint(8, 20), long_opts=True),
                                                "s": self.gen_question("score", k=r.randint(8, 20), long_opts=True)})
        for j in range(30):
            self.en = j % 2 == 0
            st = self.gen_state(r.choice(STATE_KINDS))
            self.add("stress_longins", st, {"x": self.gen_question(r.choice(["choice", "score", "noul"]), long_ins=True)})

        # structured criterion values
        for j in range(40):
            self.en = j % 2 == 0
            st = self.gen_state(r.choice(STATE_KINDS))
            self.add("struct_crit", st, {"c": self.gen_question("choice", structured=True),
                                         "s": self.gen_question("score", structured=True),
                                         "n": self.gen_question("noul", structured=True)})

        # tokenizer edge text in instructions, options and state
        self.en = False
        for e in EDGE + self.multi:
            st = r.choice([e, {"text": e, "v": self.rand_float()}, [e, e[::-1]]])
            self.add("tok_edge", st, {"n": {"type": "noul", "instructions": "Is this text about: " + e},
                                      "c": {"type": "choice", "instructions": "Pick one.", "criteria": {e[:40] or "empty": e, "other": "none of " + e[:30]}}})

        # top-level scalar states (the server refuses the null one)
        for v in [42, 3.5, True, None, "x", 1e-7]:
            self.add("scalar_state", v, {"n": self.gen_question("noul")})

        # compat: shapes the reference accepts, one question per item
        compat = [
            {"type": "choice", "instructions": "Which department?", "criteria": ["billing", "support", "sales"]},
            {"type": "choice", "instructions": "Pick a number.", "criteria": ["1", "2", "3", "4"]},
            {"type": "noul", "instructions": "Is the user angry?", "labels": {"false": "calm", "true": "angry"}},
            {"type": "noul", "instructions": "Refund requested?", "labels": {"false": "no", "true": "yes"}, "criteria": {"true": "asks for money back"}},
            {"type": "noul", "instructions": "Is it urgent?", "criteria": {"True": "needs action today", "False": "can wait"}},
            {"type": "noul", "instructions": "Is it spam?", "criteria": {"TRUE": "bulk marketing"}},
            {"type": "noul", "instructions": {"question": "Is the tone hostile?", "lang": "en"}},
            {"type": "score", "instructions": ["Rate", "severity"], "criteria": ["low", "mid", "high"]},
            {"type": "choice", "instructions": 12345, "criteria": {"a": "first", "b": "second"}},
        ]
        for q in compat:
            for st in ["I was charged twice and nobody answers my emails!", {"msg": "Hello, all good here.", "amt": 12.5}]:
                self.add("compat", st, {"q": q})
        return self.items


def ascii_letters(s):
    return all(c.isascii() for c in s if c.isalpha())


def is_english(item):
    # same rule as tests/laya/verify_reference.py --english
    return ascii_letters(json.dumps([item["state"], item["questions"]], ensure_ascii=False))


def render(items):
    return "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")


def summary(items, data):
    en = [it for it in items if is_english(it)]
    qtypes = {}
    for it in items:
        for q in it["questions"].values():
            qtypes[q["type"]] = qtypes.get(q["type"], 0) + 1
    return {
        "items_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "items": len(items),
        "questions": sum(len(it["questions"]) for it in items),
        "english_items": len(en),
        "english_questions": sum(len(it["questions"]) for it in en),
        "question_types": dict(sorted(qtypes.items())),
        "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", help="write items.jsonl here")
    ap.add_argument("--check", action="store_true", help="compare the result with corpus.json (exit 1 on a difference)")
    ap.add_argument("--record", action="store_true", help="write corpus.json from this run")
    a = ap.parse_args()
    if not (a.out or a.check or a.record):
        ap.error("nothing to do: give -o, --check or --record")

    items = Gen().build()
    data = render(items)
    s = summary(items, data)
    rec_path = os.path.join(HERE, "corpus.json")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "wb") as f:
            f.write(data)
        print("wrote %s: %d items, %d questions, sha256 %s" % (a.out, s["items"], s["questions"], s["items_sha256"]), flush=True)
    if a.record:
        with open(rec_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(s, f, indent=1)
            f.write("\n")
        print("recorded " + rec_path, flush=True)
    if a.check:
        with open(rec_path, encoding="utf-8") as f:
            want = json.load(f)
        if want != s:
            print("corpus differs from corpus.json:\n  recorded  %s\n  generated %s" % (json.dumps(want), json.dumps(s)), flush=True)
            sys.exit(1)
        print("corpus matches corpus.json (sha256 %s)" % s["items_sha256"], flush=True)


if __name__ == "__main__":
    main()
