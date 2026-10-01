"""The parity corpus regenerates byte for byte and keeps its coverage (tests/laya/parity/gen_corpus.py)."""

import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_corpus  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _build():
    items = gen_corpus.Gen().build()
    return items, gen_corpus.render(items)


def test_regenerates_recorded_sha():
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        want = json.load(f)
    items, data = _build()
    assert hashlib.sha256(data).hexdigest() == want["items_sha256"]
    assert gen_corpus.summary(items, data) == want


def test_two_runs_are_identical():
    assert _build()[1] == _build()[1]


def test_coverage():
    items, data = _build()
    assert len({it["id"] for it in items}) == len(items)
    qs = [q for it in items for q in it["questions"].values()]
    types = collections.Counter(q["type"] for q in qs)
    assert all(types[t] > 700 for t in ("choice", "score", "noul"))
    counts = {len(q["criteria"]) for q in qs if q["type"] in ("choice", "score") and isinstance(q.get("criteria"), (dict, list))}
    assert set(range(2, 21)) <= counts
    cats = collections.Counter(it["cat"].split("_k")[0] for it in items)
    for c in ("td_str_multi", "td_dict_multi", "td_dict_single", "td_list_multi", "sweep", "stress_longopt", "stress_manylongopt",
              "stress_longins", "struct_crit", "tok_edge", "scalar_state", "compat", "gen_list_turns_long", "gen_list_long_strings"):
        assert cats[c] > 0, c
    kinds = collections.Counter(type(it["state"]).__name__ for it in items)
    assert kinds["dict"] > 200 and kinds["str"] > 150 and kinds["list"] > 100 and kinds["NoneType"] == 1
    assert sum(len(json.dumps(it["state"], ensure_ascii=False)) > 6000 for it in items) > 100
    assert sum(isinstance(it["state"], list) and len(json.dumps(it["state"])) > 6000 for it in items) > 20
    en = [it for it in items if gen_corpus.is_english(it)]
    assert 400 < len(en) < len(items) - 200
    assert any("\u4e00" <= c <= "\u9fff" for c in data.decode("utf-8"))


def test_inputs_are_only_vendored_files():
    # the generator reads nothing but text/prose.txt and text/lexicon.json
    src = open(os.path.join(HERE, "gen_corpus.py"), encoding="utf-8").read()
    assert "huggingface" not in src and "glob" not in src and "urlopen" not in src
    assert sorted(os.listdir(os.path.join(HERE, "text"))) == ["lexicon.json", "prose.txt"]
    assert all(ord(c) < 128 for c in src)
