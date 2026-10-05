#!/usr/bin/env python3
"""Golden Clef prompts for test-decision-clef: encode_record of joint_schema_model.py with the HF tokenizer.

  python tests/clef/gen_golden.py <snapshot of Cloudflare/clef-flash> tests/clef/golden/prompts.jsonl

Needs transformers (tokenizer only, no model weights are read). One line per case:
{"id", "max_tokens", "request", "tokens", "question_spans": [[start, end, type]], "option_spans": [[start, end]],
"options": {qid: [option ids in prompt order]}, "n_state", "n_state_cut"} or {"id", "max_tokens", "request", "error"}.
type is the head question type: 0 noul, 1 choice, 2 score.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_corpus import CASES, long_state, many_questions  # noqa: E402
from reference import normalize  # noqa: E402


def main():
    snapshot, out_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, snapshot)
    from joint_schema_model import encode_record, render
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot)

    cases = [(f"case_{i}", c, 16384) for i, c in enumerate(CASES)]
    cases.append(("many_questions", many_questions(), 16384))
    trunc = {"state": long_state(400, 7), "questions": {"topic": {"type": "choice", "instructions": "Main topic?",
                                                                  "criteria": {"infra": "servers", "money": "billing"}},
                                                        "urgent": {"type": "noul", "instructions": "Is it urgent?"}}}
    cases.append(("cut_string", trunc, 256))
    cases.append(("cut_json", {"state": {"log": [long_state(30, i) for i in range(20)]}, "questions": trunc["questions"]}, 256))
    cases.append(("fits_exactly", trunc, 0))  # max_tokens set below to the full length
    cases.append(("too_long", trunc, 100))

    with open(out_path, "w", encoding="utf-8") as f:
        for cid, req, max_tokens in cases:
            rec_in = normalize(req)
            if max_tokens == 0:
                max_tokens = len(encode_record(tok, rec_in, max_length=1 << 20).input_ids)
            line = {"id": cid, "max_tokens": max_tokens, "request": req}
            try:
                enc = encode_record(tok, rec_in, max_length=max_tokens)
            except ValueError:
                line["error"] = "PROMPT_TOO_LONG"
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                continue
            n_state = len(tok(render(rec_in["state"]), add_special_tokens=False).input_ids)
            fixed = len(encode_record(tok, {**rec_in, "state": ""}, max_length=1 << 20).input_ids)
            line["tokens"] = list(enc.input_ids)
            line["question_spans"] = [[q.question_span[0], q.question_span[1], q.question_type] for q in enc.questions]
            line["option_spans"] = [list(s) for q in enc.questions for s in q.option_spans]
            line["options"] = {q.question_id: list(q.option_ids) for q in enc.questions}
            line["n_state"] = n_state
            line["n_state_cut"] = n_state - (len(enc.input_ids) - fixed)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
