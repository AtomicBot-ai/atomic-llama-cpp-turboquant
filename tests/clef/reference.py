#!/usr/bin/env python3
"""Clef reference outputs with joint_schema_model.py of the model repo (PyTorch).

  python tests/clef/reference.py <snapshot of Cloudflare/clef-flash> corpus.jsonl ref.jsonl \
      [--device mps|cpu|cuda] [--dtype bfloat16|float32] [--max-length 16384] [--tokens-only]

One line per request: {"id", "tokens", "questions": {qid: {"options": [...], "logits": [...],
"question_span": [a, b], "option_spans": [[a, b], ...]}}} or {"id", "error"}. Choice criteria given as a
list are passed as dict.fromkeys(str(c)) (the server normalization; the reference has no list form).
--tokens-only runs encode_record without the model (token and span parity only).
"""

import argparse
import json
import sys
import time


def normalize(req):
    req = json.loads(json.dumps(req))
    for q in req["questions"].values():
        if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), list):
            q["criteria"] = {("True" if c is True else "False" if c is False else "None" if c is None else str(c)): None
                             for c in q["criteria"]}
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--tokens-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.snapshot)
    import torch
    from joint_schema_model import collate_records, encode_record, load_release_model

    if args.tokens_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
        model = None
    else:
        model, processor = load_release_model(args.snapshot, device=args.device, dtype=getattr(torch, args.dtype))
        tokenizer = processor.tokenizer

    reqs = [json.loads(line) for line in open(args.corpus, encoding="utf-8")]
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, req in enumerate(reqs):
            rec = {"id": req["id"]}
            try:
                enc = encode_record(tokenizer, normalize(req), max_length=args.max_length)
                rec["tokens"] = list(enc.input_ids)
                qs = {}
                for q in enc.questions:
                    qs[q.question_id] = {"options": list(q.option_ids), "question_span": list(q.question_span),
                                         "option_spans": [list(s) for s in q.option_spans]}
                if model is not None:
                    batch = collate_records([enc], tokenizer.pad_token_id, torch.device(args.device))
                    with torch.inference_mode():
                        logits = model(batch)[0]
                    for q, ql in zip(enc.questions, logits):
                        qs[q.question_id]["logits"] = ql.float().cpu().tolist()
                rec["questions"] = qs
            except Exception as e:  # the reference refuses the request
                rec["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(reqs)}] {req['id']} {len(rec.get('tokens', []))} tokens {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
