#!/usr/bin/env python3
"""Send a Clef corpus to a /v1/systemone server and keep the responses.

  python tests/clef/run_server.py http://127.0.0.1:8091 corpus.jsonl out.jsonl

One line per request: {"id", "status", "body"}. Works with this fork (llama-server --decision; with
--decision-debug the answers carry raw logits and the prompt tokens) and with upstream llama-server.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("corpus")
    ap.add_argument("out")
    args = ap.parse_args()

    reqs = [json.loads(line) for line in open(args.corpus, encoding="utf-8")]
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, req in enumerate(reqs):
            body = {k: v for k, v in req.items() if k != "id"}
            body["model"] = "clef-flash"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            r = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", data=data,
                                       headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=600) as resp:
                    status, text = resp.status, resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                status, text = e.code, e.read().decode("utf-8")
            try:
                out = json.loads(text)
            except json.JSONDecodeError:
                out = text
            f.write(json.dumps({"id": req["id"], "status": status, "body": out}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(reqs)}] {req['id']} {status} {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
