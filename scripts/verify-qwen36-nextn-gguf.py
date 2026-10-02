#!/usr/bin/env python3
"""Check a GGUF for Qwen NextN tensor names (blk/layer *.nextn.*). Exit 1 if none found."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    args = ap.parse_args()

    gguf_py = Path(__file__).resolve().parent.parent / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore

    r = GGUFReader(str(args.gguf))
    nextn_like: list[str] = []
    for t in r.tensors:
        nm = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
        if ".nextn." in nm:
            nextn_like.append(nm)

    if not nextn_like:
        print("error: no NextN-style tensors found (expected names containing '.nextn.')", file=sys.stderr)
        return 1

    print(f"ok: found {len(nextn_like)} NextN-related tensors (showing up to 12):")
    for nm in sorted(nextn_like)[:12]:
        print(f"  {nm}")
    if len(nextn_like) > 12:
        print(f"  ... and {len(nextn_like) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
