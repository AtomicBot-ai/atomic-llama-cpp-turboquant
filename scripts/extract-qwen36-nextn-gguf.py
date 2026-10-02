#!/usr/bin/env python3
"""
Extract a standalone NextN draft GGUF from a combined Qwen3.6 *_MTP.gguf.

The resulting file is self-contained: it stores only the tensors that the
NextN draft graph needs (full tok_embd + output_norm + output + the
NextN tail block(s)), and re-prefixes the architecture-scoped KV pairs to
the target NextN arch ('qwen35_mtp' or 'qwen35moe_mtp'). This lets us
load the draft model on its own, with general.architecture already set
correctly, so the server does NOT need to use llama_model_params.override_arch
on the same file.

Purpose: isolation test for the suspected interaction bug between
double-mmap (target + draft pointing at same GGUF via override_arch + partial_load)
and target prefill with n_outputs > 1.

Usage:
  scripts/extract-qwen36-nextn-gguf.py <input.gguf> <output.gguf>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("extract-nextn")


def _gguf_py() -> Path:
    return Path(__file__).resolve().parent.parent / "gguf-py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="source *_MTP.gguf (combined target+draft)")
    ap.add_argument("output", type=Path, help="destination standalone NextN GGUF path")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    sys.path.insert(0, str(_gguf_py()))
    from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType  # type: ignore

    if not args.input.is_file():
        logger.error("input GGUF not found: %s", args.input)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(args.input))

    def kv_first(name: str):
        f = reader.fields.get(name)
        return f.contents() if f is not None else None

    src_arch = kv_first("general.architecture")
    if not isinstance(src_arch, str):
        logger.error("missing/invalid 'general.architecture' in %s", args.input)
        return 1

    src_arch_l = src_arch.lower()
    moe_markers = ("moe",)
    is_moe = any(m in src_arch_l for m in moe_markers)
    target_arch = "qwen35moe_mtp" if is_moe else "qwen35_mtp"

    if src_arch in (target_arch,):
        logger.info("source arch already '%s' (standalone NextN GGUF). KV prefix will be left as-is.", src_arch)

    n_layer = kv_first(f"{src_arch}.block_count")
    n_predict = kv_first(f"{src_arch}.nextn_predict_layers")

    if n_layer is None:
        logger.error("missing '%s.block_count' in source KV", src_arch)
        return 1
    if n_predict is None:
        logger.error("missing '%s.nextn_predict_layers' in source KV; this GGUF does not look like a NextN/MTP-combined file", src_arch)
        return 1

    n_main = int(n_layer) - int(n_predict)
    logger.info(
        "source arch=%s -> target arch=%s; n_layer=%d, nextn_predict_layers=%d, n_main=%d",
        src_arch, target_arch, int(n_layer), int(n_predict), n_main,
    )

    writer = GGUFWriter(str(args.output), arch=target_arch)

    skip_keys = {
        "general.architecture",
        "general.alignment",
        "GGUF.kv_count",
        "GGUF.tensor_count",
        "GGUF.version",
    }

    n_kv_copied = 0
    n_kv_skipped = 0
    n_kv_renamed = 0

    for key, field in reader.fields.items():
        if key in skip_keys:
            n_kv_skipped += 1
            continue
        if not field.types:
            n_kv_skipped += 1
            continue

        new_key = key
        if key.startswith(src_arch + ".") and src_arch != target_arch:
            new_key = target_arch + "." + key[len(src_arch) + 1 :]
            n_kv_renamed += 1

        main_type = field.types[0]

        try:
            if main_type == GGUFValueType.ARRAY:
                sub_type = field.types[-1]
                val = field.contents()
                writer.add_key_value(new_key, val, main_type, sub_type=sub_type)
            else:
                val = field.contents()
                writer.add_key_value(new_key, val, main_type)
        except Exception as exc:
            logger.warning("skip KV %s: %s", key, exc)
            n_kv_skipped += 1
            continue

        n_kv_copied += 1

    logger.info("KV: copied=%d, renamed=%d, skipped=%d", n_kv_copied, n_kv_renamed, n_kv_skipped)

    n_tensors_kept = 0
    n_tensors_dropped = 0

    keep_top_level = {"token_embd.weight", "output.weight", "output_norm.weight"}

    for t in reader.tensors:
        name = t.name if isinstance(t.name, str) else t.name.decode("utf-8")
        keep = False

        if name in keep_top_level:
            keep = True
        elif name.startswith("blk."):
            try:
                blk_idx = int(name.split(".", 2)[1])
            except (IndexError, ValueError):
                logger.warning("unparseable block tensor name: %s; dropping", name)
                blk_idx = -1
            if blk_idx >= n_main:
                keep = True

        if not keep:
            n_tensors_dropped += 1
            continue

        # For unquantized tensors (F16/F32/...) data already has the native
        # dtype + correct shape -> let writer auto-detect.
        # For quantized tensors data is a uint8 byte buffer; writer expects the
        # *byte* shape via raw_shape, plus the original quant type via
        # raw_dtype, and will convert back internally.
        unquant = {
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.F64,
            GGMLQuantizationType.BF16,
            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32,
            GGMLQuantizationType.I64,
        }
        data = np.array(t.data, copy=False)
        if t.tensor_type in unquant:
            writer.add_tensor(name, data)
        else:
            writer.add_tensor(name, data, raw_shape=list(data.shape), raw_dtype=t.tensor_type)
        n_tensors_kept += 1
        logger.debug("kept tensor: %s quant_shape=%s byte_shape=%s dtype=%s",
                     name, list(t.shape), list(data.shape), t.tensor_type.name)

    logger.info("tensors: kept=%d, dropped=%d", n_tensors_kept, n_tensors_dropped)

    if n_tensors_kept == 0:
        logger.error("no tensors kept; aborting (output not written)")
        return 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_path = args.output.resolve()
    logger.info("wrote standalone NextN GGUF: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
