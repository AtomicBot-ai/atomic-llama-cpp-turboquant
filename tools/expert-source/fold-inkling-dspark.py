#!/usr/bin/env python3
"""Build an optional DSpark sidecar for one specific Inkling 2 Mini GGUF."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize


def architecture(reader):
    return reader.get_field("general.architecture").contents()


def fold_output(target, draft):
    target_tensors = {tensor.name: tensor for tensor in target.tensors}
    draft_tensors = {tensor.name: tensor for tensor in draft.tensors}
    output = target_tensors.get("output.weight")
    if output is None:
        output = target_tensors["token_embd.weight"]
    hidden = draft_tensors["dspark_hidden_out.weight"]
    bias = draft_tensors["dspark_hidden_out.bias"]

    hidden_data = dequantize(hidden.data, hidden.tensor_type)
    bias_data = dequantize(bias.data, bias.tensor_type)
    embd, rank = hidden_data.shape
    width, vocab = map(int, output.shape)
    padded_vocab = draft.get_field("dflash.vocab_size").contents()
    if width != embd or bias_data.shape != (embd,) or vocab != padded_vocab:
        raise ValueError("target output and DSpark correction dimensions differ")

    # The bias column multiplies a constant one; remaining columns align the
    # augmented input to 32 elements.
    stride = (rank + 32) // 32 * 32
    projection = np.zeros((embd, stride), dtype=np.float32)
    projection[:, :rank] = hidden_data
    projection[:, rank] = bias_data
    folded = np.empty((vocab, stride), dtype=np.float16)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for first in range(0, vocab, 2048):
            rows = dequantize(output.data[first:first + 2048], output.tensor_type)
            folded[first:first + len(rows)] = rows @ projection
    if not np.isfinite(folded).all():
        raise ValueError("folded output exceeds the finite F16 range")
    return folded


def write_sidecar(draft, folded, path):
    writer = GGUFWriter(path, arch="dflash", endianess=draft.endianess)
    for field in draft.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        if field.name == "general.alignment":
            writer.add_custom_alignment(field.contents())
            continue
        field_type = field.types[0]
        subtype = field.types[-1] if field_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field_type, sub_type=subtype)
    for tensor in draft.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type,
                          tensor_endianess=draft.endianess)
    writer.add_tensor("dspark_folded_output.weight", folded)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Exact Inkling 2 Mini GGUF used at inference")
    parser.add_argument("draft", type=Path, help="Original Inkling 2 Mini DSpark GGUF")
    parser.add_argument("output", type=Path, help="New folded DSpark GGUF")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    target = GGUFReader(args.target)
    draft = GGUFReader(args.draft)
    if architecture(target) != "inkling2_loop" or architecture(draft) != "dflash":
        parser.error("expected an Inkling 2 Mini target and an Inkling 2 Mini DSpark draft")
    if any(tensor.name == "dspark_folded_output.weight" for tensor in draft.tensors):
        parser.error("draft already has a folded output")

    folded = fold_output(target, draft)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        temporary = Path(directory) / args.output.name
        write_sidecar(draft, folded, temporary)
        args.output.hardlink_to(temporary)
    print(f"Wrote {args.output} ({folded.nbytes / 2**20:.2f} MiB of derived weights)")


if __name__ == "__main__":
    main()
