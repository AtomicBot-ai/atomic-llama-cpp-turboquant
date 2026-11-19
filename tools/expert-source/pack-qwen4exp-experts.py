#!/usr/bin/env python3
"""Build an optional, byte-exact Qwen4Exp expert pack for the Metal reader."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader


def identity(file):
    stat = os.fstat(file.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def pack(shards, output):
    with ExitStack() as files:
        inputs = [files.enter_context(shard.open("rb")) for shard in shards]
        identities = [identity(file) for file in inputs]
        tensors = {}
        for file in inputs:
            # Parse offsets from the same open file used by pread.
            reader = GGUFReader(file)
            for tensor in reader.tensors:
                if tensor.name.endswith((".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight")):
                    if tensor.name in tensors or len(tensor.shape) != 3 or not 1 < tensor.shape[2] <= 1024:
                        raise ValueError(f"invalid expert tensor {tensor.name}")
                    count = int(tensor.shape[2])
                    if tensor.n_bytes % count:
                        raise ValueError(f"invalid slice size for {tensor.name}")
                    tensors[tensor.name] = (file, tensor.data_offset, tensor.n_bytes // count, count)

        layers = [[tensors[f"blk.{layer}.ffn_{kind}_exps.weight"] for kind in ("gate", "up", "down")]
                  for layer in range(48)]
        if len(tensors) != 3 * len(layers) or any(len({item[3] for item in row}) != 1 for row in layers):
            raise ValueError("expected 48 Qwen4Exp layers with matching expert counts")
        if output.exists():
            raise FileExistsError(output)
        pending = files.enter_context(tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".part", delete=False))
        try:
            pending.write(struct.pack("<8sQ", b"ATEXPK1\0", len(shards)))
            for state in identities:
                pending.write(struct.pack("<5Q", *state))
            for layer, row in enumerate(layers):
                for expert in range(row[0][3]):
                    crc = 0
                    for file, offset, stride, _ in row:
                        data = os.pread(file.fileno(), stride, offset + expert * stride)
                        if len(data) != stride:
                            raise OSError(f"short expert read in layer {layer}")
                        pending.write(data)
                        crc = zlib.crc32(data, crc)
                    pending.write(struct.pack("<I", crc))
                print(f"Packed layer {layer + 1}/{len(layers)}", flush=True)
            if identities != [identity(file) for file in inputs]:
                raise OSError("source shard changed during packing")
            pending.flush()
            os.fsync(pending.fileno())
            pending.close()
            os.link(pending.name, output)
        finally:
            Path(pending.name).unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+", help="original GGUF shards in split order")
    args = parser.parse_args()
    pack(args.shards, args.output)
