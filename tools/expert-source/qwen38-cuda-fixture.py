#!/usr/bin/env python3
"""Extract original packed bytes for the explicitly invoked CUDA operation probe.

No dequantization or requantization. Does not create a runnable model.
Run with PYTHONPATH=gguf-py. Output must not already exist.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('shards', nargs='+', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--include-token-embedding', action='store_true')
args = parser.parse_args()
if args.output.exists():
    parser.error('output already exists')
writer = GGUFWriter(args.output, 'qwen4exp')
seen = set()
manifest = []
for path in args.shards:
    reader = GGUFReader(path)
    for tensor in reader.tensors:
        ple = tensor.name == 'per_layer_token_embd.weight'
        token = args.include_token_embedding and tensor.name == 'token_embd.weight'
        row_table = ple or token
        routed = '_exps.weight' in tensor.name
        if not (row_table or routed):
            continue
        key = (tensor.tensor_type.name, tuple(int(n) for n in tensor.shape[:2]))
        if routed and key in seen:
            continue
        seen.add(key)
        count = int(tensor.shape[1 if row_table else 2])
        indices = [0, 1, count // 2, count - 1] if row_table else [0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 383, 511]
        if len(set(indices)) != len(indices) or max(indices) >= count:
            raise ValueError("unexpected fixture geometry")
        block_bytes = int(tensor.n_bytes) // count
        raw = np.empty((len(indices), block_bytes), dtype=np.uint8)
        with path.open('rb') as src:
            for i, index in enumerate(indices):
                src.seek(int(tensor.data_offset) + index * block_bytes)
                payload = src.read(block_bytes)
                if len(payload) != block_bytes:
                    raise ValueError('short packed read')
                raw[i] = np.frombuffer(payload, dtype=np.uint8)
        # Quantized writer infers logical columns from the packed last axis.
        raw = raw.reshape((len(indices), block_bytes) if row_table else
                          (len(indices), int(tensor.shape[1]), block_bytes // int(tensor.shape[1])))
        writer.add_tensor(tensor.name, raw, raw_dtype=tensor.tensor_type)
        manifest.append(dict(name=tensor.name, type=tensor.tensor_type.name,
                             original_shape=[int(n) for n in tensor.shape],
                             original_indices=indices, bytes=int(raw.nbytes),
                             sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
if len(manifest) != 8 + int(args.include_token_embedding):
    raise ValueError(f'unexpected fixture tensor count: {len(manifest)}')
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(args.output, args.output.stat().st_size, 'bytes')
