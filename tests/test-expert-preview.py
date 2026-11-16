#!/usr/bin/env python3
"""Targeted binding, ranking and bounded-cache checks for the P07 evaluator."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("preview", Path(__file__).resolve().parents[1] / "tools/expert-source/evaluate-preview.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PreviewTests(unittest.TestCase):
    def fixture(self, root, capacity=240):
        capture, demand = root / "capture.jsonl", root / "demand.atet"
        binary = bytearray()
        routers, tensors, rows = [], [], []

        def data(value, **extra):
            a = np.asarray(value, dtype="<f4")
            meta = dict(offset=len(binary), bytes=a.nbytes, **extra)
            binary.extend(a.tobytes())
            return meta

        for layer in range(4):
            weight = data(np.arange(16).reshape(8, 2), type="f32", input=2, output=8)
            routers.append(dict(layer=layer, weight=weight, bias=None))
            for i, name in enumerate(("gate", "up", "down")):
                tensors.append(dict(index=layer * 3 + i, layer=layer, name=f"ffn_{name}_exps.weight", payload=8, experts=8))
        for layer in range(4):
            rows.append(dict(call=0, microbatch=0, sequence=0, layer=layer, projection=layer * 3,
                             positions=[0, 1], ids=[[0, 1], [2, 3]], **data([[1, 0], [0, 1]], input=2)))
        header = dict(format="ATP2", architecture="qwen35moe", first_layer=0, last_layer=3,
                      sequence=0, output_policy="last_row_only", routers=routers, tensors=tensors)
        capture.write_text("\n".join(json.dumps(x) for x in [header] + rows) + "\n")
        Path(str(capture) + ".bin").write_bytes(binary)
        cache, lines = p.Cache("lru"), ["ATET3", "C 0 0 0 0 0 0 0 0"]
        for t in range(12):
            lines.append(f"P {t} 0 2 0 0 0 0 0")
            for expert in range(4):
                hit = cache.access((t, expert), 8, 8, capacity, "prefill", True)
                lines.append(f"A {t} {expert} 0 8 8 {capacity} {int(hit)} 8")
                cache.pins[(t, expert)] = 1
            for expert in range(4):
                lines.append(f"R {t} {expert} 0 0 0 0 0 0")
                del cache.pins[(t, expert)]
        demand.write_text("\n".join(lines) + "\n")
        return capture, demand

    def test_rank_first_global_dedup_and_bias(self):
        self.assertEqual(list(p.candidates([[7, 4, 2, 0], [7, 5, 3, 1]]))[:4], [7, 4, 5, 2])
        scores = np.zeros((1, 8), dtype=np.float32)
        self.assertEqual(p.rank(scores)[0], [0, 1, 2, 3])
        bias = np.array([0, 0, 0, 0, 4, 3, 2, 1], dtype=np.float32)
        self.assertEqual(p.rank(scores, bias)[0], [4, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            p.rank(np.array([[float("nan")]]))

    def test_exact_baseline_waste_oracle_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture, demand = self.fixture(Path(tmp))
            result = p.evaluate(capture, demand)
            baseline, predicted, oracle = (result["replay"][key] for key in ("baseline", "predicted", "oracle"))
            self.assertEqual(baseline["total"]["read_bytes"], 12 * 4 * 8)
            self.assertGreater(predicted["speculative"]["prefill"]["unused_bytes"], 0)
            self.assertGreater(predicted["extra_total_read_bytes"], 0)
            self.assertGreater(oracle["speculative"]["prefill"]["useful_bytes"], 0)
            for replay in (predicted, oracle):
                self.assertLessEqual(replay["max_optional_slices"], 12)
                self.assertLessEqual(replay["peak_cache_and_staging"], replay["capacity"])
                self.assertEqual(replay["speculative_read_bytes"], sum(s["useful_bytes"] + s["unused_bytes"] for s in replay["speculative"].values()))
            capture, demand = self.fixture(Path(tmp), capacity=80)
            replay = p.evaluate(capture, demand)["replay"]["predicted"]
            self.assertEqual(replay["max_optional_slices"], 0)
            self.assertGreater(replay["speculative"]["prefill"]["rejected_budget"], 0)

    def test_reject_stale_join_duplicate_ids_and_truncated_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field, value, error in (("projection", 1, "cursor"), ("ids", [[0, 0], [2, 3]], "IDs"),
                                         ("positions", [3, 4], "prefill")):
                capture, demand = self.fixture(root)
                records = [json.loads(line) for line in capture.read_text().splitlines()]
                records[1][field] = value
                capture.write_text("\n".join(map(json.dumps, records)))
                with self.assertRaisesRegex(ValueError, error):
                    p.evaluate(capture, demand)
            capture, demand = self.fixture(root)
            binary = Path(str(capture) + ".bin")
            binary.write_bytes(binary.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "tensor range"):
                p.evaluate(capture, demand)

    def test_reject_mismatched_demand_missing_layer_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture, demand = self.fixture(root)
            demand.write_text(demand.read_text().replace("A 0 0", "A 0 7", 1))
            with self.assertRaisesRegex(ValueError, "disagree"):
                p.evaluate(capture, demand)
            capture, demand = self.fixture(root)
            lines = capture.read_text().splitlines()
            del lines[2]
            capture.write_text("\n".join(lines))
            with self.assertRaises(ValueError):
                p.evaluate(capture, demand)
            capture, demand = self.fixture(root)
            demand.write_text(demand.read_text() + "R 0 0 0 0 0 0 0 0\n")
            with self.assertRaisesRegex(ValueError, "release without"):
                p.evaluate(capture, demand)


if __name__ == "__main__":
    unittest.main()
