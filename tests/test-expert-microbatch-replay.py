import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "microbatch_replay", Path(__file__).resolve().parents[1] / "tools/expert-source/replay-microbatches.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class MicrobatchReplayTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / "trace.tensors"
        manifest.write_text("\n".join(f"{layer * 3 + role} blk.{layer}.ffn_{name}_exps.weight"
                                     for layer in range(2) for role, name in enumerate(("gate", "up", "down"))))
        lines = ["ATET3", "C 0 0 0 8 8 0 0 0"]
        # Three slots + staging fit. Six weights do not survive a full traversal.
        for batch in range(4):
            for tensor in range(6):
                lines += [f"P {tensor} 0 8 0 0 0 0 0",
                          f"A {tensor} 0 0 5 10 35 0 10", f"R {tensor} 0 0 0 0 0 0 0"]
        path = root / "trace"
        path.write_text("\n".join(lines) + "\n")
        return path, manifest

    def test_exact_control_reuse_and_server_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.fixture(Path(directory))
            batches, meta = module.load_trace(path, manifest, 32)
            control = module.simulate((p for batch in batches for p in batch), "lru", exact=True)
            self.assertEqual(control["total"]["read_bytes"], 120)
            ordered = list(module.schedule(batches, 4, 16))
            # A four-microbatch request must still stop at two-microbatch server batches.
            self.assertEqual([p.tensor for p in ordered[:12]], [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
            candidate = module.simulate(ordered, "lru", 35)
            self.assertEqual(candidate["total"]["read_bytes"], 60)
            self.assertLessEqual(candidate["peak_cache_and_staging"], 35)
            # With less than three slots, layer reuse disappears; reserve is real.
            reduced = module.simulate(ordered, "lru", 25)
            self.assertEqual(reduced["total"]["read_bytes"], 120)
            self.assertEqual(meta["acquisitions"], 24)

    def test_reject_observation_mismatch_and_unreleased_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.fixture(Path(directory))
            original = path.read_text()
            path.write_text(original.replace("A 0 0 0 5 10 35 0 10", "A 0 0 0 5 10 35 1 10", 1))
            batches, _ = module.load_trace(path, manifest, 32)
            with self.assertRaisesRegex(ValueError, "diverged"):
                module.simulate((p for batch in batches for p in batch), "lru", exact=True)
            path.write_text(original.replace("R 0 0 0 0 0 0 0 0\n", "", 1))
            with self.assertRaisesRegex(ValueError, "outstanding lease"):
                module.load_trace(path, manifest, 32)

    def test_reject_decode_suffix_and_missing_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "token count mismatch"):
                module.load_trace(path, manifest, 31)
            lines = path.read_text().splitlines()
            # Drop the first layer from the second microbatch, not the prunable tail.
            del lines[20:29]
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "unexpected layer order"):
                module.load_trace(path, manifest, 32)


if __name__ == "__main__":
    unittest.main()
