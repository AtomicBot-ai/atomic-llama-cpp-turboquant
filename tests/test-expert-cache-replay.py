import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "replay_cache", Path(__file__).resolve().parents[1] / "tools/expert-source/replay-cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CacheReplayTests(unittest.TestCase):
    def test_prefill_frequency_cold_tail_and_pinned_candidates(self):
        cache = module.Cache("prefill_frequency")
        cache.clocks[0] = 0
        cache.access((0, 0), 5, 10, 25, "prefill")
        cache.clocks[0] = 128
        cache.access((0, 0), 5, 10, 25, "prefill")
        cache.access((0, 1), 5, 10, 25, "prefill")
        cache.access((0, 2), 5, 10, 25, "prefill")
        self.assertEqual(set(cache.entries), {(0, 0), (0, 2)})
        cache.pins[(0, 2)] = 1
        cache.access((0, 3), 5, 10, 25, "prefill")
        self.assertEqual(set(cache.entries), {(0, 2), (0, 3)})
        cache.pins[(0, 3)] = 1
        with self.assertRaisesRegex(ValueError, "no unpinned"):
            cache.evict()
        self.assertEqual(cache.used, 20)
        self.assertEqual(cache.peak_with_staging, 25)

    def test_transient_staging_evicts_even_when_payloads_fit(self):
        cache = module.Cache("lru")
        for expert in (0, 1, 2):
            self.assertFalse(cache.access((0, expert), 5, 10, 25, "decode"))
        self.assertNotIn((0, 0), cache.entries)
        self.assertEqual(cache.used, 20)
        self.assertEqual(cache.peak_with_staging, 25)
        self.assertTrue(cache.access((0, 1), 5, 10, 25, "decode"))
        self.assertEqual(cache.counts["decode"]["read_bytes"], 15)

    def test_class_lru_protects_reused_expert_from_scan(self):
        caches = [module.Cache(policy) for policy in ("lru", "class_lru")]
        results = []
        for cache in caches:
            for clock, expert in enumerate([0] * 8 + [1, 2, 0]):
                cache.clocks[0] = clock
                hit = cache.access((0, expert), 5, 10, 25, "decode")
            results.append(hit)
            self.assertLessEqual(cache.peak_with_staging, 25)
        self.assertEqual(results, [False, True])

    def test_runtime_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text("ATET1\nP 0 0 1 0 0 0 0\nA 0 0 0 5 10 25 1\n")
            with self.assertRaisesRegex(ValueError, "LRU diverged"):
                module.replay(path)

    def test_oversized_reuse_replays_physical_bytes(self):
        trace = ("ATET2\nP 0 0 1 0 0 0 0 0\n"
                 "A 0 0 0 10 20 30 0 20\n"
                 "A 0 1 0 5 20 30 0 10\n"
                 "A 0 1 0 5 20 30 1 10\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text(trace)
            result = module.replay(path)
            self.assertEqual(result['lru']['resident_bytes'], 20)
            self.assertEqual(result['lru']['total']['hits'], 1)
            self.assertEqual(result['lru']['total']['evictions'], 1)
            self.assertEqual(result['lru'], result['class_lru'])
            path.write_text(trace.replace('5 20 30 0 10', '5 10 30 0 10'))
            with self.assertRaisesRegex(ValueError, 'physical allocation diverged'):
                module.replay(path)

    def test_oversized_victim_is_freed_when_staging_cannot_fit(self):
        cache = module.Cache('lru')
        cache.access((0, 0), 5, 20, 25, 'decode', True)
        cache.access((0, 1), 10, 10, 25, 'decode', True)
        self.assertEqual(cache.used, 10)
        self.assertEqual(cache.counts['decode']['evictions'], 1)

    def test_group_leases_exclude_pinned_victims(self):
        trace = ("ATET3\nP 0 0 1 0 0 0 0 0\n"
                 "A 0 0 0 5 10 25 0 10\nA 0 1 0 5 10 25 0 10\n"
                 "R 0 1 0 0 0 0 0 0\nA 0 2 0 5 10 25 0 10\n"
                 "R 0 0 0 0 0 0 0 0\nR 0 2 0 0 0 0 0 0\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace'
            path.write_text(trace)
            result = module.replay(path)
            self.assertEqual(result['lru']['total']['evictions'], 1)
            self.assertEqual(result['lru'], result['class_lru'])
            path.write_text(trace.replace('R 0 2 0 0 0 0 0 0\n', ''))
            with self.assertRaisesRegex(ValueError, 'ended with active leases'):
                module.replay(path)


if __name__ == "__main__":
    unittest.main()
