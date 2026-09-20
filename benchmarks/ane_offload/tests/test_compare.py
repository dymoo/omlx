import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ane_compare", Path(__file__).parents[1] / "compare.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run(tps, cached=None):
    return {
        "corpus_sha256": "same",
        "request_parameters": {"max_tokens": 100},
        "groups": [
            {
                "concurrency": 1,
                "summary": {"aggregate_output_tps": tps, "ttft_seconds": {"p95": 1}},
                "rows": [
                    {
                        "id": "coding",
                        "repeat": 0,
                        "status": "ok",
                        "text_sha256": "text",
                        "cached_tokens": cached,
                    }
                ],
            }
        ],
    }


class CompareTests(unittest.TestCase):
    def test_speedup_does_not_establish_cache_or_quality(self):
        result = module.compare(run(10), run(15))[0]
        self.assertEqual(result["aggregate_output_speedup"], 1.5)
        self.assertFalse(result["cache_counts_match"])
        self.assertFalse(result["coding_correctness_evaluated"])

    def test_changed_corpus_cannot_be_compared(self):
        candidate = run(15)
        candidate["corpus_sha256"] = "different"
        with self.assertRaises(ValueError):
            module.compare(run(10), candidate)
