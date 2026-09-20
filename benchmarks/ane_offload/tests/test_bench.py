import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ane_bench", Path(__file__).parents[1] / "bench.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Response:
    def __init__(self, lines):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class Client:
    def __init__(self, lines):
        self.lines = lines

    def stream(self, *args, **kwargs):
        return Response(self.lines)


def event(data):
    return ["data: " + json.dumps(data), ""]


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_usage_and_done(self):
        lines = [": keepalive", ""]
        lines += event({"choices": [{"delta": {"role": "assistant"}}]})
        lines += event({"choices": [{"delta": {"reasoning_content": "think"}}]})
        lines += event(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        )
        lines += event(
            {
                "choices": [],
                "usage": {
                    "completion_tokens": 7,
                    "prompt_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 8},
                },
            }
        )
        lines += ["data: [DONE]", ""]
        row = await bench.stream_request(Client(lines), "http://localhost", {})
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["completion_tokens"], 7)
        self.assertEqual(row["cached_tokens"], 8)
        self.assertLessEqual(row["ttft_seconds"], row["seconds"])

    async def test_truncated_stream_is_failure_even_with_finish(self):
        lines = event(
            {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}
        )
        row = await bench.stream_request(Client(lines), "http://localhost", {})
        self.assertEqual(row["status"], "error")

    async def test_provider_error_is_failure(self):
        row = await bench.stream_request(
            Client(event({"error": {"message": "failed"}})), "", {}
        )
        self.assertEqual(row["status"], "error")

    async def test_no_usage_is_unknown_not_zero(self):
        lines = event(
            {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}
        )
        row = await bench.stream_request(Client(lines + ["data: [DONE]", ""]), "", {})
        self.assertEqual(row["status"], "ok")
        self.assertIsNone(row["completion_tokens"])
        self.assertIsNone(bench.summarize([row], 1)["aggregate_output_tps"])

    async def test_chunk_boundaries_do_not_change_text_hash(self):
        hashes = []
        for chunks in (["abc"], ["a", "bc"]):
            lines = []
            for text in chunks:
                lines += event({"choices": [{"delta": {"content": text}}]})
            lines += event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            lines += ["data: [DONE]", ""]
            hashes.append(
                (await bench.stream_request(Client(lines), "", {}))["text_sha256"]
            )
        self.assertEqual(*hashes)

    async def test_multiline_sse_and_usage_without_trailing_blank(self):
        lines = [
            'data: {"choices":',
            'data: [{"delta":{"content":"a"},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
        ]
        row = await bench.stream_request(Client(lines), "", {})
        self.assertEqual(row["status"], "ok")


class MetricsTests(unittest.TestCase):
    def test_aggregate_is_wall_clock_not_sum_of_request_tps(self):
        rows = [
            {"status": "ok", "completion_tokens": n, "ttft_seconds": 0.1, "seconds": 2}
            for n in [10, 20]
        ]
        self.assertEqual(bench.summarize(rows, 2)["aggregate_output_tps"], 15)
        rows.append({"status": "error"})
        self.assertIsNone(bench.summarize(rows, 2)["aggregate_output_tps"])

    def test_empty_and_small_distributions(self):
        self.assertIsNone(bench.percentile([], 0.95))
        self.assertEqual(bench.percentile([1, 9, 3], 0.95), 9)


if __name__ == "__main__":
    unittest.main()
