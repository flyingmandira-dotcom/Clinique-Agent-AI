import unittest
import json
from pydantic import BaseModel
from typing import List, Optional

from backend.agents.llm import (
    is_rate_limited,
    get_retry_after,
    get_backoff_time,
    _clean_and_validate_json,
    call_llm,
)


class DummySchema(BaseModel):
    summary: str
    risk_level: str
    items: Optional[List[str]] = []


class TestLLMResilience(unittest.TestCase):

    def test_rate_limit_detection(self):
        # HTTP 429 is always rate-limited
        self.assertTrue(is_rate_limited(429, {}))
        self.assertTrue(is_rate_limited(429, None))

        # HTTP 503 with Retry-After header
        self.assertTrue(is_rate_limited(503, {"Retry-After": "30"}))
        self.assertTrue(is_rate_limited(504, {"retry-after": "15"}))

        # Normal errors or 503 without Retry-After
        self.assertFalse(is_rate_limited(500, {}))
        self.assertFalse(is_rate_limited(400, {}))
        self.assertFalse(is_rate_limited(503, {}))

    def test_get_retry_after(self):
        self.assertEqual(get_retry_after({"Retry-After": "12.5"}), 12.5)
        self.assertEqual(get_retry_after({"retry-after": "45"}), 45.0)
        self.assertIsNone(get_retry_after({"other-header": "value"}))
        self.assertIsNone(get_retry_after(None))
        self.assertIsNone(get_retry_after({"Retry-After": "invalid"}))

    def test_exponential_backoff_and_jitter(self):
        # Attempt 0: base 1.0 -> >= 1.0
        t0 = get_backoff_time(0, base=1.0, jitter=True)
        self.assertGreaterEqual(t0, 1.0)

        # Exponential growth: attempt 1 > attempt 0
        t1 = get_backoff_time(1, base=1.0, jitter=False)
        self.assertEqual(t1, 2.0)

        t2 = get_backoff_time(2, base=1.0, jitter=False)
        self.assertEqual(t2, 4.0)

        t3 = get_backoff_time(3, base=1.0, jitter=False)
        self.assertEqual(t3, 8.0)

        # Jitter variation
        sampled = [get_backoff_time(1, base=1.0, jitter=True) for _ in range(5)]
        self.assertTrue(all(val >= 2.0 for val in sampled))

    def test_clean_and_validate_json_with_markdown(self):
        # Markdown fence with 'json'
        markdown_json = "```json\n{\n  \"summary\": \"All clear\",\n  \"risk_level\": \"SAFE\",\n  \"items\": [\"aspirin\"]\n}\n```"
        cleaned = _clean_and_validate_json(markdown_json, DummySchema)
        data = json.loads(cleaned)
        self.assertEqual(data["summary"], "All clear")
        self.assertEqual(data["risk_level"], "SAFE")
        self.assertEqual(data["items"], ["aspirin"])

    def test_clean_and_validate_json_with_stray_text(self):
        # Stray text before and after the JSON
        messy_response = "Here is the resulting clinical assessment:\n```\n{\"summary\": \"Moderate risk\", \"risk_level\": \"WARNING\"}\n```\nPlease let me know if you need anything else."
        cleaned = _clean_and_validate_json(messy_response, DummySchema)
        data = json.loads(cleaned)
        self.assertEqual(data["risk_level"], "WARNING")
        self.assertEqual(data["summary"], "Moderate risk")

    def test_clean_and_validate_json_invalid_schema(self):
        # Missing required field 'summary'
        invalid_json = "{\"risk_level\": \"WARNING\"}"
        with self.assertRaises(ValueError):
            _clean_and_validate_json(invalid_json, DummySchema)

    def test_call_llm_simulation_fallback(self):
        # In testing or offline scenario, fallback_to_simulation returns empty text and simulation provider
        resp_text, provider = call_llm(
            prompt="Test prompt",
            system_instruction="Test instruction",
            fallback_to_simulation=True
        )
        self.assertTrue(any(provider.startswith(prefix) for prefix in ["gemini", "groq", "openrouter", "simulation"]))
        if provider.startswith("simulation"):
            self.assertEqual(resp_text, "")


if __name__ == "__main__":
    unittest.main()
