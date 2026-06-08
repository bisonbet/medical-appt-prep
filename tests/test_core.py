from __future__ import annotations

import unittest
from unittest import mock

import app
from src.model import _get_model_cached, get_model
from src.processor import REPORT_SECTION_FALLBACK, parse_prep_report
from src.prompts import build_prep_report_prompt


class PrepReportTests(unittest.TestCase):
    def test_combined_prompt_contains_required_sections(self):
        prompt = build_prep_report_prompt(
            "Headache for three days",
            "Worse in the morning",
            "Lisinopril (Oral Pill) - 10 mg - once daily",
        )

        self.assertIn("TIMELINE:", prompt)
        self.assertIn("QUESTIONS:", prompt)
        self.assertIn("RELEVANT_INFO:", prompt)
        self.assertIn("Headache for three days", prompt)
        self.assertIn("Lisinopril", prompt)

    def test_parse_prep_report_extracts_sections(self):
        timeline, questions, relevant = parse_prep_report(
            """TIMELINE:
- Day 1 headache

QUESTIONS:
1. What could be contributing?

RELEVANT_INFO:
This is informational only."""
        )

        self.assertEqual(timeline, "- Day 1 headache")
        self.assertEqual(questions, "1. What could be contributing?")
        self.assertEqual(relevant, "This is informational only.")

    def test_parse_prep_report_falls_back_on_malformed_output(self):
        timeline, questions, relevant = parse_prep_report("A single untagged report")

        self.assertEqual(timeline, REPORT_SECTION_FALLBACK)
        self.assertEqual(questions, REPORT_SECTION_FALLBACK)
        self.assertEqual(relevant, "A single untagged report")

    def test_run_inference_uses_one_report_call(self):
        class FakeModel:
            def __init__(self):
                self.calls = 0

            def generate_report(self, _prompt: str) -> str:
                self.calls += 1
                return "TIMELINE:\nOne\nQUESTIONS:\nTwo\nRELEVANT_INFO:\nThree"

        fake_model = FakeModel()
        with mock.patch("app.get_model", return_value=fake_model):
            result = app.run_inference(
                "Headache behind eyes for three days",
                "Worse in the morning",
                "Vitamin D3 - 2000 IU daily",
            )

        self.assertEqual(fake_model.calls, 1)
        self.assertEqual(result, ("One", "Two", "Three"))


class BackendFactoryTests(unittest.TestCase):
    def tearDown(self):
        _get_model_cached.cache_clear()

    def test_factory_selects_ollama(self):
        model = get_model({"model": {"backend": "ollama", "name": "medgemma1.5"}})
        self.assertEqual(model.model_name, "medgemma1.5")

    def test_factory_selects_hf_transformers(self):
        _get_model_cached.cache_clear()
        with mock.patch("src.model.HuggingFaceTransformersModel") as model_cls:
            get_model({"model": {"backend": "hf_transformers", "name": "google/medgemma-1.5-4b-it"}})

        self.assertEqual(model_cls.call_args.kwargs["model_name"], "google/medgemma-1.5-4b-it")

    def test_factory_selects_openai_compatible(self):
        model = get_model(
            {
                "model": {
                    "backend": "openai_compatible",
                    "name": "google/medgemma-1.5-4b-it",
                    "openai_compatible_base_url": "https://example.test",
                    "openai_compatible_api_key": "secret",
                }
            }
        )

        self.assertEqual(model.model_name, "google/medgemma-1.5-4b-it")
        self.assertEqual(model.base_url, "https://example.test")


if __name__ == "__main__":
    unittest.main()
