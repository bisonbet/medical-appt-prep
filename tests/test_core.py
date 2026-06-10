from __future__ import annotations

import unittest
from unittest import mock

import app
import config_loader
from src.model_catalog import (
    get_default_model_preset_id,
    get_model_preset_choices,
    resolve_model_settings,
)
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
        with (
            mock.patch("app.get_model", return_value=fake_model),
            mock.patch("app.is_ollama_model_available", return_value=True),
        ):
            result = app.run_inference(
                "Headache behind eyes for three days",
                "Worse in the morning",
                "Vitamin D3 - 2000 IU daily",
                "medgemma-4b",
                "4096",
                "0.3",
            )

        self.assertEqual(fake_model.calls, 1)
        self.assertEqual(result, ("One", "Two", "Three"))

    def test_run_inference_resolves_selected_model_preset(self):
        class FakeModel:
            def generate_report(self, _prompt: str) -> str:
                return "TIMELINE:\nOne\nQUESTIONS:\nTwo\nRELEVANT_INFO:\nThree"

        with (
            mock.patch("app.get_model", return_value=FakeModel()) as get_model_mock,
            mock.patch("app.is_ollama_model_available", return_value=True),
        ):
            app.run_inference(
                "Headache behind eyes for three days",
                "",
                "",
                "medgemma-27b",
                "16384",
                "0.5",
            )

        model_cfg = get_model_mock.call_args.args[0]["model"]
        self.assertEqual(model_cfg["selected_preset"], "medgemma-27b")
        self.assertEqual(model_cfg["name"], "medgemma:27b")
        self.assertEqual(model_cfg["context_length"], 16384)
        self.assertEqual(model_cfg["temperature"], 0.5)

    def test_run_inference_prompts_when_ollama_model_is_missing(self):
        with (
            mock.patch("app.get_model") as get_model_mock,
            mock.patch("app.is_ollama_model_available", return_value=False),
        ):
            result = app.run_inference(
                "Headache behind eyes for three days",
                "",
                "",
                "medgemma-27b",
                "4096",
                "0.3",
            )

        self.assertFalse(get_model_mock.called)
        self.assertIn("medgemma:27b is not downloaded", result[0])
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[1], result[2])

    def test_download_selected_model_pulls_ollama_model(self):
        with mock.patch("app.pull_ollama_model", return_value="success") as pull_mock:
            status, button_update = app.download_selected_model("medgemma-27b")

        pull_mock.assert_called_once()
        self.assertIn("medgemma:27b is available locally", status)
        self.assertFalse(button_update["visible"])


class ModelCatalogTests(unittest.TestCase):
    def test_model_name_env_override_uses_custom_model_without_preset(self):
        with mock.patch.dict("os.environ", {"MODEL_NAME": "custom-med-model"}, clear=True):
            loaded_settings = config_loader.load_settings()

        self.assertEqual(loaded_settings["model"]["name"], "custom-med-model")
        self.assertNotIn("selected_preset", loaded_settings["model"])

    def test_model_preset_choices_are_filtered_by_backend(self):
        settings = {
            "model": {
                "backend": "ollama",
                "selected_preset": "medgemma-27b",
                "presets": [
                    {
                        "id": "medgemma-4b",
                        "label": "MedGemma 1.5 4B",
                        "backends": {"ollama": {"name": "medgemma1.5:4b"}},
                    },
                    {
                        "id": "hf-only",
                        "label": "HF Only",
                        "backends": {"hf_transformers": {"name": "example/model"}},
                    },
                ],
            }
        }

        self.assertEqual(get_default_model_preset_id(settings), "medgemma-4b")
        self.assertEqual(get_model_preset_choices(settings), [("MedGemma 1.5 4B", "medgemma-4b")])

    def test_resolve_model_settings_uses_backend_specific_name(self):
        settings = {
            "model": {
                "backend": "hf_transformers",
                "selected_preset": "medgemma-4b",
                "name": "fallback",
                "presets": [
                    {
                        "id": "medgemma-27b",
                        "label": "MedGemma 27B",
                        "backends": {
                            "ollama": {"name": "medgemma:27b"},
                            "hf_transformers": {"name": "google/medgemma-27b-it"},
                        },
                    }
                ],
            }
        }

        resolved = resolve_model_settings(settings, "medgemma-27b")

        self.assertEqual(resolved["model"]["selected_preset"], "medgemma-27b")
        self.assertEqual(resolved["model"]["name"], "google/medgemma-27b-it")


class BackendFactoryTests(unittest.TestCase):
    def tearDown(self):
        _get_model_cached.cache_clear()

    def test_factory_selects_ollama(self):
        model = get_model({"model": {"backend": "ollama", "name": "medgemma1.5:4b"}})
        self.assertEqual(model.model_name, "medgemma1.5:4b")

    def test_factory_selects_hf_transformers(self):
        _get_model_cached.cache_clear()
        with mock.patch("src.model.HuggingFaceTransformersModel") as model_cls:
            get_model({"model": {"backend": "hf_transformers", "name": "google/medgemma-1.5-4b-it"}})

        self.assertEqual(model_cls.call_args.kwargs["model_name"], "google/medgemma-1.5-4b-it")

    def test_factory_resolves_selected_preset(self):
        model = get_model(
            {
                "model": {
                    "backend": "ollama",
                    "selected_preset": "medgemma-27b",
                    "presets": [
                        {
                            "id": "medgemma-27b",
                            "label": "MedGemma 27B",
                            "backends": {"ollama": {"name": "medgemma:27b"}},
                        }
                    ],
                }
            }
        )

        self.assertEqual(model.model_name, "medgemma:27b")

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
