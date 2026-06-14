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
from src.processor import (
    REPORT_SECTION_FALLBACK,
    SECTION_OUTPUT_FALLBACK,
    clean_section_output,
    parse_prep_report,
)
from src.prompts import (
    build_prep_report_prompt,
    build_questions_prompt,
    build_relevant_info_prompt,
    build_timeline_prompt,
)
from src.medications import filter_medication_choices


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
        self.assertIn("Do not write thoughts", prompt)
        self.assertIn("Aim for 220-360 words total", prompt)

    def test_section_prompts_do_not_request_analysis(self):
        prompts = [
            build_timeline_prompt("Headache for three days", "", ""),
            build_questions_prompt("Headache for three days", "", ""),
            build_relevant_info_prompt("Headache for three days", "", ""),
        ]

        for prompt in prompts:
            self.assertIn("Do not write thoughts", prompt)
            self.assertIn("Do not diagnose", prompt)

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
        self.assertEqual(relevant, REPORT_SECTION_FALLBACK)

    def test_parse_prep_report_strips_preamble_before_sections(self):
        timeline, questions, relevant = parse_prep_report(
            """Here is the report:

TIMELINE:
- Day 1 headache

QUESTIONS:
1. What should I ask?

RELEVANT_INFO:
- This is informational only."""
        )

        self.assertEqual(timeline, "- Day 1 headache")
        self.assertEqual(questions, "1. What should I ask?")
        self.assertEqual(relevant, "- This is informational only.")

    def test_parse_prep_report_strips_think_block_before_sections(self):
        timeline, questions, relevant = parse_prep_report(
            """<think>I should plan the sections first.</think>

TIMELINE:
- Day 1 headache

QUESTIONS:
1. What should I ask?

RELEVANT_INFO:
- This is informational only."""
        )

        self.assertEqual(timeline, "- Day 1 headache")
        self.assertEqual(questions, "1. What should I ask?")
        self.assertEqual(relevant, "- This is informational only.")

    def test_parse_prep_report_does_not_surface_reasoning_only_output(self):
        reasoning = """thought The user wants me to prepare a report.

Patient Information Breakdown:
- Headache for two days.

Plan:
TIMELINE:
- I will organize symptoms.

QUESTIONS:
1. I will generate questions."""
        timeline, questions, relevant = parse_prep_report(reasoning)

        self.assertEqual(timeline, REPORT_SECTION_FALLBACK)
        self.assertEqual(questions, REPORT_SECTION_FALLBACK)
        self.assertEqual(relevant, REPORT_SECTION_FALLBACK)

    def test_parse_prep_report_uses_final_report_after_reasoning(self):
        timeline, questions, relevant = parse_prep_report(
            """thought I should plan first.

Plan:
TIMELINE:
- Draft plan

Final:
TIMELINE:
- Day 1 headache

QUESTIONS:
1. What should I ask?

RELEVANT_INFO:
- This is informational only."""
        )

        self.assertEqual(timeline, "- Day 1 headache")
        self.assertEqual(questions, "1. What should I ask?")
        self.assertEqual(relevant, "- This is informational only.")

    def test_clean_section_output_does_not_surface_reasoning_only_output(self):
        self.assertEqual(
            clean_section_output("thought I should plan the answer first."),
            SECTION_OUTPUT_FALLBACK,
        )

    def test_clean_section_output_does_not_surface_unused_token_reasoning(self):
        self.assertEqual(
            clean_section_output("<unused94> thought\nI should plan first.\n- Draft"),
            SECTION_OUTPUT_FALLBACK,
        )

    def test_clean_section_output_uses_final_answer_after_reasoning(self):
        self.assertEqual(
            clean_section_output("thought I should plan first.\n\nFinal:\n- Day 1 headache"),
            "- Day 1 headache",
        )

    def test_clean_section_output_strips_short_intro_before_list(self):
        self.assertEqual(
            clean_section_output("Okay, here are some questions:\n\n1. What should I ask?"),
            "1. What should I ask?",
        )

    def test_clean_section_output_drops_truncated_final_line(self):
        self.assertEqual(
            clean_section_output("- Bring medication list.\n- Tell your doctor about allergies"),
            "- Bring medication list.",
        )

    def test_clean_section_output_limits_list_items(self):
        self.assertEqual(
            clean_section_output("1. One\n2. Two\n3. Three", max_items=2),
            "1. One\n2. Two",
        )

    def test_clean_section_output_adds_required_suffix(self):
        self.assertEqual(
            clean_section_output("- Bring medication list.", required_suffix="- Informational only."),
            "- Bring medication list.\n- Informational only.",
        )

    def test_run_inference_uses_three_section_calls(self):
        class FakeModel:
            def __init__(self):
                self.calls = 0
                self.responses = ["Timeline", "Questions", "Relevant info"]

            def generate_report(self, _prompt: str) -> str:
                response = self.responses[self.calls]
                self.calls += 1
                return response

        fake_model = FakeModel()
        with (
            mock.patch("app.get_model", return_value=fake_model),
            mock.patch("app.is_ollama_model_available", return_value=True),
        ):
            result = app.run_inference(
                "Headache behind eyes for three days",
                "Worse in the morning",
                "Vitamin D3 - 2000 IU daily",
            )

        self.assertEqual(fake_model.calls, 3)
        self.assertEqual(
            result,
            (
                "Timeline",
                "Questions",
                "Relevant info\n"
                "- This is informational only and not a substitute for "
                "professional medical advice.",
            ),
        )

    def test_run_inference_uses_hidden_model_defaults(self):
        class FakeModel:
            def __init__(self):
                self.calls = 0

            def generate_report(self, _prompt: str) -> str:
                self.calls += 1
                return f"Section {self.calls}"

        with (
            mock.patch("app.get_model", return_value=FakeModel()) as get_model_mock,
            mock.patch("app.is_ollama_model_available", return_value=True),
        ):
            app.run_inference(
                "Headache behind eyes for three days",
                "",
                "",
            )

        model_cfg = get_model_mock.call_args.args[0]["model"]
        self.assertEqual(model_cfg["selected_preset"], "medgemma-4b")
        self.assertEqual(model_cfg["name"], "medgemma1.5:4b")
        self.assertEqual(model_cfg["context_length"], 8192)
        self.assertEqual(model_cfg["temperature"], 0.3)

    def test_timeline_guard_replaces_unreported_today_timing(self):
        timeline = "* **Sore throat:** Started today."

        result = app._avoid_unreported_today_timing(timeline, "Sore throat", "")

        self.assertEqual(result, "* **Sore throat:** Timing not specified.")

    def test_timeline_guard_keeps_reported_today_timing(self):
        timeline = "* **Sore throat:** Started today."

        result = app._avoid_unreported_today_timing(timeline, "Sore throat started today", "")

        self.assertEqual(result, timeline)

    def test_run_inference_prompts_when_ollama_model_is_missing(self):
        with (
            mock.patch("app.get_model") as get_model_mock,
            mock.patch("app.is_ollama_model_available", return_value=False),
        ):
            result = app.run_inference(
                "Headache behind eyes for three days",
                "",
                "",
            )

        self.assertFalse(get_model_mock.called)
        self.assertIn("medgemma1.5:4b is not downloaded", result[0])
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[1], result[2])

    def test_download_selected_model_pulls_ollama_model(self):
        with mock.patch("app.pull_ollama_model", return_value="success") as pull_mock:
            status, button_update = app.download_selected_model("medgemma-4b")

        pull_mock.assert_called_once()
        self.assertIn("medgemma1.5:4b is available locally", status)
        self.assertFalse(button_update["visible"])


class MedicationSearchTests(unittest.TestCase):
    def setUp(self):
        self.choices = [
            "Acetaminophen (Oral Pill) - 500 mg",
            "Lisinopril (Oral Pill) - 10 mg",
            "Metformin XR (Oral Pill) - 500 mg",
            "Tylenol (Oral Pill) - 325 mg",
            "Vitamin D3",
        ]

    def test_blank_query_returns_all_choices_for_scrolling(self):
        self.assertEqual(filter_medication_choices("", self.choices), self.choices)

    def test_search_is_case_insensitive(self):
        matches = filter_medication_choices("lIsi", self.choices)

        self.assertEqual(matches[0], "Lisinopril (Oral Pill) - 10 mg")

    def test_search_matches_substrings(self):
        matches = filter_medication_choices("formin", self.choices)

        self.assertEqual(matches[0], "Metformin XR (Oral Pill) - 500 mg")

    def test_search_matches_fuzzy_typos(self):
        matches = filter_medication_choices("tylenl", self.choices)

        self.assertEqual(matches[0], "Tylenol (Oral Pill) - 325 mg")

    def test_search_limit_is_applied_after_ranking(self):
        matches = filter_medication_choices("pill", self.choices, limit=2)

        self.assertEqual(len(matches), 2)
        self.assertTrue(all("Pill" in match for match in matches))

    def test_add_medication_resets_picker_to_all_choices(self):
        result = app.add_medication_entry("Lisinopril", "once daily", "", self.choices)

        self.assertEqual(result[0], "Lisinopril - once daily")
        self.assertEqual(result[3]["choices"], self.choices)
        self.assertIsNone(result[3]["value"])


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
                "selected_preset": "medgemma-4b",
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

    def test_resolve_model_settings_uses_llama_cpp_hub_config(self):
        settings = {
            "model": {
                "backend": "llama_cpp",
                "selected_preset": "medgemma-4b",
                "presets": [
                    {
                        "id": "medgemma-4b",
                        "label": "MedGemma 1.5 4B",
                        "backends": {
                            "llama_cpp": {
                                "name": "unsloth/medgemma-1.5-4b-it-GGUF",
                                "model_repo_id": "unsloth/medgemma-1.5-4b-it-GGUF",
                                "model_filename": (
                                    "medgemma-1.5-4b-it-Q4_K_M.gguf"
                                ),
                                "n_gpu_layers": -1,
                                "n_batch": 2048,
                                "n_ubatch": 1024,
                                "flash_attn": True,
                                "op_offload": True,
                                "swa_full": False,
                            }
                        },
                    }
                ],
            }
        }

        resolved = resolve_model_settings(settings, "medgemma-4b")

        self.assertEqual(resolved["model"]["selected_preset"], "medgemma-4b")
        self.assertEqual(
            resolved["model"]["model_repo_id"],
            "unsloth/medgemma-1.5-4b-it-GGUF",
        )
        self.assertEqual(resolved["model"]["n_gpu_layers"], -1)
        self.assertEqual(resolved["model"]["n_batch"], 2048)
        self.assertTrue(resolved["model"]["flash_attn"])
        self.assertFalse(resolved["model"]["swa_full"])


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

    def test_factory_selects_llama_cpp_hub_model(self):
        settings = {
            "model": {
                "backend": "llama_cpp",
                "selected_preset": "medgemma-4b",
                "context_length": 4096,
                "temperature": 0.3,
                "max_new_tokens": 768,
                "presets": [
                    {
                        "id": "medgemma-4b",
                        "label": "MedGemma 1.5 4B",
                        "backends": {
                            "llama_cpp": {
                                "model_repo_id": "unsloth/medgemma-1.5-4b-it-GGUF",
                                "model_filename": (
                                    "medgemma-1.5-4b-it-Q4_K_M.gguf"
                                ),
                                "n_gpu_layers": -1,
                                "n_batch": 2048,
                                "n_ubatch": 1024,
                                "flash_attn": True,
                                "op_offload": True,
                                "swa_full": False,
                            }
                        },
                    }
                ],
            }
        }

        with mock.patch("src.model.LlamaCppModel") as model_cls:
            get_model(settings)

        self.assertEqual(
            model_cls.call_args.kwargs["model_repo_id"],
            "unsloth/medgemma-1.5-4b-it-GGUF",
        )
        self.assertEqual(model_cls.call_args.kwargs["n_gpu_layers"], -1)
        self.assertEqual(model_cls.call_args.kwargs["n_batch"], 2048)
        self.assertEqual(model_cls.call_args.kwargs["n_ubatch"], 1024)
        self.assertTrue(model_cls.call_args.kwargs["flash_attn"])
        self.assertTrue(model_cls.call_args.kwargs["op_offload"])
        self.assertFalse(model_cls.call_args.kwargs["swa_full"])

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
