"""
Medical Appointment Prep Assistant
Main Gradio application entry point.
"""

import os
import re
import subprocess
import time

import gradio as gr
from src.model import get_model, is_ollama_model_available, pull_ollama_model
from src.model_catalog import (
    canonical_backend,
    get_default_model_preset_id,
    resolve_model_settings,
)
from src.prompts import (
    build_questions_prompt,
    build_relevant_info_prompt,
    build_timeline_prompt,
)
from src.processor import clean_section_output, validate_inputs
from src.medications import (
    filter_medication_choices,
    load_medication_choices,
    medication_index_summary,
)
from config_loader import load_settings

settings = load_settings()
_GPU_RUNTIME_LOGGED = False
_RECENT_TIMING_RE = re.compile(
    r"\b(today|tonight|this morning|this afternoon|this evening|yesterday|last night)\b",
    flags=re.IGNORECASE,
)


DEFAULT_OUTPUT = "_Your prep report will appear here._"
APPLE_CSS_PATH = "assets/apple.css"
APPLE_THEME = gr.themes.Soft()
DEFAULT_CONTEXT_LENGTH = "8192"
DEFAULT_TEMPERATURE = "0.3"
DOWNLOAD_MODEL_BUTTON_LABEL = "Download Model"
THEME_MODE_HEAD = """
<script>
(() => {
  const storageKey = "medical-appt-prep-theme";
  const validModes = new Set(["system", "light", "dark"]);

  function currentMode() {
    const storedMode = localStorage.getItem(storageKey);
    return validModes.has(storedMode) ? storedMode : "system";
  }

  function applyTheme(mode) {
    const resolvedMode = validModes.has(mode) ? mode : "system";
    document.documentElement.dataset.themeMode = resolvedMode;
    if (resolvedMode === "system") {
      document.documentElement.dataset.theme = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
      localStorage.removeItem(storageKey);
    } else {
      document.documentElement.dataset.theme = resolvedMode;
      localStorage.setItem(storageKey, resolvedMode);
    }

    document.querySelectorAll("[data-theme-option]").forEach((button) => {
      const isSelected = button.dataset.themeOption === resolvedMode;
      button.classList.toggle("selected", isSelected);
      button.setAttribute("aria-pressed", String(isSelected));
    });
  }

  function bindThemeSelector() {
    document.querySelectorAll("[data-theme-option]").forEach((button) => {
      if (button.dataset.boundThemeSelector === "true") {
        return;
      }
      button.dataset.boundThemeSelector = "true";
      button.addEventListener("click", () => applyTheme(button.dataset.themeOption));
    });
    applyTheme(currentMode());
  }

  function startThemeMode() {
    bindThemeSelector();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (currentMode() === "system") {
        applyTheme("system");
      }
    });
    new MutationObserver(bindThemeSelector).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startThemeMode);
  } else {
    startThemeMode();
  }
})();
</script>
"""


def _context_choice_value(context_length: object) -> str:
    value = str(context_length or DEFAULT_CONTEXT_LENGTH)
    return value if value.isdigit() else DEFAULT_CONTEXT_LENGTH


def _temperature_choice_value(temperature: object) -> str:
    value = str(temperature or DEFAULT_TEMPERATURE)
    try:
        float(value)
    except ValueError:
        return DEFAULT_TEMPERATURE
    return value


def _default_model_preset_id() -> str:
    return get_default_model_preset_id(settings, settings.get("model", {}).get("backend", "ollama"))


def _model_settings_for_selection(
    model_preset_id: str = "",
    context_length: str = DEFAULT_CONTEXT_LENGTH,
    temperature: str = DEFAULT_TEMPERATURE,
) -> dict:
    model_settings = resolve_model_settings(settings, model_preset_id or _default_model_preset_id())
    model_settings.setdefault("model", {})["context_length"] = int(_context_choice_value(context_length))
    model_settings.setdefault("model", {})["temperature"] = float(_temperature_choice_value(temperature))
    return model_settings


def _ollama_model_prompt(model_name: str) -> str:
    return (
        f"**{model_name} is not downloaded.** Ask the app owner to download it with Ollama "
        f"before generating a report."
    )


def _with_zero_gpu(fn):
    """Request ZeroGPU only in hosted Spaces; keep local imports lightweight."""
    if settings.get("app", {}).get("deployment") != "huggingface":
        return fn
    try:
        import spaces  # type: ignore[import]
    except ImportError:
        return fn
    return spaces.GPU(duration=120)(fn)


def _log_gpu_runtime_once(model_cfg: dict) -> None:
    """Emit non-private GPU diagnostics while inside the generation call."""
    global _GPU_RUNTIME_LOGGED
    if _GPU_RUNTIME_LOGGED or settings.get("app", {}).get("deployment") != "huggingface":
        return
    _GPU_RUNTIME_LOGGED = True


def _avoid_unreported_today_timing(timeline: str, *source_texts: str) -> str:
    """Avoid presenting invented same-day timing when the user did not report it."""
    if any(_RECENT_TIMING_RE.search(text or "") for text in source_texts):
        return timeline
    return re.sub(r"\bStarted today\b", "Timing not specified", timeline, flags=re.IGNORECASE)

    print(
        "[gpu-check] "
        f"backend={canonical_backend(model_cfg.get('backend', 'ollama'))} "
        f"preset={model_cfg.get('selected_preset', '')} "
        f"n_gpu_layers={model_cfg.get('n_gpu_layers', '')} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')!r}",
        flush=True,
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        print(f"[gpu-check] nvidia-smi unavailable: {exc}", flush=True)
        return

    if result.returncode == 0:
        print(f"[gpu-check] nvidia-smi={result.stdout.strip()}", flush=True)
    else:
        print(f"[gpu-check] nvidia-smi failed: {result.stderr.strip()}", flush=True)


def model_download_status(model_preset_id: str):
    """Return model status text and download-button visibility for internal tools."""
    model_settings = resolve_model_settings(settings, model_preset_id)
    model_cfg = model_settings.get("model", {})
    backend = canonical_backend(model_cfg.get("backend", "ollama"))
    model_name = model_cfg.get("name", "medgemma1.5:4b")

    if backend == "ollama":
        try:
            if is_ollama_model_available(model_name, model_cfg.get("ollama_base_url", "http://localhost:11434")):
                return f"**Ready.** {model_name} is available locally.", gr.update(visible=False)
        except RuntimeError as exc:
            return f"**Ollama unavailable.** {exc}", gr.update(visible=False)
        return (
            f"**{model_name} is not downloaded.** Download it now with Ollama?",
            gr.update(visible=True),
        )

    if backend == "hf_transformers":
        return "**Ready.** Hugging Face will download and cache the selected model automatically.", gr.update(visible=False)

    return f"**Ready.** The configured {backend} backend will handle model access.", gr.update(visible=False)


def download_selected_model(model_preset_id: str):
    """Download the selected Ollama model after user confirmation via button click."""
    model_settings = resolve_model_settings(settings, model_preset_id)
    model_cfg = model_settings.get("model", {})
    backend = canonical_backend(model_cfg.get("backend", "ollama"))
    model_name = model_cfg.get("name", "medgemma1.5:4b")

    if backend != "ollama":
        return "**No download needed.** This backend handles model access automatically.", gr.update(visible=False)

    try:
        status = pull_ollama_model(model_name, model_cfg.get("ollama_base_url", "http://localhost:11434"))
    except RuntimeError as exc:
        return f"**Download failed.** {exc}", gr.update(visible=True)

    return f"**Ready.** {model_name} is available locally. Ollama status: {status}.", gr.update(visible=False)


@_with_zero_gpu
def run_inference(
    symptoms: str,
    notes: str,
    medications: str,
):
    """Run one LLM inference pass and return the three report sections."""
    errors = validate_inputs(symptoms=symptoms, notes=notes, medications=medications)
    if errors:
        error_msg = "\n".join(errors)
        return error_msg, error_msg, error_msg

    model_settings = _model_settings_for_selection()
    model_cfg = model_settings.get("model", {})
    _log_gpu_runtime_once(model_cfg)
    backend = canonical_backend(model_cfg.get("backend", "ollama"))
    model_name = model_cfg.get("name", "medgemma1.5:4b")
    if backend == "ollama":
        try:
            if not is_ollama_model_available(model_name, model_cfg.get("ollama_base_url", "http://localhost:11434")):
                prompt = _ollama_model_prompt(model_name)
                return prompt, prompt, prompt
        except RuntimeError as exc:
            error_msg = f"**Ollama unavailable.** {exc}"
            return error_msg, error_msg, error_msg

    model_start = time.perf_counter()
    model = get_model(model_settings)
    model_elapsed = time.perf_counter() - model_start

    timeline_prompt = build_timeline_prompt(symptoms, notes, medications)
    questions_prompt = build_questions_prompt(symptoms, notes, medications)
    relevant_prompt = build_relevant_info_prompt(symptoms, notes, medications)

    generate_start = time.perf_counter()
    timeline_raw = model.generate_report(timeline_prompt)
    questions_raw = model.generate_report(questions_prompt)
    relevant_raw = model.generate_report(relevant_prompt)
    generate_elapsed = time.perf_counter() - generate_start
    output_chars = len(timeline_raw) + len(questions_raw) + len(relevant_raw)
    if settings.get("app", {}).get("deployment") == "huggingface":
        print(
            "[timing] "
            f"model_ready_s={model_elapsed:.2f} "
            f"generate_s={generate_elapsed:.2f} "
            f"output_chars={output_chars}",
            flush=True,
        )
    timeline = clean_section_output(timeline_raw, max_items=8)
    timeline = _avoid_unreported_today_timing(timeline, symptoms, notes)

    return (
        timeline,
        clean_section_output(questions_raw, max_items=5),
        clean_section_output(
            relevant_raw,
            max_items=4,
            required_suffix=(
                "- This is informational only and not a substitute for "
                "professional medical advice."
            ),
        ),
    )


def add_medication_entry(medication_name: str, instructions: str, current_medications: str, all_choices: list = None):
    """Append a selected or custom medication line to the medication list."""
    medication_name = (medication_name or "").strip()
    instructions = (instructions or "").strip()
    current_medications = (current_medications or "").strip()

    if not medication_name and not instructions:
        return current_medications, "", "", gr.update(choices=all_choices or [], value=None)

    if medication_name and instructions:
        entry = f"{medication_name} - {instructions}"
    else:
        entry = medication_name or instructions

    updated = f"{current_medications}\n{entry}".strip() if current_medications else entry
    return updated, "", "", gr.update(choices=all_choices or [], value=None)


def clear_all_inputs(all_choices: list):
    """Clear all input fields and reset medication picker."""
    return (
        "",  # symptoms
        "",  # notes
        "",  # medications
        "",  # medication_name
        "",  # medication_instructions
        gr.update(choices=all_choices or [], value=None),  # medication_picker
        DEFAULT_OUTPUT,  # timeline_output
        DEFAULT_OUTPUT,  # questions_output
        DEFAULT_OUTPUT,  # relevant_output
    )


def populate_medication_name(selected_medication: str, current_name: str):
    """Copy a selected RxTerms choice into the editable medication-name field."""
    return (selected_medication or current_name or "").strip()


def filter_medication_picker_choices(all_choices: list[str], evt: gr.KeyUpData):
    """Return fuzzy medication choices using the dropdown's current typed text."""
    return gr.update(choices=filter_medication_choices(evt.input_value, all_choices))


def create_ui() -> gr.Blocks:
    app_cfg = settings.get("app", {})
    model_cfg = settings.get("model", {})
    backend = model_cfg.get("backend", "ollama")
    selected_model_preset_id = get_default_model_preset_id(settings, backend)
    selected_model_settings = _model_settings_for_selection(selected_model_preset_id)
    all_medication_choices = load_medication_choices()
    medication_summary = medication_index_summary()
    deployment = app_cfg.get("deployment", "local")
    is_local_deployment = deployment == "local"
    nav_status = "Local" if is_local_deployment else "Hosted"
    about_heading = (
        "Local medical appointment preparation."
        if is_local_deployment
        else "Hosted medical appointment preparation."
    )
    about_copy = (
        "This tool organizes appointment notes with a local language model."
        if is_local_deployment
        else "This tool organizes appointment notes with a hosted language model backend."
    )

    if backend.lower() in ("hf_transformers", "huggingface", "transformers"):
        get_model(selected_model_settings)

    with gr.Blocks(
        title="Medical Appointment Prep Assistant",
        elem_id="app-shell",
        fill_width=True,
    ) as app:
        gr.HTML(
            f"""
            <header class="global-nav" aria-label="Application">
                <div class="nav-inner">
                    <span class="nav-mark" aria-hidden="true"></span>
                    <span class="nav-title">Medical Appointment Prep</span>
                    <span class="nav-status">{nav_status}</span>
                    <div class="theme-switcher" aria-label="Appearance">
                        <button type="button" data-theme-option="system">System</button>
                        <button type="button" data-theme-option="light">Light</button>
                        <button type="button" data-theme-option="dark">Dark</button>
                    </div>
                </div>
            </header>
            <section class="hero-tile">
                <h1>Medical Appointment Prep</h1>
                <h2>Arrive clear, organized, and ready.</h2>
                <p class="hero-copy">
                    Turn symptoms, notes, and medications into a concise timeline,
                    visit questions, and relevant background information.
                </p>
            </section>
            """
        )

        with gr.Tab("Prepare", elem_classes=["main-tabs"]):
            with gr.Row(elem_classes=["prep-grid"]):
                with gr.Column(scale=1, elem_classes=["input-pane"]):
                    gr.HTML(
                        """
                        <div class="pane-heading">
                            <h2>Tell us about your visit</h2>
                        </div>
                        """
                    )
                    gr.HTML('<p class="input-label">Symptoms</p>')
                    symptoms = gr.Textbox(
                        label="Symptoms",
                        show_label=False,
                        placeholder="Headache behind eyes for 3 days, worse in the morning, mild nausea",
                        lines=6,
                        max_lines=12,
                        elem_classes=["apple-input"],
                    )
                    gr.HTML('<p class="input-label">Additional Notes</p>')
                    notes = gr.Textbox(
                        label="Additional Notes",
                        show_label=False,
                        placeholder="Recent travel, stress, dietary changes, sleep changes",
                        lines=4,
                        max_lines=8,
                        elem_classes=["apple-input"],
                    )
                    gr.HTML('<p class="input-label">Current Medications</p>')
                    gr.HTML(f'<p class="fine-print medication-source">{medication_summary}</p>')
                    medication_picker = gr.Dropdown(
                        label="Find a Medication",
                        show_label=False,
                        choices=all_medication_choices,
                        value=None,
                        filterable=True,
                        allow_custom_value=True,
                        elem_classes=["apple-input", "medication-picker"],
                    )
                    gr.HTML('<p class="input-label">Medication Name</p>')
                    medication_name = gr.Textbox(
                        label="Medication Name",
                        show_label=False,
                        placeholder="Select a medication above, or type any medication, vitamin, or supplement",
                        lines=1,
                        max_lines=2,
                        elem_classes=["apple-input"],
                    )
                    gr.HTML('<p class="input-label">How You Take It</p>')
                    medication_instructions = gr.Textbox(
                        label="Medication Instructions",
                        show_label=False,
                        placeholder="How you take it, e.g. once daily, as needed, with food",
                        lines=2,
                        max_lines=4,
                        elem_classes=["apple-input"],
                    )
                    add_medication_btn = gr.Button(
                        "Add Medication",
                        variant="secondary",
                        elem_classes=["secondary-pill", "add-medication-button"],
                    )
                    medications = gr.Textbox(
                        label="Current Medications",
                        show_label=False,
                        placeholder="Selected medications will appear here. You can also edit this list directly.",
                        lines=4,
                        max_lines=8,
                        elem_classes=["apple-input"],
                    )
                    with gr.Row(elem_classes=["button-row"]):
                        submit_btn = gr.Button(
                            "Generate Prep Report",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-pill"],
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            elem_classes=["secondary-pill"],
                        )

                with gr.Column(scale=1, elem_classes=["output-pane"]):
                    gr.HTML(
                        """
                        <div class="pane-heading">
                            <h2>Your appointment prep</h2>
                        </div>
                        """
                    )
                    with gr.Tabs(elem_classes=["output-tabs"]):
                        with gr.Tab("Timeline"):
                            timeline_output = gr.Markdown(
                                label="Symptom Timeline",
                                value=DEFAULT_OUTPUT,
                                elem_classes=["report-output"],
                            )
                        with gr.Tab("Questions"):
                            questions_output = gr.Markdown(
                                label="Questions to Ask",
                                value=DEFAULT_OUTPUT,
                                elem_classes=["report-output"],
                            )
                        with gr.Tab("Relevant Info"):
                            relevant_output = gr.Markdown(
                                label="Relevant Medical Information",
                                value=DEFAULT_OUTPUT,
                                elem_classes=["report-output"],
                            )

        with gr.Tab("About", elem_classes=["main-tabs"]):
            gr.HTML(
                f"""
                <section class="about-tile">
                    <p class="section-kicker">About</p>
                    <h2>{about_heading}</h2>
                    <p>
                        {about_copy}
                        It is for informational and organizational purposes only, not diagnosis,
                        treatment, or a substitute for professional medical advice.
                    </p>
                    <p class="about-meta">
                        Source code:
                        <a href="https://github.com/bisonbet/medical-appt-prep" target="_blank" rel="noopener noreferrer">
                            GitHub repository
                        </a>
                    </p>
                    <p class="about-meta">AI assisted by Codex.</p>
                </section>
                """
            )

        submit_btn.click(
            fn=run_inference,
            inputs=[symptoms, notes, medications],
            outputs=[timeline_output, questions_output, relevant_output],
            api_name="generate",
            concurrency_limit=1,
            concurrency_id="llama-cpp-gpu",
            api_visibility="public",
        )
        # Create a state variable to hold all medication choices for filtering
        medication_choices_state = gr.State(all_medication_choices)

        medication_picker.key_up(
            fn=filter_medication_picker_choices,
            inputs=[medication_choices_state],
            outputs=[medication_picker],
            show_progress="hidden",
            trigger_mode="always_last",
            api_visibility="private",
        )
        medication_picker.change(
            fn=populate_medication_name,
            inputs=[medication_picker, medication_name],
            outputs=[medication_name],
            api_visibility="private",
        )
        add_medication_btn.click(
            fn=add_medication_entry,
            inputs=[medication_name, medication_instructions, medications, medication_choices_state],
            outputs=[medications, medication_name, medication_instructions, medication_picker],
            api_visibility="private",
        )
        clear_btn.click(
            fn=clear_all_inputs,
            inputs=[medication_choices_state],
            outputs=[
                symptoms,
                notes,
                medications,
                medication_name,
                medication_instructions,
                medication_picker,
                timeline_output,
                questions_output,
                relevant_output,
            ],
            api_visibility="private",
        )

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name=settings.get("server", {}).get("host", "127.0.0.1"),
        server_port=settings.get("server", {}).get("port", 7860),
        theme=APPLE_THEME,
        css_paths=[APPLE_CSS_PATH],
        head=THEME_MODE_HEAD,
        footer_links=["api"],
        share=False,
        inbrowser=True,
    )
