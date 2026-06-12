"""
Medical Appointment Prep Assistant
Main Gradio application entry point.
"""

import gradio as gr
from src.model import get_model, is_ollama_model_available, pull_ollama_model
from src.model_catalog import (
    canonical_backend,
    describe_model_preset,
    get_default_model_preset_id,
    get_model_preset_choices,
    resolve_model_settings,
)
from src.prompts import build_prep_report_prompt
from src.processor import validate_inputs, parse_prep_report
from src.medications import load_medication_choices, medication_index_summary
from config_loader import load_settings
import gradio as gr

settings = load_settings()


DEFAULT_OUTPUT = "_Your prep report will appear here._"
APPLE_CSS_PATH = "assets/apple.css"
APPLE_THEME = gr.themes.Soft()
CONTEXT_CHOICES = [
    ("4096", "4096"),
    ("8192", "8192"),
    ("16k", "16384"),
]
TEMPERATURE_CHOICES = [
    ("0.1 - Most steady", "0.1"),
    ("0.3 - Balanced default", "0.3"),
    ("0.5 - More varied", "0.5"),
]
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


def model_preset_details(model_preset_id: str):
    """Return display copy for the selected model preset."""
    return describe_model_preset(settings, model_preset_id)


def _context_choice_value(context_length: object) -> str:
    value = str(context_length or "4096")
    allowed_values = {choice_value for _label, choice_value in CONTEXT_CHOICES}
    return value if value in allowed_values else "4096"


def _temperature_choice_value(temperature: object) -> str:
    value = str(temperature or "0.3")
    allowed_values = {choice_value for _label, choice_value in TEMPERATURE_CHOICES}
    return value if value in allowed_values else "0.3"


def _model_settings_for_selection(
    model_preset_id: str,
    context_length: str = "4096",
    temperature: str = "0.3",
) -> dict:
    model_settings = resolve_model_settings(settings, model_preset_id)
    model_settings.setdefault("model", {})["context_length"] = int(_context_choice_value(context_length))
    model_settings.setdefault("model", {})["temperature"] = float(_temperature_choice_value(temperature))
    return model_settings


def _ollama_model_prompt(model_name: str) -> str:
    return (
        f"**{model_name} is not downloaded.** Go to Settings and choose "
        f"{DOWNLOAD_MODEL_BUTTON_LABEL} to pull it with Ollama before generating a report."
    )


def model_download_status(model_preset_id: str):
    """Return model status text and download-button visibility for Settings."""
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


def model_selection_details(model_preset_id: str):
    """Return description, status, and download visibility for the selected model."""
    status, download_visibility = model_download_status(model_preset_id)
    return model_preset_details(model_preset_id), status, download_visibility


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


def run_inference(
    symptoms: str,
    notes: str,
    medications: str,
    model_preset_id: str = "",
    context_length: str = "4096",
    temperature: str = "0.3",
):
    """Run one LLM inference pass and return the three report sections."""
    errors = validate_inputs(symptoms=symptoms, notes=notes, medications=medications)
    if errors:
        error_msg = "\n".join(errors)
        return error_msg, error_msg, error_msg

    model_settings = _model_settings_for_selection(model_preset_id, context_length, temperature)
    model_cfg = model_settings.get("model", {})
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

    model = get_model(model_settings)

    report_prompt = build_prep_report_prompt(symptoms, notes, medications)
    report_raw = model.generate_report(report_prompt)
    return parse_prep_report(report_raw)


def add_medication_entry(medication_name: str, instructions: str, current_medications: str, all_choices: list = None):
    """Append a selected or custom medication line to the medication list."""
    medication_name = (medication_name or "").strip()
    instructions = (instructions or "").strip()
    current_medications = (current_medications or "").strip()

    if not medication_name and not instructions:
        return current_medications, "", "", gr.update(choices=all_choices[:8] if all_choices else [], value=None)

    if medication_name and instructions:
        entry = f"{medication_name} - {instructions}"
    else:
        entry = medication_name or instructions

    updated = f"{current_medications}\n{entry}".strip() if current_medications else entry
    return updated, "", "", gr.update(choices=all_choices[:8] if all_choices else [], value=None)


def clear_all_inputs(all_choices: list):
    """Clear all input fields and reset medication picker."""
    return (
        "",  # symptoms
        "",  # notes
        "",  # medications
        "",  # medication_name
        "",  # medication_instructions
        gr.update(choices=all_choices[:8] if all_choices else [], value=None),  # medication_picker
        DEFAULT_OUTPUT,  # timeline_output
        DEFAULT_OUTPUT,  # questions_output
        DEFAULT_OUTPUT,  # relevant_output
    )


def populate_medication_name(selected_medication: str, current_name: str):
    """Copy a selected RxTerms choice into the editable medication-name field."""
    return (selected_medication or current_name or "").strip()


def create_ui() -> gr.Blocks:
    app_cfg = settings.get("app", {})
    model_cfg = settings.get("model", {})
    backend = model_cfg.get("backend", "ollama")
    model_preset_choices = get_model_preset_choices(settings, backend)
    selected_model_preset_id = get_default_model_preset_id(settings, backend)
    if not selected_model_preset_id:
        model_preset_choices = [("Custom configured model", "")] + model_preset_choices
    selected_model_settings = resolve_model_settings(settings, selected_model_preset_id)
    selected_model_cfg = selected_model_settings.get("model", {})
    model_preset_summary = model_preset_details(selected_model_preset_id)
    initial_model_status, initial_download_visibility = model_download_status(selected_model_preset_id)
    context_length = _context_choice_value(
        selected_model_cfg.get("context_length", model_cfg.get("context_length", 4096))
    )
    temperature = _temperature_choice_value(
        selected_model_cfg.get("temperature", model_cfg.get("temperature", 0.3))
    )
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
                            <p class="section-kicker">Input</p>
                            <h2>Appointment details</h2>
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
                        choices=all_medication_choices[:8],  # Show first 8 initially for performance
                        value=None,
                        filterable=True,
                        allow_custom_value=True,
                        elem_classes=["apple-input", "medication-picker"],
                    )
                    gr.HTML('<p class="input-label">Medication Name</p>')
                    medication_name = gr.Textbox(
                        label="Medication Name",
                        show_label=False,
                        placeholder="Select from RxTerms above, or type a medication, vitamin, or supplement",
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
                            <p class="section-kicker">Output</p>
                            <h2>Prep report</h2>
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

        with gr.Tab("Settings", elem_classes=["main-tabs"]):
            with gr.Column(elem_classes=["settings-tile"]):
                gr.HTML(
                    """
                    <div>
                        <p class="section-kicker">Configuration</p>
                        <h2>Settings</h2>
                    </div>
                    """
                )
                gr.HTML('<p class="input-label">Model</p>')
                model_preset = gr.Dropdown(
                    label="Model",
                    show_label=False,
                    choices=model_preset_choices,
                    value=selected_model_preset_id,
                    filterable=False,
                    allow_custom_value=False,
                    elem_classes=["apple-input", "model-picker"],
                )
                model_preset_helper = gr.Markdown(
                    value=model_preset_summary,
                    elem_classes=["fine-print", "model-helper"],
                )
                model_status = gr.Markdown(
                    value=initial_model_status,
                    elem_classes=["fine-print", "model-status"],
                )
                download_model_btn = gr.Button(
                    DOWNLOAD_MODEL_BUTTON_LABEL,
                    variant="secondary",
                    elem_classes=["secondary-pill", "download-model-button"],
                    visible=initial_download_visibility.get("visible", False),
                )
                with gr.Row(elem_classes=["settings-controls"]):
                    with gr.Column(scale=1):
                        gr.HTML('<p class="input-label">Context</p>')
                        context_choice = gr.Dropdown(
                            label="Context",
                            show_label=False,
                            choices=CONTEXT_CHOICES,
                            value=context_length,
                            filterable=False,
                            allow_custom_value=False,
                            elem_classes=["apple-input", "settings-picker"],
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<p class="input-label">Temperature</p>')
                        temperature_choice = gr.Dropdown(
                            label="Temperature",
                            show_label=False,
                            choices=TEMPERATURE_CHOICES,
                            value=temperature,
                            filterable=False,
                            allow_custom_value=False,
                            elem_classes=["apple-input", "settings-picker"],
                        )
                gr.HTML(f'<p class="fine-print">Backend: {backend}. Configure available models in config/settings.yaml.</p>')

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
                </section>
                """
            )

        submit_btn.click(
            fn=run_inference,
            inputs=[symptoms, notes, medications, model_preset, context_choice, temperature_choice],
            outputs=[timeline_output, questions_output, relevant_output],
            api_name="generate",
            api_visibility="public",
        )
        model_preset.change(
            fn=model_selection_details,
            inputs=[model_preset],
            outputs=[model_preset_helper, model_status, download_model_btn],
            api_visibility="private",
        )
        download_model_btn.click(
            fn=download_selected_model,
            inputs=[model_preset],
            outputs=[model_status, download_model_btn],
            api_visibility="private",
        )
        # Create a state variable to hold all medication choices for filtering
        medication_choices_state = gr.State(all_medication_choices)
        
        def handle_medication_change(selected_value, current_name, all_choices):
            """Handle medication picker change: populate name field and filter choices."""
            # Populate the medication name field
            name_result = (selected_value or current_name or "").strip()
            
            # Filter choices based on current input
            if selected_value and selected_value.strip():
                query_lower = selected_value.strip().casefold()
                matches = [choice for choice in all_choices if query_lower in choice.casefold()]
                filtered_choices = matches[:8]
            else:
                filtered_choices = all_choices[:8]
            
            return name_result, gr.update(choices=filtered_choices)
        
        medication_picker.change(
            fn=handle_medication_change,
            inputs=[medication_picker, medication_name, medication_choices_state],
            outputs=[medication_name, medication_picker],
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
