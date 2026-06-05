"""
Medical Appointment Prep Assistant
Main Gradio application entry point.
"""

import gradio as gr
from src.model import get_model
from src.prompts import (
    build_timeline_prompt,
    build_questions_prompt,
    build_relevant_info_prompt,
)
from src.processor import validate_inputs, parse_output
from config_loader import load_settings

settings = load_settings()


DEFAULT_OUTPUT = "_Your prep report will appear here._"
APPLE_CSS_PATH = "assets/apple.css"
APPLE_THEME = gr.themes.Soft()


def run_inference(symptoms: str, notes: str, medications: str):
    """Run all three LLM inference passes and return results."""
    errors = validate_inputs(symptoms=symptoms, notes=notes, medications=medications)
    if errors:
        error_msg = "\n".join(errors)
        return error_msg, error_msg, error_msg

    model = get_model(settings)

    timeline_prompt = build_timeline_prompt(symptoms, notes, medications)
    questions_prompt = build_questions_prompt(symptoms, notes, medications)
    relevant_info_prompt = build_relevant_info_prompt(symptoms, notes, medications)

    timeline_raw = model.generate(timeline_prompt)
    questions_raw = model.generate(questions_prompt)
    relevant_raw = model.generate(relevant_info_prompt)

    timeline = parse_output(timeline_raw)
    questions = parse_output(questions_raw)
    relevant = parse_output(relevant_raw)

    return timeline, questions, relevant


def create_ui() -> gr.Blocks:
    model_cfg = settings.get("model", {})
    backend = model_cfg.get("backend", "ollama")
    model_name = model_cfg.get("name", "medgemma")
    context_length = model_cfg.get("context_length", 4096)
    temperature = model_cfg.get("temperature", 0.3)

    with gr.Blocks(
        title="Medical Appointment Prep Assistant",
        elem_id="app-shell",
        fill_width=True,
    ) as app:
        gr.HTML(
            """
            <header class="global-nav" aria-label="Application">
                <div class="nav-inner">
                    <span class="nav-mark" aria-hidden="true"></span>
                    <span class="nav-title">Medical Appointment Prep</span>
                    <span class="nav-status">Local</span>
                </div>
            </header>
            <section class="hero-tile">
                <p class="eyebrow">Private prep workspace</p>
                <h1>Arrive clear, organized, and ready.</h1>
                <p class="hero-copy">
                    Turn symptoms, notes, and medications into a concise timeline,
                    visit questions, and relevant background information.
                </p>
                <p class="privacy-line">Runs locally. No appointment details leave this machine.</p>
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
                    medications = gr.Textbox(
                        label="Current Medications",
                        show_label=False,
                        placeholder="Lisinopril 10mg daily, Vitamin D 2000IU daily",
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

            submit_btn.click(
                fn=run_inference,
                inputs=[symptoms, notes, medications],
                outputs=[timeline_output, questions_output, relevant_output],
                api_name="generate",
                api_visibility="public",
            )
            clear_btn.click(
                fn=lambda: ("", "", "", DEFAULT_OUTPUT, DEFAULT_OUTPUT, DEFAULT_OUTPUT),
                inputs=[],
                outputs=[symptoms, notes, medications, timeline_output, questions_output, relevant_output],
                api_visibility="private",
            )

        with gr.Tab("Settings", elem_classes=["main-tabs"]):
            gr.HTML(
                f"""
                <section class="settings-tile">
                    <div>
                        <p class="section-kicker">Configuration</p>
                        <h2>Current model</h2>
                    </div>
                    <dl class="settings-grid">
                        <div><dt>Backend</dt><dd>{backend}</dd></div>
                        <div><dt>Model</dt><dd>{model_name}</dd></div>
                        <div><dt>Context</dt><dd>{context_length}</dd></div>
                        <div><dt>Temperature</dt><dd>{temperature}</dd></div>
                    </dl>
                    <p class="fine-print">Edit config/settings.yaml and restart the app to apply changes.</p>
                </section>
                """
            )

        with gr.Tab("About", elem_classes=["main-tabs"]):
            gr.HTML(
                """
                <section class="about-tile">
                    <p class="section-kicker">About</p>
                    <h2>Local medical appointment preparation.</h2>
                    <p>
                        This tool organizes appointment notes with a local language model.
                        It is for informational and organizational purposes only, not diagnosis,
                        treatment, or a substitute for professional medical advice.
                    </p>
                </section>
                """
            )

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name=settings.get("server", {}).get("host", "127.0.0.1"),
        server_port=settings.get("server", {}).get("port", 7860),
        theme=APPLE_THEME,
        css_paths=[APPLE_CSS_PATH],
        footer_links=["api"],
        share=False,
        inbrowser=True,
    )
