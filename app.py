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
    with gr.Blocks(title="Medical Appointment Prep Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🩺 Medical Appointment Prep Assistant
            Enter your symptoms, notes, and current medications below.
            The assistant will help you prepare for your appointment by generating
            a symptom timeline, questions to ask your doctor, and relevant background information.

            > **Privacy:** Everything runs locally. No data leaves your machine.
            """
        )

        with gr.Tab("Prepare"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    symptoms = gr.Textbox(
                        label="Symptoms",
                        placeholder="Describe your symptoms in detail...\ne.g. Headache behind eyes for 3 days, worse in the morning, mild nausea",
                        lines=6,
                        max_lines=12,
                    )
                    notes = gr.Textbox(
                        label="Additional Notes",
                        placeholder="Any other relevant info...\ne.g. Recent travel, stress, dietary changes, sleep issues",
                        lines=4,
                        max_lines=8,
                    )
                    medications = gr.Textbox(
                        label="Current Medications",
                        placeholder="List medications and dosages...\ne.g. Lisinopril 10mg daily, Vitamin D 2000IU daily",
                        lines=4,
                        max_lines=8,
                    )
                    submit_btn = gr.Button("Generate Prep Report", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    with gr.Tab("Symptom Timeline"):
                        timeline_output = gr.Markdown(
                            label="Symptom Timeline",
                            value="*Your symptom timeline will appear here...*",
                        )
                    with gr.Tab("Questions for Doctor"):
                        questions_output = gr.Markdown(
                            label="Questions to Ask",
                            value="*Suggested questions for your doctor will appear here...*",
                        )
                    with gr.Tab("Relevant Information"):
                        relevant_output = gr.Markdown(
                            label="Relevant Medical Information",
                            value="*Relevant background information will appear here...*",
                        )

            submit_btn.click(
                fn=run_inference,
                inputs=[symptoms, notes, medications],
                outputs=[timeline_output, questions_output, relevant_output],
                api_name="generate",
            )
            clear_btn.click(
                fn=lambda: ("", "", "", "*...*", "*...*", "*...*"),
                inputs=[],
                outputs=[symptoms, notes, medications, timeline_output, questions_output, relevant_output],
            )

        with gr.Tab("Settings"):
            gr.Markdown("### Model Configuration")
            gr.Markdown(
                f"""
                **Backend:** `{settings.get('model', {}).get('backend', 'ollama')}`

                **Model:** `{settings.get('model', {}).get('name', 'medgemma')}`

                **Context Length:** `{settings.get('model', {}).get('context_length', 4096)}`

                **Temperature:** `{settings.get('model', {}).get('temperature', 0.3)}`

                Edit `config/settings.yaml` to change these values, then restart the app.
                """
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Tool

                **Medical Appointment Prep Assistant** helps you organize your thoughts before
                a doctor's visit using a local, privacy-preserving LLM.

                ### Supported Backends
                - **Ollama** (recommended): Install from [ollama.ai](https://ollama.ai), then run `ollama pull medgemma`
                - **llama-cpp-python**: Download a `.gguf` model file and set the path in `config/settings.yaml`

                ### Disclaimer
                This tool is for **informational purposes only** and does not constitute medical advice.
                Always consult a qualified healthcare professional.
                """
            )

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name=settings.get("server", {}).get("host", "127.0.0.1"),
        server_port=settings.get("server", {}).get("port", 7860),
        share=False,
        inbrowser=True,
    )
