"""
Medical Appointment Prep Assistant
Main Gradio application entry point.
"""

import os
import re
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr
from fastapi.responses import FileResponse, HTMLResponse
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
_MODEL_WARMUP_LOCK = threading.Lock()
_MODEL_WARMUP_STARTED = False


DEFAULT_OUTPUT = "_Your prep report will appear here._"
APPLE_CSS_PATH = "assets/apple.css"
CUSTOM_UI_CSS_PATH = "assets/server.css"
CUSTOM_UI_JS_PATH = "assets/server.js"
CUSTOM_UI_ASSET_VERSION = "2026-06-14-user-quote"
ROBOT_IMAGE_PATH = "assets/assistant-robot.jpeg"
APPLE_THEME = gr.themes.Soft()
DEFAULT_CONTEXT_LENGTH = "8192"
DEFAULT_TEMPERATURE = "0.3"
DOWNLOAD_MODEL_BUTTON_LABEL = "Download Model"
ROOT_DIR = Path(__file__).resolve().parent
GITHUB_REPO_URL = "https://github.com/bisonbet/medical-appt-prep"
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


def _model_warmup_enabled() -> bool:
    if os.getenv("SPACE_DISABLE_MODEL_WARMUP", "").strip().lower() in {"1", "true", "yes"}:
        return False
    deployment = settings.get("app", {}).get("deployment")
    return deployment == "huggingface" or os.getenv("SPACE_ENABLE_MODEL_WARMUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _warmup_model_worker(reason: str = "startup") -> None:
    try:
        model_settings = _model_settings_for_selection()
        model_cfg = model_settings.get("model", {})
        backend = canonical_backend(model_cfg.get("backend", "ollama"))
        print(
            "[warmup] starting "
            f"reason={reason} "
            f"backend={backend} "
            f"preset={model_cfg.get('selected_preset', '')} "
            f"model={model_cfg.get('name', '')}",
            flush=True,
        )
        model = get_model(model_settings)
        warmup = getattr(model, "warmup", None)
        if callable(warmup):
            warmup()
        else:
            model.health_check()
        print("[warmup] ready", flush=True)
    except Exception as exc:
        print(f"[warmup] failed: {exc}", flush=True)


def warmup_model_async(reason: str = "startup") -> bool:
    """Start hosted model download/load/warmup in the background once."""
    global _MODEL_WARMUP_STARTED
    if not _model_warmup_enabled():
        return False
    with _MODEL_WARMUP_LOCK:
        if _MODEL_WARMUP_STARTED:
            return False
        _MODEL_WARMUP_STARTED = True
    target = _with_zero_gpu(_warmup_model_worker)
    thread = threading.Thread(target=target, args=(reason,), daemon=True)
    thread.start()
    return True


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


def _page_head(title: str = "Medical Appointment Prep") -> str:
    return f"""<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <link rel="stylesheet" href="/assets/server.css?v={CUSTOM_UI_ASSET_VERSION}" />
    <script type="module" src="/assets/server.js?v={CUSTOM_UI_ASSET_VERSION}"></script>
  </head>"""


def _topbar_html(active_page: str = "prep") -> str:
    prep_active = " selected" if active_page == "prep" else ""
    about_active = " selected" if active_page == "about" else ""
    return f"""<header class="topbar" aria-label="Application">
      <a class="brand" href="/" aria-label="Medical Appointment Prep home">
        <span class="brand-mark" aria-hidden="true">
          <span></span>
        </span>
        <span>Medical Appointment Prep</span>
      </a>
      <nav class="top-nav" aria-label="Primary">
        <a class="nav-link{prep_active}" href="/">Prep</a>
        <a class="nav-link{about_active}" href="/about">About</a>
      </nav>
      <div class="theme-switcher" aria-label="Appearance">
        <button type="button" data-theme-option="system" aria-pressed="true">System</button>
        <button type="button" data-theme-option="light" aria-pressed="false">Light</button>
        <button type="button" data-theme-option="dark" aria-pressed="false">Dark</button>
      </div>
    </header>"""


def _custom_frontend_html() -> str:
    return f"""<!doctype html>
<html lang="en" data-theme="system">
  {_page_head()}
  <body>
    {_topbar_html("prep")}

    <main class="page-shell">
      <section class="hero-workspace" aria-labelledby="page-title">
        <div class="guide-panel">
          <div class="guide-visual">
            <img src="/assets/assistant-robot.jpeg" alt="Friendly robot assistant in a medical room" />
            <div class="guide-bubble">
              <span class="pulse-dot" aria-hidden="true"></span>
              <span>Ready when you are</span>
            </div>
          </div>
          <div class="guide-copy">
            <p class="eyebrow">A calm visit-planning helper</p>
            <h1 id="page-title">Bring the important details into the room.</h1>
            <p>
              Share what has been happening, what you have noticed, and what you take.
              The assistant turns it into a clear timeline, useful questions, and background notes.
            </p>
            <div class="trust-strip" aria-label="Privacy and safety notes">
              <span>Informational only</span>
              <span>No diagnosis</span>
              <span>MedGemma 4B</span>
              <span>llama.cpp ready</span>
            </div>
          </div>
        </div>

        <form class="prep-form" id="prep-form">
          <div class="form-heading">
            <div>
              <p class="eyebrow">Step 1</p>
              <h2>Tell me what you want your clinician to know.</h2>
            </div>
            <button class="demo-action" type="button" id="demo-button">Click here for a demo</button>
          </div>

          <label class="field-label" for="symptoms">What symptoms or concerns are you having?</label>
          <textarea id="symptoms" name="symptoms" rows="6" required
            placeholder="Example: Headache behind my eyes for 3 days. Worse in the morning. Mild nausea."></textarea>

          <label class="field-label" for="notes">Anything else that might matter?</label>
          <textarea id="notes" name="notes" rows="4"
            placeholder="Example: New stress, sleep changes, recent travel, diet changes, or what makes it better or worse."></textarea>

          <div class="medication-card">
            <div>
              <p class="eyebrow">Step 2</p>
              <h3>Add medications, vitamins, or supplements.</h3>
            </div>
            <div class="medication-row">
              <div class="combo-field">
                <label class="field-label" for="medication-name">Medication name</label>
                <input id="medication-name" autocomplete="off"
                  placeholder="Search or type anything" />
                <div id="medication-suggestions" class="suggestions" role="listbox" hidden></div>
              </div>
              <div>
                <label class="field-label" for="medication-instructions">How you take it</label>
                <input id="medication-instructions"
                  placeholder="Example: 10 mg once daily" />
              </div>
              <button class="secondary-action" type="button" id="add-medication">Add</button>
            </div>
            <textarea id="medications" name="medications" rows="3"
              placeholder="Your medication list will appear here. You can edit it directly."></textarea>
          </div>

          <div class="form-actions">
            <button class="primary-action" type="submit" id="generate-button">
              <span class="button-label">Generate Prep Report</span>
              <span class="button-spinner" aria-hidden="true"></span>
            </button>
            <button class="ghost-action" type="button" id="clear-button">Clear</button>
          </div>
          <p class="form-note">
            This tool helps organize appointment notes. It is not a medical diagnosis,
            treatment recommendation, or substitute for a qualified clinician.
          </p>
        </form>
      </section>

      <section class="results-shell" aria-labelledby="results-title">
        <div class="results-heading">
          <p class="eyebrow">Step 3</p>
          <h2 id="results-title">Your appointment prep</h2>
          <p id="status-message" role="status">Your report will appear after you generate it.</p>
        </div>

        <div class="result-tabs" role="tablist" aria-label="Report sections">
          <button type="button" class="selected" role="tab" aria-selected="true" data-tab="timeline">Timeline</button>
          <button type="button" role="tab" aria-selected="false" data-tab="questions">Questions</button>
          <button type="button" role="tab" aria-selected="false" data-tab="relevant">Relevant Info</button>
        </div>

        <article class="report-panel selected" id="timeline-panel" data-panel="timeline">
          <div class="panel-icon" aria-hidden="true">1</div>
          <div class="markdown-body" id="timeline-output">Your timeline will appear here.</div>
        </article>
        <article class="report-panel" id="questions-panel" data-panel="questions">
          <div class="panel-icon" aria-hidden="true">2</div>
          <div class="markdown-body" id="questions-output">Questions for your visit will appear here.</div>
        </article>
        <article class="report-panel" id="relevant-panel" data-panel="relevant">
          <div class="panel-icon" aria-hidden="true">3</div>
          <div class="markdown-body" id="relevant-output">Relevant background information will appear here.</div>
        </article>

        <div class="export-card" aria-labelledby="export-title">
          <div class="export-heading">
            <div>
              <p class="eyebrow">Take it with you</p>
              <h3 id="export-title">Share or save the full prep report</h3>
            </div>
            <p id="export-status" role="status">Generate a report to enable full-report exports.</p>
          </div>
          <div class="export-actions" aria-label="Export options">
            <button type="button" class="export-action primary-export" id="email-report" disabled>
              <span class="export-icon" aria-hidden="true">@</span>
              <span>Email Full</span>
            </button>
            <button type="button" class="export-action" id="pdf-report" disabled>
              <span class="export-icon" aria-hidden="true">PDF</span>
              <span>Full PDF</span>
            </button>
            <button type="button" class="export-action" id="print-report" disabled>
              <span class="export-icon" aria-hidden="true">PRN</span>
              <span>Print Full</span>
            </button>
            <button type="button" class="export-action" id="copy-report" disabled>
              <span class="export-icon" aria-hidden="true">TXT</span>
              <span>Copy Full</span>
            </button>
            <button type="button" class="export-action" id="portal-copy" disabled>
              <span class="export-icon" aria-hidden="true">PT</span>
              <span>Portal Copy</span>
            </button>
            <button type="button" class="export-action" id="download-report" disabled>
              <span class="export-icon" aria-hidden="true">DL</span>
              <span>Download Full</span>
            </button>
          </div>
          <p class="export-note">
            Export actions include the symptoms, notes, medications, timeline, questions,
            and relevant info. Review before sending, saving, or pasting into another service.
          </p>
        </div>
      </section>
    </main>
  </body>
</html>"""


def _about_frontend_html() -> str:
    return f"""<!doctype html>
<html lang="en" data-theme="system">
  {_page_head("About Medical Appointment Prep")}
  <body>
    {_topbar_html("about")}

    <main class="page-shell about-page">
      <section class="about-hero" aria-labelledby="about-title">
        <p class="eyebrow">About</p>
        <h1 id="about-title">A calmer way to prepare for a medical visit.</h1>
        <p>
          Medical Appointment Prep helps people turn symptoms, notes, and
          medications into a visit-ready timeline, questions, and background
          information they can review with a clinician.
        </p>
        <div class="trust-strip" aria-label="Safety notes">
          <span>Informational only</span>
          <span>No diagnosis</span>
          <span>Patient-controlled export</span>
          <span>4B small-model build</span>
          <span>llama.cpp Space runtime</span>
        </div>
      </section>

      <section class="about-grid" aria-label="Project details">
        <article class="about-card">
          <p class="eyebrow">Purpose</p>
          <h2>Built for regular people</h2>
          <p>
            The interface is designed to feel familiar and approachable, with
            plain-language prompts and report sections that match how people
            prepare for appointments.
          </p>
        </article>
        <article class="about-card">
          <p class="eyebrow">Safety</p>
          <h2>Organization, not advice</h2>
          <p>
            The app helps organize what the user provides. It does not diagnose,
            choose treatment, or replace a qualified healthcare professional.
          </p>
        </article>
        <article class="about-card source-card">
          <p class="eyebrow">Hackathon fit</p>
          <h2>Small model, custom Gradio UI</h2>
          <p>
            The app is designed for the Backyard AI track: a specific appointment-prep
            workflow, a custom Gradio Server frontend, and a MedGemma 1.5 4B GGUF
            runtime through llama.cpp in the hosted Space.
          </p>
          <a class="github-link" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer">
            Open GitHub repository
          </a>
        </article>
      </section>

      <section class="field-note-quote" aria-labelledby="field-note-title">
        <p class="eyebrow" id="field-note-title">Field note</p>
        <blockquote>
          I always find it hard to remember what to ask or mention in the middle
          of an appointment. Helping a parent with appointments can be even more
          confusing, and this helps ensure we get the right help needed. Having
          this sheet to read from or share with my doctor will help me in my
          future appointments.
        </blockquote>
        <p class="quote-attribution">- My Spouse</p>
      </section>

      <section class="acknowledgements" aria-labelledby="acknowledgements-title">
        <div>
          <p class="eyebrow">Acknowledgements</p>
          <h2 id="acknowledgements-title">Thank you to the Build Small Hackathon sponsors.</h2>
          <p>
            This project was built for the Hugging Face Build Small Hackathon.
            Thanks to the sponsors for supporting small-model experiments, and
            especially to OpenAI's Codex for helping shape, test, and ship the app.
          </p>
        </div>
        <div class="sponsor-links" aria-label="Sponsor links">
          <a href="https://huggingface.co/" target="_blank" rel="noopener noreferrer">Hugging Face</a>
          <a href="https://openai.com/codex/" target="_blank" rel="noopener noreferrer">OpenAI Codex</a>
          <a href="https://www.nvidia.com/" target="_blank" rel="noopener noreferrer">NVIDIA</a>
          <a href="https://modal.com/" target="_blank" rel="noopener noreferrer">Modal</a>
          <a href="https://www.openbmb.cn/" target="_blank" rel="noopener noreferrer">OpenBMB</a>
          <a href="https://cohere.com/" target="_blank" rel="noopener noreferrer">Cohere</a>
          <a href="https://www.jetbrains.com/" target="_blank" rel="noopener noreferrer">JetBrains</a>
          <a href="https://blackforestlabs.ai/" target="_blank" rel="noopener noreferrer">Black Forest Labs</a>
        </div>
      </section>
    </main>
  </body>
</html>"""


def create_server_app() -> gr.Server:
    """Create a custom Server-mode app with Gradio's queued API backend."""
    server = gr.Server(title="Medical Appointment Prep Assistant")

    @server.api(
        name="generate",
        concurrency_limit=1,
        concurrency_id="llama-cpp-gpu",
        api_visibility="public",
    )
    def generate(symptoms: str, notes: str, medications: str) -> tuple[str, str, str]:
        return run_inference(symptoms, notes, medications)

    @server.get("/", response_class=HTMLResponse)
    async def homepage():
        return _custom_frontend_html()

    @server.get("/about", response_class=HTMLResponse)
    async def about_page():
        return _about_frontend_html()

    @server.get("/assets/{asset_name}")
    async def asset(asset_name: str):
        allowed_assets = {
            "server.css": CUSTOM_UI_CSS_PATH,
            "server.js": CUSTOM_UI_JS_PATH,
            "assistant-robot.jpeg": ROBOT_IMAGE_PATH,
        }
        asset_path = allowed_assets.get(asset_name)
        if asset_path is None:
            return HTMLResponse("Not found", status_code=404)
        return FileResponse(ROOT_DIR / asset_path)

    @server.get("/api/medications")
    async def medication_suggestions(q: str = ""):
        return {"choices": filter_medication_choices(q, load_medication_choices(), limit=20)}

    return server


def launch_app() -> None:
    server_host = settings.get("server", {}).get("host", "127.0.0.1")
    server_port = settings.get("server", {}).get("port", 7860)
    if os.getenv("APP_UI_MODE", "").strip().lower() == "blocks":
        ui = create_ui()
        ui.launch(
            server_name=server_host,
            server_port=server_port,
            theme=APPLE_THEME,
            css_paths=[APPLE_CSS_PATH],
            head=THEME_MODE_HEAD,
            footer_links=["api"],
            share=False,
            inbrowser=True,
        )
        return

    server = create_server_app()
    server.launch(
        server_name=server_host,
        server_port=server_port,
        show_error=True,
        inbrowser=True,
    )


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
    if is_local_deployment:
        about_heading = "Local medical appointment preparation."
        about_copy = "This tool organizes appointment notes with a local language model."
    elif deployment == "huggingface":
        about_heading = "Space-local medical appointment preparation."
        about_copy = (
            "This tool organizes appointment notes with a language model running "
            "inside this Hugging Face Space."
        )
    else:
        about_heading = "Hosted medical appointment preparation."
        about_copy = "This tool organizes appointment notes with a hosted language model backend."

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
                    <hr />
                    <h3>Field note</h3>
                    <blockquote>
                        I always find it hard to remember what to ask or mention in the middle
                        of an appointment. Helping a parent with appointments can be even more
                        confusing, and this helps ensure we get the right help needed. Having
                        this sheet to read from or share with my doctor will help me in my
                        future appointments.
                    </blockquote>
                    <p class="about-meta">- My Spouse</p>
                    <hr />
                    <h3>Acknowledgements</h3>
                    <p>
                        Built for the Hugging Face Build Small Hackathon. Thank you to the
                        sponsors supporting small-model work, especially
                        <a href="https://openai.com/codex/" target="_blank" rel="noopener noreferrer">OpenAI Codex</a>.
                    </p>
                    <p class="about-meta">
                        Sponsors:
                        <a href="https://huggingface.co/" target="_blank" rel="noopener noreferrer">Hugging Face</a>,
                        <a href="https://openai.com/codex/" target="_blank" rel="noopener noreferrer">OpenAI Codex</a>,
                        <a href="https://www.nvidia.com/" target="_blank" rel="noopener noreferrer">NVIDIA</a>,
                        <a href="https://modal.com/" target="_blank" rel="noopener noreferrer">Modal</a>,
                        <a href="https://www.openbmb.cn/" target="_blank" rel="noopener noreferrer">OpenBMB</a>,
                        <a href="https://cohere.com/" target="_blank" rel="noopener noreferrer">Cohere</a>,
                        <a href="https://www.jetbrains.com/" target="_blank" rel="noopener noreferrer">JetBrains</a>,
                        <a href="https://blackforestlabs.ai/" target="_blank" rel="noopener noreferrer">Black Forest Labs</a>.
                    </p>
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
    launch_app()
