# AGENTS.md

Guidance for future coding agents working in this repository.

## Project Snapshot

This is a Python Gradio app for preparing for medical appointments. Users enter symptoms, notes, and medications; the app generates a three-part report:

- `TIMELINE`
- `QUESTIONS`
- `RELEVANT_INFO`

The app is informational only. Do not add diagnosis, treatment, triage certainty, or emergency-care decision logic unless the user explicitly asks and the behavior is carefully framed as non-diagnostic guidance.

## Repository Layout

- `app.py` is the main Gradio UI and request orchestration.
- `src/model.py` owns LLM backend implementations and backend selection.
- `src/prompts.py` owns prompt templates and required report markers.
- `src/processor.py` owns validation, output cleanup, and report parsing.
- `src/medications.py` loads the committed local RxTerms medication index.
- `config_loader.py` loads `config/settings.yaml` and environment overrides.
- `assets/apple.css` owns the custom Apple-like Gradio styling and theme behavior.
- `scripts/export_hf_space.py` exports a Hugging Face Space repository.
- `scripts/sync_rxterms.py` refreshes the committed medication autocomplete JSON.
- `deploy/huggingface-space/` contains the hosted Space wrapper and hosted requirements.
- `tests/test_core.py` covers the prompt/parser/inference/backend factory basics.

## Local Commands

Use a virtual environment if dependencies are not already installed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run the app:

```bash
python3 app.py
```

The default local URL is `http://127.0.0.1:7860`.

Refresh the medication index:

```bash
python3 scripts/sync_rxterms.py
```

Export a Hugging Face Space:

```bash
python3 scripts/export_hf_space.py /path/to/hf-space-repo
```

## Architecture Rules

- Keep UI wiring in `app.py`; keep reusable logic in `src/`.
- Preserve the one-call report generation path unless the user asks for a different UX. `run_inference()` calls `generate_report()` once and expects all three sections back.
- If changing report format, update both `build_prep_report_prompt()` and `parse_prep_report()`, then add/adjust tests in `tests/test_core.py`.
- The parser depends on these exact section markers: `TIMELINE:`, `QUESTIONS:`, and `RELEVANT_INFO:`.
- Keep input validation in `src/processor.py`; do not scatter validation checks through UI event handlers.
- `get_model(settings)` is cached through `_get_model_cached()`. Clear the cache in tests when changing backend selection behavior.
- Do not import heavyweight optional dependencies at module import time. `llama_cpp`, `torch`, `transformers`, and `spaces` should remain optional or hosted-only.

## Backend Conventions

Supported backends are:

- `ollama`
- `llama_cpp`
- `hf_transformers`
- `openai_compatible`

When adding or changing a backend:

- Implement the `BaseLLM` interface in `src/model.py`.
- Add factory support in `get_model()` / `_get_model_cached()`.
- Add or update tests for backend selection without loading real models.
- Update `README.md`, `config/settings.yaml`, `.env.example`, and Space docs if user-visible configuration changes.
- Avoid logging prompts, raw model responses, API keys, medication entries, or appointment details.

## Privacy, Safety, and Medical Boundaries

This app handles sensitive health-related user input. Treat all symptoms, notes, medication entries, prompts, and generated reports as private.

- Do not persist user appointment content unless the user explicitly requests it.
- Do not send user content to a remote backend unless the selected configuration already implies hosted processing.
- Preserve local-mode privacy copy when `app.deployment` is `local`.
- Make hosted-mode copy explicit that a remote backend processes entries.
- Never commit secrets. `.env` is gitignored; API keys should come from environment variables.
- Keep generated text framed as organization and appointment preparation, not diagnosis or medical advice.
- Keep disclaimers visible in prompt/system behavior and app copy.

## UI and Styling

The app uses Gradio 6 with custom CSS in `assets/apple.css`.

- Keep the current restrained, Apple-like visual style unless the user asks for a redesign.
- Prefer changing CSS classes in `assets/apple.css` over large inline style blocks in `app.py`.
- Preserve light/dark/system theme switching. The JavaScript lives in `THEME_MODE_HEAD` in `app.py`.
- Check mobile layout when touching nav, tabs, buttons, or output panels. Existing breakpoints are at `900px` and `640px`.
- Avoid adding explanatory marketing copy; the primary screen should remain the usable preparation workflow.

## Medication Data

Medication autocomplete is local and committed at `data/medications/rxterms_medications.json`.

- Do not introduce a runtime network dependency for medication search.
- `scripts/sync_rxterms.py` downloads a monthly RxTerms release and writes the compact JSON index.
- Downloaded ZIP archives under `data/medications/*.zip` are ignored and should not be committed.
- Keep custom medication, vitamin, and supplement entry support; the RxTerms index is not exhaustive.

## Hugging Face Space Export

Hosted deployment is intentionally separated from the local dependency path.

- Root `requirements.txt` stays lightweight and should not include Torch/Transformers by default.
- Hosted-only dependencies belong in `deploy/huggingface-space/requirements.txt`.
- The Space wrapper imports `shared_app.py`, which is copied from root `app.py` by `scripts/export_hf_space.py`.
- If core files, config, assets, data, or package layout change, verify the export script still copies everything needed.
- For hosted defaults, update `deploy/huggingface-space/app.py` and `deploy/huggingface-space/README.md` together.

## Testing Expectations

Before finishing changes that affect Python behavior, run:

```bash
python3 -m unittest discover -s tests
```

Add focused tests when changing:

- prompt markers or prompt content contracts
- report parsing behavior
- validation limits or messages
- backend factory behavior
- medication loading behavior
- export script behavior

Avoid tests that require a live Ollama daemon, Hugging Face model download, GPU, network access, or real API key unless the user explicitly asks for integration testing.

## Dependency and Config Policy

- Keep local dependencies minimal.
- Do not add heavyweight ML packages to root `requirements.txt` unless the user explicitly chooses that tradeoff.
- Keep environment overrides documented in `.env.example`.
- When adding a setting, update `config/settings.yaml`, `config_loader.py`, and relevant docs.
- Prefer standard-library utilities where practical; the current backend HTTP clients use `urllib`.

## Change Hygiene

- Keep edits scoped to the requested behavior.
- Do not reformat unrelated files.
- Do not regenerate `data/medications/rxterms_medications.json` unless medication data is part of the task.
- Do not edit files inside an exported Space repository unless the user is working in that repository. Edit the template or export script here instead.
- Preserve existing user changes in the worktree.

## Review Checklist

For non-trivial changes, verify:

- Unit tests pass.
- Local import path still works from repo root.
- The app can still launch with `python3 app.py` if dependencies and backend are available.
- Local vs hosted privacy copy is still accurate.
- Generated report sections still parse into all three tabs.
- Hosted export still has the files and dependencies it needs.
