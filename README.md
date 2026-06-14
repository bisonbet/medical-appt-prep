# Medical Appointment Prep Assistant

A dual-mode AI assistant that helps you prepare for doctor's appointments.
Enter your symptoms, notes, and medications, then get back a symptom timeline,
questions to ask your doctor, and relevant background information.

Local mode runs on your machine with Ollama or llama.cpp. Hosted mode is designed
for Hugging Face Spaces and OpenAI-compatible serverless endpoints such as
Nebius.

---

## Features

- Gradio 6 web UI (runs in your browser)
- Custom `gradio.Server` frontend for a warmer, non-technical appointment-prep experience
- One-click fictional sample test data so reviewers can try the flow without inventing inputs
- Three focused model calls produce the Timeline, Questions, and Relevant Info tabs
- Full-report export actions for email drafts, PDF/print, clipboard, portal copy, and text download
- Print/PDF export opens a printable full-report view and triggers the browser print dialog
- Four LLM backend options: **Ollama**, **llama-cpp-python**, **Hugging Face Transformers**, or **OpenAI-compatible**
- Uses MedGemma 1.5 4B by default, with model settings hidden from non-technical users
- Local RxTerms medication autocomplete with custom entry fallback
- Hugging Face Space startup warmup for the hosted llama.cpp model
- Works on Windows, macOS, Linux, and Hugging Face Spaces

## Hackathon Positioning

This project is aimed at the Build Small Hackathon's **Backyard AI** track: it
helps a real person prepare for a specific kind of stressful everyday task, a
medical appointment, without trying to diagnose or choose treatment.

The competition build is intentionally small-model first:

- **Tiny Titan fit:** MedGemma 1.5 4B default model
- **Llama Champion fit:** hosted Space path runs the GGUF model through `llama.cpp`
- **Off-Brand / Custom UI fit:** custom `gradio.Server` frontend instead of default Blocks
- **Local-first fit:** local desktop path uses Ollama or llama.cpp without cloud APIs
- **Field Notes fit:** the app is designed to be explained as an appointment-prep workflow, not a medical-advice product

---

## Field Note

Project links:

- [Hugging Face Field Notes article](https://huggingface.co/blog/bisonnetworking/medical-asst-prep-june26)
- [Video walkthrough](https://youtu.be/kRHPwQGC6aU)
- [LinkedIn project update](https://www.linkedin.com/posts/timothy-champ_build-small-hackathon-starts-in-6-hours-activity-7471918851132207104-Wh7T?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEBVNSMBksj_L93Oketlrphm5iSG9BGmBqU)

> I always find it hard to remember what to ask or mention in the middle of an
> appointment. Helping a parent with appointments can be even more confusing,
> and this helps ensure we get the right help needed. Having this sheet to read
> from or share with my doctor will help me in my future appointments.
>
> - My Spouse

---

## Quick Start (Ollama — recommended)

### 1. Install Ollama

| Platform | Instructions |
|----------|-------------|
| **macOS** | `brew install ollama` or download from [ollama.ai](https://ollama.ai) |
| **Windows** | Download installer from [ollama.ai](https://ollama.ai) |
| **Linux** | `curl -fsSL https://ollama.ai/install.sh \| sh` |

Start the Ollama daemon:
```bash
ollama serve
```

Pull the model (in a new terminal):
```bash
ollama pull medgemma1.5:4b
```

If the Ollama model has not been pulled yet, the app will prompt you to download
it before generation. Hugging Face and llama.cpp models are downloaded and
cached automatically by their backends when used.

### 2. Clone and set up the project

```bash
git clone https://github.com/bisonbet/medical-appt-prep.git
cd medical-appt-prep
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Open your browser to **http://127.0.0.1:7860**

The default UI uses `gradio.Server`: a custom HTML/CSS/JS frontend backed by
Gradio's queued API. To launch the older Blocks UI for comparison or fallback:

```bash
APP_UI_MODE=blocks python app.py
```

---

## Alternative Backend: llama-cpp-python

Use this if you want to load a `.gguf` file directly without Ollama.

### Install llama-cpp-python

**CPU only (all platforms):**
```bash
pip install llama-cpp-python
```

**Apple Silicon (Metal GPU acceleration):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

**NVIDIA GPU (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Configure

Edit `config/settings.yaml`:
```yaml
model:
  backend: llama_cpp
  model_path: /path/to/your/medgemma.gguf
  n_gpu_layers: 0  # increase to offload layers to GPU
```

Download a compatible `.gguf` from [Hugging Face](https://huggingface.co) or
configure `model_repo_id` and `model_filename` so the llama.cpp backend can
download it from the Hub.

---

## Hosted Backends

Hosted mode changes the privacy posture: appointment entries are processed by the
configured remote model backend. Set `APP_DEPLOYMENT=huggingface` or
`APP_DEPLOYMENT=hosted` so the UI shows hosted-mode copy.

### Hugging Face Space

Use the separate Space export folder to avoid adding Torch and Transformers to the
lightweight local install path.

```bash
python scripts/export_hf_space.py /path/to/hf-space-repo
```

The hosted competition Space is `build-small-hackathon/medical-appt-prep`.
For a full export, upload, and runtime verification pass, use:

```bash
python scripts/deploy_hf_space.py
```

The deploy helper downloads the live Space README before export so hackathon
track, prize, and badge tags committed on Hugging Face are preserved.

Configure the Space with:

```text
HF_TOKEN=<token>
MODEL_BACKEND=llama_cpp
MODEL_PRESET=medgemma-4b
LLAMA_CPP_MODEL_REPO_ID=unsloth/medgemma-1.5-4b-it-GGUF
LLAMA_CPP_MODEL_FILENAME=medgemma-1.5-4b-it-Q4_K_M.gguf
LLAMA_CPP_N_GPU_LAYERS=-1
LLAMA_CPP_N_BATCH=2048
LLAMA_CPP_N_UBATCH=1024
LLAMA_CPP_FLASH_ATTN=1
LLAMA_CPP_OP_OFFLOAD=1
LLAMA_CPP_SWA_FULL=0
MODEL_CONTEXT_LENGTH=8192
MODEL_MAX_NEW_TOKENS=256
MODEL_TEMPERATURE=0.3
APP_DEPLOYMENT=huggingface
```

The Space template lives in `deploy/huggingface-space/` and pins Gradio 6. It
uses ZeroGPU for report generation and a CUDA-enabled `llama-cpp-python` wheel.
The template enforces these hosted defaults unless `SPACE_USE_ENV_MODEL_CONFIG=1`
is set, which prevents stale Space variables from switching the competition
build back to another backend. On startup, the Space starts a background model
preload/warmup so the first visitor is less likely to pay the full model
download/load cost. Set `SPACE_DISABLE_MODEL_WARMUP=1` if you need to disable
that behavior temporarily.

### OpenAI-Compatible / Nebius

The `openai_compatible` backend calls `/v1/chat/completions` and is intended for
Nebius Serverless AI or any compatible hosted endpoint.

```text
MODEL_BACKEND=openai_compatible
MODEL_NAME=<hosted-model-name>
OPENAI_COMPATIBLE_BASE_URL=<endpoint-base-url>
OPENAI_COMPATIBLE_API_KEY=<api-key>
APP_DEPLOYMENT=hosted
```

---

## Configuration

All settings live in `config/settings.yaml`. You can also override them with a `.env` file
(copy `.env.example` to `.env`).

| Key | Default | Description |
|-----|---------|-------------|
| `model.backend` | `ollama` | `ollama`, `llama_cpp`, `hf_transformers`, or `openai_compatible` |
| `model.selected_preset` | `medgemma-4b` | Default model preset |
| `model.presets` | MedGemma 1.5 4B | Backend-specific model catalog |
| `model.name` | `medgemma1.5:4b` | Fallback backend model name |
| `model.ollama_base_url` | `http://localhost:11434` | Ollama API URL |
| `model.model_path` | — | Path to `.gguf` file (llama_cpp only) |
| `model.model_repo_id` | — | Hugging Face repo for a GGUF model download (llama_cpp only) |
| `model.model_filename` | — | GGUF filename inside `model.model_repo_id` (llama_cpp only) |
| `model.max_new_tokens` | `256` | Generation budget for the full prep report |
| `model.temperature` | `0.3` | Generation temperature |
| `model.context_length` | `8192` | Context window size |
| `app.deployment` | `local` | UI copy mode: `local`, `huggingface`, or `hosted` |
| `server.port` | `7860` | Local web server port |
| `APP_UI_MODE` | Server UI | Set to `blocks` to use the fallback Gradio Blocks interface |
| `SPACE_DISABLE_MODEL_WARMUP` | unset | Set to `1` to disable hosted startup model warmup |

To add another selectable medical model later, add a new entry under
`model.presets` with backend-specific names for the backends you want to support.

---

## Medication Autocomplete

Medication autocomplete uses a bundled RxTerms-derived JSON index at
`data/medications/rxterms_medications.json`. This keeps medication entry local for
desktop builds and Hugging Face Spaces. The app also allows custom medication,
vitamin, and supplement entries when an item is not in the index.

Refresh the RxTerms index with:

```bash
python scripts/sync_rxterms.py
```

The source ZIP download is ignored by git; the compact JSON index is committed.

---

## Project Structure

```
medical-appt-prep/
├── app.py               # Gradio UI and inference orchestration
├── assets/
│   ├── assistant-robot.jpeg # Friendly guide image used by the Server UI
│   ├── server.css       # Custom Server UI styling
│   ├── server.js        # Custom Server UI behavior
│   └── apple.css        # Fallback Blocks UI styling
├── config_loader.py     # Loads settings.yaml + .env overrides
├── config/
│   └── settings.yaml    # Model and server configuration
├── data/
│   └── medications/     # Bundled RxTerms autocomplete index
├── deploy/
│   └── huggingface-space/ # Gradio Space template
├── scripts/
│   ├── export_hf_space.py # Build a standalone HF Space repo folder
│   └── sync_rxterms.py  # Refreshes the local RxTerms index
├── src/
│   ├── __init__.py
│   ├── medications.py   # Medication autocomplete data helpers
│   ├── model.py         # LLM backends
│   ├── prompts.py       # Prompt template and required report markers
│   └── processor.py     # Input validation and output post-processing
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Disclaimer

This tool is for **informational and organizational purposes only**.
It does not provide medical diagnoses or replace professional medical advice.
Always consult a qualified healthcare provider for medical decisions.
