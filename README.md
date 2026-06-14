# Medical Appointment Prep Assistant

A dual-mode AI assistant that helps you prepare for doctor's appointments.
Enter your symptoms, notes, and medications — get back a symptom timeline, questions
to ask your doctor, and relevant background information.

Local mode runs on your machine with Ollama or llama.cpp. Hosted mode is designed
for Hugging Face Spaces now and OpenAI-compatible serverless endpoints such as
Nebius later.

---

## Features

- Gradio 6 web UI (runs in your browser)
- Custom `gradio.Server` frontend for a warmer, non-technical appointment-prep experience
- Separate focused generation calls fill each report tab: Timeline, Questions, Relevant Info
- Export actions for email drafts, PDF/print, clipboard, portal copy, and text download
- Four LLM backend options: **Ollama**, **llama-cpp-python**, **Hugging Face Transformers**, or **OpenAI-compatible**
- Uses MedGemma 1.5 4B by default, with model settings hidden from non-technical users
- Local RxTerms medication autocomplete with custom entry fallback
- Works on Windows, macOS, Linux, and Hugging Face Spaces

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
git clone https://github.com/YOUR_USERNAME/medical-appt-prep.git
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

Download a compatible `.gguf` from [Hugging Face](https://huggingface.co) (search for
`medgemma gguf` or `MediPhi gguf`).

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
| `model.max_new_tokens` | `256` | Generation budget for each generated section |
| `app.deployment` | `local` | UI copy mode: `local`, `huggingface`, or `hosted` |
| `model.temperature` | `0.3` | Generation temperature |
| `model.context_length` | `8192` | Context window size |
| `server.port` | `7860` | Local web server port |
| `APP_UI_MODE` | Server UI | Set to `blocks` to use the fallback Gradio Blocks interface |

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
│   ├── prompts.py       # Prompt templates for each output type
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
