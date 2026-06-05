# Medical Appointment Prep Assistant

A locally-running AI assistant that helps you prepare for doctor's appointments.
Enter your symptoms, notes, and medications — get back a symptom timeline, questions
to ask your doctor, and relevant background information.

**Everything runs on your machine. No data leaves your device.**

---

## Features

- Gradio web UI (runs in your browser)
- Two LLM backend options: **Ollama** (recommended) or **llama-cpp-python**
- Targets medical-focused models: MedGemma, MediPhi
- Works on Windows, macOS, and Linux

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
ollama pull medgemma
```

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

## Configuration

All settings live in `config/settings.yaml`. You can also override them with a `.env` file
(copy `.env.example` to `.env`).

| Key | Default | Description |
|-----|---------|-------------|
| `model.backend` | `ollama` | `ollama` or `llama_cpp` |
| `model.name` | `medgemma` | Ollama model name |
| `model.ollama_base_url` | `http://localhost:11434` | Ollama API URL |
| `model.model_path` | — | Path to `.gguf` file (llama_cpp only) |
| `model.temperature` | `0.3` | Generation temperature |
| `model.context_length` | `4096` | Context window size |
| `server.port` | `7860` | Local web server port |

---

## Project Structure

```
medical-appt-prep/
├── app.py               # Gradio UI and inference orchestration
├── config_loader.py     # Loads settings.yaml + .env overrides
├── config/
│   └── settings.yaml    # Model and server configuration
├── src/
│   ├── __init__.py
│   ├── model.py         # LLM backends (Ollama, llama-cpp-python)
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
