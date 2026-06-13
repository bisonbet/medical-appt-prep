---
title: Medical Appointment Prep
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
---

# Medical Appointment Prep

Hosted Gradio Space for the Medical Appointment Prep Assistant.

Required Space secret:

- `HF_TOKEN`: Hugging Face token, used for Hub model downloads when needed

Hosted model defaults:

- `MODEL_BACKEND=llama_cpp`
- `MODEL_PRESET=medgemma-4b`
- `LLAMA_CPP_MODEL_REPO_ID=unsloth/medgemma-1.5-4b-it-GGUF`
- `LLAMA_CPP_MODEL_FILENAME=medgemma-1.5-4b-it-Q4_K_M.gguf`
- `LLAMA_CPP_N_GPU_LAYERS=-1`
- `LLAMA_CPP_N_BATCH=2048`
- `LLAMA_CPP_N_UBATCH=1024`
- `LLAMA_CPP_FLASH_ATTN=1`
- `LLAMA_CPP_OP_OFFLOAD=1`
- `LLAMA_CPP_SWA_FULL=0`
- `MODEL_CONTEXT_LENGTH=8192`
- `MODEL_MAX_NEW_TOKENS=256`
- `MODEL_TEMPERATURE=0.3`
- `APP_DEPLOYMENT=huggingface`

These defaults are enforced by `app.py` so stale Space Variables do not switch
the competition build back to another backend. Set `SPACE_USE_ENV_MODEL_CONFIG=1`
only if you intentionally want Space Variables to override the model backend.

The hosted Space installs the CUDA 13.0 `llama-cpp-python` wheel, requests
ZeroGPU around report generation, and downloads the Unsloth MedGemma 1.5 4B
Q4_K_M GGUF from the Hub at startup/cache time.
Set `LLAMA_CPP_VERBOSE=1` temporarily if you need full llama.cpp CUDA layer
offload logs for verification.

Build this Space folder from the main repo with:

```bash
python scripts/export_hf_space.py /path/to/hf-space-repo
```
