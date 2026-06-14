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

Hackathon focus: Backyard AI appointment preparation with a custom Gradio Server
UI, one-click fictional sample test data, full-report export actions, and a small
MedGemma 1.5 4B GGUF model running through llama.cpp.

Project links:

- Field Notes article: https://huggingface.co/blog/bisonnetworking/medical-asst-prep-june26
- Video walkthrough: https://youtu.be/kRHPwQGC6aU
- LinkedIn project update: https://www.linkedin.com/posts/timothy-champ_build-small-hackathon-starts-in-6-hours-activity-7471918851132207104-Wh7T?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEBVNSMBksj_L93Oketlrphm5iSG9BGmBqU

> I always find it hard to remember what to ask or mention in the middle of an
> appointment. Helping a parent with appointments can be even more confusing,
> and this helps ensure we get the right help needed. Having this sheet to read
> from or share with my doctor will help me in my future appointments.
>
> - My Spouse

The Space launches the custom `gradio.Server` interface by default: a warm
HTML/CSS/JS frontend backed by Gradio's queued API. Set `APP_UI_MODE=blocks`
only if you need the fallback Gradio Blocks interface.

At startup, the Space begins a background llama.cpp model download/load/warmup
so the first user request is less likely to pay the full cold-start cost. Set
`SPACE_DISABLE_MODEL_WARMUP=1` to turn that off temporarily.

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
