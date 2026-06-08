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

Required Space secrets / variables:

- `HF_TOKEN`: Hugging Face token with accepted access to `google/medgemma-1.5-4b-it`
- `MODEL_BACKEND=hf_transformers`
- `MODEL_NAME=google/medgemma-1.5-4b-it`
- `APP_DEPLOYMENT=huggingface`

Build this Space folder from the main repo with:

```bash
python scripts/export_hf_space.py /path/to/hf-space-repo
```
