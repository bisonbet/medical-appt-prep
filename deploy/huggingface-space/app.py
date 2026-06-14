"""
Hugging Face Space entrypoint.

This wrapper sets hosted defaults, then launches the shared application code
exported into the Space repository by scripts/export_hf_space.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_DEPLOYMENT", "huggingface")

HOSTED_LLAMA_CPP_DEFAULTS = {
    "MODEL_BACKEND": "llama_cpp",
    "MODEL_PRESET": "medgemma-4b",
    "LLAMA_CPP_MODEL_REPO_ID": "unsloth/medgemma-1.5-4b-it-GGUF",
    "LLAMA_CPP_MODEL_FILENAME": "medgemma-1.5-4b-it-Q4_K_M.gguf",
    "LLAMA_CPP_N_GPU_LAYERS": "-1",
    "LLAMA_CPP_N_BATCH": "2048",
    "LLAMA_CPP_N_UBATCH": "1024",
    "LLAMA_CPP_FLASH_ATTN": "1",
    "LLAMA_CPP_OP_OFFLOAD": "1",
    "LLAMA_CPP_SWA_FULL": "0",
    "MODEL_CONTEXT_LENGTH": "8192",
    "MODEL_MAX_NEW_TOKENS": "256",
    "MODEL_TEMPERATURE": "0.3",
}

if os.getenv("SPACE_USE_ENV_MODEL_CONFIG", "").strip().lower() in {"1", "true", "yes"}:
    for key, value in HOSTED_LLAMA_CPP_DEFAULTS.items():
        os.environ.setdefault(key, value)
else:
    for key, value in HOSTED_LLAMA_CPP_DEFAULTS.items():
        os.environ[key] = value

from shared_app import (  # noqa: E402
    APPLE_CSS_PATH,
    APPLE_THEME,
    THEME_MODE_HEAD,
    create_server_app,
    create_ui,
    settings,
)


if __name__ == "__main__":
    if os.getenv("APP_UI_MODE", "").strip().lower() == "blocks":
        ui = create_ui()
        ui.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=settings.get("server", {}).get("port", 7860),
            theme=APPLE_THEME,
            css_paths=[APPLE_CSS_PATH],
            head=THEME_MODE_HEAD,
            footer_links=["api"],
            share=False,
        )
    else:
        server = create_server_app()
        server.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=settings.get("server", {}).get("port", 7860),
            show_error=True,
        )
