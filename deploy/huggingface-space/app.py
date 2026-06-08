"""
Hugging Face Space entrypoint.

This wrapper sets hosted defaults, then launches the shared application code
exported into the Space repository by scripts/export_hf_space.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_DEPLOYMENT", "huggingface")
os.environ.setdefault("MODEL_BACKEND", "hf_transformers")
os.environ.setdefault("MODEL_NAME", "google/medgemma-1.5-4b-it")

from shared_app import APPLE_CSS_PATH, APPLE_THEME, THEME_MODE_HEAD, create_ui, settings  # noqa: E402


if __name__ == "__main__":
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
