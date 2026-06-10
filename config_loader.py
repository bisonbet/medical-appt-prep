"""
Config loader — reads config/settings.yaml and merges .env overrides.
Kept at project root so `app.py` can import it without package machinery.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

_SETTINGS_PATH = Path(__file__).parent / "config" / "settings.yaml"


def load_settings() -> dict:
    with open(_SETTINGS_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Allow env-var overrides for common keys
    env_map = {
        "OLLAMA_BASE_URL": ("model", "ollama_base_url"),
        "MODEL_NAME": ("model", "name"),
        "MODEL_PRESET": ("model", "selected_preset"),
        "MODEL_PATH": ("model", "model_path"),
        "MODEL_BACKEND": ("model", "backend"),
        "BACKEND": ("model", "backend"),
        "APP_DEPLOYMENT": ("app", "deployment"),
        "OPENAI_COMPATIBLE_BASE_URL": ("model", "openai_compatible_base_url"),
        "OPENAI_COMPATIBLE_API_KEY": ("model", "openai_compatible_api_key"),
    }
    model_name_override = os.getenv("MODEL_NAME")
    model_preset_override = os.getenv("MODEL_PRESET")
    for env_key, (section, key) in env_map.items():
        val = os.getenv(env_key)
        if val:
            cfg.setdefault(section, {})[key] = val
    if model_name_override and not model_preset_override:
        cfg.setdefault("model", {}).pop("selected_preset", None)

    return cfg
