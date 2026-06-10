"""Model preset catalog helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BACKEND_ALIASES = {
    "ollama": "ollama",
    "llama_cpp": "llama_cpp",
    "llama-cpp": "llama_cpp",
    "llamacpp": "llama_cpp",
    "hf_transformers": "hf_transformers",
    "huggingface": "hf_transformers",
    "transformers": "hf_transformers",
    "openai_compatible": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "nebius": "openai_compatible",
}


def canonical_backend(backend: str | None) -> str:
    """Normalize backend aliases used in config and environment variables."""
    return BACKEND_ALIASES.get((backend or "ollama").lower(), (backend or "ollama").lower())


def get_model_presets(settings: dict[str, Any], backend: str | None = None) -> list[dict[str, Any]]:
    """Return configured model presets, optionally filtered by backend support."""
    model_cfg = settings.get("model", {})
    presets = model_cfg.get("presets") or []
    if backend is None:
        return [preset for preset in presets if preset.get("id")]

    backend_key = canonical_backend(backend)
    supported = []
    for preset in presets:
        if not preset.get("id"):
            continue
        backends = preset.get("backends") or {}
        if backend_key in backends:
            supported.append(preset)
    return supported


def get_default_model_preset_id(settings: dict[str, Any], backend: str | None = None) -> str:
    """Pick the configured preset if available for the backend, otherwise the first one."""
    model_cfg = settings.get("model", {})
    backend_key = canonical_backend(backend or model_cfg.get("backend", "ollama"))
    presets = get_model_presets(settings, backend_key)
    if "selected_preset" not in model_cfg:
        return ""
    selected_preset = model_cfg.get("selected_preset")
    if selected_preset and any(preset["id"] == selected_preset for preset in presets):
        return selected_preset
    if selected_preset and presets:
        return presets[0]["id"]
    return ""


def get_model_preset_choices(settings: dict[str, Any], backend: str | None = None) -> list[tuple[str, str]]:
    """Return Gradio dropdown choices for presets available to the selected backend."""
    model_cfg = settings.get("model", {})
    backend_key = canonical_backend(backend or model_cfg.get("backend", "ollama"))
    return [
        (preset.get("label") or preset["id"], preset["id"])
        for preset in get_model_presets(settings, backend_key)
    ]


def get_model_preset(settings: dict[str, Any], preset_id: str | None) -> dict[str, Any] | None:
    """Find a configured model preset by id."""
    for preset in get_model_presets(settings):
        if preset.get("id") == preset_id:
            return preset
    return None


def describe_model_preset(settings: dict[str, Any], preset_id: str | None) -> str:
    """Build a concise label for the selected preset and backend model."""
    model_cfg = settings.get("model", {})
    backend_key = canonical_backend(model_cfg.get("backend", "ollama"))
    preset = get_model_preset(settings, preset_id)
    if not preset:
        return f"Configured model: {model_cfg.get('name', 'medgemma')}."

    backend_cfg = (preset.get("backends") or {}).get(backend_key, {})
    backend_model_name = backend_cfg.get("name", model_cfg.get("name", "medgemma"))
    description = preset.get("description", "")
    if description:
        return f"{description} Backend model: {backend_model_name}."
    return f"Backend model: {backend_model_name}."


def resolve_model_settings(settings: dict[str, Any], preset_id: str | None = None) -> dict[str, Any]:
    """Return a settings copy with the selected preset resolved to backend-specific keys."""
    resolved = deepcopy(settings)
    model_cfg = resolved.setdefault("model", {})
    backend_key = canonical_backend(model_cfg.get("backend", "ollama"))
    selected_preset = preset_id or get_default_model_preset_id(resolved, backend_key)
    preset = get_model_preset(resolved, selected_preset)

    if not preset:
        return resolved

    backend_cfg = (preset.get("backends") or {}).get(backend_key)
    if not backend_cfg:
        return resolved

    model_cfg["selected_preset"] = preset["id"]
    for key, value in backend_cfg.items():
        if value is not None:
            model_cfg[key] = value
    return resolved
