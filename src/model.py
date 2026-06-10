"""
LLM interface layer.

Backends:
  - OllamaModel  : uses the Ollama REST API (cross-platform, recommended)
  - LlamaCppModel: uses llama-cpp-python with a local .gguf file
"""

from __future__ import annotations

import abc
import os
import json
import urllib.request
import urllib.error
from functools import lru_cache
from typing import Any

from src.model_catalog import resolve_model_settings


try:
    import spaces  # type: ignore[import]
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args: Any, **_kwargs: Any):
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()


_HF_MODEL: Any | None = None
_HF_PROCESSOR: Any | None = None


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: int = 5) -> set[str]:
    """Return locally available Ollama model names."""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url.rstrip('/')}. Is Ollama running?"
        ) from exc

    names = set()
    for model in body.get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            names.add(name)
    return names


def is_ollama_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Return True when the requested Ollama model is already pulled locally."""
    available = list_ollama_models(base_url)
    if model_name in available:
        return True
    if ":" not in model_name and f"{model_name}:latest" in available:
        return True
    return False


def pull_ollama_model(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 1800,
) -> str:
    """Pull an Ollama model using the local Ollama REST API."""
    url = f"{base_url.rstrip('/')}/api/pull"
    payload = {"model": model_name, "stream": False}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {model_name} from Ollama.") from exc

    status = body.get("status", "downloaded")
    return str(status)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLM(abc.ABC):
    """Minimal interface every backend must implement."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Run inference and return the raw response string."""
        ...

    def generate_report(self, prompt: str) -> str:
        """Run one report-generation inference call."""
        return self.generate(prompt)

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable / loaded."""
        ...


# ---------------------------------------------------------------------------
# Ollama backend (recommended — cross-platform, no Python bindings needed)
# ---------------------------------------------------------------------------

class OllamaModel(BaseLLM):
    """
    Talks to a locally-running Ollama daemon via its REST API.

    Install Ollama: https://ollama.ai
    Pull a model:  ollama pull medgemma1.5
    """

    def __init__(
        self,
        model_name: str = "medgemma1.5:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        context_length: int = 4096,
        max_new_tokens: int = 2048,
        system_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_length,
                "num_predict": self.max_new_tokens,
            },
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from exc

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# llama-cpp-python backend (direct GGUF loading, no daemon required)
# ---------------------------------------------------------------------------

class LlamaCppModel(BaseLLM):
    """
    Loads a .gguf model file directly via llama-cpp-python.

    Install:  pip install llama-cpp-python
              (GPU: see https://github.com/abetlen/llama-cpp-python for build flags)

    Usage: set backend: llama_cpp in config/settings.yaml and provide model_path.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.3,
        context_length: int = 4096,
        max_new_tokens: int = 2048,
        n_gpu_layers: int = 0,
        system_prompt: str = "",
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            ) from exc

        self.temperature = temperature
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self._llm = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        return self._llm is not None


# ---------------------------------------------------------------------------
# Hugging Face Transformers backend (Spaces / ZeroGPU)
# ---------------------------------------------------------------------------

class HuggingFaceTransformersModel(BaseLLM):
    """Runs MedGemma through Transformers for Hugging Face Spaces."""

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        temperature: float = 0.3,
        max_new_tokens: int = 2048,
        system_prompt: str = "",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "hf_transformers backend requires torch, transformers, and accelerate."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
        except ValueError:
            from transformers import AutoModelForMultimodalLM

            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
        global _HF_MODEL, _HF_PROCESSOR
        _HF_MODEL = self.model
        _HF_PROCESSOR = self.processor

    def generate(self, prompt: str) -> str:
        return _hf_generate(
            prompt,
            self.system_prompt,
            self.temperature,
            self.max_new_tokens,
        )

    def health_check(self) -> bool:
        return self.model is not None and self.processor is not None


@spaces.GPU(duration=120)
def _hf_generate(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    if _HF_MODEL is None or _HF_PROCESSOR is None:
        raise RuntimeError("Hugging Face model is not loaded.")

    import torch

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    inputs = _HF_PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_HF_MODEL.device)
    input_len = inputs["input_ids"].shape[-1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generation = _HF_MODEL.generate(**inputs, **generation_kwargs)
    return _HF_PROCESSOR.decode(generation[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (Nebius / serverless endpoints)
# ---------------------------------------------------------------------------

class OpenAICompatibleModel(BaseLLM):
    """Calls an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.3,
        max_new_tokens: int = 2048,
        system_prompt: str = "",
    ) -> None:
        if not base_url:
            raise ValueError("openai_compatible.base_url must be configured.")
        if not api_key:
            raise ValueError("OPENAI_COMPATIBLE_API_KEY must be configured.")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach OpenAI-compatible endpoint at {self.base_url}.") from exc
        return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def health_check(self) -> bool:
        return bool(self.base_url and self.api_key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant helping a patient prepare "
    "for a doctor's appointment. Provide clear, organized, and accurate "
    "information. Always remind the user to consult their healthcare provider "
    "for medical decisions. Use plain language."
)


def _model_cfg_key(settings_json: str) -> str:
    return settings_json


def get_model(settings: dict) -> BaseLLM:
    """Instantiate the correct backend from settings dict."""
    return _get_model_cached(json.dumps(settings, sort_keys=True))


@lru_cache(maxsize=4)
def _get_model_cached(settings_json: str) -> BaseLLM:
    settings = resolve_model_settings(json.loads(_model_cfg_key(settings_json)))
    model_cfg = settings.get("model", {})
    backend = model_cfg.get("backend", "ollama").lower()
    max_new_tokens = int(model_cfg.get("max_new_tokens", 2048))

    if backend == "ollama":
        return OllamaModel(
            model_name=model_cfg.get("name", "medgemma1.5:4b"),
            base_url=model_cfg.get("ollama_base_url", "http://localhost:11434"),
            temperature=float(model_cfg.get("temperature", 0.3)),
            context_length=int(model_cfg.get("context_length", 4096)),
            max_new_tokens=max_new_tokens,
            system_prompt=_SYSTEM_PROMPT,
        )
    elif backend in ("llama_cpp", "llama-cpp", "llamacpp"):
        model_path = model_cfg.get("model_path", "")
        if not model_path:
            raise ValueError(
                "model.model_path must be set in config/settings.yaml when using llama_cpp backend"
            )
        return LlamaCppModel(
            model_path=model_path,
            temperature=float(model_cfg.get("temperature", 0.3)),
            context_length=int(model_cfg.get("context_length", 4096)),
            max_new_tokens=max_new_tokens,
            n_gpu_layers=int(model_cfg.get("n_gpu_layers", 0)),
            system_prompt=_SYSTEM_PROMPT,
        )
    elif backend in ("hf_transformers", "huggingface", "transformers"):
        return HuggingFaceTransformersModel(
            model_name=model_cfg.get("name", "google/medgemma-1.5-4b-it"),
            temperature=float(model_cfg.get("temperature", 0.3)),
            max_new_tokens=max_new_tokens,
            system_prompt=_SYSTEM_PROMPT,
        )
    elif backend in ("openai_compatible", "openai-compatible", "nebius"):
        return OpenAICompatibleModel(
            model_name=model_cfg.get("name", "google/medgemma-1.5-4b-it"),
            base_url=model_cfg.get("openai_compatible_base_url", ""),
            api_key=model_cfg.get("openai_compatible_api_key", ""),
            temperature=float(model_cfg.get("temperature", 0.3)),
            max_new_tokens=max_new_tokens,
            system_prompt=_SYSTEM_PROMPT,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'ollama', 'llama_cpp', "
            "'hf_transformers', or 'openai_compatible'."
        )
