"""
LLM interface layer.

Backends:
  - OllamaModel  : uses the Ollama REST API (cross-platform, recommended)
  - LlamaCppModel: uses llama-cpp-python with a local .gguf file
"""

from __future__ import annotations

import abc
import json
import urllib.request
import urllib.error
from typing import Any


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLM(abc.ABC):
    """Minimal interface every backend must implement."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Run inference and return the raw response string."""
        ...

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
        model_name: str = "medgemma1.5",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        context_length: int = 4096,
        system_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.context_length = context_length
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
            max_tokens=1024,
        )
        return response["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        return self._llm is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant helping a patient prepare "
    "for a doctor's appointment. Provide clear, organized, and accurate "
    "information. Always remind the user to consult their healthcare provider "
    "for medical decisions. Use plain language."
)


def get_model(settings: dict) -> BaseLLM:
    """Instantiate the correct backend from settings dict."""
    model_cfg = settings.get("model", {})
    backend = model_cfg.get("backend", "ollama").lower()

    if backend == "ollama":
        return OllamaModel(
            model_name=model_cfg.get("name", "medgemma1.5"),
            base_url=model_cfg.get("ollama_base_url", "http://localhost:11434"),
            temperature=float(model_cfg.get("temperature", 0.3)),
            context_length=int(model_cfg.get("context_length", 4096)),
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
            n_gpu_layers=int(model_cfg.get("n_gpu_layers", 0)),
            system_prompt=_SYSTEM_PROMPT,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'llama_cpp'.")
