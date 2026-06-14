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
import re
import threading
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
_MODEL_FACTORY_LOCK = threading.Lock()
ZERO_GPU_DURATION_SECONDS = 60


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
        """Run one report-generation inference call for a prompt."""
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
        model_path: str = "",
        model_repo_id: str = "",
        model_filename: str = "",
        temperature: float = 0.3,
        context_length: int = 4096,
        max_new_tokens: int = 2048,
        n_gpu_layers: int = 0,
        n_batch: int = 512,
        n_ubatch: int = 512,
        flash_attn: bool = False,
        op_offload: bool | None = None,
        swa_full: bool | None = None,
        system_prompt: str = "",
    ) -> None:
        try:
            import llama_cpp  # type: ignore[import]
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            ) from exc

        self.temperature = temperature
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.model_name = model_repo_id or model_path
        self._warmed = False
        self._completion_lock = threading.Lock()

        if model_repo_id and model_filename:
            model_path = self._download_hub_gguf(model_repo_id, model_filename)
        if not model_path:
            raise ValueError("llama_cpp requires either model_path or model_repo_id/model_filename.")

        self.model_path = model_path
        supports_gpu_fn = getattr(llama_cpp, "llama_supports_gpu", None)
        supports_gpu = supports_gpu_fn() if callable(supports_gpu_fn) else "unknown"
        verbose = os.getenv("LLAMA_CPP_VERBOSE", "").strip().lower() in {"1", "true", "yes"}
        print(
            "[llama-cpp-check] "
            f"supports_gpu={supports_gpu} "
            f"n_gpu_layers={n_gpu_layers} "
            f"n_ctx={context_length} "
            f"n_batch={n_batch} "
            f"n_ubatch={n_ubatch} "
            f"flash_attn={flash_attn} "
            f"op_offload={op_offload} "
            f"swa_full={swa_full} "
            f"verbose={verbose}",
            flush=True,
        )
        self._llm = Llama(
            model_path=model_path,
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attn=flash_attn,
            op_offload=op_offload,
            swa_full=swa_full,
            verbose=verbose,
        )

    @staticmethod
    def _download_hub_gguf(repo_id: str, filename: str) -> str:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Loading llama_cpp models from Hugging Face requires huggingface-hub."
            ) from exc

        token = os.getenv("HF_TOKEN") or None
        split_pattern = re.sub(r"-\d{5}-of-\d{5}(\.gguf)$", r"-*of-*\1", filename)
        if split_pattern != filename:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=[split_pattern],
                token=token,
            )
            return os.path.join(snapshot_dir, filename)

        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        with self._completion_lock:
            if self._warmed:
                return
            self._llm.create_completion("Warmup:", max_tokens=1, temperature=0.0)
            self._warmed = True

    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        with self._completion_lock:
            response = self._llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            self._warmed = True
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


@spaces.GPU(duration=ZERO_GPU_DURATION_SECONDS)
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


def _optional_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def get_model(settings: dict) -> BaseLLM:
    """Instantiate the correct backend from settings dict."""
    with _MODEL_FACTORY_LOCK:
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
        model_repo_id = model_cfg.get("model_repo_id", "")
        model_filename = model_cfg.get("model_filename", "")
        if not model_path and not (model_repo_id and model_filename):
            raise ValueError(
                "model.model_path or model.model_repo_id/model.model_filename must be set "
                "when using llama_cpp backend"
            )
        return LlamaCppModel(
            model_path=model_path,
            model_repo_id=model_repo_id,
            model_filename=model_filename,
            temperature=float(model_cfg.get("temperature", 0.3)),
            context_length=int(model_cfg.get("context_length", 4096)),
            max_new_tokens=max_new_tokens,
            n_gpu_layers=int(model_cfg.get("n_gpu_layers", 0)),
            n_batch=int(model_cfg.get("n_batch", 512)),
            n_ubatch=int(model_cfg.get("n_ubatch", 512)),
            flash_attn=bool(_optional_bool(model_cfg.get("flash_attn"), False)),
            op_offload=_optional_bool(model_cfg.get("op_offload"), None),
            swa_full=_optional_bool(model_cfg.get("swa_full"), None),
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
