"""Unified LLM client implementations for VPR quality pipeline.

This module provides:
- Text-only LLM clients for Step 1 (caption → objects extraction)
- Vision-language model (VLM) clients for Step 3 (visual validation)
- Text-only LLM clients for Step 4 (final filtering)

Backends supported:
- Local HuggingFace (transformers): text-only and vision-language models
- OpenAI-compatible HTTP servers (e.g., vLLM): for VLMs
- Google Gemini API: for VLMs
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol, List
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# ============================================================================
#  Marker for OOM skips (Step 1)
# ============================================================================

OOM_SKIP_MARKER = "[SKIPPED_OOM_TOO_LONG]"


# ============================================================================
#  Configuration dataclasses
# ============================================================================

@dataclass
class TextLLMConfig:
    """Configuration for text-only LLM (Step 1: caption → objects)."""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    device: Optional[int] = None  # GPU index; -1 for CPU; None=auto
    use_4bit: bool = False  # Use 4-bit quantization


@dataclass
class VLMConfig:
    """Configuration for vision-language model (Step 3: VLM validation)."""
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    device: Optional[int] = None


@dataclass
class OpenAICompatConfig:
    """Configuration for OpenAI-compatible HTTP backend."""
    base_url: str = "http://localhost:8000"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini API."""
    api_key: str
    model: str = "gemini-2.5-flash"
    max_output_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


# Backwards-compatible aliases used by step1_caption_to_objects.py
LLMConfig = TextLLMConfig
# LLMClient alias is defined after the class below.

# ============================================================================
#  Text-only LLM Client (Step 1)
# ============================================================================

class TextLLMClient:
    """Local HuggingFace transformers client for text-only LLMs."""

    def __init__(self, config: TextLLMConfig) -> None:
        self.config = config

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        from transformers.utils import logging as hf_logging

        # Make transformers quiet
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        use_cuda = torch.cuda.is_available()
        model_kwargs = {"low_cpu_mem_usage": True}

        if use_cuda and self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Configure generation defaults
        try:
            gen_cfg = model.generation_config
            if gen_cfg is not None:
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = self.config.temperature
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = self.config.top_p
        except Exception:
            pass

        # Decide device for the pipeline
        if self.config.use_4bit and use_cuda:
            pipeline_device = None
        elif self.config.device is not None:
            pipeline_device = int(self.config.device)
        else:
            pipeline_device = 0 if use_cuda else -1

        self.generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            device=pipeline_device,
        )
        self._torch = torch

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text from a prompt."""
        actual_max_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=actual_max_tokens,
                do_sample=self.config.temperature > 0.0,
                num_return_sequences=1,
            )
        except (self._torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[TextLLM] CUDA OOM error - skipping", file=sys.stderr)
                self._cleanup_gpu_memory()
                return OOM_SKIP_MARKER
            else:
                raise

        if not outputs or "generated_text" not in outputs[0]:
            return ""

        return outputs[0]["generated_text"].strip()

    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[str]:
        """Generate text for a batch of prompts."""
        if not prompts:
            return []

        actual_max_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

        try:
            outputs = self.generator(
                prompts,
                max_new_tokens=actual_max_tokens,
                do_sample=self.config.temperature > 0.0,
                num_return_sequences=1,
            )

            results: List[str] = []
            for item in outputs:
                if not item or "generated_text" not in item[0]:
                    results.append("")
                    continue
                results.append(item[0]["generated_text"].strip())

            self._cleanup_gpu_memory()
            return results

        except (self._torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                raise

            print(f"[TextLLM] CUDA OOM during batch - falling back to one-by-one", file=sys.stderr)
            self._cleanup_gpu_memory()

            results: List[str] = []
            for prompt in prompts:
                results.append(self.generate(prompt, max_new_tokens=actual_max_tokens))
            return results

    def _postprocess_generated_text(self, generated_text: str) -> str:
        """Clean up and normalize raw model output into a single dot-separated list."""
        lines = generated_text.split("\n")
        cleaned_lines = []

        prefixes_to_remove = [
            "- Support:", "- support:", "Support:", "support:",
            "- Response:", "- response:", "Response:", "response:",
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            for prefix in prefixes_to_remove:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line.startswith("- ") and len(line) > 2:
                line = line[2:].strip()
            if line:
                cleaned_lines.append(line)

        full_text = " ".join(cleaned_lines)

        # Truncate at END_OF_LIST marker (or any prefix of it)
        full_marker = "### END_OF_LIST ###"
        prefixes = [full_marker[:i] for i in range(1, len(full_marker) + 1)]
        cut_idx = -1
        for marker in prefixes:
            idx = full_text.find(marker)
            if idx != -1:
                if cut_idx == -1 or idx < cut_idx:
                    cut_idx = idx
        if cut_idx != -1:
            full_text = full_text[:cut_idx].strip()

        # Parse into parts
        if ". " in full_text:
            parts = [p.strip() for p in full_text.split(". ") if p.strip()]
        elif ", " in full_text:
            parts = [p.strip() for p in full_text.split(", ") if p.strip()]
        else:
            parts = [full_text] if full_text else []

        # Deduplicate (case-insensitive, order-preserving)
        seen: set = set()
        unique_parts: List[str] = []
        for part in parts:
            part_clean = part.strip()
            if not part_clean or len(part_clean) < 2:
                continue
            part_lower = part_clean.lower()
            if part_lower not in seen:
                seen.add(part_lower)
                unique_parts.append(part_clean)

        # Detect repeated subsequence and truncate
        if len(unique_parts) > 6:
            first_three = tuple(unique_parts[:3])
            for i in range(3, len(unique_parts) - 2):
                if tuple(unique_parts[i:i + 3]) == first_three:
                    unique_parts = unique_parts[:i]
                    break

        return ". ".join(unique_parts)

    # ------------------------------------------------------------------
    # Aliases used by step1_caption_to_objects.py
    # ------------------------------------------------------------------
    def get_objects_from_caption(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate + post-process (single prompt)."""
        raw = self.generate(prompt, max_new_tokens=max_new_tokens)
        if raw == OOM_SKIP_MARKER:
            return raw
        return self._postprocess_generated_text(raw)

    def get_objects_from_captions_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate + post-process (batch of prompts)."""
        raws = self.generate_batch(prompts, max_new_tokens=max_new_tokens)
        return [
            r if r == OOM_SKIP_MARKER else self._postprocess_generated_text(r)
            for r in raws
        ]

    def _cleanup_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# Backwards-compatible alias so step1 can do: from src.utils.llm_clients import LLMClient
LLMClient = TextLLMClient

# ============================================================================
#  VLM Protocol and Utilities (Step 3)
# ============================================================================

class VLMClient(Protocol):
    """Protocol for vision-language model clients."""
    def is_object_in_image(self, *, image_path: str, object_name: str, description: Optional[str] = None) -> bool: ...
    def is_object_in_image_batch(self, queries: List[dict]) -> List[bool]: ...


_MODEL_ALIASES = {
    "qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-72b-instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}


def _normalize_model_name(model: str) -> str:
    """Normalize model name."""
    m = (model or "").strip()
    if not m:
        return m
    key = m.strip().lower()
    return _MODEL_ALIASES.get(key, m)


def _prompt_style() -> str:
    """Get active prompt style from environment."""
    style = (os.environ.get("VLLM_PROMPT_STYLE") or "strict_yn").strip().lower()
    if style in {"strict_yn", "yn", "yesno", "yes_no"}:
        return "strict_yn"
    if style in {"describe_then_yesno", "describe_then_yn", "describe_first", "cot"}:
        return "describe_then_yesno"
    return "strict_yn"


def _max_output_tokens_for_style(default_tokens: int) -> int:
    """Choose output token budget based on prompt style."""
    style = _prompt_style()
    if style == "describe_then_yesno":
        try:
            v = int(os.environ.get("VLLM_MAX_TOKENS_DESCRIBE_THEN_YESNO", "64"))
            return max(8, v)
        except Exception:
            return 64
    try:
        v = int(os.environ.get("VLLM_MAX_TOKENS_STRICT_YN", str(default_tokens)))
        return max(1, v)
    except Exception:
        return int(default_tokens)


def _build_object_presence_prompt(*, object_name: str) -> str:
    """Build prompt for object presence check."""
    obj = object_name.strip()
    style = _prompt_style()

    if style == "describe_then_yesno":
        return (
            "Describe the contents of the image briefly. Then determine if the target object is present.\n"
            + f"Target object: '{obj}'\n\n"
            + "Respond in exactly this format:\n"
            + "Description: <1-3 sentences>\n"
            + "Answer: yes|no"
        )

    # Legacy strict yes/no prompt
    base_question = f"Answer strictly with 'yes' or 'no'. Is there a '{obj}' clearly visible in this image?"
    return (
        "Example:\n"
        + "Input: Answer strictly with 'yes' or 'no'. Is there a 'tree canopies' clearly visible in this image?\n"
        + "Expected output: yes.\n\n"
        + f"Input: {base_question}\n"
        + "Expected output:"
    )


_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*answer\s*[:\-]\s*(yes|no)\b", re.IGNORECASE)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _extract_yes_no(raw_output: str) -> Optional[bool]:
    """Extract yes/no decision from model output."""
    raw = (raw_output or "").strip().lower()
    if not raw:
        return None

    m = _ANSWER_LINE_RE.search(raw)
    if m:
        return m.group(1).lower() == "yes"

    matches = _YESNO_RE.findall(raw)
    if matches:
        return matches[-1].lower() == "yes"

    first_token = raw.split()[0] if raw.split() else ""
    if first_token.startswith("y"):
        return True
    if first_token.startswith("n"):
        return False
    return None


# ============================================================================
#  Local HuggingFace VLM Client
# ============================================================================

class LocalHFVLMClient:
    """Vision-language client backed by HuggingFace VLM (e.g., Qwen2-VL)."""

    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText
        except Exception:
            AutoModelForImageTextToText = None
        try:
            from transformers import AutoModelForVision2Seq
        except Exception:
            AutoModelForVision2Seq = None
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as hf_logging
        from PIL import Image

        self._torch = torch
        self._Image = Image
        self._AutoProcessor = AutoProcessor
        self._hf_logging = hf_logging

        self.config = config or VLMConfig()
        self._device = torch.device("cpu")

        # Make transformers quiet
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # Decide device
        if self.config.device is not None:
            if self.config.device >= 0 and torch.cuda.is_available():
                device_str = f"cuda:{self.config.device}"
            else:
                device_str = "cpu"
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device_str)
        torch_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        # Choose model class
        model_cls = AutoModelForCausalLM
        if AutoModelForVision2Seq is not None:
            model_cls = AutoModelForVision2Seq
        if AutoModelForImageTextToText is not None:
            model_cls = AutoModelForImageTextToText

        model = model_cls.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self._device)

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._model = model

    def is_object_in_image(
        self, *, image_path: str, object_name: str, description: Optional[str] = None
    ) -> bool:
        """Check if object is present in image."""
        object_name = object_name.strip()
        if not object_name:
            return False

        try:
            image = self._Image.open(image_path).convert("RGB")
        except Exception:
            return False

        prompt = _build_object_presence_prompt(object_name=object_name)

        # Use chat templating if available
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                templated = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                proc_inputs = self._processor(
                    text=[templated], images=[image], return_tensors="pt", padding=True
                )
            except Exception:
                proc_inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        else:
            proc_inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        inputs = proc_inputs.to(self._device)
        generated_ids = self._model.generate(
            **inputs, max_new_tokens=_max_output_tokens_for_style(self.config.max_new_tokens)
        )

        # Decode only generated continuation
        input_len = 0
        try:
            input_ids = None
            if hasattr(inputs, "get"):
                input_ids = inputs.get("input_ids")
            if input_ids is None and hasattr(inputs, "input_ids"):
                input_ids = getattr(inputs, "input_ids")
            if input_ids is not None and hasattr(input_ids, "shape"):
                input_len = int(input_ids.shape[1])
        except Exception:
            input_len = 0

        to_decode = generated_ids[:, input_len:] if input_len > 0 else generated_ids
        raw = self._processor.batch_decode(to_decode, skip_special_tokens=True)[0]

        parsed = _extract_yes_no(raw)
        return bool(parsed) if parsed is not None else False

    def is_object_in_image_batch(self, queries: List[dict]) -> List[bool]:
        """Batch variant of is_object_in_image."""
        if not queries:
            return []

        results: List[bool] = []
        for q in queries:
            image_path = (q.get("image_path") or "").strip()
            object_name = (q.get("object_name") or "").strip()
            description = q.get("description") or None
            if not object_name:
                results.append(False)
                continue

            present = self.is_object_in_image(
                image_path=image_path, object_name=object_name, description=description
            )
            results.append(present)

        return results


# ============================================================================
#  OpenAI-Compatible HTTP VLM Client
# ============================================================================

def _guess_mime_type(image_path: str) -> str:
    """Guess MIME type from file extension."""
    suffix = Path(image_path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _image_to_data_url(image_path: str) -> str:
    """Convert image to data URL."""
    mime = _guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenAICompatVLMClient:
    """VLM client using OpenAI-compatible HTTP API."""

    def __init__(self, config: Optional[OpenAICompatConfig] = None) -> None:
        self.config = config or OpenAICompatConfig()
        self._base_url = self.config.base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/v1/chat/completions"

    def is_object_in_image(
        self, *, image_path: str, object_name: str, description: Optional[str] = None
    ) -> bool:
        """Check if object is present in image."""
        object_name = object_name.strip()
        if not object_name:
            return False

        try:
            img_url = _image_to_data_url(image_path)
        except Exception:
            return False

        prompt = _build_object_presence_prompt(object_name=object_name)

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": _max_output_tokens_for_style(self.config.max_tokens),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        raw = self._post_chat_completions(payload)
        parsed = _extract_yes_no(raw)
        return bool(parsed) if parsed is not None else False

    def is_object_in_image_batch(self, queries: List[dict]) -> List[bool]:
        """Batch variant of is_object_in_image."""
        if not queries:
            return []

        out: List[bool] = []
        for q in queries:
            out.append(
                self.is_object_in_image(
                    image_path=(q.get("image_path") or "").strip(),
                    object_name=(q.get("object_name") or "").strip(),
                    description=q.get("description") or None,
                )
            )
        return out

    def _post_chat_completions(self, payload: dict) -> str:
        """POST to /v1/chat/completions endpoint."""
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib_request.Request(self._endpoint, data=body, headers=headers, method="POST")

        transient_http = {408, 409, 425, 429, 500, 502, 503, 504}
        last_exc: Optional[Exception] = None
        attempts = max(0, int(self.config.max_retries)) + 1

        for attempt in range(attempts):
            try:
                with urllib_request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    data = resp.read().decode("utf-8")
                last_exc = None
                break
            except HTTPError as e:
                last_exc = e
                try:
                    details = e.read().decode("utf-8")
                except Exception:
                    details = ""
                if int(getattr(e, "code", 0) or 0) in transient_http and attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible request failed (HTTP {e.code}).\nResponse: {details}"
                ) from e
            except (URLError, TimeoutError) as e:
                last_exc = e
                if attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Failed to reach endpoint at {self._endpoint}.\nReason: {e}") from e

        if last_exc is not None:
            raise RuntimeError(f"Failed to reach endpoint at {self._endpoint}.\nReason: {last_exc}") from last_exc

        try:
            parsed = json.loads(data)
            return parsed.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as e:
            raise RuntimeError(f"Failed to parse response as JSON.\nRaw: {data[:2000]}") from e


# ============================================================================
#  Google Gemini VLM Client
# ============================================================================

def _image_to_inline_data(image_path: str) -> tuple[str, str]:
    """Return (mime_type, base64_data) for Gemini inline_data payloads."""
    mime = _guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64


class GeminiVLMClient:
    """VLM client using Google Gemini API."""

    def __init__(self, config: GeminiConfig) -> None:
        if not (config.api_key or "").strip():
            raise ValueError("Gemini API key is missing.")
        self.config = config

    def is_object_in_image(
        self, *, image_path: str, object_name: str, description: Optional[str] = None
    ) -> bool:
        """Check if object is present in image."""
        object_name = object_name.strip()
        if not object_name:
            return False

        try:
            mime, b64 = _image_to_inline_data(image_path)
        except Exception:
            return False

        prompt = _build_object_presence_prompt(object_name=object_name)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": float(self.config.temperature),
                "topP": float(self.config.top_p),
                "maxOutputTokens": int(_max_output_tokens_for_style(self.config.max_output_tokens)),
            },
        }

        raw = self._post_generate_content(payload)
        parsed = _extract_yes_no(raw)
        return bool(parsed) if parsed is not None else False

    def is_object_in_image_batch(self, queries: List[dict]) -> List[bool]:
        """Batch variant of is_object_in_image."""
        if not queries:
            return []

        out: List[bool] = []
        for q in queries:
            out.append(
                self.is_object_in_image(
                    image_path=(q.get("image_path") or "").strip(),
                    object_name=(q.get("object_name") or "").strip(),
                    description=q.get("description") or None,
                )
            )
        return out

    def _post_generate_content(self, payload: dict) -> str:
        """POST to Gemini generateContent endpoint."""
        model = (self.config.model or "").strip()
        if not model:
            raise ValueError("Gemini model name is empty.")

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={self.config.api_key}"
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        req = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")

        transient_http = {408, 409, 425, 429, 500, 502, 503, 504}
        last_exc: Optional[Exception] = None
        attempts = max(0, int(self.config.max_retries)) + 1

        for attempt in range(attempts):
            try:
                with urllib_request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    data = resp.read().decode("utf-8")
                last_exc = None
                break
            except HTTPError as e:
                last_exc = e
                try:
                    details = e.read().decode("utf-8")
                except Exception:
                    details = ""
                code = int(getattr(e, "code", 0) or 0)
                if code in transient_http and attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Gemini request failed (HTTP {code}).\nResponse: {details[:4000]}") from e
            except (URLError, TimeoutError) as e:
                last_exc = e
                if attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"Failed to reach Gemini endpoint. Reason: {e}") from e

        if last_exc is not None:
            raise RuntimeError(f"Failed to reach Gemini endpoint. Reason: {last_exc}") from last_exc

        try:
            parsed = json.loads(data)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Gemini response as JSON.\nRaw: {data[:2000]}") from e

        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed.get("error") or {}
            msg = (err.get("message") or str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Gemini API error: {msg}")

        try:
            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            if not candidates:
                return ""
            content = (candidates[0] or {}).get("content", {}) or {}
            parts = content.get("parts", []) or []
            texts: List[str] = []
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    texts.append(p["text"])
            return "\n".join(texts).strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected Gemini response shape.\nRaw: {data[:2000]}") from e


# ============================================================================
#  Client builders
# ============================================================================

def build_vlm_client(provider: Optional[str] = None) -> VLMClient:
    """Build a VLM client based on provider."""
    provider = (provider or os.environ.get("VLLM_PROVIDER") or "auto").strip().lower()

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        model = os.environ.get("GEMINI_MODEL") or GeminiConfig.model
        return GeminiVLMClient(GeminiConfig(api_key=str(api_key), model=str(model)))

    if provider in ("openai", "openai_compat"):
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("VLLM_OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("OpenAI-compatible provider requires OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        model = os.environ.get("OPENAI_MODEL") or OpenAICompatConfig.model
        model = _normalize_model_name(str(model))
        return OpenAICompatVLMClient(OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model))

    if provider in ("local", "local_hf", "hf"):
        local_model = os.environ.get("VLLM_HF_MODEL") or VLMConfig.model_name
        local_model = _normalize_model_name(str(local_model))
        return LocalHFVLMClient(VLMConfig(model_name=str(local_model)))

    # Auto: check for OPENAI_BASE_URL, else use local
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("VLLM_OPENAI_BASE_URL")
    if base_url:
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        model = os.environ.get("OPENAI_MODEL") or OpenAICompatConfig.model
        model = _normalize_model_name(str(model))
        return OpenAICompatVLMClient(OpenAICompatConfig(base_url=base_url, api_key=api_key, model=model))

    # Default to local HF
    local_model = os.environ.get("VLLM_HF_MODEL") or VLMConfig.model_name
    local_model = _normalize_model_name(str(local_model))
    return LocalHFVLMClient(VLMConfig(model_name=str(local_model)))


# Backwards-compatible aliases used by step3 / checker
build_default_client = build_vlm_client
TextOnlyLLMClient = LocalHFVLMClient
