"""Backend contracts for optional script enhancement providers."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Protocol

from .enhancement_policy import EnhancementPolicy
from .script_analysis import ScriptAnalysis


@dataclass(frozen=True)
class BackendEnhancement:
    """A candidate enhancement returned by a backend."""

    text: str
    backend_name: str
    available: bool
    diagnostic: str


class EnhancementBackend(Protocol):
    """Interface implemented by optional enhancement providers."""

    def enhance(
        self,
        text: str,
        *,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
    ) -> BackendEnhancement:
        """Return a candidate enhanced script without performing narration optimization."""


@dataclass(frozen=True)
class HuggingFaceEnhancementConfig:
    """Runtime configuration for the optional Transformers enhancement backend."""

    model_id: str = "Qwen/Qwen3-1.7B"
    max_new_tokens: int = 320
    device_map: str = "auto"
    minimum_free_vram_gb: float | None = None
    # Greedy decoding (do_sample=False) always picks the single highest-probability
    # token at each step, which tends to stay close to a near-paraphrase of the
    # input even when the prompt explicitly asks for a bolder rewrite (e.g. a
    # hook-driven restructure). Sampling is on by default so the model can
    # actually take the creative liberty the writing goals ask for; the
    # protected-span validator remains the hard safety net regardless of
    # decoding strategy, so this doesn't weaken fact protection.
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9

    @classmethod
    def from_environment(cls) -> "HuggingFaceEnhancementConfig":
        """Load optional configuration without importing Transformers or Torch."""
        model_id = os.getenv("HOOPERTTS_ENHANCEMENT_MODEL_ID", cls.model_id)
        max_new_tokens = int(
            os.getenv("HOOPERTTS_ENHANCEMENT_MAX_NEW_TOKENS", str(cls.max_new_tokens))
        )
        device_map = os.getenv("HOOPERTTS_ENHANCEMENT_DEVICE_MAP", cls.device_map)
        minimum = os.getenv("HOOPERTTS_ENHANCEMENT_MIN_FREE_VRAM_GB")
        do_sample = os.getenv("HOOPERTTS_ENHANCEMENT_DO_SAMPLE")
        temperature = os.getenv("HOOPERTTS_ENHANCEMENT_TEMPERATURE")
        top_p = os.getenv("HOOPERTTS_ENHANCEMENT_TOP_P")
        return cls(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            minimum_free_vram_gb=float(minimum) if minimum else None,
            do_sample=do_sample.lower() not in ("0", "false", "no") if do_sample else cls.do_sample,
            temperature=float(temperature) if temperature else cls.temperature,
            top_p=float(top_p) if top_p else cls.top_p,
        )


class HuggingFaceEnhancementBackend:
    """Lazy, single-use Qwen3 text-generation backend for script enhancement."""

    name = "huggingface"

    def __init__(
        self,
        config: HuggingFaceEnhancementConfig | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or HuggingFaceEnhancementConfig.from_environment()
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self.last_device: str | None = None
        self.last_latency_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether a model is currently retained in memory."""
        return self._model is not None and self._tokenizer is not None

    def enhance(
        self,
        text: str,
        *,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
    ) -> BackendEnhancement:
        """Generate one candidate, then immediately release model resources."""
        started_at = perf_counter()
        try:
            self._load()
            prompt = self._build_prompt(text, analysis, policy)
            candidate = self._generate(prompt)
            if not candidate:
                return BackendEnhancement(
                    text=text,
                    backend_name=self.name,
                    available=True,
                    diagnostic="The enhancement model returned no usable script; the original was preserved.",
                )
            elapsed = perf_counter() - started_at
            self.last_latency_seconds = elapsed
            self.last_device = self._device_label()
            return BackendEnhancement(
                text=candidate,
                backend_name=self.name,
                available=True,
                diagnostic=(
                    f"Generated a candidate with {self.config.model_id} in {elapsed:.2f}s "
                    f"on {self._device_label()}."
                ),
            )
        except Exception as exc:
            self.last_latency_seconds = perf_counter() - started_at
            self.last_device = None
            return BackendEnhancement(
                text=text,
                backend_name=self.name,
                available=False,
                diagnostic=(
                    f"Hugging Face enhancement failed for {self.config.model_id}: {exc}. "
                    "The original script was preserved."
                ),
            )
        finally:
            self.release()

    def release(self) -> None:
        """Release the enhancement model before any later TTS model load."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self) -> None:
        if self.is_loaded:
            return
        self._check_resources()
        if self._tokenizer_loader is None or self._model_loader is None:
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            self._tokenizer_loader = AutoTokenizer.from_pretrained
            self._model_loader = AutoModelForCausalLM.from_pretrained
        load_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "torch_dtype": "auto",
        }
        self._tokenizer = self._retry_on_transient_error(
            lambda: self._tokenizer_loader(self.config.model_id)
        )
        self._model = self._retry_on_transient_error(
            lambda: self._model_loader(self.config.model_id, **load_kwargs)
        )

    def _retry_on_transient_error(
        self,
        load_fn: Callable[[], Any],
        max_retries: int = 4,
        backoff_seconds: float = 10.0,
    ) -> Any:
        """Retry a Hugging Face Hub download on transient connection failures
        (e.g. a dropped connection mid-download, surfaced as IncompleteRead),
        matching the retry-with-backoff pattern already used for the
        Qwen3-TTS model download in the Colab notebook. Colab's free-tier
        networking is known to drop connections mid-download; without this,
        a single flaky moment silently discards the whole enhancement attempt
        and falls back to the original script."""
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return load_fn()
            except Exception as exc:  # noqa: BLE001 - deliberately broad; this
                # wraps a network download step, not business logic, and the
                # failure modes (connection resets, incomplete reads, CDN
                # hiccups) surface as many different exception types.
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    def _check_resources(self) -> None:
        """Fail early on insufficient CUDA memory instead of risking a TTS crash."""
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Missing optional dependencies. Install hoopertts[enhancement] first"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Script enhancement is disabled to avoid slow CPU "
                "inference; use a GPU runtime or keep Optimize Only selected"
            )
        free_bytes, _ = torch.cuda.mem_get_info()
        required_gb = self.config.minimum_free_vram_gb
        if required_gb is None:
            if "0.6B" in self.config.model_id:
                required_gb = 2.0
            elif "Phi-3.5" in self.config.model_id:
                required_gb = 6.0
            else:
                required_gb = 4.5
        if free_bytes < required_gb * 1024**3:
            raise RuntimeError(
                f"Only {free_bytes / 1024**3:.1f} GiB of CUDA memory is free; "
                f"{required_gb:.1f} GiB is required for {self.config.model_id}. "
                "Release other GPU models or set HOOPERTTS_ENHANCEMENT_MODEL_ID="
                "Qwen/Qwen3-0.6B"
            )

    def _build_prompt(
        self, text: str, analysis: ScriptAnalysis, policy: EnhancementPolicy
    ) -> str:
        issues = "\n".join(
            f"- {issue.category}: {issue.recommendation}" for issue in analysis.issues
        ) or "- No mandatory changes. Leave strong sentences unchanged."
        goals = "\n".join(f"- {goal}" for goal in policy.writing_goals)
        avoid = "\n".join(f"- {item}" for item in policy.avoid)
        return f"""Rewrite this script only if a targeted improvement is useful.

Return ONLY the revised script, with no commentary, labels, markdown, or explanation.
Preserve every factual claim. Do not invent, remove, or alter game features, names,
organizations, dates, numbers, prices, platforms, URLs, or quotations. Do not use
creator-specific wording.
Your response must be a genuine rewrite, not a copy of the original wording. Returning
the original script unchanged, or changing only a word or two, is not acceptable unless
the writing goals below explicitly call for minimal changes — reread the writing goals
and actually apply them to the sentence structure and phrasing.

Example of the kind of transformation expected (structure only — do not reuse this
example's wording or topic in your actual answer):
Original: "A local bakery opened in 1998. It sells 200 loaves a day. The owner learned
baking from her grandmother."
Rewritten: "Everything the owner knows about baking, she learned from her grandmother —
and today it adds up to 200 loaves a day, rolling out of a bakery that's been open
since 1998."
Notice every fact (1998, 200 loaves, grandmother) survives exactly, but the sentence
order, structure, and phrasing are substantially different. Apply this same kind of
restructuring to the script below, in the direction the writing goals describe.

If the source has line breaks separating list items or bullet points, keep each item
as its own sentence or clause with clear ending punctuation. Never merge separate
list items into a single run-on sentence with no punctuation between them.
Follow the writing goals below even if it means substantially restructuring the
script (e.g. reordering for a stronger opening hook) — restraint vs. boldness is set
by the writing goals and avoid list, not by a fixed rule to leave sentences as-is.

Writing goals for the selected profile:
{goals}

Avoid:
{avoid}

Deterministic analysis observations:
{issues}

Original script:
{text}
"""

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            model_inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            model_inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        model_inputs = model_inputs.to(self._model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": getattr(self._tokenizer, "eos_token_id", None),
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        generated_ids = self._model.generate(**model_inputs, **generation_kwargs)
        input_length = model_inputs["input_ids"].shape[-1]
        output_ids = generated_ids[0][input_length:]
        return self._clean_candidate(
            self._tokenizer.decode(output_ids, skip_special_tokens=True)
        )

    def _clean_candidate(self, text: str) -> str:
        candidate = text.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if candidate.lower().startswith("enhanced script:"):
            candidate = candidate.split(":", 1)[1].strip()
        return candidate

    def _device_label(self) -> str:
        device = getattr(self._model, "device", "unknown")
        return str(device)


class UnavailableEnhancementBackend:
    """Safe default backend used when no text-generation provider is configured."""

    name = "unavailable"

    def enhance(
        self,
        text: str,
        *,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
    ) -> BackendEnhancement:
        """Return the original script and explain that no backend is configured."""
        return BackendEnhancement(
            text=text,
            backend_name=self.name,
            available=False,
            diagnostic=(
                "Script enhancement is unavailable because no optional enhancement "
                "backend is configured. The original script was preserved."
            ),
        )
