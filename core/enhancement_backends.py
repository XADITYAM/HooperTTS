"""Backend contracts for optional script enhancement providers."""

from __future__ import annotations

import gc
import os
import re
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
    validation_retry_count: int = 1
    retry_temperature: float = 0.25
    retry_top_p: float = 0.9

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
        validation_retry_count = os.getenv("HOOPERTTS_ENHANCEMENT_VALIDATION_RETRIES")
        retry_temperature = os.getenv("HOOPERTTS_ENHANCEMENT_RETRY_TEMPERATURE")
        retry_top_p = os.getenv("HOOPERTTS_ENHANCEMENT_RETRY_TOP_P")
        return cls(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            minimum_free_vram_gb=float(minimum) if minimum else None,
            do_sample=do_sample.lower() not in ("0", "false", "no") if do_sample else cls.do_sample,
            temperature=float(temperature) if temperature else cls.temperature,
            top_p=float(top_p) if top_p else cls.top_p,
            validation_retry_count=(
                max(0, int(validation_retry_count))
                if validation_retry_count
                else cls.validation_retry_count
            ),
            retry_temperature=(
                float(retry_temperature) if retry_temperature else cls.retry_temperature
            ),
            retry_top_p=float(retry_top_p) if retry_top_p else cls.retry_top_p,
        )


class HuggingFaceEnhancementBackend:
    """Lazy, single-use Transformers text-generation backend for script enhancement."""

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
        return self._generate_backend_result(
            text,
            analysis=analysis,
            policy=policy,
            feedback=None,
            retry=False,
        )

    def enhance_with_feedback(
        self,
        text: str,
        *,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
        diagnostics: tuple[str, ...],
        previous_candidate: str,
    ) -> BackendEnhancement:
        """Retry generation with explicit validator feedback.

        This is intentionally an optional backend capability. ScriptEnhancer checks
        for it only after a protected-span validation failure, so other backend
        implementations do not need to know about retries.
        """
        feedback = self._build_retry_feedback(diagnostics, previous_candidate)
        return self._generate_backend_result(
            text,
            analysis=analysis,
            policy=policy,
            feedback=feedback,
            retry=True,
        )

    def _generate_backend_result(
        self,
        text: str,
        *,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
        feedback: str | None,
        retry: bool,
    ) -> BackendEnhancement:
        started_at = perf_counter()
        try:
            self._load()
            prompt = self._build_prompt(
                text, analysis, policy, feedback=feedback, retry=retry
            )
            candidate = self._generate(
                prompt,
                temperature=self.config.retry_temperature if retry else self.config.temperature,
                top_p=self.config.retry_top_p if retry else self.config.top_p,
            )
            if not candidate:
                return BackendEnhancement(
                    text=text,
                    backend_name=self.name,
                    available=True,
                    diagnostic=(
                        "The enhancement model returned no usable script; "
                        "the original was preserved."
                    ),
                )
            elapsed = perf_counter() - started_at
            self.last_latency_seconds = elapsed
            self.last_device = self._device_label()
            attempt_label = "validation retry" if retry else "initial generation"
            return BackendEnhancement(
                text=candidate,
                backend_name=self.name,
                available=True,
                diagnostic=(
                    f"Generated a candidate with {self.config.model_id} ({attempt_label}) "
                    f"in {elapsed:.2f}s on {self._device_label()}."
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
        self,
        text: str,
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
        *,
        feedback: str | None = None,
        retry: bool = False,
    ) -> str:
        issues = "\n".join(
            f"- {issue.category}: {issue.recommendation}" for issue in analysis.issues
        ) or "- No mandatory changes. Leave strong sentences unchanged."
        goals = "\n".join(f"- {goal}" for goal in policy.writing_goals)
        avoid = "\n".join(f"- {item}" for item in policy.avoid)
        protected_facts = self._extract_prompt_facts(text)
        model_note = self._model_specific_instruction()
        retry_note = ""
        if retry and feedback:
            retry_note = f"""

VALIDATION RETRY — the previous candidate was rejected.
Fix every problem below while rewriting from the ORIGINAL SOURCE.
Do not mention the validation process in your output.
{feedback}
"""
        return f"""{model_note}

Rewrite this script for a natural spoken-video delivery only when a targeted improvement is useful.

OUTPUT CONTRACT
- Return ONLY the final revised script.
- No commentary, labels, markdown, notes, or explanation.
- Do not summarize the source. Rewrite the source. The result is not a copy of the original.
- Keep every factual claim and every concrete identifier.

FACT PRESERVATION CONTRACT
The checklist below is an immutable factual ledger. Every item must appear in the final output,
with the same factual meaning. Copy the exact wording where practical. You may change surrounding
grammar, punctuation, sentence position, or possessives, but you may NOT omit, generalize away,
replace with a vague phrase, or alter any checklist item.
- Numbers and years must survive.
- Dates may expand abbreviations (for example, Aug. 27 -> August 27) but the date itself must not change.
- Numbered titles must retain their numbers (for example, Red Dead Redemption 2 keeps the 2).
- Named people, companies, games, and organizations must remain identifiable.
- Do not invent new facts, entities, products, features, dates, prices, or platforms.

IMMUTABLE FACTS
{protected_facts}
{retry_note}

REWRITE GOALS
{goals}

AVOID
{avoid}

DETERMINISTIC ANALYSIS
{issues}

STRUCTURE RULES
- Preserve the source's paragraph/list-item boundaries unless combining them is clearly necessary for flow. Keep list items separate.
- Every list item or paragraph beat must remain clearly separated by punctuation; never flatten list items into a run-on sentence.
- Do not create sentence fragments by breaking immediately before/after words such as of, to, at, for, with, the, a, an, that, which, who, and, or, but.
- Prefer natural spoken clauses over arbitrary word-count chunks.

REWRITE EXAMPLE (structure only; do not reuse its wording):
Original: "A bakery opened in 1998. It sells 200 loaves a day."
Rewritten: "That bakery has been open since 1998 — and today, it turns out 200 loaves every day."
The numbers and meaning remain intact while the sentence structure changes.

ORIGINAL SOURCE
{text}

FINAL SELF-CHECK BEFORE OUTPUT
1. Every item in IMMUTABLE FACTS appears in the final script.
2. No number, date, title identifier, price, URL, or platform was dropped or changed.
3. The result is a genuine rewrite, not a summary.
4. The result contains no explanation or labels.
"""

    def _model_specific_instruction(self) -> str:
        model_id = self.config.model_id.casefold()
        if "phi-3.5" in model_id or "phi3.5" in model_id:
            return (
                "You are Phi-3.5-mini acting as a careful script editor. "
                "Follow explicit constraints literally. When a source fact appears in the immutable "
                "ledger, treat it as text that must be copied into the final rewrite rather than as "
                "information that may be paraphrased away. Preserve completeness before creativity."
            )
        return (
            "You are a careful script editor. Follow explicit constraints literally and preserve "
            "all immutable facts before applying stylistic improvements."
        )

    @staticmethod
    def _build_retry_feedback(
        diagnostics: tuple[str, ...], previous_candidate: str
    ) -> str:
        problems = "\n".join(f"- {item}" for item in diagnostics)
        return (
            "Validator failures from the previous attempt:\n"
            f"{problems}\n\n"
            "Previous attempt (use only to identify what went wrong; do not copy it blindly):\n"
            f"{previous_candidate}\n"
        )

    @staticmethod
    def _extract_prompt_facts(text: str) -> str:
        """Create a compact checklist of high-risk factual spans for small models."""
        patterns = (
            re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE),
            re.compile(r"(?:[$€£¥]\s?\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s?(?:USD|EUR|GBP|INR))\b", re.IGNORECASE),
            re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
                r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
                r"\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE
            ),
            re.compile(r"\b(?:19|20)\d{2}\b"),
            re.compile(r"\b(?:Grand Theft Auto 6|GTA 6|Red Dead Redemption 2)\b", re.IGNORECASE),
        )
        found: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(0).strip(".,;: ")
                key = value.casefold()
                if value and key not in seen:
                    seen.add(key)
                    found.append(value)
        return "\n".join(f"- {value}" for value in found) or "- No high-risk factual spans detected."

    def _generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
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
            generation_kwargs["temperature"] = (
                self.config.temperature if temperature is None else temperature
            )
            generation_kwargs["top_p"] = self.config.top_p if top_p is None else top_p
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
