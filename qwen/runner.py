"""Native Qwen3-TTS generation runner."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.enhancement_backends import HuggingFaceEnhancementBackend, HuggingFaceEnhancementConfig
from core.planner import NarrationPlanner
from core.profile import ProfileManager
from core.script_enhancement import EnhancementMode, ScriptEnhancer, ScriptIntelligence
from .environment import diagnose, format_diagnostics
from .prompt_builder import QwenPrompt, build_prompt


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Script-enhancer model tiers offered in the CLI and Gradio UI. Quality is the
# default: Qwen3-1.7B gives materially better pacing/emphasis output than the
# 0.6B model for a modest VRAM cost. Fast exists for constrained Colab
# sessions (e.g. when the free-tier T4 is already tight on memory).
ENHANCEMENT_MODEL_TIERS: dict[str, str] = {
    "quality": "Qwen/Qwen3-1.7B",
    "fast": "Qwen/Qwen3-0.6B",
}
DEFAULT_ENHANCEMENT_MODEL_TIER = "quality"


@dataclass(frozen=True)
class GenerationResult:
    """Result of an attempted Qwen generation."""

    success: bool
    output_path: str | None
    diagnostics: str
    prompt: QwenPrompt | None = None
    enhancement_diagnostic: str | None = None


def generate(
    script_path: str | Path,
    reference_audio: str | Path | None,
    profile: str,
    output_path: str | Path,
    enhancement_mode: EnhancementMode | str = EnhancementMode.OPTIMIZE_ONLY,
    enhancement_model_tier: str = DEFAULT_ENHANCEMENT_MODEL_TIER,
) -> GenerationResult:
    """Optimize (and optionally enhance) a script, build a Qwen prompt, and generate audio.

    Script enhancement is opt-in and off by default (``EnhancementMode.OPTIMIZE_ONLY``),
    so existing callers keep their current behavior unchanged. Passing
    ``ENHANCE_ONLY`` or ``ENHANCE_AND_OPTIMIZE`` lazily loads the Hugging Face
    Qwen3 backend for a single generation call, then releases it before the
    Qwen3-TTS model is loaded.
    """
    script = Path(script_path)
    output = Path(output_path)
    if not script.exists():
        return GenerationResult(False, None, f"Script not found: {script}")

    mode = EnhancementMode(enhancement_mode)
    narration_profile = ProfileManager().load(profile)
    original_text = script.read_text(encoding="utf-8")

    intelligence = ScriptIntelligence(
        enhancer=ScriptEnhancer(backend=_build_enhancement_backend(mode, enhancement_model_tier))
    )
    intelligence_result = intelligence.process(
        original_text, mode=mode, profile=narration_profile.name
    )
    optimized_text = intelligence_result.output_text
    enhancement_diagnostic = (
        intelligence_result.enhancement.diagnostic
        if intelligence_result.enhancement is not None
        else None
    )

    narration_plan = NarrationPlanner(narration_profile).plan(optimized_text)
    prompt = build_prompt(narration_plan, narration_profile)

    diagnostics = diagnose()
    if not diagnostics.ready:
        return GenerationResult(
            success=False,
            output_path=None,
            diagnostics=format_diagnostics(diagnostics),
            prompt=prompt,
            enhancement_diagnostic=enhancement_diagnostic,
        )

    try:
        model = load_model(diagnostics.model_location)
        wavs, sample_rate = run_inference(
            model=model,
            prompt=prompt,
            reference_audio=Path(reference_audio) if reference_audio else None,
        )
        save_wav(output, wavs[0], sample_rate)
    except Exception:
        return GenerationResult(
            success=False,
            output_path=None,
            diagnostics=f"Qwen generation failed:\n{traceback.format_exc()}",
            prompt=prompt,
            enhancement_diagnostic=enhancement_diagnostic,
        )

    return GenerationResult(
        success=True,
        output_path=str(output),
        diagnostics=f"Wrote {output}",
        prompt=prompt,
        enhancement_diagnostic=enhancement_diagnostic,
    )


def _build_enhancement_backend(mode: EnhancementMode, model_tier: str):
    """Return the enhancement backend to use, without importing Torch/Transformers unless needed."""
    if mode is EnhancementMode.OPTIMIZE_ONLY:
        from core.enhancement_backends import UnavailableEnhancementBackend

        return UnavailableEnhancementBackend()

    model_id = ENHANCEMENT_MODEL_TIERS.get(
        model_tier, ENHANCEMENT_MODEL_TIERS[DEFAULT_ENHANCEMENT_MODEL_TIER]
    )
    return HuggingFaceEnhancementBackend(config=HuggingFaceEnhancementConfig(model_id=model_id))


def load_model(model_location: str | None) -> Any:
    """Load a Qwen3-TTS model with the official qwen_tts wrapper."""
    checkpoint = resolve_model_checkpoint(model_location)

    import torch  # type: ignore[import-not-found]
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    register_qwen_tts_model()

    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_map != "cpu" else torch.float32
    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": dtype,
    }
    if device_map != "cpu":
        load_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        return Qwen3TTSModel.from_pretrained(checkpoint, **load_kwargs)
    except Exception as exc:
        if load_kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        load_kwargs.pop("attn_implementation", None)
        try:
            return Qwen3TTSModel.from_pretrained(checkpoint, **load_kwargs)
        except Exception:
            raise exc


def register_qwen_tts_model() -> None:
    """Register Qwen3-TTS classes with Transformers when available."""
    try:
        from qwen_tts.core.models import (  # type: ignore[import-not-found]
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
            Qwen3TTSProcessor,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoConfig,
            AutoModel,
            AutoProcessor,
        )
    except ImportError:
        return

    register_calls = (
        lambda: AutoConfig.register("qwen3_tts", Qwen3TTSConfig),
        lambda: AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration),
        lambda: AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor),
    )
    for register_call in register_calls:
        try:
            register_call()
        except ValueError as exc:
            if "already" not in str(exc).lower():
                raise


def resolve_model_checkpoint(model_location: str | None) -> str:
    """Return an official model id or concrete local snapshot path."""
    if not model_location:
        return DEFAULT_QWEN_MODEL_ID

    candidate = Path(model_location)
    if not candidate.exists():
        return model_location

    if (candidate / "config.json").exists():
        return str(candidate)

    snapshots_dir = candidate / "snapshots"
    if snapshots_dir.exists():
        snapshots = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ]
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
            return str(latest_snapshot)

    return str(candidate)


def run_inference(
    model: Any, prompt: QwenPrompt, reference_audio: Path | None
) -> tuple[Any, int]:
    """Run the official Qwen3-TTS 12Hz Base voice-clone example path."""
    if reference_audio is None:
        raise ValueError("Reference audio is required for the official Base example.")

    ref_audio_single = load_reference_audio(reference_audio)
    ref_text_single = None
    syn_text_single = prompt.optimized_text
    syn_lang_single = "Auto"

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    xvec_only = True
    return model.generate_voice_clone(
        text=syn_text_single,
        language=syn_lang_single,
        ref_audio=ref_audio_single,
        ref_text=ref_text_single,
        x_vector_only_mode=xvec_only,
        **common_gen_kwargs,
    )


def load_reference_audio(reference_audio: Path) -> str:
    """Return a reference audio path in the form expected by Qwen voice clone."""
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    return str(reference_audio)


def save_wav(output_path: Path, wav: Any, sample_rate: int) -> None:
    """Write generated audio to a WAV file."""
    import soundfile as sf  # type: ignore[import-not-found]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, sample_rate)
