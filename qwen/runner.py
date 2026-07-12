"""Native Qwen3-TTS generation runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner
from core.profile import ProfileManager
from .environment import diagnose, format_diagnostics
from .prompt_builder import QwenPrompt, build_prompt


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@dataclass(frozen=True)
class GenerationResult:
    """Result of an attempted Qwen generation."""

    success: bool
    output_path: str | None
    diagnostics: str
    prompt: QwenPrompt | None = None


def generate(
    script_path: str | Path,
    reference_audio: str | Path | None,
    profile: str,
    output_path: str | Path,
) -> GenerationResult:
    """Optimize a script, build a Qwen prompt, generate audio, and save WAV."""
    script = Path(script_path)
    output = Path(output_path)
    if not script.exists():
        return GenerationResult(False, None, f"Script not found: {script}")

    narration_profile = ProfileManager().load(profile)
    original_text = script.read_text(encoding="utf-8")
    optimized_text = ScriptOptimizer().optimize(
        original_text, profile=narration_profile.name
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
        )

    try:
        model = load_model(diagnostics.model_location)
        wavs, sample_rate = run_inference(
            model=model,
            prompt=prompt,
            reference_audio=Path(reference_audio) if reference_audio else None,
        )
        save_wav(output, wavs[0], sample_rate)
    except Exception as exc:
        return GenerationResult(
            success=False,
            output_path=None,
            diagnostics=f"Qwen generation failed: {exc}",
            prompt=prompt,
        )

    return GenerationResult(
        success=True,
        output_path=str(output),
        diagnostics=f"Wrote {output}",
        prompt=prompt,
    )


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
    """Run the best available Qwen inference path for the provided inputs."""
    if reference_audio is not None:
        ref_audio_path = load_reference_audio(reference_audio)
        if hasattr(model, "generate_voice_clone"):
            return model.generate_voice_clone(
                text=prompt.optimized_text,
                language="English",
                ref_audio=ref_audio_path,
                ref_text=None,
                x_vector_only_mode=True,
                max_new_tokens=2048,
            )

    if hasattr(model, "generate_voice_design"):
        return model.generate_voice_design(
            text=prompt.optimized_text,
            language="English",
            instruct=prompt.style_prompt,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

    if hasattr(model, "generate_custom_voice"):
        return model.generate_custom_voice(
            text=prompt.optimized_text,
            language="English",
            speaker="ryan",
            instruct=prompt.style_prompt,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

    raise RuntimeError("Loaded Qwen model does not expose a supported API.")


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
