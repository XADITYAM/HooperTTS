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
    """Load a Qwen3-TTS model from a local model location."""
    if model_location is None:
        raise RuntimeError("Model location is not available.")

    import torch  # type: ignore[import-not-found]
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

    return Qwen3TTSModel.from_pretrained(
        model_location,
        device_map="cuda",
        dtype=torch.bfloat16,
    )


def run_inference(
    model: Any, prompt: QwenPrompt, reference_audio: Path | None
) -> tuple[Any, int]:
    """Run the best available Qwen inference path for the provided inputs."""
    if reference_audio is not None:
        ref_audio = load_reference_audio(reference_audio)
        if hasattr(model, "generate_voice_clone"):
            return model.generate_voice_clone(
                text=prompt.optimized_text,
                language="English",
                ref_audio=ref_audio,
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


def load_reference_audio(reference_audio: Path) -> tuple[Any, int]:
    """Load reference audio as the tuple expected by Qwen voice clone."""
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    import soundfile as sf  # type: ignore[import-not-found]

    wav, sample_rate = sf.read(reference_audio)
    return wav, int(sample_rate)


def save_wav(output_path: Path, wav: Any, sample_rate: int) -> None:
    """Write generated audio to a WAV file."""
    import soundfile as sf  # type: ignore[import-not-found]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, sample_rate)
