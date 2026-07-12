"""Environment diagnostics for optional Qwen3-TTS generation."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class EnvironmentDiagnostics:
    """Readable Qwen generation environment diagnostics."""

    cuda_available: bool
    torch_available: bool
    qwen_tts_available: bool
    soundfile_available: bool
    model_location: str | None
    model_available: bool
    ffmpeg_available: bool
    messages: list[str]

    @property
    def ready(self) -> bool:
        """Return whether the local environment can attempt generation."""
        return (
            self.cuda_available
            and self.torch_available
            and self.qwen_tts_available
            and self.soundfile_available
            and self.model_available
        )


def diagnose(
    model_location: str | Path | None = None,
    model_size: str = "1.7B",
    model_type: str = "Base",
) -> EnvironmentDiagnostics:
    """Return diagnostics for Qwen3-TTS generation dependencies."""
    messages: list[str] = []
    torch_available = importlib.util.find_spec("torch") is not None
    qwen_tts_available = importlib.util.find_spec("qwen_tts") is not None
    soundfile_available = importlib.util.find_spec("soundfile") is not None
    ffmpeg_available = shutil.which("ffmpeg") is not None
    cuda_available = False

    if torch_available:
        try:
            import torch  # type: ignore[import-not-found]

            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            messages.append(f"Torch is installed but CUDA check failed: {exc}")

    resolved_model = find_model_location(model_location, model_size, model_type)
    model_available = resolved_model is not None

    if not torch_available:
        messages.append("Missing torch. Install PyTorch with CUDA support.")
    if not cuda_available:
        messages.append("CUDA is not available. Qwen3-TTS generation expects a GPU.")
    if not qwen_tts_available:
        messages.append(
            "Missing qwen_tts. Install Qwen3-TTS or qwen-tts before generation."
        )
    if not soundfile_available:
        messages.append("Missing soundfile. It is required to write WAV output.")
    if not model_available:
        messages.append(
            "Qwen model files were not found locally. Download the selected model "
            "from Hugging Face first."
        )
    if not ffmpeg_available:
        messages.append("ffmpeg was not found. Basic WAV writing can still work.")

    if not messages:
        messages.append("Qwen generation environment looks ready.")

    return EnvironmentDiagnostics(
        cuda_available=cuda_available,
        torch_available=torch_available,
        qwen_tts_available=qwen_tts_available,
        soundfile_available=soundfile_available,
        model_location=str(resolved_model) if resolved_model else None,
        model_available=model_available,
        ffmpeg_available=ffmpeg_available,
        messages=messages,
    )


def find_model_location(
    model_location: str | Path | None = None,
    model_size: str = "1.7B",
    model_type: str = "Base",
) -> Path | None:
    """Return a local Qwen model path if one can be found."""
    if model_location:
        candidate = Path(model_location)
        return candidate if candidate.exists() else None

    repo_name = f"Qwen3-TTS-12Hz-{model_size}-{model_type}"
    local_candidates = (
        Path("qwen_tts_model") / repo_name,
        DEFAULT_MODEL_ROOT / f"models--Qwen--{repo_name}",
    )
    for candidate in local_candidates:
        if candidate.exists():
            return candidate
    return None


def format_diagnostics(diagnostics: EnvironmentDiagnostics) -> str:
    """Return diagnostics as human-readable text."""
    lines = [
        "Qwen3-TTS Environment",
        "=====================",
        f"Torch installed: {diagnostics.torch_available}",
        f"CUDA available: {diagnostics.cuda_available}",
        f"qwen_tts installed: {diagnostics.qwen_tts_available}",
        f"soundfile installed: {diagnostics.soundfile_available}",
        f"ffmpeg available: {diagnostics.ffmpeg_available}",
        f"Model available: {diagnostics.model_available}",
        f"Model location: {diagnostics.model_location or 'not found'}",
        "",
        "Diagnostics:",
    ]
    lines.extend(f"- {message}" for message in diagnostics.messages)
    return "\n".join(lines)
