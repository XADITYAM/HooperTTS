"""Gradio web interface for HooperTTS native Qwen generation."""

from __future__ import annotations

import inspect
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from uuid import uuid4

import gradio as gr

from qwen.runner import generate


PROFILE_CHOICES = [
    "default",
    "documentary",
    "gaming_news",
    "podcast",
    "youtube_shorts",
]
OUTPUT_DIR = Path(gettempdir()) / "hoopertts_gradio"


def _component(component_type: type[Any], **kwargs: Any) -> Any:
    """Create a Gradio component with only kwargs supported by this version."""
    parameters = inspect.signature(component_type.__init__).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return component_type(**supported_kwargs)


def _uploaded_path(uploaded_file: Any) -> Path | None:
    """Return a pathlib path for a Gradio uploaded file value."""
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    if hasattr(uploaded_file, "name"):
        return Path(str(uploaded_file.name))
    return None


def generate_speech(
    script_file: Any,
    reference_audio: Any,
    profile: str,
) -> tuple[str, str, str | None, str | None, str]:
    """Generate speech and return UI-ready optimized text, prompt, audio, and logs."""
    script_path = _uploaded_path(script_file)
    reference_path = _uploaded_path(reference_audio)

    if script_path is None or not script_path.exists():
        return "", "", None, None, "Upload a .txt script before generating speech."

    if script_path.suffix.lower() != ".txt":
        return "", "", None, None, "The script upload must be a .txt file."

    if reference_path is not None and reference_path.suffix.lower() != ".wav":
        return "", "", None, None, "The reference voice upload must be a .wav file."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"hoopertts_{uuid4().hex}.wav"

    result = generate(
        script_path=script_path,
        reference_audio=reference_path,
        profile=profile,
        output_path=output_path,
    )

    optimized_script = result.prompt.optimized_text if result.prompt else ""
    style_prompt = result.prompt.style_prompt if result.prompt else ""
    audio_path = result.output_path if result.success else None
    download_path = result.output_path if result.success else None

    return (
        optimized_script,
        style_prompt,
        audio_path,
        download_path,
        result.diagnostics,
    )


def build_interface() -> gr.Blocks:
    """Build and return the HooperTTS Gradio interface."""
    with gr.Blocks(title="HooperTTS Qwen Generator") as interface:
        gr.Markdown("# HooperTTS Qwen Generator")
        gr.Markdown(
            "Upload a narration script and optional reference voice, then generate "
            "Qwen3-TTS audio from the optimized HooperTTS plan."
        )

        with gr.Row():
            script_input = _component(
                gr.File,
                label="Script (.txt)",
                file_types=[".txt"],
                type="filepath",
            )
            reference_input = _component(
                gr.File,
                label="Reference Voice (.wav)",
                file_types=[".wav"],
                type="filepath",
            )

        profile_input = _component(
            gr.Dropdown,
            choices=PROFILE_CHOICES,
            value="default",
            label="Narration Profile",
        )
        generate_button = _component(
            gr.Button,
            value="Generate Speech",
            variant="primary",
        )

        with gr.Row():
            optimized_output = _component(
                gr.Textbox,
                label="Optimized Script",
                lines=14,
            )
            style_output = _component(
                gr.Textbox,
                label="Generated Style Prompt",
                lines=8,
            )

        audio_output = _component(gr.Audio, label="Generated WAV", type="filepath")
        download_output = _component(gr.File, label="Download WAV")
        diagnostics_output = _component(gr.Textbox, label="Diagnostics", lines=8)

        generate_button.click(
            fn=generate_speech,
            inputs=[script_input, reference_input, profile_input],
            outputs=[
                optimized_output,
                style_output,
                audio_output,
                download_output,
                diagnostics_output,
            ],
        )

    return interface


if __name__ == "__main__":
    build_interface().launch(share=True)
