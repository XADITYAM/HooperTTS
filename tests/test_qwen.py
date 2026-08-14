from pathlib import Path
from tempfile import TemporaryDirectory

import qwen.runner as runner
from core.profile import ProfileManager
from core.script_enhancement import EnhancementMode
from qwen.environment import EnvironmentDiagnostics, format_diagnostics
from qwen.prompt_builder import build_prompt


def test_prompt_builder_uses_planner_output() -> None:
    profile = ProfileManager().load("gaming_news")
    from core.planner import NarrationPlanner

    plan = NarrationPlanner(profile).plan("Imagine HooperTTS. Officially confirmed.")
    prompt = build_prompt(plan, profile)

    assert "Imagine" in prompt.optimized_text
    assert "gaming news" in prompt.style_prompt
    assert "gaming news narrator" in prompt.speaker_prompt


def test_friendslop_gaming_prompt_uses_casual_delivery() -> None:
    profile = ProfileManager().load("friendslop_gaming")
    prompt = build_prompt([], profile)

    assert "friendslop gaming narration style" in prompt.style_prompt
    assert "Casual gaming friend" in prompt.speaker_prompt


def test_environment_diagnostics_format() -> None:
    diagnostics = EnvironmentDiagnostics(
        cuda_available=False,
        torch_available=False,
        qwen_tts_available=False,
        soundfile_available=False,
        model_location=None,
        model_available=False,
        ffmpeg_available=False,
        messages=["Missing dependencies."],
    )

    output = format_diagnostics(diagnostics)

    assert "Qwen3-TTS Environment" in output
    assert "Missing dependencies." in output
    assert not diagnostics.ready


def test_runner_resolves_hugging_face_snapshot_path() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        snapshot = root / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            '{"model_type": "qwen3_tts"}', encoding="utf-8"
        )

        assert runner.resolve_model_checkpoint(str(root)) == str(snapshot)


def test_run_inference_passes_reference_audio_path_string() -> None:
    class FakeVoiceCloneModel:
        def generate_voice_clone(self, **kwargs):
            assert isinstance(kwargs["ref_audio"], str)
            assert kwargs["x_vector_only_mode"] is True
            return [[0.0]], 24000

    with TemporaryDirectory() as temp_dir:
        reference_path = Path(temp_dir) / "voice.wav"
        reference_path.write_text("fake wav", encoding="utf-8")
        profile = ProfileManager().load("default")
        prompt = build_prompt([], profile)

        wavs, sample_rate = runner.run_inference(
            FakeVoiceCloneModel(),
            prompt,
            reference_path,
        )

        assert wavs == [[0.0]]
        assert sample_rate == 24000


def test_runner_generate_with_mocked_qwen() -> None:
    original_diagnose = runner.diagnose
    original_load_model = runner.load_model
    original_run_inference = runner.run_inference
    original_save_wav = runner.save_wav

    try:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            output_path = root / "output.wav"
            script_path.write_text(
                "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
            )

            runner.diagnose = lambda: EnvironmentDiagnostics(
                cuda_available=True,
                torch_available=True,
                qwen_tts_available=True,
                soundfile_available=True,
                model_location=str(root / "model"),
                model_available=True,
                ffmpeg_available=True,
                messages=["Ready."],
            )
            runner.load_model = lambda model_location: object()
            runner.run_inference = lambda model, prompt, reference_audio: (
                [[0.0, 0.1, -0.1]],
                24000,
            )

            def fake_save_wav(path, wav, sample_rate):
                path.write_text(f"{sample_rate}:{len(wav)}", encoding="utf-8")

            runner.save_wav = fake_save_wav

            result = runner.generate(
                script_path=script_path,
                reference_audio=None,
                profile="default",
                output_path=output_path,
            )

            assert result.success
            assert result.output_path == str(output_path)
            assert output_path.read_text(encoding="utf-8") == "24000:3"
            assert result.prompt is not None
            assert "Imagine" in result.prompt.optimized_text
    finally:
        runner.diagnose = original_diagnose
        runner.load_model = original_load_model
        runner.run_inference = original_run_inference
        runner.save_wav = original_save_wav


def test_runner_generate_defaults_to_optimize_only_without_loading_enhancer() -> None:
    """Default generate() must not build a Hugging Face backend at all."""
    original_build_backend = runner._build_enhancement_backend
    calls: list[tuple[EnhancementMode, str]] = []

    def spying_build_backend(mode, model_tier):
        calls.append((mode, model_tier))
        return original_build_backend(mode, model_tier)

    original_diagnose = runner.diagnose
    original_load_model = runner.load_model
    original_run_inference = runner.run_inference
    original_save_wav = runner.save_wav
    runner._build_enhancement_backend = spying_build_backend

    try:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            output_path = root / "output.wav"
            script_path.write_text(
                "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
            )

            runner.diagnose = lambda: EnvironmentDiagnostics(
                cuda_available=True,
                torch_available=True,
                qwen_tts_available=True,
                soundfile_available=True,
                model_location=str(root / "model"),
                model_available=True,
                ffmpeg_available=True,
                messages=["Ready."],
            )
            runner.load_model = lambda model_location: object()
            runner.run_inference = lambda model, prompt, reference_audio: (
                [[0.0, 0.1, -0.1]],
                24000,
            )
            runner.save_wav = lambda path, wav, sample_rate: path.write_text(
                f"{sample_rate}:{len(wav)}", encoding="utf-8"
            )

            result = runner.generate(
                script_path=script_path,
                reference_audio=None,
                profile="default",
                output_path=output_path,
            )

            assert result.success
            assert result.enhancement_diagnostic is None
            assert calls == [(EnhancementMode.OPTIMIZE_ONLY, "quality")]
    finally:
        runner._build_enhancement_backend = original_build_backend
        runner.diagnose = original_diagnose
        runner.load_model = original_load_model
        runner.run_inference = original_run_inference
        runner.save_wav = original_save_wav


def test_runner_generate_enhance_mode_populates_enhancement_diagnostic() -> None:
    """Requesting enhancement should surface the backend's diagnostic message."""

    class FakeBackend:
        name = "fake"

        def enhance(self, text, *, analysis, policy):
            from core.enhancement_backends import BackendEnhancement

            return BackendEnhancement(
                text=text,
                backend_name=self.name,
                available=True,
                diagnostic="Generated a candidate with the fake backend.",
            )

    original_build_backend = runner._build_enhancement_backend
    original_diagnose = runner.diagnose
    original_load_model = runner.load_model
    original_run_inference = runner.run_inference
    original_save_wav = runner.save_wav
    runner._build_enhancement_backend = lambda mode, model_tier: FakeBackend()

    try:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            output_path = root / "output.wav"
            script_path.write_text(
                "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
            )

            runner.diagnose = lambda: EnvironmentDiagnostics(
                cuda_available=True,
                torch_available=True,
                qwen_tts_available=True,
                soundfile_available=True,
                model_location=str(root / "model"),
                model_available=True,
                ffmpeg_available=True,
                messages=["Ready."],
            )
            runner.load_model = lambda model_location: object()
            runner.run_inference = lambda model, prompt, reference_audio: (
                [[0.0, 0.1, -0.1]],
                24000,
            )
            runner.save_wav = lambda path, wav, sample_rate: path.write_text(
                f"{sample_rate}:{len(wav)}", encoding="utf-8"
            )

            result = runner.generate(
                script_path=script_path,
                reference_audio=None,
                profile="default",
                output_path=output_path,
                enhancement_mode=EnhancementMode.ENHANCE_AND_OPTIMIZE,
                enhancement_model_tier="fast",
            )

            assert result.success
            assert result.enhancement_diagnostic == (
                "Generated a candidate with the fake backend."
            )
    finally:
        runner._build_enhancement_backend = original_build_backend
        runner.diagnose = original_diagnose
        runner.load_model = original_load_model
        runner.run_inference = original_run_inference
        runner.save_wav = original_save_wav


if __name__ == "__main__":
    test_prompt_builder_uses_planner_output()
    test_friendslop_gaming_prompt_uses_casual_delivery()
    test_environment_diagnostics_format()
    test_runner_resolves_hugging_face_snapshot_path()
    test_run_inference_passes_reference_audio_path_string()
    test_runner_generate_with_mocked_qwen()
    test_runner_generate_defaults_to_optimize_only_without_loading_enhancer()
    test_runner_generate_enhance_mode_populates_enhancement_diagnostic()
