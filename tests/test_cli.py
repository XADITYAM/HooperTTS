from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import qwen.runner as qwen_runner
from core.cli import main
from qwen.runner import GenerationResult


def run_cli(args: list[str]) -> str:
    output = StringIO()
    with redirect_stdout(output):
        exit_code = main(args)
    assert exit_code == 0
    return output.getvalue()


def test_cli_optimize() -> None:
    with TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "script.txt"
        script_path.write_text(
            "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
        )

        output = run_cli(["optimize", str(script_path)])

    assert "Imagine" in output
    assert "OFFICIALLY" in output


def test_cli_benchmark() -> None:
    with TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "script.txt"
        script_path.write_text(
            "Imagine HooperTTS. Officially confirmed.", encoding="utf-8"
        )

        output = run_cli(["benchmark", str(script_path)])

    assert "HooperTTS Benchmark" in output
    assert "Profile Used" in output
    assert Path("output/original.txt").exists()
    assert Path("output/optimized.txt").exists()


def test_cli_compare() -> None:
    with TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "script.txt"
        script_path.write_text("Imagine HooperTTS.", encoding="utf-8")

        output = run_cli(["compare", str(script_path)])

    assert "Original" in output
    assert "Optimized" in output


def test_cli_profiles() -> None:
    output = run_cli(["profiles"])

    assert "default" in output
    assert "gaming_news" in output


def test_cli_doctor() -> None:
    output = run_cli(["doctor"])

    assert "Qwen3-TTS Environment" in output


def test_cli_generate_accepts_command_profile() -> None:
    original_generate = qwen_runner.generate

    try:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_path = root / "script.txt"
            output_path = root / "output.wav"
            script_path.write_text("Imagine HooperTTS.", encoding="utf-8")

            def fake_generate(script_path, reference_audio, profile, output_path):
                assert profile == "gaming_news"
                Path(output_path).write_text("wav", encoding="utf-8")
                return GenerationResult(
                    success=True,
                    output_path=str(output_path),
                    diagnostics="Wrote output.wav",
                    prompt=None,
                )

            qwen_runner.generate = fake_generate

            output = run_cli(
                [
                    "generate",
                    "--script",
                    str(script_path),
                    "--profile",
                    "gaming_news",
                    "--output",
                    str(output_path),
                ]
            )

            assert "Wrote output.wav" in output
            assert output_path.exists()
    finally:
        qwen_runner.generate = original_generate


def test_cli_validate() -> None:
    with TemporaryDirectory() as temp_dir:
        sample_dir = Path(temp_dir)
        (sample_dir / "one.txt").write_text("Imagine HooperTTS.", encoding="utf-8")

        output = run_cli(["validate", str(sample_dir)])

    assert "Validation" in output
    assert "one.txt: OK" in output


if __name__ == "__main__":
    test_cli_optimize()
    test_cli_benchmark()
    test_cli_compare()
    test_cli_profiles()
    test_cli_doctor()
    test_cli_generate_accepts_command_profile()
    test_cli_validate()
