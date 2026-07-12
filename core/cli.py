"""Command-line interface for HooperTTS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from benchmark import run_benchmark

from .optimizer import ScriptOptimizer
from .profile import ProfileManager


def main(argv: Sequence[str] | None = None) -> int:
    """Run the HooperTTS command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.command(args))


def build_parser() -> argparse.ArgumentParser:
    """Create the HooperTTS argument parser."""
    parser = argparse.ArgumentParser(
        prog="hoopertts",
        description="Optimize narration scripts for HooperTTS.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Narration profile to use. Defaults to default.",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize a script.")
    optimize_parser.add_argument("script", type=Path)
    optimize_parser.set_defaults(command=optimize_command)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark a script and write output files."
    )
    benchmark_parser.add_argument("script", type=Path)
    benchmark_parser.set_defaults(command=benchmark_command)

    compare_parser = subparsers.add_parser(
        "compare", help="Print original and optimized text."
    )
    compare_parser.add_argument("script", type=Path)
    compare_parser.set_defaults(command=compare_command)

    profiles_parser = subparsers.add_parser("profiles", help="List profiles.")
    profiles_parser.set_defaults(command=profiles_command)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check optional Qwen3-TTS generation environment."
    )
    doctor_parser.set_defaults(command=doctor_command)

    generate_parser = subparsers.add_parser(
        "generate", help="Optimize a script and generate WAV audio with Qwen3-TTS."
    )
    generate_parser.add_argument("--script", required=True, type=Path)
    generate_parser.add_argument("--reference", type=Path)
    generate_parser.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        help="Narration profile to use for generation.",
    )
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.set_defaults(command=generate_command)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate every .txt file in a directory."
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.set_defaults(command=validate_command)

    return parser


def optimize_command(args: argparse.Namespace) -> int:
    """Optimize a script and print the result."""
    source_path = require_file(args.script)
    text = source_path.read_text(encoding="utf-8")
    print(ScriptOptimizer().optimize(text, profile=args.profile))
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    """Run the benchmark for a script."""
    source_path = require_file(args.script)
    run_benchmark(source_path, profile_name=args.profile)
    return 0


def compare_command(args: argparse.Namespace) -> int:
    """Print original and optimized script text."""
    source_path = require_file(args.script)
    original = source_path.read_text(encoding="utf-8")
    optimized = ScriptOptimizer().optimize(original, profile=args.profile)

    print("Original")
    print("========")
    print(original.strip())
    print()
    print("Optimized")
    print("=========")
    print(optimized)
    return 0


def profiles_command(args: argparse.Namespace) -> int:
    """Print available narration profiles."""
    manager = ProfileManager()
    print("Available Profiles")
    print("==================")
    for profile_name in manager.list_profiles():
        marker = " (default)" if profile_name == "default" else ""
        print(f"{profile_name}{marker}")
    return 0


def doctor_command(args: argparse.Namespace) -> int:
    """Print Qwen generation environment diagnostics."""
    from qwen.environment import diagnose, format_diagnostics

    print(format_diagnostics(diagnose()))
    return 0


def generate_command(args: argparse.Namespace) -> int:
    """Generate speech with the optional Qwen backend."""
    from qwen.runner import generate

    script_path = require_file(args.script)
    reference_audio = require_file(args.reference) if args.reference else None
    result = generate(
        script_path=script_path,
        reference_audio=reference_audio,
        profile=args.profile,
        output_path=args.output,
    )
    print(result.diagnostics)
    if result.prompt:
        print()
        print("Qwen Prompt")
        print("===========")
        print(f"Style: {result.prompt.style_prompt}")
        print(f"Speaker: {result.prompt.speaker_prompt}")
    return 0 if result.success else 1


def validate_command(args: argparse.Namespace) -> int:
    """Validate that every text sample can be optimized."""
    target_path = args.path
    if not target_path.exists() or not target_path.is_dir():
        raise SystemExit(f"Directory not found: {target_path}")

    text_files = sorted(target_path.glob("*.txt"))
    if not text_files:
        raise SystemExit(f"No .txt files found in {target_path}")

    optimizer = ScriptOptimizer()
    print("Validation")
    print("==========")
    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8")
        optimized = optimizer.optimize(text, profile=args.profile)
        status = "OK" if optimized else "EMPTY"
        print(f"{text_file.name}: {status}")
    return 0


def require_file(path: Path) -> Path:
    """Return path if it is an existing file, otherwise exit."""
    if not path.exists() or not path.is_file():
        raise SystemExit(f"File not found: {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
