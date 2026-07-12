"""Benchmark HooperTTS script optimization output.

Run from the project root:

    python benchmark.py

The benchmark reads the first `.txt` file in `samples/`, writes the original
and optimized scripts to `output/`, and prints comparison and rhythm metrics.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner, SentencePlan
from core.profile import NarrationProfile, ProfileManager
from core.pronunciation import PronunciationEngine
from core.style_rules import STYLE_RULES

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_STYLE = "documentary"
DEFAULT_PROFILE = "default"

WORD_PATTERN = re.compile(r"\b[\w']+\b")
UPPERCASE_WORD_PATTERN = re.compile(r"\b[A-Z][A-Z']+\b")


def main() -> None:
    """Run the benchmark and print a readable optimization summary."""
    source_path = find_sample_text(SAMPLES_DIR)
    run_benchmark(source_path, profile_name=DEFAULT_PROFILE)


def run_benchmark(
    source_path: Path,
    profile_name: str = DEFAULT_PROFILE,
    output_dir: Path = OUTPUT_DIR,
) -> str:
    """Run the benchmark for a source file and return optimized text."""
    original_text = source_path.read_text(encoding="utf-8")
    narration_profile = ProfileManager().load(profile_name)

    optimizer = ScriptOptimizer()
    optimized_text = optimizer.optimize(
        original_text, style=DEFAULT_STYLE, profile=narration_profile.name
    )

    write_outputs(original_text, optimized_text, output_dir)
    print_summary(
        source_path, original_text, optimized_text, DEFAULT_STYLE, narration_profile
    )
    return optimized_text


def find_sample_text(samples_dir: Path) -> Path:
    """Return the first text sample from the samples directory."""
    text_files = sorted(samples_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"No .txt sample files found in {samples_dir}")
    return text_files[0]


def display_path(path: Path) -> str:
    """Return a readable path for benchmark output."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def write_outputs(original_text: str, optimized_text: str, output_dir: Path) -> None:
    """Write original and optimized benchmark files to the output directory."""
    if output_dir.exists() and not output_dir.is_dir():
        output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original.txt").write_text(original_text, encoding="utf-8")
    (output_dir / "optimized.txt").write_text(optimized_text, encoding="utf-8")


def print_summary(
    source_path: Path,
    original_text: str,
    optimized_text: str,
    style: str,
    profile: NarrationProfile,
) -> None:
    """Print word, pause, emphasis, and narration group metrics."""
    pronunciation_replacements = PronunciationEngine().get_replacements(original_text)
    sentence_plans = NarrationPlanner(profile).plan(original_text)
    speaking_speed = estimate_speaking_speed(profile, sentence_plans)
    summary = {
        "Source file": display_path(source_path),
        "Profile Used": profile.name,
        "Original word count": count_words(original_text),
        "Optimized word count": count_words(optimized_text),
        "Inserted pauses": count_inserted_pauses(original_text, optimized_text),
        "Emphasized words": count_emphasized_words(optimized_text, style),
        "Estimated narration groups": estimate_narration_groups(optimized_text),
        "Estimated Speaking Speed": f"{speaking_speed} wpm",
        "Estimated Speaking Time": estimate_speaking_time(
            optimized_text, speaking_speed, profile
        ),
    }
    rhythm_summary = calculate_rhythm_stats(optimized_text)
    planner_summary = calculate_planner_stats(sentence_plans)
    chunk_summary = calculate_chunk_stats(sentence_plans)

    print("HooperTTS Benchmark")
    print("===================")
    for label, value in summary.items():
        print(f"{label}: {value}")
    for label, value in rhythm_summary.items():
        print(f"{label}: {value}")
    for label, value in planner_summary.items():
        print(f"{label}: {value}")
    for label, value in chunk_summary.items():
        print(f"{label}: {value}")
    print_pronunciation_replacements(pronunciation_replacements)


def count_words(text: str) -> int:
    """Return the number of words in text."""
    return len(WORD_PATTERN.findall(text))


def count_inserted_pauses(original_text: str, optimized_text: str) -> int:
    """Return how many pause markers were added by optimization."""
    return max(optimized_text.count("...") - original_text.count("..."), 0)


def count_emphasized_words(text: str, style: str) -> int:
    """Return the number of style-configured words emphasized in text."""
    emphasis_words = {
        word.upper()
        for word, action in STYLE_RULES.get(style, {}).items()
        if action in {"emphasis", "dramatic"}
    }
    return sum(
        1 for word in UPPERCASE_WORD_PATTERN.findall(text) if word in emphasis_words
    )


def estimate_narration_groups(text: str) -> int:
    """Return the number of non-empty narration groups in optimized text."""
    return len([group for group in re.split(r"\n\s*\n", text.strip()) if group.strip()])


def calculate_rhythm_stats(text: str) -> dict[str, int | float]:
    """Return breath group counts and word-count spread for optimized text."""
    counts = [count_words(group) for group in get_breath_groups(text)]
    if not counts:
        return {
            "Estimated Breath Groups": 0,
            "Average Words Per Group": 0.0,
            "Longest Group": 0,
            "Shortest Group": 0,
        }

    return {
        "Estimated Breath Groups": len(counts),
        "Average Words Per Group": round(sum(counts) / len(counts), 2),
        "Longest Group": max(counts),
        "Shortest Group": min(counts),
    }


def get_breath_groups(text: str) -> list[str]:
    """Return non-empty spoken breath groups, excluding standalone pauses."""
    groups = []
    for line in text.splitlines():
        group = line.strip()
        if group and group != "...":
            groups.append(group)
    return groups


def calculate_planner_stats(sentence_plans: list[SentencePlan]) -> dict[str, int]:
    """Return sentence type counts from narration plans."""
    return {
        "Hooks detected": count_sentence_type(sentence_plans, "HOOK"),
        "Reveals detected": count_sentence_type(sentence_plans, "REVEAL"),
        "Questions detected": count_sentence_type(sentence_plans, "QUESTION"),
        "CTAs detected": count_sentence_type(sentence_plans, "CTA"),
    }


def count_sentence_type(
    sentence_plans: list[SentencePlan], sentence_type: str
) -> int:
    """Return how many plans match a sentence type."""
    return sum(1 for plan in sentence_plans if plan.sentence_type == sentence_type)


def calculate_chunk_stats(sentence_plans: list[SentencePlan]) -> dict[str, int | float]:
    """Return aggregate chunk size statistics from sentence plans."""
    counts = [count_words(chunk) for plan in sentence_plans for chunk in plan.chunks]
    if not counts:
        return {
            "Average Chunk Size": 0.0,
            "Longest Chunk": 0,
            "Shortest Chunk": 0,
            "Chunk Count": 0,
        }
    return {
        "Average Chunk Size": round(sum(counts) / len(counts), 2),
        "Longest Chunk": max(counts),
        "Shortest Chunk": min(counts),
        "Chunk Count": len(counts),
    }


def estimate_speaking_speed(
    profile: NarrationProfile, sentence_plans: list[SentencePlan]
) -> int:
    """Return estimated speaking speed in words per minute."""
    if sentence_plans:
        average_energy = sum(plan.estimated_energy for plan in sentence_plans) / len(
            sentence_plans
        )
    else:
        average_energy = profile.energy_curve["NORMAL"]
    speed = 155 + (average_energy - 5) * 4 - profile.pause_strength * 10
    return max(95, min(210, round(speed)))


def estimate_speaking_time(
    text: str, speaking_speed: int, profile: NarrationProfile
) -> str:
    """Return estimated speaking time for optimized text."""
    spoken_seconds = count_words(text) / speaking_speed * 60
    pause_seconds = text.count("...") * 0.35 * profile.pause_strength
    total_seconds = spoken_seconds + pause_seconds
    return f"{total_seconds:.1f}s"


def print_pronunciation_replacements(replacements: list[tuple[str, str]]) -> None:
    """Print the pronunciation replacements found in the source text."""
    print("Pronunciation Replacements:")
    if not replacements:
        print("None")
        return

    for written_form, spoken_form in replacements:
        print(f"{written_form} -> {spoken_form}")


if __name__ == "__main__":
    main()
