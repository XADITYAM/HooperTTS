"""Evaluate HooperTTS over a directory of scripts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from benchmark import (
    calculate_chunk_stats,
    calculate_planner_stats,
    count_inserted_pauses,
    count_words,
    estimate_speaking_speed,
    estimate_speaking_time,
)
from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner
from core.profile import NarrationProfile, ProfileManager
from core.pronunciation import PronunciationEngine

DEFAULT_OUTPUT_DIR = Path("evaluation")
CSV_COLUMNS = (
    "Script Name",
    "Word Count",
    "Profile",
    "Pause Count",
    "Chunk Count",
    "Narration Score",
    "Estimated Speaking Time",
    "Pronunciation Replacements",
    "Hooks",
    "Reveals",
    "Questions",
    "CTAs",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one evaluated script."""

    script_name: str
    word_count: int
    profile: str
    pause_count: int
    chunk_count: int
    narration_score: float
    estimated_speaking_time: str
    pronunciation_replacements: int
    hooks: int
    reveals: int
    questions: int
    ctas: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluate_dataset(args.dataset, args.output, args.profile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluator argument parser."""
    parser = argparse.ArgumentParser(
        prog="python evaluate.py",
        description="Evaluate HooperTTS over a dataset directory.",
    )
    parser.add_argument("dataset", type=Path, help="Directory containing .txt scripts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results.csv, summary.json, and summary.md.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Narration profile to use for all scripts.",
    )
    return parser


def evaluate_dataset(
    dataset_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile_name: str = "default",
) -> list[EvaluationResult]:
    """Evaluate every .txt script in a dataset directory."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    profile = ProfileManager().load(profile_name)
    script_paths = sorted(dataset_dir.rglob("*.txt"))
    results = [evaluate_script(path, dataset_dir, profile) for path in script_paths]
    warnings = build_warnings(dataset_dir, results)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(results, output_dir / "results.csv")
    summary = build_summary(results, warnings)
    write_summary_json(summary, output_dir / "summary.json")
    write_summary_md(summary, output_dir / "summary.md")
    return results


def evaluate_script(
    script_path: Path, dataset_dir: Path, profile: NarrationProfile
) -> EvaluationResult:
    """Evaluate one script and return its metrics."""
    original_text = script_path.read_text(encoding="utf-8")
    optimized_text = ScriptOptimizer().optimize(original_text, profile=profile.name)
    sentence_plans = NarrationPlanner(profile).plan(original_text)
    planner_stats = calculate_planner_stats(sentence_plans)
    chunk_stats = calculate_chunk_stats(sentence_plans)
    pronunciation_replacements = PronunciationEngine().get_replacements(original_text)
    speaking_speed = estimate_speaking_speed(profile, sentence_plans)

    word_count = count_words(original_text)
    pause_count = count_inserted_pauses(original_text, optimized_text)
    chunk_count = int(chunk_stats["Chunk Count"])
    return EvaluationResult(
        script_name=script_path.relative_to(dataset_dir).as_posix(),
        word_count=word_count,
        profile=profile.name,
        pause_count=pause_count,
        chunk_count=chunk_count,
        narration_score=calculate_narration_score(
            word_count=word_count,
            pause_count=pause_count,
            chunk_count=chunk_count,
            hooks=planner_stats["Hooks detected"],
            reveals=planner_stats["Reveals detected"],
        ),
        estimated_speaking_time=estimate_speaking_time(
            optimized_text, speaking_speed, profile
        ),
        pronunciation_replacements=len(pronunciation_replacements),
        hooks=planner_stats["Hooks detected"],
        reveals=planner_stats["Reveals detected"],
        questions=planner_stats["Questions detected"],
        ctas=planner_stats["CTAs detected"],
    )


def calculate_narration_score(
    word_count: int, pause_count: int, chunk_count: int, hooks: int, reveals: int
) -> float:
    """Return a simple 0-100 narration readiness score."""
    if word_count == 0:
        return 0.0

    chunk_density = min(chunk_count / max(word_count / 7, 1), 1.2)
    pause_density = min(pause_count / max(word_count / 18, 1), 1.2)
    intent_bonus = min((hooks + reveals) * 5, 15)
    score = 45 + chunk_density * 25 + pause_density * 15 + intent_bonus
    return round(min(score, 100), 1)


def write_results_csv(results: list[EvaluationResult], output_path: Path) -> None:
    """Write per-script evaluation results to CSV."""
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "Script Name": result.script_name,
                    "Word Count": result.word_count,
                    "Profile": result.profile,
                    "Pause Count": result.pause_count,
                    "Chunk Count": result.chunk_count,
                    "Narration Score": result.narration_score,
                    "Estimated Speaking Time": result.estimated_speaking_time,
                    "Pronunciation Replacements": result.pronunciation_replacements,
                    "Hooks": result.hooks,
                    "Reveals": result.reveals,
                    "Questions": result.questions,
                    "CTAs": result.ctas,
                }
            )


def build_summary(
    results: list[EvaluationResult], warnings: list[str]
) -> dict[str, object]:
    """Return aggregate evaluation summary data."""
    profile_distribution = Counter(result.profile for result in results)
    script_count = len(results)
    word_total = sum(result.word_count for result in results)
    pause_total = sum(result.pause_count for result in results)
    chunk_total = sum(result.chunk_count for result in results)
    score_total = sum(result.narration_score for result in results)

    return {
        "script_count": script_count,
        "average_statistics": {
            "word_count": average(word_total, script_count),
            "pause_count": average(pause_total, script_count),
            "chunk_count": average(chunk_total, script_count),
            "narration_score": average(score_total, script_count),
        },
        "profile_distribution": dict(profile_distribution),
        "optimization_totals": {
            "word_count": word_total,
            "pause_count": pause_total,
            "chunk_count": chunk_total,
            "pronunciation_replacements": sum(
                result.pronunciation_replacements for result in results
            ),
            "hooks": sum(result.hooks for result in results),
            "reveals": sum(result.reveals for result in results),
            "questions": sum(result.questions for result in results),
            "ctas": sum(result.ctas for result in results),
        },
        "warnings": warnings,
        "results": [asdict(result) for result in results],
    }


def build_warnings(dataset_dir: Path, results: list[EvaluationResult]) -> list[str]:
    """Return evaluation warnings for summary outputs."""
    warnings: list[str] = []
    if not results:
        warnings.append(f"No .txt scripts found in {dataset_dir}.")
    for result in results:
        if result.word_count == 0:
            warnings.append(f"{result.script_name} is empty.")
        if result.chunk_count == 0:
            warnings.append(f"{result.script_name} produced no chunks.")
        if result.narration_score < 60:
            warnings.append(f"{result.script_name} has a low narration score.")
    return warnings


def write_summary_json(summary: dict[str, object], output_path: Path) -> None:
    """Write aggregate summary data to JSON."""
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_summary_md(summary: dict[str, object], output_path: Path) -> None:
    """Write aggregate summary data to Markdown."""
    average_stats = summary["average_statistics"]
    profile_distribution = summary["profile_distribution"]
    totals = summary["optimization_totals"]
    warnings = summary["warnings"]

    lines = [
        "# HooperTTS Evaluation Summary",
        "",
        f"Scripts evaluated: {summary['script_count']}",
        "",
        "## Average Statistics",
        "",
        f"- Word Count: {average_stats['word_count']}",
        f"- Pause Count: {average_stats['pause_count']}",
        f"- Chunk Count: {average_stats['chunk_count']}",
        f"- Narration Score: {average_stats['narration_score']}",
        "",
        "## Profile Distribution",
        "",
    ]
    lines.extend(
        f"- {profile}: {count}" for profile, count in profile_distribution.items()
    )
    lines.extend(
        [
            "",
            "## Optimization Totals",
            "",
            f"- Word Count: {totals['word_count']}",
            f"- Pause Count: {totals['pause_count']}",
            f"- Chunk Count: {totals['chunk_count']}",
            f"- Pronunciation Replacements: {totals['pronunciation_replacements']}",
            f"- Hooks: {totals['hooks']}",
            f"- Reveals: {totals['reveals']}",
            f"- Questions: {totals['questions']}",
            f"- CTAs: {totals['ctas']}",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def average(total: int | float, count: int) -> float:
    """Return rounded average for aggregate metrics."""
    if count == 0:
        return 0.0
    return round(total / count, 2)


if __name__ == "__main__":
    raise SystemExit(main())
