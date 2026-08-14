"""Read-only Script Intelligence experiment for a fixed input script."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Sequence

from benchmark import count_words
from core.enhancement_backends import (
    HuggingFaceEnhancementBackend,
    HuggingFaceEnhancementConfig,
)
from core.script_enhancement import EnhancementMode, ScriptEnhancer, ScriptIntelligence


def main(argv: Sequence[str] | None = None) -> int:
    """Run one optional enhancement experiment without changing the source file."""
    parser = argparse.ArgumentParser(
        prog="python enhancement_benchmark.py",
        description="Run a read-only Script Intelligence enhancement experiment.",
    )
    parser.add_argument("script", type=Path, help="Input .txt script; it is never modified.")
    parser.add_argument("--profile", default="friendslop_gaming")
    parser.add_argument("--model-id", default=HuggingFaceEnhancementConfig().model_id)
    args = parser.parse_args(argv)

    source_path = args.script
    if not source_path.exists() or not source_path.is_file():
        raise SystemExit(f"Script not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    backend = HuggingFaceEnhancementBackend(
        config=HuggingFaceEnhancementConfig(model_id=args.model_id)
    )
    intelligence = ScriptIntelligence(enhancer=ScriptEnhancer(backend=backend))

    started_at = perf_counter()
    result = intelligence.process(
        source,
        mode=EnhancementMode.ENHANCE_ONLY,
        profile=args.profile,
    )
    elapsed = perf_counter() - started_at
    enhancement = result.enhancement
    assert enhancement is not None
    change_ratio = (
        len(enhancement.changes) / enhancement.analysis.sentence_count
        if enhancement.analysis.sentence_count
        else 0.0
    )

    print("HooperTTS Script Enhancement Experiment")
    print("======================================")
    print(f"Source file: {source_path}")
    print(f"Profile: {args.profile}")
    print(f"Model: {backend.config.model_id}")
    print(f"Device used: {backend.last_device or 'not loaded'}")
    print(f"Original word count: {count_words(source)}")
    print(f"Enhanced word count: {count_words(result.output_text)}")
    print(f"Changed sentence count: {len(enhancement.changes)}")
    print(f"Change ratio: {change_ratio:.2f}")
    print(f"Protected-span validation: {enhancement.validation.passed}")
    print(f"Enhancement latency: {elapsed:.2f}s")
    print(f"Backend available: {enhancement.backend_available}")
    print(f"Diagnostic: {enhancement.diagnostic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
