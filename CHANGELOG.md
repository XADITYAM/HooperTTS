# Changelog

## Unreleased

- Added optional native Qwen3-TTS generation backend with environment
  diagnostics, prompt building, mocked tests, and CLI commands.
- Added `evaluate.py` for recursive dataset evaluation with CSV, JSON, and
  Markdown summaries.
- Added evaluation tests and documentation.
- Added Ruff and Black configuration for consistent local cleanup.
- Added GitHub Actions for tests and Python compilation on Python 3.10, 3.11,
  and 3.12.
- Added public-release documentation, including contributing guidelines,
  architecture notes, profile docs, and roadmap.
- Added an argparse-powered `hoopertts` CLI with optimize, benchmark, compare,
  profiles, and validate commands.
- Added `pyproject.toml` packaging support for `pip install -e .`.
- Added narration profiles with JSON configuration for default, documentary,
  gaming news, YouTube Shorts, and podcast delivery.
- Added `ProfileManager` and wired profiles into optimizer, planner, chunker,
  rhythm rendering, and benchmark reporting.
- Added benchmark estimates for profile, speaking speed, and speaking time.
- Added `SemanticChunker` for protected-phrase, punctuation-aware spoken idea
  groups.
- Updated `SentencePlan` to carry chunks and the rhythm engine to render those
  chunks directly.
- Updated the benchmark to report chunk count and chunk size statistics.
- Added `NarrationPlanner` and `SentencePlan` metadata for sentence-level
  narration intent.
- Updated the optimizer to pass planned sentences into the rhythm engine while
  keeping the public optimizer API unchanged.
- Updated the benchmark to report hook, reveal, question, and CTA counts.
- Added a configurable `PronunciationEngine` backed by `pronunciation.json`.
- Updated optimization to apply pronunciation replacements before rhythm
  grouping without changing the public optimizer API.
- Updated the benchmark to list pronunciation replacements found in the sample.
- Added the first `RhythmEngine` implementation for protected phrases, natural
  thought groups, opening pauses, contrast pauses, reveal emphasis, and cleaner
  final-sentence cadence.
- Updated the benchmark to report breath group rhythm statistics.
- Refactored the script optimizer into typed normalization, thought grouping,
  and style-rule stages while keeping `ScriptOptimizer.optimize(text, style="documentary")`.
- Moved style behavior definitions into configuration-only data.
- Added logging for unknown styles and unsupported style actions.
- Tightened whitespace handling to avoid unnecessary blank lines.
