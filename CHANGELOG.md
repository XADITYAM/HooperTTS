# Changelog

## Unreleased

- Fixed a second protected-span validator bug found via a real script upload:
  the capitalized-phrase pattern used `\s+` between words, so it could match
  across a line break — e.g. the last word of one bullet-list item fused with
  the first word of the next into a single fake protected phrase (bullet
  items often have no ending punctuation). Any reformatting of that
  whitespace then failed validation on both sides ("missing" and
  "invented" simultaneously). The pattern now only matches within a line,
  and the sentence-opener exclusion also recognizes list markers (`-`, `*`,
  `1.`, etc.), not just `.!?` punctuation.
- Fixed a bug where `ProtectedSpanValidator` treated every capitalized
  sentence-opening word (e.g. "Imagine", "Officially") as a protected proper
  noun, so almost any real rewrite was auto-rejected and enhancement silently
  returned the original script unchanged. Now only genuine names/titles are
  protected — multi-word capitalized phrases, and single capitalized words
  that aren't just sentence-initial. Genuine fact drops (dates, prices,
  platforms, real names) are still rejected as before.
- Rejected-candidate diagnostics now include the specific missing/invented
  spans instead of a bare "rejected by protected-span validation" message.
- Wired the existing Script Intelligence enhancement pipeline into the actual
  generation path: `qwen.runner.generate()` now accepts `enhancement_mode`
  and `enhancement_model_tier`, the Gradio app exposes a Script Enhancement
  dropdown (default off) and a Quality/Fast model-tier dropdown, and the CLI
  `generate` command gains matching `--enhance` / `--enhance-model` flags.
  Enhancement diagnostics are surfaced in both the app and CLI output.
- Added the v0.3-beta optional Hugging Face Qwen3 enhancement backend with
  lazy loading, CUDA-memory checks, immediate resource release, and audited
  change records.
- Added the v0.3-alpha Script Intelligence foundation: deterministic script
  analysis, profile-aware enhancement policies, protected-span validation, and
  an unavailable-by-default enhancement backend contract.
- Added the `friendslop_gaming` profile for casual, brisk short-form gaming
  narration with controlled energy and natural reveals.
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
