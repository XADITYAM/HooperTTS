# Changelog

## Unreleased

- Added validator-aware enhancement retries for Phi-3.5-mini-instruct. When a generated rewrite is rejected because it drops protected facts (for example `GTA 6` or `2022`), the strict validator remains unchanged and the backend gets one or more configurable retry attempts with the exact validation failures fed back into a correction prompt. Retry decoding uses a lower default temperature (0.25) to favor factual recovery over extra creativity. Added an explicit immutable-fact ledger to the enhancement prompt, including dates, years, numbered titles, and URLs/prices where present. This is model-agnostic at the backend level but includes a Phi-specific instruction block for stronger literal constraint following.
- Added environment controls `HOOPERTTS_ENHANCEMENT_VALIDATION_RETRIES`, `HOOPERTTS_ENHANCEMENT_RETRY_TEMPERATURE`, and `HOOPERTTS_ENHANCEMENT_RETRY_TOP_P`. Default is one validation retry.
- Added regression coverage for Phi prompt facts, successful validator retry, repeated retry rejection, and environment configuration. Full test suite now passes with 73 tests.
- Fixed a sentence-splitting bug found via a real script upload: the naive
  `[^.!?]+[.!?]?` sentence pattern in `NarrationPlanner` treated abbreviation
  periods (e.g. "Aug.") as sentence-ending, creating a fake sentence boundary
  right in the middle of "...on Aug. 27, but fans...". That fake boundary
  then triggered two separate pause heuristics on a boundary that was never
  really there (pre-sentence pause + contrast-word pause before "but"),
  producing stray isolated "..." pause-marker lines in the optimized output.
  Added an abbreviation list (months, honorifics) that's protected before
  sentence splitting and restored after. This is the accepted trade-off of
  any lightweight tokenizer: "Aug." immediately followed by a genuine new
  sentence now gets merged too, rather than risking the more visible failure
  mode of a wrongly split, visibly fragmented sentence. Added
  `tests/test_planner.py` — the planner previously had zero direct test
  coverage, which is part of why this went unnoticed.
- Fixed the real blocker behind Phi-3.5-mini's first rejected candidate: the
  bare single-quote alternative in `_QUOTATION_PATTERN` had no word-boundary
  guard, so two unrelated possessive/contraction apostrophes (e.g. "GTA 6's
  ... the game's") got greedily paired into one fake multi-sentence
  "quotation" spanning everything between them — this alone produced most of
  a long missing/invented diagnostic list. Also normalized curly ("smart")
  vs. straight quote/apostrophe typography before span extraction, since the
  source script uses curly quotes but model output commonly defaults to
  straight ones even when a quotation's actual content is preserved exactly
  — that typography difference alone was causing false rejections
  independent of the pairing bug. Genuine dropped/altered quotes are still
  correctly rejected either way.
- Added a third enhancement model tier, "Creative" (`microsoft/Phi-3.5-mini-instruct`),
  selectable in the app's Enhancement Model dropdown and via `--enhance-model creative`
  on the CLI. Added after real Colab testing: Qwen3-1.7B stayed 96.8%-99.6%
  word-for-word identical to the input across multiple runs, even at
  temperature 1.2 with top_p 0.97 — it applied mechanical instructions
  (list-item punctuation) but not structural ones (hooks, reordering),
  regardless of sampling settings. That ruled out decoding randomness as the
  cause and pointed to a genuine instruction-following capability gap at
  this model size. Phi-3.5-mini-instruct is specifically known for stronger
  multi-constraint instruction-following at a similar parameter count. VRAM
  requirement set to 6 GiB free (vs. 4.5 for Qwen3-1.7B); the existing
  sequential load/release pattern still applies so this doesn't change
  concurrent VRAM usage, only the size of whichever single model is loaded
  at a time. Generation will take noticeably longer than the 45-85s range
  seen with Qwen3-1.7B.
- Added retry-with-backoff around the enhancement model's Hugging Face
  download (tokenizer + model), matching the pattern the notebook already
  uses for the Qwen3-TTS snapshot download. Root cause of a real failure:
  `AutoModelForCausalLM.from_pretrained` had zero retry logic, so a single
  dropped connection mid-download (`IncompleteRead`) on Colab's free-tier
  networking discarded the entire enhancement attempt immediately, even
  though the exact same class of failure was already known to happen (it's
  why the Qwen3-TTS download has retries). 4 attempts, 10s/20s/30s backoff.
- Diffed a real run and found the model was applying the one *mechanical*
  instruction it had a concrete rule for (add a period between former bullet
  items) while leaving sentence order and phrasing 96.8% word-for-word
  identical to the source — it wasn't ignoring instructions, it just had no
  concrete example of what "restructure boldly" actually looks like as
  opposed to a paraphrase-safety instruction to leave things alone. Added a
  worked before/after example to the prompt (generic topic, so it doesn't
  bias any profile's actual subject matter) demonstrating real sentence
  reordering while every fact stays exact. Small instruction models
  generally imitate a shown example far better than they execute an
  abstract stylistic directive.
- Found (via direct diffing) that the "enhanced" output being reported was,
  word-for-word, the untouched original script — not a conservative
  paraphrase. The enhancer prompt now explicitly forbids returning the
  original wording unchanged, since a heavily constrained "do not
  invent/remove/alter X, Y, Z..." prompt can push a cautious model toward
  copying the input verbatim as the only way to guarantee it breaks none of
  the listed rules. Also added a diagnostic check: if an accepted candidate
  turns out to be identical to the source at the sentence level, the
  diagnostic now says so explicitly instead of looking identical to a real
  accepted rewrite ("Generated a candidate with...").
- Switched the enhancement backend from greedy decoding (`do_sample=False`) to
  sampling by default (`do_sample=True`, `temperature=0.8`, `top_p=0.9`),
  configurable via `HOOPERTTS_ENHANCEMENT_DO_SAMPLE` /
  `HOOPERTTS_ENHANCEMENT_TEMPERATURE` / `HOOPERTTS_ENHANCEMENT_TOP_P`. Root
  cause of "the rewrite is basically the same as the input" even after the
  friendslop_gaming policy was loosened: greedy decoding always picks the
  single highest-probability token at each step, which stays close to a
  near-paraphrase of a well-formed input regardless of what the prompt asks
  for. Protected-span validation is unaffected by decoding strategy and
  remains the hard safety net against invented/dropped facts.
- Loosened the `friendslop_gaming` policy, which was structurally preventing
  any real creative rewrite: `max_changed_sentences_ratio` was 0.4, so a
  rewrite touching most of the script (e.g. adding a hook, reordering for
  a surprise beat) got discarded wholesale and silently fell back to the
  original text. Raised the ratio to 0.9, added explicit "open with a hook"
  / "build toward a surprising detail" writing goals, and removed the
  "unnecessary full-script rewrites" restriction that conflicted with them.
  Also removed the hardcoded "keep a strong sentence unchanged" line from
  the shared prompt template (it was overriding per-profile intent) —
  restraint vs. boldness is now controlled entirely by each profile's own
  writing_goals/avoid list. `default` and any unlisted profile are
  unaffected (already self-regulate via their own "avoid unnecessary
  rewriting"). Fact protection (protected-span validation) is unchanged.
- Fixed the enhancer prompt flattening bulleted/list-style scripts into a
  single run-on sentence with no punctuation between former items (found via
  real generation output). The prompt now explicitly tells the model to keep
  each list item as its own sentence/clause with clear ending punctuation.
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
