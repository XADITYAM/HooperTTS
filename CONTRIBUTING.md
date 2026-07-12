# Contributing

Thanks for helping improve HooperTTS. Keep changes small, tested, and focused
on narration quality.

## Project Structure

- `core/optimizer.py` is the public optimization entry point.
- `core/pronunciation.py` applies configured spoken forms.
- `core/planner.py` classifies sentences and builds `SentencePlan` metadata.
- `core/chunker.py` splits sentences into semantic spoken chunks.
- `core/rhythm.py` renders planned chunks into narration-friendly text.
- `core/profile.py` loads narration profiles from `profiles/`.
- `core/cli.py` powers the `hoopertts` command.
- `benchmark.py` contains benchmark metrics shared by CLI and script usage.
- `evaluate.py` evaluates large script datasets and writes CSV/JSON/Markdown
  reports.
- `tests/` contains dependency-free test scripts.

## Coding Style

- Use type hints for public and private helpers.
- Keep modules dependency-free unless the project explicitly adds a dependency.
- Prefer configuration data over hardcoded behavior when behavior is profile-
  or style-specific.
- Keep public API compatibility for `ScriptOptimizer.optimize(...)`.
- Run the test scripts and `py_compile` before opening a pull request.
- Ruff and Black configuration live in `pyproject.toml`.

## Commit Messages

Use short imperative messages:

- `Add profile validation tests`
- `Clean up rhythm helpers`
- `Document CLI usage`

Avoid bundling unrelated cleanup, behavior changes, and docs in one commit.

## Adding Profiles

1. Create a new JSON file in `profiles/`.
2. Include every required key:
   - `pause_strength`
   - `hook_style`
   - `reveal_style`
   - `ending_style`
   - `chunk_target`
   - `energy_curve`
   - `question_style`
3. Add the profile to `pyproject.toml` under `tool.setuptools.data-files`.
4. Add or update tests if the profile changes expected behavior.
5. Run `hoopertts profiles` and confirm the profile appears.
