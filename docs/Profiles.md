# Profiles

Narration profiles live in `profiles/` and are loaded by
`core.profile.ProfileManager`.

## Bundled Profiles

- `default`
- `documentary`
- `gaming_news`
- `youtube_shorts`
- `podcast`

## Required Fields

```json
{
  "pause_strength": 1.0,
  "hook_style": "intro_pause",
  "reveal_style": "emphasis",
  "ending_style": "clean",
  "chunk_target": 7,
  "energy_curve": {
    "HOOK": 8,
    "REVEAL": 8,
    "QUESTION": 7,
    "CTA": 7,
    "EVIDENCE": 5,
    "CONTRAST": 6,
    "NORMAL": 4
  },
  "question_style": "curious"
}
```

## Adding a Profile

1. Copy `profiles/default.json`.
2. Rename it to a lowercase, underscore-separated profile name.
3. Tune pause strength, chunk target, and energy values.
4. Add the JSON file to `pyproject.toml`.
5. Run:

```bash
hoopertts profiles
hoopertts --profile your_profile optimize samples/benchmark.txt
```
