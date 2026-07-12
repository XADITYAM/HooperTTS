"""Configurable pronunciation replacement engine."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


class PronunciationEngine:
    """Replace configured written terms with spoken pronunciation forms."""

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "pronunciation.json"

    def __init__(self, config_path: Path | None = None) -> None:
        """Load pronunciation replacements from a JSON configuration file."""
        self.config_path = config_path or self._default_config_path()
        self.replacements = self._load_replacements(self.config_path)
        self._pattern = self._build_pattern(self.replacements)

    def process(self, text: str) -> str:
        """Return text with configured pronunciation replacements applied."""
        if self._pattern is None:
            return text

        return self._pattern.sub(
            lambda match: self._replacement_for(match.group(0)), text
        )

    def get_replacements(self, text: str) -> list[tuple[str, str]]:
        """Return unique pronunciation replacements that would apply to text."""
        if self._pattern is None:
            return []

        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for match in self._pattern.finditer(text):
            original = match.group(0)
            lookup_key = original.lower()
            if lookup_key in seen:
                continue
            seen.add(lookup_key)
            found.append((original, self._replacement_for(original)))
        return found

    def _load_replacements(self, config_path: Path) -> dict[str, str]:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_replacements = json.load(config_file)

        return {
            str(term): str(spoken_form)
            for term, spoken_form in raw_replacements.items()
        }

    def _default_config_path(self) -> Path:
        candidates = (
            self.DEFAULT_CONFIG_PATH,
            Path.cwd() / "pronunciation.json",
            Path(sys.prefix) / "pronunciation.json",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.DEFAULT_CONFIG_PATH

    def _build_pattern(
        self, replacements: dict[str, str]
    ) -> re.Pattern[str] | None:
        if not replacements:
            return None

        alternatives = sorted(replacements, key=len, reverse=True)
        escaped_terms = "|".join(re.escape(term) for term in alternatives)
        return re.compile(rf"(?<!\w)({escaped_terms})(?!\w)", flags=re.IGNORECASE)

    def _replacement_for(self, original: str) -> str:
        for configured_term, spoken_form in self.replacements.items():
            if configured_term.lower() == original.lower():
                return self._match_capitalization(
                    original, configured_term, spoken_form
                )
        return original

    def _match_capitalization(
        self, original: str, configured_term: str, spoken_form: str
    ) -> str:
        if original == configured_term:
            return spoken_form
        if original.islower():
            return spoken_form.lower()
        if original.isupper():
            return spoken_form.upper()
        if original.istitle() and not self._looks_like_acronym(spoken_form):
            return spoken_form.title()
        return spoken_form

    def _looks_like_acronym(self, spoken_form: str) -> bool:
        compact = spoken_form.replace(" ", "")
        return compact.isupper() and len(compact) > 1
