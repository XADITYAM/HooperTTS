"""Text optimization pipeline for narration scripts."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from re import Match

from .planner import NarrationPlanner
from .profile import ProfileManager
from .pronunciation import PronunciationEngine
from .rhythm import RhythmEngine
from .style_rules import EMOTIONAL_WORDS, STYLE_RULES

logger = logging.getLogger(__name__)


class ScriptOptimizer:
    """Optimize plain text scripts for spoken TTS delivery."""

    _SPACE_PATTERN = re.compile(r"[ \t]+")
    _BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
    _TRAILING_SPACE_PATTERN = re.compile(r"[ \t]+\n")

    def optimize(
        self, text: str, style: str = "documentary", profile: str = "default"
    ) -> str:
        """Return an optimized version of text for the requested style."""
        logger.debug(
            "Optimizing script with style '%s' and profile '%s'.", style, profile
        )
        narration_profile = ProfileManager().load(profile)

        optimized = self._apply_pronunciations(text)
        optimized = self._normalize_whitespace(optimized)
        sentence_plans = NarrationPlanner(narration_profile).plan(optimized)
        optimized = RhythmEngine(profile=narration_profile).process_plans(
            sentence_plans
        )
        return self._clean_blank_lines(optimized)

    def normalize(self, text: str) -> str:
        """Return text with consistent spacing and blank lines."""
        return self._normalize_whitespace(text)

    def normalize_whitespace(self, text: str) -> str:
        """Return text with consistent spacing and blank lines."""
        return self._normalize_whitespace(text)

    def create_thought_groups(self, text: str) -> str:
        """Return text split into short spoken thought groups."""
        return self._create_thought_groups(text)

    def insert_pauses(self, text: str) -> str:
        """Return text split with narration-friendly pauses."""
        return self._create_thought_groups(text)

    def apply_style(self, text: str) -> str:
        """Return text with the default documentary style rules applied."""
        return self._apply_style_rules(text, "documentary")

    def apply_style_rules(self, text: str, style: str = "documentary") -> str:
        """Return text with the requested style rules applied."""
        return self._apply_style_rules(text, style)

    def _normalize_whitespace(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._SPACE_PATTERN.sub(" ", normalized.strip())
        return self._clean_blank_lines(normalized)

    def _apply_pronunciations(self, text: str) -> str:
        pronounced = PronunciationEngine().process(text)
        return self._clean_blank_lines(pronounced)

    def _create_thought_groups(self, text: str) -> str:
        grouped = RhythmEngine().process(text)
        return self._clean_blank_lines(grouped)

    def _apply_style_rules(self, text: str, style: str) -> str:
        rules = STYLE_RULES.get(style)
        if rules is None:
            logger.warning("Unknown optimizer style '%s'; using documentary.", style)
            rules = EMOTIONAL_WORDS

        styled = text
        for word, action in rules.items():
            pattern = re.compile(rf"\b({re.escape(word)})\b", flags=re.IGNORECASE)
            transformer = self._get_rule_transformer(action)
            if transformer is None:
                logger.debug("Skipping unsupported style action '%s'.", action)
                continue
            styled = pattern.sub(transformer, styled)

        return self._clean_blank_lines(styled)

    def _get_rule_transformer(
        self, action: str
    ) -> Callable[[Match[str]], str] | None:
        transformers: dict[str, Callable[[Match[str]], str]] = {
            "pause_before": lambda match: f"...\n{match.group(1)}",
            "contrast": lambda match: f"\n...\n{match.group(1)}",
            "emphasis": lambda match: match.group(1).upper(),
            "soft": lambda match: f"{match.group(1)}...",
            "dramatic": lambda match: f"\n...\n{match.group(1).upper()}",
        }
        return transformers.get(action)

    def _clean_blank_lines(self, text: str) -> str:
        cleaned = self._TRAILING_SPACE_PATTERN.sub("\n", text)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
        cleaned = self._BLANK_LINE_PATTERN.sub("\n\n", cleaned)
        return cleaned.strip()
