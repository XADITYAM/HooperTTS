"""Rhythm engine for natural spoken phrase grouping."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .planner import NarrationPlanner, SentencePlan
from .profile import NarrationProfile, ProfileManager


class RhythmEngine:
    """Group script text into natural narration breath groups."""

    OPENING_WORDS: tuple[str, ...] = (
        "Imagine",
        "Suppose",
        "Picture this",
        "What if",
    )
    CONTRAST_WORDS: tuple[str, ...] = (
        "But",
        "However",
        "Instead",
        "Yet",
        "Although",
    )
    REVEAL_WORDS: tuple[str, ...] = (
        "Finally",
        "Officially",
        "Breaking",
        "Confirmed",
        "Exclusive",
    )
    FINAL_BREAK_WORDS: tuple[str, ...] = (
        "and",
        "but",
        "for",
        "from",
        "in",
        "of",
        "to",
        "with",
    )

    _BLANK_LINE_PATTERN = re.compile(r"\n{3,}")

    def __init__(
        self,
        profile: NarrationProfile | None = None,
    ) -> None:
        """Create a rhythm engine for a narration profile."""
        self.profile = profile or ProfileManager().load()

    def process(self, text: str) -> str:
        """Return text shaped into natural spoken thought groups."""
        return self.process_plans(NarrationPlanner(self.profile).plan(text))

    def process_plans(self, sentence_plans: Sequence[SentencePlan]) -> str:
        """Return planned sentences shaped into natural spoken thought groups."""
        plans = list(sentence_plans)

        shaped_sentences: list[str] = []
        final_index = len(plans) - 1
        for index, sentence_plan in enumerate(plans):
            if not sentence_plan.text:
                continue

            groups = list(sentence_plan.chunks)
            if index == final_index:
                groups = self._shape_final_sentence(groups)

            shaped = "\n".join(groups)
            shaped = self._apply_planned_pauses(shaped, sentence_plan)
            shaped = self._apply_contrast_pauses(shaped)
            shaped = self._apply_reveal_emphasis(shaped, sentence_plan.emphasized_words)
            shaped_sentences.append(shaped.strip())

        return self._clean("\n\n".join(shaped_sentences))

    def _apply_opening_pause(self, text: str) -> str:
        if self.profile.hook_style == "conversational":
            return text
        for opening in self.OPENING_WORDS:
            pattern = re.compile(rf"^({re.escape(opening)})\b", flags=re.IGNORECASE)
            if pattern.search(text):
                pause = "..." if self.profile.hook_style != "fast_hook" else ".."
                return pattern.sub(
                    lambda match: f"{pause}\n{match.group(1)}", text, count=1
                )
        return text

    def _apply_contrast_pauses(self, text: str) -> str:
        contrast_pattern = "|".join(re.escape(word) for word in self.CONTRAST_WORDS)
        pattern = re.compile(rf"(?<!\w)({contrast_pattern})\b", flags=re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            prefix = text[: match.start()]
            if prefix.endswith("...\n") or prefix.endswith("...\n\n"):
                return match.group(1)
            if prefix and not prefix.endswith("\n"):
                return f"\n...\n{match.group(1)}"
            return f"...\n{match.group(1)}"

        return pattern.sub(replace, text)

    def _apply_planned_pauses(self, text: str, sentence_plan: SentencePlan) -> str:
        if sentence_plan.sentence_type == "HOOK":
            return self._apply_opening_pause(text)
        threshold = 0.3 * self.profile.pause_strength
        if sentence_plan.estimated_pause_before >= threshold:
            return f"...\n{text}"
        return text

    def _apply_reveal_emphasis(self, text: str, emphasized_words: Sequence[str]) -> str:
        if self.profile.reveal_style == "natural":
            return text
        if emphasized_words:
            reveal_pattern = "|".join(re.escape(word) for word in emphasized_words)
        else:
            reveal_pattern = "|".join(re.escape(word) for word in self.REVEAL_WORDS)
        pattern = re.compile(rf"\b({reveal_pattern})\b", flags=re.IGNORECASE)
        return pattern.sub(lambda match: match.group(1).upper(), text)

    def _shape_final_sentence(self, groups: list[str]) -> list[str]:
        if self.profile.ending_style == "punchy":
            return groups
        if len(groups) > 1:
            return groups

        words = groups[0].split()
        minimum_length = 5 if self.profile.ending_style == "slow" else 6
        if len(words) <= minimum_length:
            return groups

        split_at = self._find_final_break_index(words)
        if split_at is None:
            return groups
        return [" ".join(words[:split_at]).strip(), " ".join(words[split_at:]).strip()]

    def _find_final_break_index(self, words: Sequence[str]) -> int | None:
        upper_bound = min(6, len(words) - 1)
        lower_bound = 1
        for index in range(upper_bound, lower_bound - 1, -1):
            previous_word = words[index - 1].strip("\"'()[]{}").rstrip(".!?;:")
            if words[index - 1].endswith(","):
                return index
            if previous_word.lower() in self.FINAL_BREAK_WORDS:
                return index
        return None

    def _clean(self, text: str) -> str:
        cleaned = re.sub(r"[ \t]+\n", "\n", text)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
        cleaned = self._BLANK_LINE_PATTERN.sub("\n\n", cleaned)
        return cleaned.strip()
