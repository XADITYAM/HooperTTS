"""Narration planning heuristics for HooperTTS scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .chunker import SemanticChunker
from .profile import NarrationProfile, ProfileManager

SentenceType = Literal[
    "HOOK",
    "REVEAL",
    "QUESTION",
    "CTA",
    "EVIDENCE",
    "CONTRAST",
    "NORMAL",
]


@dataclass(frozen=True)
class SentencePlan:
    """Narration metadata for a single sentence."""

    text: str
    sentence_type: SentenceType
    estimated_energy: int
    estimated_pause_before: float
    estimated_pause_after: float
    emphasized_words: list[str]
    chunks: list[str]


class NarrationPlanner:
    """Analyze script sentences and assign narration metadata."""

    HOOK_OPENERS: tuple[str, ...] = (
        "imagine",
        "suppose",
        "picture this",
        "what if",
    )
    REVEAL_WORDS: tuple[str, ...] = (
        "finally",
        "officially",
        "breaking",
        "confirmed",
        "exclusive",
        "revealed",
        "announced",
    )
    CONTRAST_WORDS: tuple[str, ...] = (
        "but",
        "however",
        "instead",
        "yet",
        "although",
        "despite",
    )
    CTA_WORDS: tuple[str, ...] = (
        "subscribe",
        "follow",
        "like",
        "comment",
        "share",
        "watch",
        "click",
        "check out",
        "tell me",
        "let me know",
    )
    EVIDENCE_WORDS: tuple[str, ...] = (
        "according to",
        "data",
        "report",
        "reports",
        "source",
        "sources",
        "study",
        "confirmed by",
        "evidence",
    )

    _SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")
    _NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    _SPACE_PATTERN = re.compile(r"[ \t]+")

    def __init__(self, profile: NarrationProfile | None = None) -> None:
        """Create a planner for a narration profile."""
        self.profile = profile or ProfileManager().load()
        self.chunker = SemanticChunker(chunk_target=self.profile.chunk_target)

    def plan(self, text: str) -> list[SentencePlan]:
        """Return narration plans for every sentence in text."""
        sentences = self._split_sentences(text)
        return [
            self._plan_sentence(sentence, index)
            for index, sentence in enumerate(sentences)
        ]

    def _split_sentences(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._SPACE_PATTERN.sub(" ", normalized.strip())
        return [
            match.group(0).strip()
            for match in self._SENTENCE_PATTERN.finditer(normalized)
            if match.group(0).strip()
        ]

    def _plan_sentence(self, sentence: str, index: int) -> SentencePlan:
        sentence_type = self._detect_type(sentence, index)
        emphasized_words = self._find_emphasized_words(sentence)
        return SentencePlan(
            text=sentence,
            sentence_type=sentence_type,
            estimated_energy=self._estimate_energy(sentence_type, sentence),
            estimated_pause_before=self._estimate_pause_before(sentence_type),
            estimated_pause_after=self._estimate_pause_after(sentence_type),
            emphasized_words=emphasized_words,
            chunks=self.chunker.chunk(sentence),
        )

    def _detect_type(self, sentence: str, index: int) -> SentenceType:
        lowered = sentence.lower().strip()
        if lowered.endswith("?"):
            return "QUESTION"
        if self._contains_phrase(lowered, self.CTA_WORDS):
            return "CTA"
        if index == 0 and self._starts_with_phrase(lowered, self.HOOK_OPENERS):
            return "HOOK"
        if self._contains_phrase(lowered, self.REVEAL_WORDS):
            return "REVEAL"
        if self._contains_phrase(lowered, self.CONTRAST_WORDS):
            return "CONTRAST"
        if self._contains_phrase(
            lowered, self.EVIDENCE_WORDS
        ) or self._NUMBER_PATTERN.search(sentence):
            return "EVIDENCE"
        return "NORMAL"

    def _estimate_energy(self, sentence_type: SentenceType, sentence: str) -> int:
        energy = self.profile.energy_curve[sentence_type]
        if "!" in sentence:
            energy += 1
        return max(1, min(10, energy))

    def _estimate_pause_before(self, sentence_type: SentenceType) -> float:
        base_pauses: dict[SentenceType, float] = {
            "HOOK": 0.8,
            "REVEAL": 0.4,
            "QUESTION": 0.3,
            "CTA": 0.4,
            "EVIDENCE": 0.2,
            "CONTRAST": 0.3,
            "NORMAL": 0.0,
        }
        return round(base_pauses[sentence_type] * self.profile.pause_strength, 2)

    def _estimate_pause_after(self, sentence_type: SentenceType) -> float:
        base_pauses: dict[SentenceType, float] = {
            "HOOK": 0.5,
            "REVEAL": 0.5,
            "QUESTION": 0.5,
            "CTA": 0.6,
            "EVIDENCE": 0.3,
            "CONTRAST": 0.4,
            "NORMAL": 0.3,
        }
        if sentence_type == "QUESTION" and self.profile.question_style == "urgent":
            return round(base_pauses[sentence_type] * 0.8, 2)
        return round(base_pauses[sentence_type] * self.profile.pause_strength, 2)

    def _find_emphasized_words(self, sentence: str) -> list[str]:
        lowered = sentence.lower()
        emphasized: list[str] = []
        for word in self.REVEAL_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                emphasized.append(word)
        return emphasized

    def _starts_with_phrase(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(text.startswith(phrase) for phrase in phrases)

    def _contains_phrase(self, text: str, phrases: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)
