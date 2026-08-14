"""Deterministic script-quality analysis for optional Script Intelligence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


IssueCategory = Literal[
    "hook",
    "clarity",
    "exposition",
    "sentence_length",
    "pacing",
    "repetition",
    "transition",
    "information_density",
    "reveal",
    "ending",
    "short_form",
]
IssueSeverity = Literal["info", "suggestion", "warning"]


@dataclass(frozen=True)
class ScriptIssue:
    """A concrete, non-rewriting observation about a source script."""

    category: IssueCategory
    severity: IssueSeverity
    sentence_index: int | None
    evidence: str
    recommendation: str


@dataclass(frozen=True)
class ScriptAnalysis:
    """Structured, deterministic assessment of a script's editorial readiness."""

    hook_score: int
    clarity_score: int
    pacing_score: int
    repetition_score: int
    short_form_score: int
    information_density_score: int
    ending_score: int
    word_count: int
    sentence_count: int
    average_sentence_words: float
    issues: tuple[ScriptIssue, ...]

    @property
    def scores(self) -> dict[str, int]:
        """Return scores in a JSON-friendly form."""
        return {
            "hook": self.hook_score,
            "clarity": self.clarity_score,
            "pacing": self.pacing_score,
            "repetition": self.repetition_score,
            "short_form": self.short_form_score,
            "information_density": self.information_density_score,
            "ending": self.ending_score,
        }


class ScriptAnalyzer:
    """Assess script structure with transparent heuristics and no LLM dependency."""

    HOOK_OPENERS = ("imagine", "what if", "picture this", "suppose", "did you")
    TRANSITIONS = (
        "but",
        "however",
        "instead",
        "meanwhile",
        "because",
        "so",
        "then",
        "while",
    )
    CTA_WORDS = (
        "subscribe",
        "follow",
        "like",
        "comment",
        "share",
        "tell me",
        "let me know",
    )
    REVEAL_WORDS = (
        "finally",
        "officially",
        "confirmed",
        "revealed",
        "announced",
    )

    _SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")
    _WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
    _CONTENT_WORD_PATTERN = re.compile(r"\b(?:[A-Za-z][\w'-]*|\d+(?:\.\d+)?)\b")
    _FILLER_PATTERN = re.compile(
        r"\b(?:really|very|actually|basically|literally|just|kind of|sort of)\b",
        flags=re.IGNORECASE,
    )

    def analyze(self, text: str) -> ScriptAnalysis:
        """Return structured analysis without changing the source text."""
        sentences = self._split_sentences(text)
        word_counts = [self._word_count(sentence) for sentence in sentences]
        word_count = sum(word_counts)
        average_sentence_words = (
            round(word_count / len(sentences), 2) if sentences else 0.0
        )
        issues: list[ScriptIssue] = []

        hook_score = self._analyze_hook(sentences, issues)
        clarity_score = self._analyze_clarity(sentences, word_counts, issues)
        pacing_score = self._analyze_pacing(sentences, word_counts, issues)
        repetition_score = self._analyze_repetition(sentences, issues)
        self._analyze_exposition(sentences, word_counts, issues)
        self._analyze_transitions(sentences, issues)
        self._analyze_reveal_placement(sentences, issues)
        information_density_score = self._analyze_density(text, word_count, issues)
        ending_score = self._analyze_ending(sentences, issues)
        short_form_score = self._short_form_score(
            word_count,
            average_sentence_words,
            hook_score,
            pacing_score,
            information_density_score,
            issues,
        )

        return ScriptAnalysis(
            hook_score=hook_score,
            clarity_score=clarity_score,
            pacing_score=pacing_score,
            repetition_score=repetition_score,
            short_form_score=short_form_score,
            information_density_score=information_density_score,
            ending_score=ending_score,
            word_count=word_count,
            sentence_count=len(sentences),
            average_sentence_words=average_sentence_words,
            issues=tuple(issues),
        )

    def _analyze_hook(self, sentences: list[str], issues: list[ScriptIssue]) -> int:
        if not sentences:
            return 0
        opening = sentences[0].strip()
        lowered = opening.lower()
        opening_words = self._word_count(opening)
        if lowered.startswith(self.HOOK_OPENERS) or opening.endswith("?"):
            return 9
        if opening_words <= 14:
            return 7
        issues.append(
            ScriptIssue(
                category="hook",
                severity="suggestion",
                sentence_index=0,
                evidence=opening,
                recommendation="Lead with the premise, question, or most specific detail sooner.",
            )
        )
        return 4

    def _analyze_clarity(
        self,
        sentences: list[str],
        word_counts: list[int],
        issues: list[ScriptIssue],
    ) -> int:
        long_sentences = [
            (index, sentence)
            for index, (sentence, count) in enumerate(zip(sentences, word_counts))
            if count > 28
        ]
        for index, sentence in long_sentences:
            issues.append(
                ScriptIssue(
                    category="sentence_length",
                    severity="suggestion",
                    sentence_index=index,
                    evidence=sentence,
                    recommendation="Split or tighten this sentence while preserving its factual detail.",
                )
            )
        return max(1, 9 - min(len(long_sentences) * 2, 7)) if sentences else 0

    def _analyze_pacing(
        self,
        sentences: list[str],
        word_counts: list[int],
        issues: list[ScriptIssue],
    ) -> int:
        if not sentences:
            return 0
        long_count = sum(count > 24 for count in word_counts)
        short_count = sum(count <= 7 for count in word_counts)
        if long_count:
            issues.append(
                ScriptIssue(
                    category="pacing",
                    severity="suggestion",
                    sentence_index=None,
                    evidence=f"{long_count} sentence(s) exceed 24 words.",
                    recommendation="Mix compact factual beats with longer explanatory sentences.",
                )
            )
        if len(sentences) >= 4 and short_count == 0:
            issues.append(
                ScriptIssue(
                    category="pacing",
                    severity="info",
                    sentence_index=None,
                    evidence="The script contains no short beat sentences.",
                    recommendation="Consider a concise reaction or transition where it improves rhythm.",
                )
            )
        return max(1, 9 - long_count * 2 - (1 if len(sentences) >= 4 and not short_count else 0))

    def _analyze_repetition(
        self, sentences: list[str], issues: list[ScriptIssue]
    ) -> int:
        seen: dict[str, int] = {}
        repeats: list[tuple[int, str]] = []
        for index, sentence in enumerate(sentences):
            normalized = " ".join(self._WORD_PATTERN.findall(sentence.lower()))
            if normalized and normalized in seen:
                repeats.append((index, sentence))
            seen[normalized] = index
        for index, sentence in repeats:
            issues.append(
                ScriptIssue(
                    category="repetition",
                    severity="warning",
                    sentence_index=index,
                    evidence=sentence,
                    recommendation="Remove or consolidate this repeated sentence.",
                )
            )
        return max(1, 10 - len(repeats) * 3) if sentences else 0

    def _analyze_exposition(
        self,
        sentences: list[str],
        word_counts: list[int],
        issues: list[ScriptIssue],
    ) -> None:
        if not sentences or word_counts[0] <= 20:
            return
        issues.append(
            ScriptIssue(
                category="exposition",
                severity="suggestion",
                sentence_index=0,
                evidence=sentences[0],
                recommendation="Condense opening background and surface the central premise first.",
            )
        )

    def _analyze_transitions(
        self, sentences: list[str], issues: list[ScriptIssue]
    ) -> None:
        if len(sentences) < 3:
            return
        has_transition = any(
            re.search(rf"\b{re.escape(word)}\b", sentence, flags=re.IGNORECASE)
            for sentence in sentences[1:]
            for word in self.TRANSITIONS
        )
        if not has_transition:
            issues.append(
                ScriptIssue(
                    category="transition",
                    severity="info",
                    sentence_index=None,
                    evidence="No explicit transition cue was found after the opening sentence.",
                    recommendation="Add a transition only where it clarifies the relationship between factual beats.",
                )
            )

    def _analyze_reveal_placement(
        self, sentences: list[str], issues: list[ScriptIssue]
    ) -> None:
        reveal_index = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if any(
                    re.search(rf"\b{re.escape(word)}\b", sentence, re.IGNORECASE)
                    for word in self.REVEAL_WORDS
                )
            ),
            None,
        )
        if reveal_index is not None and reveal_index > 1:
            issues.append(
                ScriptIssue(
                    category="reveal",
                    severity="info",
                    sentence_index=reveal_index,
                    evidence=sentences[reveal_index],
                    recommendation="Consider whether this verified reveal belongs closer to the premise.",
                )
            )

    def _analyze_density(
        self, text: str, word_count: int, issues: list[ScriptIssue]
    ) -> int:
        if word_count == 0:
            return 0
        filler_count = len(self._FILLER_PATTERN.findall(text))
        density = max(1, 9 - filler_count)
        if filler_count:
            issues.append(
                ScriptIssue(
                    category="information_density",
                    severity="suggestion",
                    sentence_index=None,
                    evidence=f"{filler_count} filler phrase(s) detected.",
                    recommendation="Remove filler only where doing so keeps the factual claim intact.",
                )
            )
        return density

    def _analyze_ending(self, sentences: list[str], issues: list[ScriptIssue]) -> int:
        if not sentences:
            return 0
        ending = sentences[-1].strip()
        lowered = ending.lower()
        if any(word in lowered for word in self.CTA_WORDS) or ending.endswith("?"):
            return 8
        if self._word_count(ending) <= 16:
            return 7
        issues.append(
            ScriptIssue(
                category="ending",
                severity="info",
                sentence_index=len(sentences) - 1,
                evidence=ending,
                recommendation="Consider ending on the clearest takeaway or a natural audience prompt.",
            )
        )
        return 5

    def _short_form_score(
        self,
        word_count: int,
        average_sentence_words: float,
        hook_score: int,
        pacing_score: int,
        density_score: int,
        issues: list[ScriptIssue],
    ) -> int:
        if word_count == 0:
            return 0
        length_score = 9 if word_count <= 150 else 6 if word_count <= 250 else 3
        sentence_score = 9 if average_sentence_words <= 18 else 5 if average_sentence_words <= 25 else 2
        if word_count > 250:
            issues.append(
                ScriptIssue(
                    category="short_form",
                    severity="suggestion",
                    sentence_index=None,
                    evidence=f"The script contains {word_count} words.",
                    recommendation="Prioritize the strongest verified details for short-form delivery.",
                )
            )
        return round((length_score + sentence_score + hook_score + pacing_score + density_score) / 5)

    def _split_sentences(self, text: str) -> list[str]:
        return [
            match.group(0).strip()
            for match in self._SENTENCE_PATTERN.finditer(text)
            if match.group(0).strip()
        ]

    def _word_count(self, text: str) -> int:
        return len(self._CONTENT_WORD_PATTERN.findall(text))
