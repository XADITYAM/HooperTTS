"""Semantic chunking for natural spoken idea groups."""

from __future__ import annotations

import re
from collections.abc import Sequence


class SemanticChunker:
    """Split sentences into short natural spoken idea groups."""

    DEFAULT_PROTECTED_PHRASES: tuple[str, ...] = (
        "Grand Theft Auto 6",
        "GTA 6",
        "Rockstar Games",
        "Take-Two Interactive",
        "Vice City",
        "Lucia and Jason",
    )
    CONJUNCTIONS: tuple[str, ...] = (
        "and",
        "but",
        "or",
        "because",
        "so",
        "then",
        "while",
        "although",
        "yet",
    )
    TRANSITIONS: tuple[str, ...] = (
        "actually",
        "finally",
        "however",
        "instead",
        "meanwhile",
        "officially",
        "seriously",
        "suddenly",
        "unexpectedly",
    )
    RELATIVE_WORDS: tuple[str, ...] = (
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
    )
    PREPOSITIONS: tuple[str, ...] = (
        "after",
        "around",
        "at",
        "before",
        "between",
        "by",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "of",
        "on",
        "over",
        "through",
        "to",
        "under",
        "with",
        "without",
    )
    DRAMATIC_WORDS: tuple[str, ...] = (
        "breaking",
        "finally",
        "confirmed",
        "exclusive",
        "suddenly",
    )

    MAX_CHUNK_WORDS = 7
    MIN_CHUNK_WORDS = 2

    _SPACE_PATTERN = re.compile(r"[ \t]+")
    _WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

    def __init__(
        self, protected_phrases: Sequence[str] | None = None, chunk_target: int = 7
    ) -> None:
        """Create a chunker with optional extra protected phrases."""
        phrases = protected_phrases or self.DEFAULT_PROTECTED_PHRASES
        self.protected_phrases = tuple(sorted(phrases, key=len, reverse=True))
        self.chunk_target = max(
            self.MIN_CHUNK_WORDS, min(self.MAX_CHUNK_WORDS, chunk_target)
        )

    def chunk(self, sentence: str) -> list[str]:
        """Return natural spoken idea groups for a sentence."""
        normalized = self._normalize(sentence)
        if not normalized:
            return []
        if self._is_dramatic_word(normalized):
            return [normalized]

        protected_sentence, placeholders = self._protect_phrases(normalized)
        tokens = protected_sentence.split()
        chunks: list[str] = []

        while self._count_tokens(tokens, placeholders) > self.chunk_target:
            break_index = self._find_break_index(tokens, placeholders)
            chunks.append(
                self._restore_phrases(" ".join(tokens[:break_index]), placeholders)
            )
            tokens = tokens[break_index:]

        if tokens:
            chunks.append(self._restore_phrases(" ".join(tokens), placeholders))

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _normalize(self, sentence: str) -> str:
        normalized = sentence.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._SPACE_PATTERN.sub(" ", normalized.strip())
        return normalized

    def _protect_phrases(self, sentence: str) -> tuple[str, dict[str, str]]:
        protected = sentence
        placeholders: dict[str, str] = {}
        for phrase in self.protected_phrases:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)

            def replace(match: re.Match[str]) -> str:
                placeholder = f"__PHRASE_{len(placeholders)}__"
                placeholders[placeholder] = match.group(0)
                return placeholder

            protected = pattern.sub(replace, protected)
        return protected, placeholders

    def _restore_phrases(self, text: str, placeholders: dict[str, str]) -> str:
        restored = text
        for placeholder, phrase in placeholders.items():
            restored = restored.replace(placeholder, phrase)
        return restored

    def _find_break_index(
        self, tokens: Sequence[str], placeholders: dict[str, str]
    ) -> int:
        candidates: list[tuple[int, int]] = []
        running_count = 0
        total_count = self._count_tokens(tokens, placeholders)

        for index, token in enumerate(tokens[:-1], start=1):
            running_count += self._token_word_count(token, placeholders)
            remaining_count = total_count - running_count
            if running_count < self.MIN_CHUNK_WORDS:
                continue
            if running_count > self.chunk_target:
                break
            if remaining_count == 1 and not self._is_dramatic_word(tokens[index]):
                continue
            score = self._boundary_score(token, tokens[index])
            candidates.append((score, index))

        scored_candidates = [candidate for candidate in candidates if candidate[0] > 0]
        if scored_candidates:
            scored_candidates.sort(reverse=True)
            return scored_candidates[0][1]
        if candidates:
            return candidates[-1][1]
        return min(len(tokens), self.chunk_target)

    def _boundary_score(self, current_token: str, next_token: str) -> int:
        current = self._clean_token(current_token)
        next_clean = self._clean_token(next_token)
        if current_token.rstrip().endswith((",", ";", ":", ")", "]", "}")):
            return 5
        if current in self.TRANSITIONS:
            return 4
        if current in self.CONJUNCTIONS:
            return 3
        if current in self.RELATIVE_WORDS or next_clean in self.RELATIVE_WORDS:
            return 2
        if current in self.PREPOSITIONS:
            return 1
        return 0

    def _count_tokens(
        self, tokens: Sequence[str], placeholders: dict[str, str]
    ) -> int:
        return sum(self._token_word_count(token, placeholders) for token in tokens)

    def _token_word_count(self, token: str, placeholders: dict[str, str]) -> int:
        cleaned = token.strip(",.!?;:\"'()[]{}")
        if cleaned in placeholders:
            return len(self._WORD_PATTERN.findall(placeholders[cleaned]))
        return len(self._WORD_PATTERN.findall(token))

    def _clean_token(self, token: str) -> str:
        return token.strip(",.!?;:\"'()[]{}").lower()

    def _is_dramatic_word(self, text: str) -> bool:
        words = self._WORD_PATTERN.findall(text)
        return len(words) == 1 and words[0].lower() in self.DRAMATIC_WORDS
