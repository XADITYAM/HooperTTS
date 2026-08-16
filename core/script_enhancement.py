"""Optional Script Intelligence orchestration and safety validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from math import ceil
from difflib import SequenceMatcher
from typing import Literal

from .enhancement_backends import EnhancementBackend, UnavailableEnhancementBackend
from .enhancement_policy import EnhancementPolicy, EnhancementPolicyManager
from .optimizer import ScriptOptimizer
from .script_analysis import ScriptAnalysis, ScriptAnalyzer


class EnhancementMode(str, Enum):
    """Supported Script Intelligence processing modes."""

    OPTIMIZE_ONLY = "optimize_only"
    ENHANCE_ONLY = "enhance_only"
    ENHANCE_AND_OPTIMIZE = "enhance_and_optimize"


ProtectedSpanKind = Literal[
    "date",
    "number",
    "organization",
    "platform",
    "price",
    "proper_noun",
    "quotation",
    "title",
    "url",
]


@dataclass(frozen=True)
class ProtectedSpan:
    """A source detail that an enhancement candidate must preserve exactly."""

    kind: ProtectedSpanKind
    value: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of protected-span and no-invention validation."""

    passed: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class ChangeRecord:
    """Reserved audit record for a future available enhancement backend."""

    kind: str
    sentence_index: int | None
    before: str
    after: str
    rationale: str


@dataclass(frozen=True)
class EnhancementResult:
    """Analysis and safe enhancement outcome for one source script."""

    original_text: str
    enhanced_text: str
    analysis: ScriptAnalysis
    policy: EnhancementPolicy
    changes: tuple[ChangeRecord, ...]
    backend_name: str
    backend_available: bool
    diagnostic: str
    validation: ValidationResult


@dataclass(frozen=True)
class ScriptIntelligenceResult:
    """Result of one requested Script Intelligence mode."""

    mode: EnhancementMode
    original_text: str
    enhancement: EnhancementResult | None
    output_text: str


class ProtectedSpanValidator:
    """Reject candidates that lose source facts or add protected-looking details."""

    PLATFORMS = (
        "PlayStation",
        "PS5",
        "PS4",
        "Xbox",
        "Nintendo Switch",
        "Steam",
        "Epic Games Store",
        "PC",
        "iOS",
        "Android",
    )
    _URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
    _QUOTATION_PATTERN = re.compile(r'"[^"\n]+"|“[^”\n]+”|\'[^\'\n]+\'')
    _PRICE_PATTERN = re.compile(r"(?:[$€£¥]\s?\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s?(?:USD|EUR|GBP|INR))\b", re.IGNORECASE)
    _DATE_PATTERN = re.compile(
        r"\b(?:\d{4}-\d{1,2}-\d{1,2}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|"
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)(?:\s+\d{4})?)\b",
        re.IGNORECASE,
    )
    _NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?\b")
    # [ \t]+ (not \s+) between words: a run of capitalized words must stay on
    # one line to be treated as a single phrase. Without this, two unrelated
    # capitalized words on either side of a line break (e.g. the last word of
    # one bullet list item and the first word of the next) get fused into one
    # fake "protected phrase" that no reformatting can ever match exactly.
    _CAPITALIZED_PHRASE_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z0-9'’-]*(?:[ \t]+(?:[A-Z][A-Za-z0-9'’-]*|\d+|[IVX]+)){0,5}\b"
    )
    _ORG_SUFFIX_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z0-9'’-]*(?:[ \t]+[A-Z][A-Za-z0-9'’-]*)*[ \t]+"
        r"(?:Games|Studios|Interactive|Entertainment|Inc\.?|Ltd\.?)\b"
    )

    def extract(self, text: str) -> tuple[ProtectedSpan, ...]:
        """Extract exact source spans that must remain present after enhancement."""
        spans: list[ProtectedSpan] = []
        spans.extend(self._find("url", self._URL_PATTERN, text))
        spans.extend(self._find("quotation", self._QUOTATION_PATTERN, text))
        spans.extend(self._find("price", self._PRICE_PATTERN, text))
        spans.extend(self._find("date", self._DATE_PATTERN, text))
        spans.extend(self._find("number", self._NUMBER_PATTERN, text))
        spans.extend(self._find("organization", self._ORG_SUFFIX_PATTERN, text))
        for platform in self.PLATFORMS:
            pattern = re.compile(rf"(?<!\w){re.escape(platform)}(?!\w)")
            spans.extend(self._find("platform", pattern, text))
        for match in self._CAPITALIZED_PHRASE_PATTERN.finditer(text):
            value = match.group(0)
            if self._is_ordinary_sentence_opener(text, match.start(), value):
                continue
            kind: ProtectedSpanKind = "title" if any(char.isdigit() for char in value) else "proper_noun"
            spans.append(ProtectedSpan(kind=kind, value=value))
        return self._unique(spans)

    def _is_ordinary_sentence_opener(self, text: str, start: int, value: str) -> bool:
        """Return True for a single capitalized word that is only capitalized
        because it opens a sentence or a list/paragraph line (e.g. "Imagine",
        "Officially", or the first word after a bullet marker like "- " or
        "1)"), not because it is a genuine name or title. Multi-word
        capitalized phrases are kept protected regardless of position, since
        those are almost always real titles or names (e.g. "Grand Theft
        Auto 6 arrives...")."""
        if " " in value:
            return False
        line_start = text.rfind("\n", 0, start) + 1
        line_prefix = text[line_start:start]
        # Strip whitespace and common list-marker characters (bullets, dashes,
        # numbering, punctuation) from the start of the line up to this word.
        # If nothing meaningful is left, this word is the first real content
        # on its line/list item, not a mid-sentence proper noun.
        if not line_prefix.strip(" \t-*\u2022\u2023\u25e6\u25aa\u00b7()0123456789.:"):
            return True
        prefix = text[:start].rstrip()
        return not prefix or prefix[-1] in ".!?\"'\u201d)"

    def validate(self, original_text: str, candidate_text: str) -> ValidationResult:
        """Ensure source protected spans survive and candidates add none of their own."""
        if candidate_text == original_text:
            return ValidationResult(passed=True, diagnostics=())

        original_spans = self.extract(original_text)
        candidate_spans = self.extract(candidate_text)
        candidate_values = {span.value for span in candidate_spans}
        original_values = {span.value for span in original_spans}
        missing = [span for span in original_spans if span.value not in candidate_values]
        invented = [span for span in candidate_spans if span.value not in original_values]
        diagnostics = [
            f"Missing protected {span.kind}: {span.value}" for span in missing
        ] + [f"Unexpected protected {span.kind}: {span.value}" for span in invented]
        return ValidationResult(passed=not diagnostics, diagnostics=tuple(diagnostics))

    def _find(
        self, kind: ProtectedSpanKind, pattern: re.Pattern[str], text: str
    ) -> list[ProtectedSpan]:
        return [ProtectedSpan(kind=kind, value=match.group(0)) for match in pattern.finditer(text)]

    def _unique(self, spans: list[ProtectedSpan]) -> tuple[ProtectedSpan, ...]:
        unique: list[ProtectedSpan] = []
        seen: set[tuple[str, str]] = set()
        for span in spans:
            key = (span.kind, span.value)
            if key not in seen:
                seen.add(key)
                unique.append(span)
        return tuple(unique)


class ScriptEnhancer:
    """Run optional enhancement behind factual safety and policy boundaries."""

    def __init__(
        self,
        analyzer: ScriptAnalyzer | None = None,
        policy_manager: EnhancementPolicyManager | None = None,
        backend: EnhancementBackend | None = None,
        validator: ProtectedSpanValidator | None = None,
    ) -> None:
        self.analyzer = analyzer or ScriptAnalyzer()
        self.policy_manager = policy_manager or EnhancementPolicyManager()
        self.backend = backend or UnavailableEnhancementBackend()
        self.validator = validator or ProtectedSpanValidator()

    def enhance(self, text: str, profile: str = "default") -> EnhancementResult:
        """Analyze then safely request a candidate enhancement from the configured backend."""
        analysis = self.analyzer.analyze(text)
        policy = self.policy_manager.load(profile)
        candidate = self.backend.enhance(text, analysis=analysis, policy=policy)
        validation = self.validator.validate(text, candidate.text)
        if not candidate.available:
            return EnhancementResult(
                original_text=text,
                enhanced_text=text,
                analysis=analysis,
                policy=policy,
                changes=(),
                backend_name=candidate.backend_name,
                backend_available=False,
                diagnostic=candidate.diagnostic,
                validation=validation,
            )
        if not validation.passed:
            return EnhancementResult(
                original_text=text,
                enhanced_text=text,
                analysis=analysis,
                policy=policy,
                changes=(),
                backend_name=candidate.backend_name,
                backend_available=True,
                diagnostic=(
                    "Enhancement was rejected by protected-span validation: "
                    + "; ".join(validation.diagnostics)
                ),
                validation=validation,
            )
        changes = self._build_change_records(text, candidate.text)
        if not self._within_revision_limit(changes, analysis, policy):
            return EnhancementResult(
                original_text=text,
                enhanced_text=text,
                analysis=analysis,
                policy=policy,
                changes=(),
                backend_name=candidate.backend_name,
                backend_available=True,
                diagnostic=(
                    "Enhancement was rejected because it changed more sentences than "
                    "the selected policy permits."
                ),
                validation=validation,
            )
        if not changes:
            return EnhancementResult(
                original_text=text,
                enhanced_text=candidate.text,
                analysis=analysis,
                policy=policy,
                changes=changes,
                backend_name=candidate.backend_name,
                backend_available=True,
                diagnostic=(
                    f"{candidate.diagnostic} Note: the accepted candidate is identical "
                    "to the original script at the sentence level — the model did not "
                    "actually apply a rewrite this time, even though validation passed "
                    "(nothing changed, so there was nothing to reject)."
                ),
                validation=validation,
            )
        return EnhancementResult(
            original_text=text,
            enhanced_text=candidate.text,
            analysis=analysis,
            policy=policy,
            changes=changes,
            backend_name=candidate.backend_name,
            backend_available=True,
            diagnostic=candidate.diagnostic,
            validation=validation,
        )

    def _build_change_records(
        self, original_text: str, candidate_text: str
    ) -> tuple[ChangeRecord, ...]:
        """Create an auditable record for every changed source sentence."""
        original_sentences = self.analyzer._split_sentences(original_text)
        candidate_sentences = self.analyzer._split_sentences(candidate_text)
        changes: list[ChangeRecord] = []
        matcher = SequenceMatcher(a=original_sentences, b=candidate_sentences)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            old_text = " ".join(original_sentences[old_start:old_end])
            new_text = " ".join(candidate_sentences[new_start:new_end])
            changes.append(
                ChangeRecord(
                    kind=tag,
                    sentence_index=old_start if old_start < len(original_sentences) else None,
                    before=old_text,
                    after=new_text,
                    rationale="Targeted backend revision accepted after protected-span validation.",
                )
            )
        return tuple(changes)

    def _within_revision_limit(
        self,
        changes: tuple[ChangeRecord, ...],
        analysis: ScriptAnalysis,
        policy: EnhancementPolicy,
    ) -> bool:
        if not changes:
            return True
        changed_sentence_count = sum(
            max(1, len(self.analyzer._split_sentences(change.before)))
            for change in changes
        )
        limit = max(1, ceil(analysis.sentence_count * policy.max_changed_sentences_ratio))
        return changed_sentence_count <= limit


class ScriptIntelligence:
    """Mode dispatcher that composes enhancement with the unchanged optimizer."""

    def __init__(
        self,
        enhancer: ScriptEnhancer | None = None,
        optimizer: ScriptOptimizer | None = None,
    ) -> None:
        self.enhancer = enhancer or ScriptEnhancer()
        self.optimizer = optimizer or ScriptOptimizer()

    def process(
        self,
        text: str,
        mode: EnhancementMode = EnhancementMode.OPTIMIZE_ONLY,
        profile: str = "default",
    ) -> ScriptIntelligenceResult:
        """Process text for the selected mode without changing optimizer behavior."""
        if mode is EnhancementMode.OPTIMIZE_ONLY:
            return ScriptIntelligenceResult(
                mode=mode,
                original_text=text,
                enhancement=None,
                output_text=self.optimizer.optimize(text, profile=profile),
            )

        enhancement = self.enhancer.enhance(text, profile=profile)
        if mode is EnhancementMode.ENHANCE_ONLY:
            return ScriptIntelligenceResult(
                mode=mode,
                original_text=text,
                enhancement=enhancement,
                output_text=enhancement.enhanced_text,
            )
        if mode is EnhancementMode.ENHANCE_AND_OPTIMIZE:
            return ScriptIntelligenceResult(
                mode=mode,
                original_text=text,
                enhancement=enhancement,
                output_text=self.optimizer.optimize(
                    enhancement.enhanced_text, profile=profile
                ),
            )
        raise ValueError(f"Unsupported enhancement mode: {mode}")
