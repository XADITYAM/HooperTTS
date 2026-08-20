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
    # No word-boundary guard on a bare straight quote ' means two unrelated
    # possessive/contraction apostrophes (e.g. "GTA 6's ... the game's") get
    # greedily paired into one fake multi-sentence "quotation" spanning
    # everything between them. (?<!\w)...(?!\w) requires the opening quote to
    # NOT be stuck onto a preceding word and the closing quote to NOT be
    # stuck onto a following word, which a real quotation mark satisfies but
    # a possessive/contraction apostrophe never does.
    _QUOTATION_PATTERN = re.compile(r'"[^"\n]+"|(?<!\w)\'[^\'\n]+\'(?!\w)')
    _PRICE_PATTERN = re.compile(r"(?:[$€£¥]\s?\d+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?\s?(?:USD|EUR|GBP|INR))\b", re.IGNORECASE)
    _DATE_PATTERN = re.compile(
        r"\b(?:\d{4}-\d{1,2}-\d{1,2}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|"
        r"Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|"
        r"Nov\.?|Dec\.?)\s+\d{1,2}(?:,\s*\d{4})?|"
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan\.?|Feb\.?|Mar\.?|"
        r"Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|"
        r"Nov\.?|Dec\.?)(?:\s+\d{4})?)\b",
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

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """Canonicalize typographic ("smart") quotes and apostrophes to their
        ASCII equivalents before span extraction. The source script may use
        curly punctuation while a model's own output commonly defaults to
        straight punctuation (or vice versa) even when the quoted content
        itself is preserved exactly — without this, that typography
        difference alone causes a false "missing"/"invented" mismatch on an
        otherwise correctly preserved quotation or possessive."""
        return (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

    def extract(self, text: str) -> tuple[ProtectedSpan, ...]:
        """Extract exact source spans that must remain present after enhancement."""
        text = self._normalize_quotes(text)
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
        existing_spans = tuple(spans)
        for match in self._CAPITALIZED_PHRASE_PATTERN.finditer(text):
            value = match.group(0)
            if self._is_ordinary_sentence_opener(text, match.start(), value):
                continue
            if self._is_descriptive_capitalized_span(value, existing_spans):
                continue
            kind: ProtectedSpanKind = "title" if any(char.isdigit() for char in value) else "proper_noun"
            spans.append(ProtectedSpan(kind=kind, value=value))
        return self._unique(spans)

    @classmethod
    def _is_descriptive_capitalized_span(
        cls, value: str, existing_spans: tuple[ProtectedSpan, ...]
    ) -> bool:
        """Avoid treating dates and similar factual phrases as names/titles.

        For example, ``August 27`` and ``On August 27`` can match the broad
        capitalized-phrase regex. The dedicated date extractor is the source of
        truth for those values, so they should not also become protected titles
        or proper nouns. Numeric game titles such as ``Red Dead Redemption 2``
        do not contain a full date span and therefore remain protected.
        """
        normalized_value = cls._normalize_fact_value(value)
        for span in existing_spans:
            if span.kind not in {"date", "price", "url", "quotation"}:
                continue
            normalized_span = (
                cls._normalize_date(span.value)
                if span.kind == "date"
                else cls._normalize_fact_value(span.value)
            )
            if normalized_span in normalized_value:
                return True
            if span.kind == "date" and cls._normalize_date(value) in set(normalized_span.split()):
                return True
        return False

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
        """Ensure hard facts survive while allowing natural entity rewording.

        The old validator compared every extracted capitalized phrase as an exact
        string. That is too strict for LLM rewriting: ``Rockstar Games`` may
        naturally become ``Rockstar's``, ``Jason's`` may become ``Jason``, and
        ``Aug. 27`` may become ``August 27`` without changing the underlying
        fact. Hard facts (numbers, dates, prices, URLs, titles, quotations, and
        platforms) remain strict; proper nouns/organizations are matched with a
        conservative token-overlap rule and possessive normalization.
        """
        if candidate_text == original_text:
            return ValidationResult(passed=True, diagnostics=())

        original_spans = self.extract(original_text)
        candidate_spans = self.extract(candidate_text)

        hard_kinds = {
            "url",
            "quotation",
            "price",
            "date",
            "number",
            "title",
            "platform",
        }
        original_hard = [span for span in original_spans if span.kind in hard_kinds]
        candidate_hard = [span for span in candidate_spans if span.kind in hard_kinds]
        original_soft = [
            span for span in original_spans if span.kind in {"proper_noun", "organization"}
        ]
        candidate_soft = [
            span for span in candidate_spans if span.kind in {"proper_noun", "organization"}
        ]

        missing: list[ProtectedSpan] = []
        for span in original_hard:
            if not any(self._hard_span_matches(span, candidate) for candidate in candidate_hard):
                missing.append(span)

        missing_soft: list[ProtectedSpan] = []
        for span in original_soft:
            if not any(self._soft_span_matches(span, candidate) for candidate in candidate_soft):
                missing_soft.append(span)

        invented: list[ProtectedSpan] = []
        candidate_source_tokens = set(self._entity_tokens(original_text))
        for span in [*candidate_hard, *candidate_soft]:
            if span.kind in {"url", "quotation", "price", "date", "number", "title", "platform"}:
                if not any(self._hard_span_matches(original, span) for original in original_hard):
                    invented.append(span)
                continue
            if not any(self._soft_span_matches(span, original) for original in original_soft):
                # A named entity that was merely moved from sentence-opening
                # position into the middle of a rewritten sentence may not have
                # been extracted from the source as a soft span. If its lexical
                # tokens already occur in the source text, it is not an invented
                # entity and should be allowed.
                if not any(token in candidate_source_tokens for token in self._entity_tokens(span.value)):
                    invented.append(span)

        diagnostics = [
            f"Missing protected {span.kind}: {span.value}" for span in [*missing, *missing_soft]
        ] + [
            f"Unexpected protected {span.kind}: {span.value}" for span in invented
        ]
        return ValidationResult(passed=not diagnostics, diagnostics=tuple(diagnostics))

    @classmethod
    def _hard_span_matches(cls, original: ProtectedSpan, candidate: ProtectedSpan) -> bool:
        if original.kind != candidate.kind and not {original.kind, candidate.kind} <= {"number", "title"}:
            return False
        if original.kind == candidate.kind == "date":
            return cls._normalize_date(original.value) == cls._normalize_date(candidate.value)
        return cls._normalize_fact_value(original.value) == cls._normalize_fact_value(candidate.value)

    @classmethod
    def _soft_span_matches(cls, original: ProtectedSpan, candidate: ProtectedSpan) -> bool:
        source_tokens = cls._entity_tokens(original.value)
        candidate_tokens = cls._entity_tokens(candidate.value)
        if not source_tokens or not candidate_tokens:
            return False
        if source_tokens == candidate_tokens:
            return True
        # Allow grammatical possessives such as ``Jason's`` / ``Jason``.
        if len(candidate_tokens) == 1 and candidate_tokens[0].rstrip("'s") in source_tokens:
            return True
        if len(source_tokens) == 1 and source_tokens[0].rstrip("'s") in candidate_tokens:
            return True
        # Conservative shorthand support: at least half of the source entity's
        # lexical tokens must survive in the candidate, with the first token
        # present. This accepts ``Rockstar Games`` -> ``Rockstar's`` but does not
        # accept unrelated newly invented names. Hard numeric/title spans still
        # catch dropped identifiers such as the ``2`` in ``Red Dead Redemption 2``.
        overlap = len(set(source_tokens) & set(candidate_tokens))
        return source_tokens[0] in candidate_tokens and overlap >= max(1, (len(source_tokens) + 1) // 2)

    @staticmethod
    def _normalize_fact_value(value: str) -> str:
        return re.sub(r"\s+", " ", ProtectedSpanValidator._normalize_quotes(value).strip()).casefold()

    @staticmethod
    def _normalize_date(value: str) -> str:
        text = ProtectedSpanValidator._normalize_fact_value(value).replace(".", "")
        abbreviations = {
            "jan": "january", "feb": "february", "mar": "march",
            "apr": "april", "jun": "june", "jul": "july",
            "aug": "august", "sep": "september", "sept": "september",
            "oct": "october", "nov": "november", "dec": "december",
        }
        words = text.split()
        if words and words[0] in abbreviations:
            words[0] = abbreviations[words[0]]
        elif len(words) >= 2 and words[1] in abbreviations:
            words[1] = abbreviations[words[1]]
        return " ".join(words)

    @staticmethod
    def _entity_tokens(value: str) -> tuple[str, ...]:
        value = ProtectedSpanValidator._normalize_quotes(value).casefold()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value)
        return tuple(token.rstrip("'s") if token.endswith("'s") else token for token in tokens)

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

        # Optional validator-aware retries for backends that support them. Keep the
        # validator strict: retries exist to give a small model another explicit
        # chance to restore facts it accidentally dropped, not to weaken acceptance.
        retry_fn = getattr(self.backend, "enhance_with_feedback", None)
        retry_attempts = 0
        max_retries = max(0, int(getattr(getattr(self.backend, "config", None), "validation_retry_count", 0)))
        while (
            candidate.available
            and not validation.passed
            and callable(retry_fn)
            and retry_attempts < max_retries
        ):
            retry_attempts += 1
            retry_candidate = retry_fn(
                text,
                analysis=analysis,
                policy=policy,
                diagnostics=validation.diagnostics,
                previous_candidate=candidate.text,
            )
            if not retry_candidate.available:
                candidate = retry_candidate
                break
            candidate = retry_candidate
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
                    "Enhancement was rejected by protected-span validation"
                    + (" after a validator-aware retry"
                       if retry_attempts == 1 else
                       f" after {retry_attempts} validator-aware retries"
                       if retry_attempts > 1 else "")
                    + ": "
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
