from core.enhancement_backends import BackendEnhancement
from core.script_enhancement import (
    EnhancementMode,
    ProtectedSpanValidator,
    ScriptEnhancer,
    ScriptIntelligence,
)


class UnsafeBackend:
    def enhance(self, text: str, *, analysis: object, policy: object) -> BackendEnhancement:
        return BackendEnhancement(
            text="A new game launches next year for $70.",
            backend_name="unsafe-test",
            available=True,
            diagnostic="Candidate returned.",
        )


def test_protected_span_validator_preserves_factual_details() -> None:
    source = (
        'Rockstar Games said "Grand Theft Auto 6" arrives on August 27, 2027 '
        "for PS5 at $69.99: https://example.com/gta6"
    )
    validator = ProtectedSpanValidator()

    assert validator.validate(source, source).passed

    invalid = validator.validate(source, "Grand Theft Auto 6 arrives next year.")
    assert not invalid.passed
    assert any("Rockstar Games" in diagnostic for diagnostic in invalid.diagnostics)


def test_protected_span_validator_does_not_flag_ordinary_sentence_openers() -> None:
    """Regression test: a single capitalized word is often just capitalized
    because it opens a sentence (e.g. "Imagine", "Officially"), not because it
    is a genuine name. Rejecting every such reword made enhancement a no-op in
    practice, since almost any rewrite changes at least one sentence-opening
    word. Real names/titles must still be protected even when they happen to
    open a sentence, as long as they are multi-word (e.g. "Grand Theft Auto")."""
    validator = ProtectedSpanValidator()

    original = "Imagine opening HooperTTS. Officially confirmed, it works great."
    reworded = "Picture opening HooperTTS \u2014 officially confirmed to work great."
    assert validator.validate(original, reworded).passed

    original_with_title = "Grand Theft Auto 6 arrives this summer. Players are thrilled."
    dropped_title = "A new game arrives this summer. Players are thrilled."
    assert not validator.validate(original_with_title, dropped_title).passed


def test_protected_span_validator_does_not_fuse_across_bullet_list_lines() -> None:
    """Regression test: a multi-word capitalized-phrase match must not cross a
    line break. Bullet-list items often end without punctuation, so the last
    word of one item and the first word of the next (e.g. "...seen in
    Fortnite" followed on a new line by "The game...") were getting fused
    into a single fake protected phrase like "Fortnite\\n    The" that no
    reformatting could ever match exactly. Real cross-line reformatting
    (e.g. normalizing indentation to dash bullets) must validate cleanly."""
    validator = ProtectedSpanValidator()
    original = (
        "Future content could go beyond what we've seen in Fortnite\n"
        "    The game reportedly has major PS5 features."
    )
    reformatted = (
        "Future content could go beyond what we've seen in Fortnite\n"
        "- The game reportedly has major PS5 features."
    )

    spans = validator.extract(original)
    assert not any("\n" in span.value for span in spans)
    assert validator.validate(original, reformatted).passed


def test_protected_span_validator_does_not_pair_unrelated_apostrophes() -> None:
    """Regression test: found via a real Phi-3.5-mini generation. Possessive
    and contraction apostrophes ("GTA 6's ... the game's ... isn't") were
    being greedily paired by the bare-single-quote pattern into one fake
    multi-sentence "quotation" spanning everything between two unrelated
    apostrophes, which then could never match on either side."""
    validator = ProtectedSpanValidator()
    text = (
        "GTA 6's innovative gameplay emerge from the insights of YouTuber "
        "HipHopGamer, ahead of the Extended Look Netflix stream on August 27. "
        "Unveiling a series of groundbreaking features, HipHopGamer notes the "
        "game's technology isn't just for movies."
    )
    spans = [s for s in validator.extract(text) if s.kind == "quotation"]
    assert spans == []


def test_protected_span_validator_matches_quotations_across_quote_styles() -> None:
    """Regression test: the source script uses curly ("smart") quotes, but a
    model's own output commonly defaults to straight ASCII quotes even when
    the quoted content itself is preserved exactly. That typography
    difference alone must not cause a false rejection."""
    validator = ProtectedSpanValidator()
    original = "Gameplay supposedly feels \u201ccompletely different\u201d from previous games."
    candidate = 'The developer said gameplay feels "completely different" this time around.'
    assert validator.validate(original, candidate).passed

    # A genuinely dropped/altered quote must still be rejected.
    dropped = validator.validate(
        original, "The developer said gameplay feels totally unique this time around."
    )
    assert not dropped.passed



def test_protected_span_validator_allows_natural_entity_rewording_and_date_abbreviations() -> None:
    """LLM rewrites may expand dates and shorten entity references without
    changing the underlying facts. Those changes must not be rejected as
    invented/missing proper nouns."""
    validator = ProtectedSpanValidator()
    source = (
        "Netflix is set to reveal Red Dead Redemption 2 starring Jason's "
        "from Rockstar Games on Aug. 27."
    )
    candidate = (
        "On August 27, Netflix reveals Red Dead Redemption 2, starring Jason "
        "with Rockstar's new details."
    )
    result = validator.validate(source, candidate)
    assert result.passed, result.diagnostics


def test_protected_span_validator_rejects_dropped_numeric_title_fact() -> None:
    validator = ProtectedSpanValidator()
    source = "Red Dead Redemption 2 is finally getting a new trailer."
    candidate = "Red Dead Redemption is finally getting a new trailer."
    result = validator.validate(source, candidate)
    assert not result.passed
    assert any("number: 2" in diagnostic for diagnostic in result.diagnostics)


def test_protected_span_validator_rejects_new_hard_fact() -> None:
    validator = ProtectedSpanValidator()
    source = "Grand Theft Auto 6 arrives on August 27."
    candidate = "Grand Theft Auto 6 arrives on August 28."
    result = validator.validate(source, candidate)
    assert not result.passed
    assert any("August 28" in diagnostic for diagnostic in result.diagnostics)

def test_enhancement_diagnostic_explains_why_validation_rejected_a_candidate() -> None:
    """Regression test: a bare 'rejected by protected-span validation' message
    gave no way to tell what actually broke. The diagnostic must include the
    specific missing/invented spans."""

    class DroppingBackend:
        def enhance(self, text, *, analysis, policy):
            return BackendEnhancement(
                text="A new game arrives next year for $70.",
                backend_name="dropping-test",
                available=True,
                diagnostic="Candidate returned.",
            )

    source = "Grand Theft Auto 6 arrives on August 27 for PS5 at $69.99."
    result = ScriptEnhancer(backend=DroppingBackend()).enhance(source)

    assert not result.validation.passed
    assert "Grand Theft Auto 6" in result.diagnostic
    assert result.enhanced_text == source


def test_diagnostic_flags_when_accepted_candidate_is_identical_to_original() -> None:
    """Regression test: a candidate that trivially passes validation by being
    byte-identical to the input looked, from the diagnostic alone, exactly
    like a real accepted rewrite ("Generated a candidate with..."). This
    required manual text diffing to catch. The diagnostic must say so."""

    class EchoBackend:
        def enhance(self, text, *, analysis, policy):
            return BackendEnhancement(
                text=text,  # echoes the input back verbatim
                backend_name="echo-test",
                available=True,
                diagnostic="Generated a candidate with echo-test in 1.00s on cpu.",
            )

    source = "Grand Theft Auto 6 arrives on August 27 for PS5 at $69.99."
    result = ScriptEnhancer(backend=EchoBackend()).enhance(source, profile="friendslop_gaming")

    assert result.validation.passed
    assert result.changes == ()
    assert "identical to the original" in result.diagnostic


def test_validation_rejects_unsafe_candidate_and_returns_original() -> None:
    source = "Grand Theft Auto 6 arrives on August 27 for PS5."
    result = ScriptEnhancer(backend=UnsafeBackend()).enhance(
        source, profile="friendslop_gaming"
    )

    assert result.enhanced_text == source
    assert not result.validation.passed
    assert "rejected" in result.diagnostic.lower()


def test_enhance_only_never_invokes_qwen() -> None:
    import qwen.runner as qwen_runner

    intelligence = ScriptIntelligence()
    source = "Grand Theft Auto 6 arrives on August 27."
    original_generate = qwen_runner.generate

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("Enhance Only must not invoke Qwen generation.")

    try:
        qwen_runner.generate = fail_if_called
        result = intelligence.process(
            source,
            mode=EnhancementMode.ENHANCE_ONLY,
            profile="friendslop_gaming",
        )
    finally:
        qwen_runner.generate = original_generate

    assert result.output_text == source
    assert result.enhancement is not None
    assert not result.enhancement.backend_available
    assert result.enhancement.backend_name == "unavailable"


def test_optimize_only_matches_existing_optimizer() -> None:
    source = "Imagine opening GTA 6. Rockstar officially revealed new details."
    intelligence = ScriptIntelligence()

    result = intelligence.process(
        source, mode=EnhancementMode.OPTIMIZE_ONLY, profile="youtube_shorts"
    )

    assert result.enhancement is None
    assert result.output_text == intelligence.optimizer.optimize(
        source, profile="youtube_shorts"
    )


def test_enhance_and_optimize_preserves_source_when_backend_unavailable() -> None:
    source = "Imagine opening GTA 6. Rockstar officially revealed new details."
    intelligence = ScriptIntelligence()

    result = intelligence.process(
        source, mode=EnhancementMode.ENHANCE_AND_OPTIMIZE, profile="friendslop_gaming"
    )

    assert result.enhancement is not None
    assert result.enhancement.enhanced_text == source
    assert result.output_text == intelligence.optimizer.optimize(
        source, profile="friendslop_gaming"
    )


if __name__ == "__main__":
    test_protected_span_validator_preserves_factual_details()
    test_protected_span_validator_does_not_flag_ordinary_sentence_openers()
    test_protected_span_validator_does_not_fuse_across_bullet_list_lines()
    test_protected_span_validator_does_not_pair_unrelated_apostrophes()
    test_protected_span_validator_matches_quotations_across_quote_styles()
    test_protected_span_validator_allows_natural_entity_rewording_and_date_abbreviations()
    test_protected_span_validator_rejects_dropped_numeric_title_fact()
    test_protected_span_validator_rejects_new_hard_fact()
    test_enhancement_diagnostic_explains_why_validation_rejected_a_candidate()
    test_diagnostic_flags_when_accepted_candidate_is_identical_to_original()
    test_validation_rejects_unsafe_candidate_and_returns_original()
    test_enhance_only_never_invokes_qwen()
    test_optimize_only_matches_existing_optimizer()
    test_enhance_and_optimize_preserves_source_when_backend_unavailable()
