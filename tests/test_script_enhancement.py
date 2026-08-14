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
    test_enhancement_diagnostic_explains_why_validation_rejected_a_candidate()
    test_validation_rejects_unsafe_candidate_and_returns_original()
    test_enhance_only_never_invokes_qwen()
    test_optimize_only_matches_existing_optimizer()
    test_enhance_and_optimize_preserves_source_when_backend_unavailable()
