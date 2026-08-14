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
    test_validation_rejects_unsafe_candidate_and_returns_original()
    test_enhance_only_never_invokes_qwen()
    test_optimize_only_matches_existing_optimizer()
    test_enhance_and_optimize_preserves_source_when_backend_unavailable()
