from core.enhancement_backends import UnavailableEnhancementBackend


def test_unavailable_backend_preserves_source_with_diagnostic() -> None:
    source = "Grand Theft Auto 6 launches on August 27."

    result = UnavailableEnhancementBackend().enhance(
        source, analysis=object(), policy=object()
    )

    assert result.text == source
    assert not result.available
    assert "unavailable" in result.diagnostic.lower()


if __name__ == "__main__":
    test_unavailable_backend_preserves_source_with_diagnostic()
