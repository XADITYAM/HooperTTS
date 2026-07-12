from core.chunker import SemanticChunker


def test_chunker_protects_phrases() -> None:
    chunker = SemanticChunker()
    chunks = chunker.chunk(
        "Grand Theft Auto 6 brings Lucia and Jason back to Vice City, "
        "but fans noticed the detail."
    )

    assert any("Grand Theft Auto 6" in chunk for chunk in chunks)
    assert any("Lucia and Jason" in chunk for chunk in chunks)
    assert any("Vice City" in chunk for chunk in chunks)


def test_chunker_respects_word_limits() -> None:
    chunker = SemanticChunker()
    chunks = chunker.chunk(
        "The trailer looks calm, but the reveal suddenly changes everything "
        "for longtime fans."
    )

    assert len(chunks) > 1
    assert all(2 <= len(chunk.split()) <= 7 for chunk in chunks)


def test_chunker_allows_dramatic_word() -> None:
    assert SemanticChunker().chunk("Breaking!") == ["Breaking!"]


if __name__ == "__main__":
    test_chunker_protects_phrases()
    test_chunker_respects_word_limits()
    test_chunker_allows_dramatic_word()
