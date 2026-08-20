from core.planner import NarrationPlanner


def test_abbreviation_period_does_not_split_sentence() -> None:
    """Regression test: found via a real script upload. "Aug." was treated as
    a sentence-ending period by the naive [^.!?]+[.!?]? splitter, creating a
    fake sentence boundary right after it. That fake boundary then made the
    rest of the real sentence ("27, but fans...") look like it opened a new
    sentence, which triggered narration pause heuristics on a boundary that
    was never really there, producing stray "..." pause markers in the
    optimized output."""
    planner = NarrationPlanner()
    text = (
        "Netflix is set to reveal an extended look at Grand Theft Auto 6 on "
        "Aug. 27, but fans who are hankering to get a closer look at "
        "Rockstar Games' Florida simulator don't have to wait until then."
    )
    sentences = planner._split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_other_common_abbreviations_do_not_split_sentences() -> None:
    planner = NarrationPlanner()
    cases = [
        "The event happens on Jan. 5 next year.",
        "Dr. Smith confirmed the release date.",
        "Mr. Rockstar himself would not comment.",
    ]
    for text in cases:
        sentences = planner._split_sentences(text)
        assert len(sentences) == 1, f"unexpectedly split: {text!r} -> {sentences!r}"


def test_genuine_sentence_boundary_after_abbreviation_is_a_known_limitation() -> None:
    """Documents an accepted trade-off, not a bug: a lightweight abbreviation
    list can't distinguish "Aug. 27" (mid-sentence) from "Aug. <New
    sentence>" (genuine boundary) — the same ambiguity exists in every
    simple sentence tokenizer without full NLP disambiguation. We chose to
    always treat these abbreviations as non-terminating, since a merged
    sentence just gets read with a brief pause where the period was (mildly
    suboptimal pacing), whereas a wrongly split sentence produced visibly
    broken output (stray "..." pause markers, fragmented "Aug." / "27,"
    chunks) — the failure mode we're actually protecting against."""
    planner = NarrationPlanner()
    text = "The meeting is scheduled for Aug. It was moved up a week."
    sentences = planner._split_sentences(text)
    assert sentences == [text]


def test_abbreviation_period_is_restored_not_dropped() -> None:
    planner = NarrationPlanner()
    text = "The reveal airs on Aug. 27 this year."
    sentences = planner._split_sentences(text)
    assert sentences == [text]
    assert "Aug." in sentences[0]


if __name__ == "__main__":
    test_abbreviation_period_does_not_split_sentence()
    test_other_common_abbreviations_do_not_split_sentences()
    test_genuine_sentence_boundary_after_abbreviation_is_a_known_limitation()
    test_abbreviation_period_is_restored_not_dropped()
