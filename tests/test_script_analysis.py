from core.script_analysis import ScriptAnalyzer


def test_analyzer_returns_structured_scores_without_tts() -> None:
    analysis = ScriptAnalyzer().analyze(
        "This new co-op horror game lets four players control the same monster. "
        "What happens when the hunters become the threat?"
    )

    assert analysis.word_count == 20
    assert analysis.sentence_count == 2
    assert 0 <= analysis.hook_score <= 10
    assert 0 <= analysis.short_form_score <= 10
    assert isinstance(analysis.scores, dict)


def test_analyzer_flags_long_opening_and_duplicate_sentence() -> None:
    long_opening = (
        "This is a very long opening sentence that spends too much time explaining "
        "background context before it tells the audience what the game actually does "
        "or why the detail matters to them."
    )
    analysis = ScriptAnalyzer().analyze(f"{long_opening} {long_opening}")

    categories = {issue.category for issue in analysis.issues}

    assert "hook" in categories
    assert "sentence_length" in categories
    assert "repetition" in categories


def test_analyzer_handles_empty_text() -> None:
    analysis = ScriptAnalyzer().analyze("")

    assert analysis.word_count == 0
    assert analysis.sentence_count == 0
    assert analysis.scores == {
        "hook": 0,
        "clarity": 0,
        "pacing": 0,
        "repetition": 0,
        "short_form": 0,
        "information_density": 0,
        "ending": 0,
    }


def test_analyzer_reports_missing_transitions_and_late_reveal() -> None:
    analysis = ScriptAnalyzer().analyze(
        "The game has four classes. Players explore a flooded city. "
        "The developer officially revealed co-op at the end."
    )

    categories = {issue.category for issue in analysis.issues}

    assert "transition" in categories
    assert "reveal" in categories


if __name__ == "__main__":
    test_analyzer_returns_structured_scores_without_tts()
    test_analyzer_flags_long_opening_and_duplicate_sentence()
    test_analyzer_handles_empty_text()
    test_analyzer_reports_missing_transitions_and_late_reveal()
