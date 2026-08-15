from core.enhancement_policy import EnhancementPolicyManager


def test_friendslop_policy_uses_high_level_characteristics() -> None:
    policy = EnhancementPolicyManager().load("friendslop_gaming")

    assert policy.name == "friendslop_gaming"
    assert any("curiosity-driven" in goal for goal in policy.writing_goals)
    assert any("creator-specific" in item for item in policy.avoid)
    assert policy.allow_unchanged_sentences


def test_friendslop_policy_permits_a_real_hook_driven_rewrite() -> None:
    """Regression test: the original policy capped changes at 40% of
    sentences and told the model to leave strong sentences unchanged, so a
    real hook/surprise-driven rewrite (touching most of the script) was
    rejected wholesale by the revision-limit check. friendslop_gaming should
    now allow that, while still forbidding invented facts."""
    policy = EnhancementPolicyManager().load("friendslop_gaming")

    assert any("hook" in goal for goal in policy.writing_goals)
    assert policy.max_changed_sentences_ratio >= 0.9
    assert any("invented" in item for item in policy.avoid)


def test_unknown_profile_falls_back_to_default_policy() -> None:
    policy = EnhancementPolicyManager().load("unknown_profile")

    assert policy.name == "default"
    assert policy.max_changed_sentences_ratio == 0.4


if __name__ == "__main__":
    test_friendslop_policy_uses_high_level_characteristics()
    test_friendslop_policy_permits_a_real_hook_driven_rewrite()
    test_unknown_profile_falls_back_to_default_policy()
