from core.enhancement_policy import EnhancementPolicyManager


def test_friendslop_policy_uses_high_level_characteristics() -> None:
    policy = EnhancementPolicyManager().load("friendslop_gaming")

    assert policy.name == "friendslop_gaming"
    assert any("curiosity-driven" in goal for goal in policy.writing_goals)
    assert any("creator-specific" in item for item in policy.avoid)
    assert policy.allow_unchanged_sentences


def test_unknown_profile_falls_back_to_default_policy() -> None:
    policy = EnhancementPolicyManager().load("unknown_profile")

    assert policy.name == "default"
    assert policy.max_changed_sentences_ratio == 0.4


if __name__ == "__main__":
    test_friendslop_policy_uses_high_level_characteristics()
    test_unknown_profile_falls_back_to_default_policy()
