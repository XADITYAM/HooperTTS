from core.optimizer import ScriptOptimizer
from core.planner import NarrationPlanner
from core.profile import ProfileManager


def test_profile_manager_loads_default() -> None:
    profile = ProfileManager().load("default")

    assert profile.name == "default"
    assert profile.pause_strength > 0
    assert profile.chunk_target == 7
    assert profile.energy_curve["HOOK"] >= profile.energy_curve["NORMAL"]


def test_profile_manager_loads_all_bundled_profiles() -> None:
    manager = ProfileManager()

    for profile_name in (
        "default",
        "documentary",
        "gaming_news",
        "youtube_shorts",
        "podcast",
    ):
        profile = manager.load(profile_name)
        assert profile.name == profile_name
        assert profile.pause_strength > 0
        assert 2 <= profile.chunk_target <= 7
        assert set(profile.energy_curve) == {
            "HOOK",
            "REVEAL",
            "QUESTION",
            "CTA",
            "EVIDENCE",
            "CONTRAST",
            "NORMAL",
        }


def test_optimizer_accepts_profile_keyword() -> None:
    output = ScriptOptimizer().optimize(
        "Imagine opening HooperTTS. Officially confirmed.",
        profile="gaming_news",
    )

    assert "Imagine" in output
    assert "OFFICIALLY" in output


def test_planner_uses_profile_chunk_target() -> None:
    profile = ProfileManager().load("youtube_shorts")
    plans = NarrationPlanner(profile).plan(
        "Imagine a long gaming update that suddenly changes everything for players."
    )

    assert plans
    assert all(
        len(chunk.split()) <= profile.chunk_target
        for plan in plans
        for chunk in plan.chunks
    )


if __name__ == "__main__":
    test_profile_manager_loads_default()
    test_profile_manager_loads_all_bundled_profiles()
    test_optimizer_accepts_profile_keyword()
    test_planner_uses_profile_chunk_target()
