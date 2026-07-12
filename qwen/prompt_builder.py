"""Build Qwen prompts from HooperTTS narration plans."""

from __future__ import annotations

from dataclasses import dataclass

from core.planner import SentencePlan
from core.profile import NarrationProfile


@dataclass(frozen=True)
class QwenPrompt:
    """Prompt payload for Qwen3-TTS generation."""

    optimized_text: str
    style_prompt: str
    speaker_prompt: str


def build_prompt(
    narration_plan: list[SentencePlan], profile: NarrationProfile
) -> QwenPrompt:
    """Return optimized narration text plus Qwen style and speaker prompts."""
    optimized_text = "\n\n".join(render_plan(plan) for plan in narration_plan).strip()
    return QwenPrompt(
        optimized_text=optimized_text,
        style_prompt=build_style_prompt(profile, narration_plan),
        speaker_prompt=build_speaker_prompt(profile),
    )


def render_plan(plan: SentencePlan) -> str:
    """Return narration text for one sentence plan."""
    return "\n".join(plan.chunks).strip()


def build_style_prompt(
    profile: NarrationProfile, narration_plan: list[SentencePlan]
) -> str:
    """Return a compact style prompt for Qwen instructions."""
    average_energy = 5.0
    if narration_plan:
        average_energy = sum(plan.estimated_energy for plan in narration_plan) / len(
            narration_plan
        )
    return (
        f"Read in a {profile.name.replace('_', ' ')} narration style. "
        f"Hook style: {profile.hook_style}. "
        f"Reveal style: {profile.reveal_style}. "
        f"Question style: {profile.question_style}. "
        f"Average energy: {average_energy:.1f}/10. "
        "Keep pacing natural and preserve the script's dramatic pauses."
    )


def build_speaker_prompt(profile: NarrationProfile) -> str:
    """Return a speaker prompt derived from the narration profile."""
    if profile.name == "gaming_news":
        return "Energetic gaming news narrator with clear, fast delivery."
    if profile.name == "youtube_shorts":
        return "Punchy short-form narrator with high urgency and crisp diction."
    if profile.name == "podcast":
        return "Warm conversational host with relaxed pacing."
    if profile.name == "documentary":
        return "Cinematic documentary narrator with measured authority."
    return "Clear narrator with balanced energy and natural pacing."
