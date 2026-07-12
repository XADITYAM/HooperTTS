"""Narration profile loading for HooperTTS."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SentenceType = Literal[
    "HOOK",
    "REVEAL",
    "QUESTION",
    "CTA",
    "EVIDENCE",
    "CONTRAST",
    "NORMAL",
]


@dataclass(frozen=True)
class NarrationProfile:
    """Configuration values that shape planning and rhythm rendering."""

    name: str
    pause_strength: float
    hook_style: str
    reveal_style: str
    ending_style: str
    chunk_target: int
    energy_curve: dict[SentenceType, int]
    question_style: str


class ProfileManager:
    """Load narration profiles from the project profiles directory."""

    DEFAULT_PROFILE = "default"
    PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

    def __init__(self, profiles_dir: Path | None = None) -> None:
        """Create a manager for loading profile JSON files."""
        self.profiles_dir = profiles_dir or self.PROFILES_DIR

    def load(self, profile: str = DEFAULT_PROFILE) -> NarrationProfile:
        """Return a narration profile, falling back to default if missing."""
        profile_name = profile or self.DEFAULT_PROFILE
        profile_path = self._profile_path(profile_name)
        if not profile_path.exists():
            profile_name = self.DEFAULT_PROFILE
            profile_path = self._profile_path(profile_name)

        with profile_path.open("r", encoding="utf-8") as profile_file:
            raw_profile = json.load(profile_file)

        return NarrationProfile(
            name=profile_name,
            pause_strength=float(raw_profile["pause_strength"]),
            hook_style=str(raw_profile["hook_style"]),
            reveal_style=str(raw_profile["reveal_style"]),
            ending_style=str(raw_profile["ending_style"]),
            chunk_target=int(raw_profile["chunk_target"]),
            energy_curve=self._load_energy_curve(raw_profile["energy_curve"]),
            question_style=str(raw_profile["question_style"]),
        )

    def list_profiles(self) -> list[str]:
        """Return available profile names."""
        profile_dir = self._profiles_dir()
        return sorted(path.stem for path in profile_dir.glob("*.json"))

    def _profile_path(self, profile_name: str) -> Path:
        return self._profiles_dir() / f"{profile_name}.json"

    def _profiles_dir(self) -> Path:
        candidates = (
            self.profiles_dir,
            Path.cwd() / "profiles",
            Path(sys.prefix) / "profiles",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.profiles_dir

    def _load_energy_curve(self, raw_curve: dict[str, int]) -> dict[SentenceType, int]:
        return {
            "HOOK": int(raw_curve.get("HOOK", 8)),
            "REVEAL": int(raw_curve.get("REVEAL", 8)),
            "QUESTION": int(raw_curve.get("QUESTION", 7)),
            "CTA": int(raw_curve.get("CTA", 7)),
            "EVIDENCE": int(raw_curve.get("EVIDENCE", 5)),
            "CONTRAST": int(raw_curve.get("CONTRAST", 6)),
            "NORMAL": int(raw_curve.get("NORMAL", 4)),
        }
