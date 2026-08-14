"""Profile-aware policy loading for optional script enhancement."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnhancementPolicy:
    """Writing constraints and goals for one enhancement profile."""

    name: str
    writing_goals: tuple[str, ...]
    avoid: tuple[str, ...]
    max_changed_sentences_ratio: float
    allow_unchanged_sentences: bool


class EnhancementPolicyManager:
    """Load profile-specific writing policies with a safe default fallback."""

    DEFAULT_POLICY = "default"
    POLICIES_DIR = Path(__file__).resolve().parent.parent / "enhancement_policies"

    def __init__(self, policies_dir: Path | None = None) -> None:
        self.policies_dir = policies_dir or self.POLICIES_DIR

    def load(self, profile: str = DEFAULT_POLICY) -> EnhancementPolicy:
        """Return a policy for a profile, falling back to the default policy."""
        policy_name = profile or self.DEFAULT_POLICY
        policy_path = self._policy_path(policy_name)
        if not policy_path.exists():
            policy_name = self.DEFAULT_POLICY
            policy_path = self._policy_path(policy_name)

        with policy_path.open(encoding="utf-8") as policy_file:
            raw_policy = json.load(policy_file)

        limits = raw_policy.get("revision_limits", {})
        return EnhancementPolicy(
            name=policy_name,
            writing_goals=tuple(str(item) for item in raw_policy.get("writing_goals", [])),
            avoid=tuple(str(item) for item in raw_policy.get("avoid", [])),
            max_changed_sentences_ratio=float(
                limits.get("max_changed_sentences_ratio", 0.4)
            ),
            allow_unchanged_sentences=bool(
                limits.get("allow_unchanged_sentences", True)
            ),
        )

    def list_policies(self) -> list[str]:
        """Return available policy names."""
        return sorted(path.stem for path in self._policies_dir().glob("*.json"))

    def _policy_path(self, policy_name: str) -> Path:
        return self._policies_dir() / f"{policy_name}.json"

    def _policies_dir(self) -> Path:
        candidates = (
            self.policies_dir,
            Path.cwd() / "enhancement_policies",
            Path(sys.prefix) / "enhancement_policies",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.policies_dir
