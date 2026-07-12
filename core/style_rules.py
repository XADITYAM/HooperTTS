"""Style rule configuration for script optimization."""

from typing import Final

STYLE_RULES: Final[dict[str, dict[str, str]]] = {
    "documentary": {
        "imagine": "pause_before",
        "finally": "emphasis",
        "officially": "emphasis",
        "breaking": "emphasis",
        "confirmed": "emphasis",
        "exclusive": "emphasis",
        "however": "contrast",
        "but": "contrast",
        "instead": "contrast",
        "yet": "contrast",
        "although": "contrast",
        "actually": "soft",
        "honestly": "soft",
        "seriously": "soft",
        "unexpectedly": "dramatic",
        "suddenly": "dramatic",
    }
}

EMOTIONAL_WORDS: Final[dict[str, str]] = STYLE_RULES["documentary"]
