"""
HooperTTS Optimizer
Version 0.1

This module converts a normal script into a narration-friendly script
by improving rhythm and readability for TTS models.
"""

import re


class ScriptOptimizer:

    def __init__(self):
        pass

    def optimize(self, text: str) -> str:
        """
        Main optimization function.
        """

        text = self.normalize_whitespace(text)
        text = self.insert_pauses(text)

        return text.strip()

    def normalize_whitespace(self, text: str) -> str:
        """
        Remove extra spaces and normalize line breaks.
        """

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def insert_pauses(self, text: str) -> str:
        """
        Insert dramatic pauses after sentence endings.
        """

        text = re.sub(r"\.\s+", ".\n\n...\n\n", text)

        text = re.sub(r"\?\s+", "?\n\n...\n\n", text)

        text = re.sub(r"!\s+", "!\n\n...\n\n", text)

        return text
