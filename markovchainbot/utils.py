"""Processors and readers."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class MessageReader(ABC):
    """Abstract message reader."""

    @abstractmethod
    def get_messages(self, filepath: Path):
        """Get messages from a file."""
        pass


@dataclass
class DiscordMessageReader(MessageReader):
    """Message reader, discord format."""

    def get_messages(self, filepath: Path):
        with open(filepath, "r") as file:
            messages: dict[str, dict[str, int | str]] = json.load(file)

        return [c["content"] for c in messages.values()]


@dataclass
class MessageProcessor:
    """Processes a string message."""

    def process(self, message: str):
        """Process message."""
        message = re.sub(r"[^\w\d\s\?\!\"\':<>]", r"", message)
        message = re.sub(r"(\?|\!|;|~~)", r" \1", message)
        words = re.split(r"\s+|,|\.|;", message)
        words = ["<start>"] + words + ["<end>"]
        words = [
            c
            for c in words
            if c is not None and not c.startswith("http") and len(c) > 0
        ]

        words = [
            word if (word.upper() == word or "<" in word) else word.lower()
            for word in words
        ]
        return words
