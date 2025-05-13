"""Processors and readers."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class MessageReader(ABC):
    """Abstract message reader."""

    @abstractmethod
    def get_messages(self, filepath: Path) -> list[str]:
        """Get messages from a file."""
        pass


@dataclass
class DiscordMessageReader(MessageReader):
    """Message reader, discord format."""

    def get_messages(self, filepath: Path) -> list[str]:
        """Get discord messages contained in a json."""
        with open(filepath) as file:
            messages: dict[str, dict[str, int | str]] = json.load(file)

        return [str(c["content"]) for c in messages.values()]


@dataclass
class DiscordPackageMessageReader(MessageReader):
    """Message reader, discord format."""

    def get_messages(self, filepath: Path) -> list[str]:
        """Get discord messages contained in a json."""
        with open(filepath) as file:
            messages: list[dict[str, int | str]] = json.load(file)

        return [str(c["Contents"]) for c in messages]


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


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]
