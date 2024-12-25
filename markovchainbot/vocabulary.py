"""Vocabulary builder."""

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
class Tokenizer:
    """Tokenizer.

    Tokenizes vocabulary and forms mappings.
    """

    vocabulary_dir_path: Path
    message_reader: MessageReader
    _word_to_token: dict[str, str] = None
    _token_to_word: dict[str, str] = None

    def tokenize(self):
        """Tokenizer."""
        # current_token_id = 0

        for filepath in self.vocabulary_dir_path.iterdir():
            messages = self.message_reader.get_messages(filepath)

            for message in messages:
                print(message)
                message = re.sub(r"(\?|\!|;|~~)", r" \1", message)
                words = re.split(r"\s+|,|\.|;", message)
                words = [c for c in words if c is not None]
                print(words)
                input()
