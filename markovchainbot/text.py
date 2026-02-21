"""Text utils."""

import re


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    previous_row = list(range(len(s1) + 1))
    for idx_s2, char_s2 in enumerate(s2):
        current_row = [idx_s2 + 1]
        for idx_s1, char_s1 in enumerate(s1):
            if char_s1 == char_s2:
                current_row.append(previous_row[idx_s1])
            else:
                current_row.append(
                    1
                    + min(
                        previous_row[idx_s1],
                        previous_row[idx_s1 + 1],
                        current_row[-1],
                    )
                )
        previous_row = current_row
    return previous_row[-1]


def process_message(message: str) -> list[str]:
    """Process a message into a word list with start/end markers."""
    message = re.sub(r"[^\w\d\s\?\!\"\':<>]", r"", message)
    message = re.sub(r"(\?|\!|;|~~)", r" \1", message)
    words = re.split(r"\s+|,|\.|;", message)
    words = ["<start>"] + words + ["<end>"]
    words = [
        word
        for word in words
        if word is not None and not word.startswith("http") and len(word) > 0
    ]
    return [
        word if (word.upper() == word or "<" in word) else word.lower()
        for word in words
    ]
