"""Configuration classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration."""

    max_order: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    """Controls sentence generation behavior."""

    max_length: int = 200
    min_length: int = 3
    randomize_probability_base: float = 0.03
    randomize_probability_increment: float = 0.03
    short_sentence_retry_probability: float = 0.5
