"""msgpack serialization for MarkovChain."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import msgpack
import msgpack.exceptions

if TYPE_CHECKING:
    from markovchainbot.chain import MarkovChain

from markovchainbot.config import TrainingConfig

_VERSION = 1


def save(chain: MarkovChain, filepath: Path) -> None:
    """Save a trained MarkovChain to msgpack."""
    data = {
        "version": _VERSION,
        "training_config": {
            "max_order": chain.training_config.max_order,
        },
        "word_to_token": chain._word_to_token,
        "chains": {
            order: {
                ",".join(map(str, tuple_key)): transitions
                for tuple_key, transitions in chain_values.items()
            }
            for order, chain_values in chain._chains.items()
        },
    }
    filepath.write_bytes(msgpack.packb(data))


def load(filepath: Path) -> MarkovChain:
    """Load a MarkovChain from msgpack."""
    from markovchainbot.chain import MarkovChain

    try:
        data = msgpack.unpackb(filepath.read_bytes(), strict_map_key=False)
    except (
        ValueError,
        msgpack.exceptions.FormatError,
        msgpack.exceptions.StackError,
    ) as e:
        msg = f"Failed to load model from {filepath}"
        raise ValueError(msg) from e

    token_to_word: dict[int, str] = {
        v: k for k, v in data["word_to_token"].items()
    }
    vocabulary_token_ids: set[int] = set(data["word_to_token"].values())
    current_token_id = max(data["word_to_token"].values()) + 1

    chains = {
        order: {
            tuple(int(t) for t in context_str.split(",")): v
            for context_str, v in chain_data.items()
        }
        for order, chain_data in data["chains"].items()
    }

    training_config = TrainingConfig(
        max_order=data["training_config"]["max_order"],
    )

    return MarkovChain(
        training_config=training_config,
        _word_to_token=data["word_to_token"],
        _token_to_word=token_to_word,
        _vocabulary_token_ids=vocabulary_token_ids,
        _chains=chains,
        _current_token_id=current_token_id,
    )
