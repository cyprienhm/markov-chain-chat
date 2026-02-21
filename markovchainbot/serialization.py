"""JSON serialization for MarkovChain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markovchainbot.chain import MarkovChain

from markovchainbot.config import TrainingConfig

_VERSION = 1


def save(chain: MarkovChain, filepath: Path) -> None:
    """Save a trained MarkovChain to JSON."""
    serialized_chains: dict[str, dict[str, dict[str, int]]] = {}
    for order, chain_data in chain._chains.items():
        serialized_order: dict[str, dict[str, int]] = {}
        for context, transitions in chain_data.items():
            key = ",".join(str(t) for t in context)
            serialized_order[key] = {
                str(token): count for token, count in transitions.items()
            }
        serialized_chains[str(order)] = serialized_order

    data = {
        "version": _VERSION,
        "training_config": {
            "max_order": chain.training_config.max_order,
        },
        "word_to_token": chain._word_to_token,
        "chains": serialized_chains,
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def load(filepath: Path) -> MarkovChain:
    """Load a MarkovChain from JSON."""
    from markovchainbot.chain import MarkovChain

    try:
        data = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError) as e:
        msg = f"Failed to load model from {filepath}"
        raise ValueError(msg) from e

    word_to_token: dict[str, int] = {
        word: int(token_id) for word, token_id in data["word_to_token"].items()
    }
    token_to_word: dict[int, str] = {v: k for k, v in word_to_token.items()}
    vocabulary_token_ids: set[int] = set(word_to_token.values())
    current_token_id = max(word_to_token.values()) + 1

    chains: dict[int, dict[tuple[int, ...], dict[int, int]]] = {}
    for order_str, chain_data in data["chains"].items():
        order = int(order_str)
        chains[order] = {}
        for context_str, transitions in chain_data.items():
            context = tuple(int(t) for t in context_str.split(","))
            chains[order][context] = {
                int(token): count for token, count in transitions.items()
            }

    training_config = TrainingConfig(
        max_order=data["training_config"]["max_order"],
    )

    return MarkovChain(
        training_config=training_config,
        _word_to_token=word_to_token,
        _token_to_word=token_to_word,
        _vocabulary_token_ids=vocabulary_token_ids,
        _chains=chains,
        _current_token_id=current_token_id,
    )
