"""Markov chain implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from markovchainbot.config import GenerationConfig, TrainingConfig
from markovchainbot.text import levenshtein_distance, process_message


@dataclass
class MarkovChain:
    """N-gram Markov chain text generator.

    Tokenizes vocabulary, builds transition chains up to the order specified in
    training_config.
    """

    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    generation_config: GenerationConfig = field(
        default_factory=GenerationConfig
    )
    _word_to_token: dict[str, int] = field(default_factory=dict)
    _token_to_word: dict[int, str] = field(default_factory=dict)
    _vocabulary_token_ids: set[int] = field(default_factory=set)
    _chains: dict[int, dict[tuple[int, ...], dict[int, int]]] = field(
        default_factory=dict
    )
    _current_token_id: int = 0
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def add_messages(self, messages: list[str]) -> None:
        """Add messages to vocabulary and build chains."""
        for message in messages:
            words = process_message(message)
            for word in words:
                self._add_as_token(word)
            self._add_to_chain(words)
        self._sort_chains()

    def _add_as_token(self, word: str) -> None:
        """Register a word in the vocabulary if not already present."""
        if word not in self._word_to_token:
            self._word_to_token[word] = self._current_token_id
            self._token_to_word[self._current_token_id] = word
            self._vocabulary_token_ids.add(self._current_token_id)
            self._current_token_id += 1

    def _add_to_chain(self, words: list[str]) -> None:
        """Build n-gram chains from a word sequence."""
        tokens = [self._word_to_token[w] for w in words]
        for order in range(1, self.training_config.max_order + 1):
            if order not in self._chains:
                self._chains[order] = {}
            chain = self._chains[order]
            for i in range(len(tokens) - order):
                context = tuple(tokens[i : i + order])
                next_token = tokens[i + order]
                if context not in chain:
                    chain[context] = {}
                chain[context][next_token] = (
                    chain[context].get(next_token, 0) + 1
                )

    def _sort_chains(self) -> None:
        """Sort all chain transitions by frequency (descending)."""
        for order in self._chains:
            self._chains[order] = {
                context: {
                    token: counts[token]
                    for token in sorted(counts, key=counts.get, reverse=True)
                }
                for context, counts in self._chains[order].items()
            }

    def _get_random_token(self) -> int:
        """Get a random token from vocabulary."""
        return int(self._rng.choice(tuple(self._vocabulary_token_ids)))

    def predict_next_token(
        self,
        context: list[int],
    ) -> int:
        """Predict the next token given a context sequence.

        Tries the highest available n-gram order first,
        falling back to lower orders.
        """
        max_order = self.training_config.max_order
        for order in range(min(max_order, len(context)), 0, -1):
            chain = self._chains.get(order, {})
            key = tuple(context[-order:])
            if key in chain:
                return self._pick_randomly(chain[key])
        return self._get_random_token()

    def _pick_randomly(self, transition_weights: dict[int, int]) -> int:
        """Pick a token weighted by transition counts."""
        total = sum(transition_weights.values())
        threshold = self._rng.random() * total
        cumulative = 0
        for token, count in transition_weights.items():
            cumulative += count
            if cumulative >= threshold:
                return token
        return next(iter(transition_weights))

    def tokenize_sentence(self, sentence: str) -> list[int]:
        """Tokenize a sentence, using fuzzy match for unknown words."""
        words = process_message(sentence)
        return [self._resolve_token(w) for w in words]

    def untokenize_list(self, tokens: list[int]) -> list[str]:
        """Convert a list of token IDs back to words."""
        return [self._token_to_word[t] for t in tokens]

    def _resolve_token(self, word: str) -> int:
        """Resolve a word to its token ID, fuzzy fallback."""
        if word in self._word_to_token:
            return self._word_to_token[word]
        closest = min(
            self._word_to_token,
            key=lambda w: levenshtein_distance(word, w),
        )
        return self._word_to_token[closest]

    def continue_sentence(self, sentence: str) -> str:
        """Continue a sentence using the trained model."""
        if not self._word_to_token:
            msg = "Model has no vocabulary. Train before generating."
            raise ValueError(msg)

        words = process_message(sentence)[:-1]  # drop <end>
        tokens = [self._resolve_token(w) for w in words]

        if not tokens:
            tokens = [self._get_random_token()]

        cfg = self.generation_config
        generated_words: list[str] = []
        context = tokens[:]
        randomize_probability = cfg.randomize_probability_base
        end_token = self._word_to_token["<end>"]

        while len(generated_words) < cfg.max_length:
            coin_flip = self._rng.random()
            if coin_flip < randomize_probability:
                next_token = self._get_random_token()
                randomize_probability = cfg.randomize_probability_base
            else:
                next_token = self.predict_next_token(context)

            randomize_probability += (
                self._rng.random() * cfg.randomize_probability_increment
            )
            context.append(next_token)

            if next_token == end_token:
                if (
                    len(generated_words) < cfg.min_length
                    and self._rng.random()
                    < cfg.short_sentence_retry_probability
                ):
                    continue
                break

            generated_words.append(self._token_to_word[next_token])

        return " ".join(generated_words)

    def save(self, filepath: str | Path) -> None:
        """Save the trained model to JSON."""
        from markovchainbot.serialization import save

        save(self, Path(filepath))

    @classmethod
    def load(cls, filepath: str | Path) -> MarkovChain:
        """Load a trained model from JSON."""
        from markovchainbot.serialization import load

        return load(Path(filepath))
