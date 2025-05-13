"""Markov chain."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from markovchainbot.utils import (
    DiscordMessageReader,
    MessageProcessor,
    MessageReader,
    levenshtein_distance,
)


@dataclass
class MarkovChain:
    """Markov Chain.

    Tokenizes vocabulary and forms mappings.
    """

    message_reader: MessageReader = field(default_factory=DiscordMessageReader)
    message_processor: MessageProcessor = field(
        default_factory=MessageProcessor
    )
    _word_to_token: dict[str, int] = field(default_factory=dict)
    _token_to_word: dict[int, str] = field(default_factory=dict)
    _vocabulary_tokens_ids: set[int] = field(default_factory=set)
    _unigram_chain: dict[int, dict[int, int]] = field(default_factory=dict)
    _bigram_chain: dict[tuple[int, int], dict[int, int]] = field(
        default_factory=dict
    )
    _current_token_id: int = 0
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def add_messages_to_vocabulary(self, messages: list[str]):
        """Add list of messages to vocabulary."""
        for message in messages:
            words = self.message_processor.process(message)
            for word in words:
                self.add_as_token(word)
            self.add_to_chain(words)
        self.sort_chains()

    def add_file_to_vocabulary(self, filepath: Path):
        """Add file to vocabulary."""
        messages = self.message_reader.get_messages(filepath)

        for message in messages:
            words = self.message_processor.process(message)
            for word in words:
                self.add_as_token(word)
            self.add_to_chain(words)
        self.sort_chains()

    def add_to_chain(self, words):
        """Build chains."""
        for i in range(len(words) - 1):
            current_token = self._word_to_token[words[i]]
            next_token = self._word_to_token[words[i + 1]]

            if current_token not in self._unigram_chain:
                self._unigram_chain[current_token] = {}
            self._unigram_chain[current_token][next_token] = (
                self._unigram_chain[current_token].get(next_token, 0) + 1
            )

            if i <= len(words) - 3:
                next_next_token = self._word_to_token[words[i + 2]]
                if (current_token, next_token) not in self._bigram_chain:
                    self._bigram_chain[(current_token, next_token)] = {}
                self._bigram_chain[(current_token, next_token)][
                    next_next_token
                ] = (
                    self._bigram_chain[(current_token, next_token)].get(
                        next_next_token, 0
                    )
                    + 1
                )

    def sort_chains(self):
        """Reverse order sort chains."""
        self._unigram_chain = {
            k: {kk: v[kk] for kk in sorted(v, key=v.get, reverse=True)}
            for k, v in self._unigram_chain.items()
        }
        self._bigram_chain = {
            k: {kk: v[kk] for kk in sorted(v, key=v.get, reverse=True)}
            for k, v in self._bigram_chain.items()
        }

    def get_random_token(self):
        """Get a random token from vocabulary."""
        return self._rng.choice(tuple(self._vocabulary_tokens_ids))

    def predict_next_token(self, *, prev_token, prev_prev_token=None) -> int:
        """Predict based on previous tokens."""
        # 1 gram case
        if prev_prev_token is None:
            if prev_token not in self._unigram_chain:
                # return random.choice(tuple(self._vocabulary))
                return self.get_random_token()

            to_pick = self._unigram_chain[prev_token]
            return self.pick_randomly_from_dict(to_pick)
        else:
            if (prev_prev_token, prev_token) not in self._bigram_chain:
                return self.predict_next_token(
                    prev_token=prev_token, prev_prev_token=None
                )

            to_pick = self._bigram_chain[(prev_prev_token, prev_token)]
            return self.pick_randomly_from_dict(to_pick)

    def pick_randomly_from_dict(self, to_pick: dict[int, int]) -> int:
        """Generate random number, pick when cumulative/total > random."""
        total_count = sum(to_pick.values())
        random_number = self._rng.random() * total_count
        cumulative = 0
        for token, value in to_pick.items():
            cumulative += value
            if cumulative >= random_number:
                return token
        return next(iter(to_pick))

    def tokenize_sentence(self, sentence):
        """Single sentence."""
        words = self.message_processor.process(sentence)
        return [self._word_to_token[w] for w in words]

    def untokenize_list(self, tokens):
        """Single sentence."""
        return [self._token_to_word[t] for t in tokens]

    def add_as_token(self, word):
        """Add word and token_id++."""
        if word not in self._word_to_token:
            self._word_to_token[word] = self._current_token_id
            self._token_to_word[self._current_token_id] = word
            self._vocabulary_tokens_ids.add(self._current_token_id)
            self._current_token_id += 1

    def continue_sentence(self, sentence: str):
        """Continue sentence."""
        words = self.message_processor.process(sentence)[:-1]
        tokens = []
        for w in words:
            if w not in self._word_to_token:
                tokens.append(
                    self._word_to_token[
                        min(
                            self._word_to_token,
                            key=lambda x: levenshtein_distance(w, x),
                        )
                    ]
                )
            else:
                tokens.append(self._word_to_token[w])

        continued = []
        if len(tokens) == 1:
            prev_token = None
            current_token = tokens[0]
        else:
            prev_token = tokens[-2]
            current_token = tokens[-1]

        probability_randomize = 0.03
        while len(continued) < 200:
            coin_flip = self._rng.random()
            if coin_flip < probability_randomize:
                prev_token, current_token = (
                    current_token,
                    self.get_random_token(),
                )
                probability_randomize = 0.03
            else:
                prev_token, current_token = (
                    current_token,
                    self.predict_next_token(
                        prev_token=current_token, prev_prev_token=prev_token
                    ),
                )
            probability_randomize += self._rng.random() * 0.03
            if current_token == self._word_to_token["<end>"]:
                if len(continued) < 3 and self._rng.random() < 0.5:
                    continue
                else:
                    continued.append(self._token_to_word[current_token])
                    break
            continued.append(self._token_to_word[current_token])
        return " ".join(continued[:-1])
