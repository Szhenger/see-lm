"""
A deliberately simple BPE tokenizer, to serve as an oracle in tests.

Encoding follows the definition directly: split the text on the special
tokens, pre-tokenize each remaining piece with the GPT-2 pattern, turn each
pre-token into single bytes, then apply every merge in training order.
Nothing is cached or reordered, so it is slow (every pre-token costs one
pass per merge) but easy to trust.

It is also the base class of OptimalTokenizer, which keeps everything here
except encode(): the setup and checks of the vocabulary, merges and special
tokens, loading from files, decoding, and the streaming encode_iterable.
"""

import os
import pickle
from collections.abc import Iterable, Iterator

import regex as re

from tokenization.trainer.optimal_bpe import PAT_RE  # the pre-tokenization pattern the trainer uses

_WS_RE = re.compile(r"\s")  # the pattern's own idea of whitespace, which str.isspace does not quite match


class NaiveTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Construct a tokenizer from a vocabulary, a list of merges, and optional special tokens."""
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(dict.fromkeys(special_tokens or []))  # deduplicated, order kept

        # A special token that is not in the vocabulary gets the next free ID.
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[max(self.vocab, default=-1) + 1] = token_bytes

        # The inverse table, bytes -> ID. Do not assume a byte's ID is its value:
        # that holds for our trainer's vocabularies but not for GPT-2's.
        self.id_of: dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}
        if len(self.id_of) != len(self.vocab):
            raise ValueError("vocabulary contains duplicate byte strings")
        for a, b in self.merges:
            if a + b not in self.id_of:
                raise ValueError(f"merge {(a, b)!r} produces {a + b!r}, which is not in the vocabulary")

        # Special tokens: ID by string, and a pattern that finds them in text.
        # Longest first, so a special token that contains another one wins.
        # The capturing group makes re.split keep the tokens it splits on.
        self._special_id: dict[str, int] = {t: self.id_of[t.encode("utf-8")] for t in self.special_tokens}
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self.special_re: re.Pattern[str] | None = re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")
            self._longest_special = len(ordered[0])  # in characters
        else:
            self.special_re = None
            self._longest_special = 0

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
        **kwargs,
    ):
        """Construct a tokenizer from the pickle files our training scripts write."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens, **kwargs)

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        The bytes are joined first and decoded once, because one character can
        be split across tokens. Malformed bytes become U+FFFD instead of raising.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily yield token IDs for the strings of an iterable, such as the lines of a file.

        The IDs are exactly those of encode() on the concatenation: a pre-token
        or a special token that spans two strings is encoded whole. Only the
        current string and a short tail of the earlier ones are held at a time.
        """
        carry = ""
        for piece in iterable:
            text = carry + piece
            cut = self._stable_prefix_end(text)
            if cut:
                yield from self.encode(text[:cut])
            carry = text[cut:]
        if carry:
            yield from self.encode(carry)

    def _stable_prefix_end(self, text: str) -> int:
        """
        Length of the longest prefix whose encoding cannot change when more text follows.

        The prefix ends on a special token or pre-token boundary. The last
        character is held back because the pre-token pattern looks one character
        ahead, and the last (longest special token - 1) characters are held back
        because text that follows could complete a special token starting there
        or lengthen one ending there. A cut between a whitespace character and a
        non-whitespace one is never made either: whitespace before a non-whitespace
        character is split differently from whitespace at the end of the text.
        A cut just before a special token is always safe, since encode() treats
        the text between special tokens as a string of its own.
        """
        limit = len(text) - max(1, self._longest_special - 1)
        if limit <= 0:
            return 0
        cut = 0
        pos = 0
        specials = self.special_re.finditer(text) if self.special_re is not None else iter(())
        while True:
            match = next(specials, None)
            gap_end = len(text) if match is None else match.start()
            for pretoken in PAT_RE.finditer(text, pos, gap_end):
                end = pretoken.end()
                if end > limit:
                    return cut
                if end == gap_end or not (_WS_RE.match(text, end - 1) and not _WS_RE.match(text, end)):
                    cut = end
            if match is None or match.end() > limit:
                return cut
            cut = pos = match.end()

    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token IDs."""
        ids: list[int] = []
        for is_special, piece in self._split_on_special_tokens(text):
            if is_special:
                ids.append(self.id_of[piece.encode("utf-8")])
            else:
                for match in PAT_RE.finditer(piece):
                    ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return ids

    def _split_on_special_tokens(self, text: str) -> Iterator[tuple[bool, str]]:
        """Yield (is_special, piece) in order. Pieces are never empty."""
        if self.special_re is None:
            if text:
                yield False, text
            return
        # With one capturing group, split returns text and special tokens
        # alternately, so odd positions hold the special tokens.
        for position, piece in enumerate(self.special_re.split(text)):
            if piece:
                yield position % 2 == 1, piece

    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        """Apply every merge, in training order, to one pre-token."""
        word = [bytes([b]) for b in pretoken]
        for a, b in self.merges:
            if len(word) < 2:
                break  # nothing left to merge
            word = _merge(word, a, b)
        return [self.id_of[token] for token in word]


def _merge(word: list[bytes], a: bytes, b: bytes) -> list[bytes]:
    """Replace each non-overlapping occurrence of the pair (a, b), scanning left to right."""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out