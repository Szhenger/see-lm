"""
Naive byte-pair encoding (BPE) training.

That is O(num_merges * corpus_size): fine for small files, slow for big ones.
"""

import os
from collections import Counter
import regex as re
from pathlib import Path


# GPT-2 pre-tokenization pattern.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Word = tuple[bytes, ...]  # one pre-token, as a sequence of current tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer."""
    # 1. Initialize the vocabulary: 256 single bytes, then the special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2. Pre-tokenize.
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    counts = _pretokenize(text, special_tokens)

    # 3. Merge the most frequent adjacent pair until the vocabulary is full.
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, freq in counts.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq

        if not pair_counts:  # every pre-token is already a single token
            break

        # Highest count wins; ties go to the lexicographically greater pair.
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        new_counts: Counter[Word] = Counter()
        for word, freq in counts.items():
            new_counts[_merge(word, best)] += freq
        counts = new_counts

    return vocab, merges


def _pretokenize(text: str, special_tokens: list[str]) -> Counter[Word]:
    """Count pre-tokens, each stored as a tuple of single bytes."""
    if special_tokens:
        # Longest first, so a token that contains another one wins the match.
        ordered = sorted(special_tokens, key=len, reverse=True)
        chunks = re.split("|".join(re.escape(t) for t in ordered), text)
    else:
        chunks = [text]

    counts: Counter[Word] = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            encoded = match.group().encode("utf-8")
            counts[tuple(bytes([b]) for b in encoded)] += 1
    return counts


def _merge(word: Word, pair: tuple[bytes, bytes]) -> Word:
    """Replace each non-overlapping occurrence of `pair`, scanning left to right."""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)