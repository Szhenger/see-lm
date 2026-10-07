"""
Streaming and chunking must give the same IDs as encode() on the whole text.

These use the trained TinyStories tokenizer. Its vocabulary has a (b" ", b"\\n")
merge, so whitespace that is split differently at a line break gives
different IDs, which the GPT-2 vocabulary in the other tests would hide.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from tokenization.tokenizer import optimal_tokenizer
from tokenization.tokenizer.naive_tokenizer import NaiveTokenizer
from tokenization.tokenizer.optimal_tokenizer import OptimalTokenizer
from tokenization.trainer.optimal_bpe import find_chunk_boundaries, special_pattern

ARTIFACTS = Path(__file__).resolve().parent.parent / "tokenization" / "trainer" / "artifacts"
EOT = "<|endoftext|>"
SPECIALS = [EOT, EOT + EOT]  # the doubled form is its own token, and contains the single one

TEXT = (
    "Once upon a time, there was a little girl. \n"
    "She loved to play outside.  \n\n"
    "The end.   \n" + EOT + "Second story: a boy and his dog.\n" + EOT + EOT + "Third story\n\n\n   with odd   spacing\t\n"
    + EOT + "Last words, it's over"
)


def _tokenizer(cls):
    return cls.from_files(ARTIFACTS / "tinystories_vocab.pkl", ARTIFACTS / "tinystories_merges.pkl", SPECIALS)


@pytest.fixture(scope="module")
def optimal():
    return _tokenizer(OptimalTokenizer)


@pytest.fixture(scope="module")
def naive():
    return _tokenizer(NaiveTokenizer)


def _random_pieces(text: str, rng: random.Random, max_len: int = 9) -> list[str]:
    pieces, i = [], 0
    while i < len(text):
        n = rng.randint(1, max_len)
        pieces.append(text[i : i + n])
        i += n
    return pieces


def test_vocab_has_the_merge_this_file_relies_on(optimal):
    assert (b" ", b"\n") in optimal.merges


def test_naive_and_optimal_agree(optimal, naive):
    assert naive.encode(TEXT) == optimal.encode(TEXT)
    assert optimal.decode(optimal.encode(TEXT)) == TEXT


@pytest.mark.parametrize("fixture", ["optimal", "naive"])
def test_encode_iterable_over_lines_matches_encode(request, fixture):
    tokenizer = request.getfixturevalue(fixture)
    lines = TEXT.splitlines(keepends=True)
    assert "".join(lines) == TEXT
    assert list(tokenizer.encode_iterable(lines)) == tokenizer.encode(TEXT)


def test_encode_iterable_whitespace_run_over_three_pieces(optimal):
    pieces = ["end. ", " ", " \nNext"]
    assert list(optimal.encode_iterable(pieces)) == optimal.encode("".join(pieces))


def test_encode_iterable_special_token_split_across_pieces(optimal, naive):
    cases = [
        ["abc<|endof", "text|>def"],  # a special token cut in the middle
        ["abc" + EOT, EOT + "def"],  # a doubled special token cut between its halves
        ["abc" + EOT + "<|endo", "ftext|>def"],  # the doubled token's second half cut
        [EOT, EOT, EOT],  # three in a row: doubled then single
        ["x", "", "y"],  # an empty piece
    ]
    for pieces in cases:
        whole = "".join(pieces)
        assert list(optimal.encode_iterable(pieces)) == optimal.encode(whole), pieces
        assert list(naive.encode_iterable(pieces)) == naive.encode(whole), pieces


def test_encode_iterable_random_pieces_match_encode(optimal):
    expected = optimal.encode(TEXT)
    for seed in range(40):
        pieces = _random_pieces(TEXT, random.Random(seed))
        assert list(optimal.encode_iterable(pieces)) == expected, seed


def test_encode_iterable_without_special_tokens(optimal):
    plain = OptimalTokenizer(optimal.vocab, optimal.merges)
    text = TEXT.replace(EOT, " | ")
    for seed in range(10):
        pieces = _random_pieces(text, random.Random(seed))
        assert list(plain.encode_iterable(pieces)) == plain.encode(text), seed


@pytest.mark.parametrize("num_processes", [1, 2])
def test_encode_file_matches_encode(optimal, tmp_path, monkeypatch, num_processes):
    monkeypatch.setattr(optimal_tokenizer, "CHUNK_BYTES", 150)  # so a small file becomes many chunks
    text = TEXT * 12
    source = tmp_path / "in.txt"
    source.write_text(text, encoding="utf-8")
    out = tmp_path / "out.npy"
    count = optimal.encode_file(source, out, num_processes=num_processes)
    ids = np.load(out, mmap_mode="r")
    expected = optimal.encode(text)
    assert count == len(expected) == len(ids)
    assert ids.tolist() == expected


@pytest.mark.parametrize("num_chunks", [1, 2, 3, 5, 8, 13, 21, 50])
def test_find_chunk_boundaries_only_cuts_where_a_whole_file_scan_finds_a_token(tmp_path, num_chunks):
    specials = (EOT, "x" + EOT, EOT + EOT)
    data = (("story " * 7 + "x" + EOT) * 3 + "tale " * 5 + EOT + EOT) * 11
    data = data.encode("utf-8")
    path = tmp_path / "data.txt"
    path.write_bytes(data)

    matches = [(m.start(), m.end()) for m in special_pattern(specials).finditer(data)]
    starts = {s for s, _ in matches}
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, list(specials))

    assert boundaries[0] == 0 and boundaries[-1] == len(data)
    assert boundaries == sorted(set(boundaries))
    for b in boundaries[1:-1]:
        assert b in starts, f"boundary {b} is not where a special token starts"
        assert not any(s < b < e for s, e in matches), f"boundary {b} splits a special token"


def test_find_chunk_boundaries_needs_a_special_token(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"no markers here")
    with open(path, "rb") as f, pytest.raises(ValueError):
        find_chunk_boundaries(f, 4, [])
