"""
Optimized byte-level BPE tokenizer.

Same results as naive_tokenizer.NaiveTokenizer, obtained differently:

Compute
  * Merges are applied by rank instead of by scanning the whole merge list.
    Within a pre-token, the adjacent pair with the lowest rank is merged
    (all of its occurrences, left to right) until no pair has a rank. This
    gives the same result as applying the merges in training order: a merge
    with a lower rank than the lowest one present cannot apply, and every
    pair a merge creates contains the new token, so its rank is higher.
  * Tokens are integer IDs throughout, so a finished pre-token needs no
    further lookup.
  * A bounded cache maps each pre-token string to its IDs. Text is Zipfian,
    so most pre-token occurrences are hits.
  * The text is scanned in place: special tokens are located with finditer,
    and the pattern runs on the gaps through findall's pos/endpos, so no
    copies of the text are made.
  * encode_file spreads a large file over worker processes. Chunks are cut
    at special tokens, so the output is identical to a single pass.
  * encode_iterable holds back the tail of each string until the next one
    arrives, so a pre-token or special token spanning two strings is encoded
    whole and the output is identical to encode() on the concatenation.

Memory
  * encode returns list[int] (8 bytes per token, plus the int objects). For
    anything large, encode_file writes a uint16 (or uint32) .npy file in
    batches and never holds all the IDs at once.
  * Workers read chunks of CHUNK_BYTES and decode one document at a time,
    so no large str is ever built (a str holding one emoji costs 4 bytes per
    character).
  * The cache is bounded (max_cache_size entries, roughly 150-200 bytes each);
    when full, the oldest half is dropped.
"""

import os
import struct
import sys
from array import array
from collections.abc import Iterable
from itertools import islice
from multiprocessing import Pool

import regex as re

from tokenization.tokenizer.naive_tokenizer import NaiveTokenizer
from tokenization.trainer.optimal_bpe import PAT_RE, find_chunk_boundaries, special_pattern

CHUNK_BYTES = 64 * 1024 * 1024  # bytes of input per worker task in encode_file
NPY_HEADER_BYTES = 128  # fixed header size, so it can be rewritten once the length is known
_INF = float("inf")


class OptimalTokenizer(NaiveTokenizer):
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        max_cache_size: int = 1_000_000,
    ) -> None:
        """Construct a tokenizer from a vocabulary, a list of merges, and optional special tokens."""
        super().__init__(vocab, merges, special_tokens)
        self.max_cache_size = max_cache_size
        try:
            # ID of each single byte. Do not assume it equals the byte value:
            # that holds for our trainer's vocabularies but not for GPT-2's.
            self._byte_id: list[int] = [self.id_of[bytes([b])] for b in range(256)]
        except KeyError as e:
            raise ValueError(f"vocabulary must contain every single byte; missing {e.args[0]!r}") from None

        # Merge tables keyed by ID pairs: the pair's rank, and the ID it merges into.
        self._rank: dict[tuple[int, int], int] = {}
        self._merged: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):  # NaiveTokenizer checked that a + b is in the vocabulary
            pair = (self.id_of[a], self.id_of[b])
            self._rank.setdefault(pair, rank)  # a repeated merge keeps its first rank
            self._merged[pair] = self.id_of[a + b]

        # The trainer's special-token pattern over bytes, for encode_file. The
        # capturing group makes split keep the tokens it splits on.
        if self.special_tokens:
            self._special_bytes_re = re.compile(b"(" + special_pattern(tuple(self.special_tokens)).pattern + b")")
        else:
            self._special_bytes_re = None

        # Decode table: a list when IDs are 0..n-1, which makes lookups cheaper.
        if sorted(self.vocab) == list(range(len(self.vocab))):
            self._bytes_of = [self.vocab[i] for i in range(len(self.vocab))]
        else:
            self._bytes_of = self.vocab

        # Output element type for encode_file: 2 bytes per token when the IDs fit.
        max_id = max(self.vocab)
        self._typecode, self._npy_dtype = ("H", "<u2") if max_id < 1 << 16 else ("I", "<u4")

        self._cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token IDs."""
        out: list[int] = []
        self._encode_text(text, out)
        return out

    def encode_file(
        self,
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
        num_processes: int | None = None,
        decode_errors: str = "strict",
    ) -> int:
        """
        Encode a whole file to a .npy array of token IDs and return the token count.

        The output has dtype uint16 when every ID fits, else uint32, and loads
        with numpy.load(output_path, mmap_mode="r"). Chunks of CHUNK_BYTES are
        encoded by worker processes, cut at special tokens so the result is
        identical to encode() on the whole text. With no special tokens there
        is no safe place to cut, so the file is one chunk in one process.
        """
        input_path = os.fspath(input_path)
        num_processes = num_processes or os.cpu_count() or 1

        if self.special_tokens:
            with open(input_path, "rb") as f:
                num_chunks = os.path.getsize(input_path) // CHUNK_BYTES + 1
                boundaries = find_chunk_boundaries(f, num_chunks, self.special_tokens)
        else:
            boundaries = [0, os.path.getsize(input_path)]
        tasks = [(input_path, start, end, decode_errors) for start, end in zip(boundaries[:-1], boundaries[1:])]

        count = 0
        with open(output_path, "wb") as out:
            out.write(_npy_header(0, self._npy_dtype))  # placeholder, rewritten below
            if num_processes == 1 or len(tasks) <= 1:
                for task in tasks:
                    blob = self._encode_chunk(task)
                    out.write(blob)
                    count += len(blob) // _ITEM_SIZE[self._typecode]
            else:
                init_args = (self.vocab, self.merges, self.special_tokens, self.max_cache_size)
                with Pool(num_processes, initializer=_init_worker, initargs=init_args) as pool:
                    for blob in pool.imap(_encode_chunk_in_worker, tasks):  # ordered
                        out.write(blob)
                        count += len(blob) // _ITEM_SIZE[self._typecode]
            out.seek(0)
            out.write(_npy_header(count, self._npy_dtype))
        return count

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        The bytes are joined first and decoded once, because one character can
        be split across tokens. Malformed bytes become U+FFFD instead of raising.
        """
        return b"".join(map(self._bytes_of.__getitem__, ids)).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _encode_text(self, text: str, out: list[int]) -> None:
        """Append the IDs for text, special tokens included, to out."""
        pos = 0
        if self.special_re is not None:
            for match in self.special_re.finditer(text):
                self._encode_plain(text, pos, match.start(), out)
                out.append(self._special_id[match.group()])
                pos = match.end()
        self._encode_plain(text, pos, len(text), out)

    def _encode_plain(self, text: str, start: int, end: int, out: list[int]) -> None:
        """Append the IDs for text[start:end], which contains no special token, to out."""
        if start >= end:
            return
        cache = self._cache
        lookup = cache.get
        extend = out.extend
        merge = self._merge_pretoken
        # pos/endpos make the pattern see text[start:end] without copying it.
        for pretoken in PAT_RE.findall(text, start, end):
            ids = lookup(pretoken)
            if ids is None:
                ids = merge(pretoken)
                if len(cache) >= self.max_cache_size:
                    self._evict()
                cache[pretoken] = ids
            extend(ids)

    def _merge_pretoken(self, pretoken: str) -> tuple[int, ...]:
        """Merge one pre-token by rank until no adjacent pair has a merge."""
        byte_id = self._byte_id
        word = [byte_id[b] for b in pretoken.encode("utf-8")]
        rank = self._rank.get
        merged = self._merged
        while len(word) > 1:
            best = min(zip(word, word[1:]), key=lambda pair: rank(pair, _INF))
            if best not in merged:
                break
            a, b = best
            c = merged[best]
            # Replace every non-overlapping occurrence, left to right.
            new_word: list[int] = []
            i, n = 0, len(word)
            while i < n:
                if word[i] == a and i + 1 < n and word[i + 1] == b:
                    new_word.append(c)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return tuple(word)

    def _evict(self) -> None:
        """Drop the oldest half of the cache (dicts keep insertion order)."""
        for key in list(islice(self._cache, len(self._cache) // 2)):
            del self._cache[key]

    def _encode_chunk(self, task: tuple[str, int, int, str]) -> bytes:
        """
        Encode one byte range of a file; return the IDs packed as the output element type.

        The chunk is split on the special tokens while still bytes, and each
        piece is decoded on its own, so no large str is built.
        """
        path, start, end, decode_errors = task
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        packed = array(self._typecode)
        if self._special_bytes_re is None:
            pieces = [chunk]
        else:
            pieces = self._special_bytes_re.split(chunk)  # odd positions are the special tokens
        del chunk

        buffer: list[int] = []
        for position, piece in enumerate(pieces):
            if not piece:
                continue
            if position % 2 == 1:
                buffer.append(self.id_of[piece])
            else:
                text = piece.decode("utf-8", errors=decode_errors)
                self._encode_plain(text, 0, len(text), buffer)
            if len(buffer) >= 1 << 20:
                packed.extend(buffer)
                buffer.clear()
        packed.extend(buffer)
        if sys.byteorder == "big":
            packed.byteswap()  # the .npy header declares little-endian
        return packed.tobytes()


# ----------------------------------------------------------------------
# Worker-process support for encode_file
# ----------------------------------------------------------------------
_ITEM_SIZE = {"H": 2, "I": 4}
_worker_tokenizer: OptimalTokenizer | None = None


def _init_worker(vocab, merges, special_tokens, max_cache_size) -> None:
    """Build this worker's tokenizer once, so every chunk it handles shares the cache."""
    global _worker_tokenizer
    _worker_tokenizer = OptimalTokenizer(vocab, merges, special_tokens, max_cache_size)


def _encode_chunk_in_worker(task: tuple[str, int, int, str]) -> bytes:
    assert _worker_tokenizer is not None, "worker was not initialized"
    return _worker_tokenizer._encode_chunk(task)


def _npy_header(length: int, dtype: str) -> bytes:
    """
    A .npy (format 1.0) header for a 1-D array, padded to NPY_HEADER_BYTES.

    The fixed size lets encode_file write a placeholder first and rewrite the
    header with the real length when the data is complete.
    """
    magic = b"\x93NUMPY\x01\x00"
    body = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({length},), }}".encode("latin1")
    header_len = NPY_HEADER_BYTES - len(magic) - 2
    assert len(body) < header_len, "header does not fit"
    body = body + b" " * (header_len - len(body) - 1) + b"\n"
    return magic + struct.pack("<H", header_len) + body