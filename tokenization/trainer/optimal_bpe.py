"""
Optimized BPE training.

Four optimizations over the naive trainer:

1. Pre-tokenization runs in parallel across worker processes. Each worker
   splits its chunk on the special tokens while it is still bytes and decodes
   one document at a time, so it never builds a large string. Counting is
   done in C (regex findall + Counter.update).
2. Inside the merge loop, tokens are integer IDs and each distinct pre-token
   is stored once as a list of IDs next to its frequency.
3. Pair counts are cached and updated incrementally. After a merge, only the
   words that contained the pair can change, and within such a word only the
   pairs touching a merged position. The pair-to-words index is lazy: entries
   are added when a pair appears in a word and never removed, so a visit may
   find no occurrence and simply does nothing.
4. The cyclic garbage collector is paused while the caches exist. Nothing here
   forms reference cycles, but every full collection would walk the millions
   of lists, sets and tuples that make up the caches.
"""

import gc
import heapq
import os
import regex as re
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import get_context
from collections.abc import Sequence
from typing import BinaryIO


# GPT-2 pre-tokenization pattern. It must contain no capturing group: findall
# would then return the group instead of the whole match.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)

# Aim for chunks of about this size. A worker holds roughly two copies of its
# chunk as bytes plus its counter, so this bounds memory per worker.
TARGET_CHUNK_BYTES = 64 * 1024 * 1024

Pair = tuple[int, int]  # two token IDs


# --------------------------------------------------------------------------
# Training (greedy)
# --------------------------------------------------------------------------
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer.

    Returns:
        vocab:  token ID -> token bytes
        merges: merged pairs, in the order they were created
    """
    # 1. Initialize the vocabulary: 256 single bytes, then the special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # 2. Pre-tokenize, in parallel.
    counts = parallel_pretokenize(input_path, special_tokens, num_processes)

    # The caches built below hold millions of container objects. They form no
    # reference cycles, so pause the cyclic collector rather than let each of
    # its full collections walk all of them.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _merge_loop(vocab, counts, vocab_size)
    finally:
        if gc_was_enabled:
            gc.enable()


# --------------------------------------------------------------------------
# Chunking (serial)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def special_pattern(special_tokens: tuple[str, ...]) -> re.Pattern[bytes]:
    """
    A bytes pattern matching any special token. Longest first, so a token
    that contains another one wins the match. No capturing group, so split
    drops the tokens and findall returns whole matches.
    """
    ordered = sorted((t.encode("utf-8") for t in special_tokens), key=len, reverse=True)
    return re.compile(b"|".join(re.escape(t) for t in ordered))


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: Sequence[str],
) -> list[int]:
    """
    Byte offsets that cut the file into parts which can be processed independently.

    Every interior boundary is where a special token starts in a scan of the
    whole file with the same longest-first pattern the workers split on, so
    no document and no special token is ever split between two chunks. May
    return fewer chunks than asked for when boundaries coincide.
    """
    if not special_tokens:
        raise ValueError("at least one special token is needed to cut the file safely")
    pattern = special_pattern(tuple(special_tokens))
    token_bytes = [t.encode("utf-8") for t in special_tokens]
    alphabet = frozenset(b"".join(token_bytes))
    longest = max(len(t) for t in token_bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    # Initial guesses, uniformly spaced; each is moved to the start of the
    # first special token that ends after it. Boundaries are found in order,
    # because each search may fall back on the previous boundary.
    chunk_size = file_size // desired_num_chunks
    boundaries = [0]
    for i in range(1, desired_num_chunks):
        boundaries.append(_token_start_after(file, pattern, alphabet, longest, boundaries[-1], i * chunk_size, file_size))
    boundaries.append(file_size)

    # Unique and ordered; there may be fewer than desired_num_chunks.
    return sorted(set(boundaries))


def _token_start_after(
    file: BinaryIO,
    pattern: re.Pattern[bytes],
    alphabet: frozenset[int],
    longest: int,
    previous: int,
    guess: int,
    file_size: int,
    read_size: int = 4096,
) -> int:
    """
    Offset of the first special token that ends after `guess`, or file_size if there is none.

    "Special token" means a match of the longest-first pattern in a scan of
    the whole file. A scan started at an arbitrary offset can disagree with
    it, e.g. by starting inside a token and matching its second half, so the
    scan starts right after the nearest byte before the guess that no special
    token contains: no match can cover that byte, so from there on the two
    scans agree. If there is no such byte since the previous boundary, which
    is itself a token start (or the start of the file), the scan starts there.
    """
    # 1. Find the synchronization point, reading backwards from the guess.
    sync = previous
    position = guess
    while position > previous:
        low = max(previous, position - read_size)
        file.seek(low)
        block = file.read(position - low)
        index = next((j for j in range(len(block) - 1, -1, -1) if block[j] not in alphabet), None)
        if index is not None:
            sync = low + index + 1
            break
        position = low

    # 2. Scan forwards from there. A match is trusted only once `longest` bytes
    #    from its start are in the window, so a longer token starting at the
    #    same place cannot be missed; otherwise the window is extended first.
    file.seek(sync)
    position = sync  # file offset of window[0]
    window = b""
    while True:
        block = file.read(read_size)
        at_eof = not block
        window += block
        last_end = 0
        for match in pattern.finditer(window):
            if not at_eof and match.start() + longest > len(window):
                resume = match.start()  # undecided: read more and look at it again
                break
            if position + match.end() > guess:
                return position + match.start()
            last_end = match.end()
        else:
            if at_eof:
                return file_size
            # No token starts before here: everything earlier was fully visible.
            resume = max(last_end, len(window) - longest + 1, 0)
        position += resume
        window = window[resume:]


# --------------------------------------------------------------------------
# Pre-tokenization (parallel)
# --------------------------------------------------------------------------
# Set this environment variable to a folder and every pool worker traces its
# counting with cProfile, writing worker-<pid>.prof there. train_bpe.py
# --profile uses it. Nothing happens unless the variable is set.
PROFILE_DIR_VAR = "BPE_PROFILE_DIR"
_profiler = None  # one per worker process, created on first use


def _count_chunk(task: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    """
    Worker entry point: count the pre-tokens in one byte range of the file.

    Must be a top-level function so worker processes can import it.
    """
    if os.environ.get(PROFILE_DIR_VAR):
        return _profiled(_count_pretokens, task)
    return _count_pretokens(task)


def _profiled(function, *args):
    """
    Run function under this process's profiler, then write the stats so far.

    The stats are rewritten after every task because a pool worker is never
    told which task is its last.

    The pool spawns its workers, so each starts without an active profiler.
    Should a worker ever inherit one (a forked child of a profiled parent), it
    cannot enable a second one and just counts, unprofiled.
    """
    global _profiler
    import cProfile

    if _profiler is False:
        return function(*args)
    if _profiler is None:
        _profiler = cProfile.Profile()
    try:
        _profiler.enable()
    except ValueError:  # another profiler is already active in this process
        _profiler = False
        return function(*args)
    try:
        return function(*args)
    finally:
        _profiler.disable()
        try:
            _profiler.dump_stats(os.path.join(os.environ[PROFILE_DIR_VAR], f"worker-{os.getpid()}.prof"))
        except OSError:
            pass  # the profile is a side channel; the count still goes back to the parent


def _count_pretokens(task: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    """Count the pre-tokens in one byte range of the file."""
    path, start, end, special_tokens = task

    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # Remove the special tokens by splitting on them, so the pattern never
    # sees them and nothing can merge across them. Splitting happens on bytes,
    # and each document is decoded on its own: a str is stored at the width
    # of its widest character, so decoding a whole chunk that contains one
    # emoji would take four bytes per character.
    pieces = special_pattern(special_tokens).split(chunk) if special_tokens else [chunk]
    del chunk

    # findall and Counter.update both loop in C, so there is no Python-level
    # work per pre-token.
    counts: Counter[str] = Counter()
    findall = PAT_RE.findall
    for piece in pieces:
        counts.update(findall(piece.decode("utf-8", errors="ignore")))

    # Encode each distinct pre-token once, here in the worker.
    return Counter({token.encode("utf-8"): n for token, n in counts.items()})


def parallel_pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
    chunks_per_process: int = 4,
) -> Counter[bytes]:
    """
    Count pre-tokens across worker processes. Keys are UTF-8 encoded pre-tokens.

    The file is cut at special tokens, so no document is ever split between
    two chunks. With no special tokens there is no safe
    place to cut, and the file is processed as a single chunk.
    """
    num_processes = num_processes or os.cpu_count() or 1
    path = os.fspath(input_path)
    specials = tuple(special_tokens)

    if special_tokens:
        with open(path, "rb") as f:
            # Enough chunks to even out the load and to keep each one small.
            by_size = os.path.getsize(path) // TARGET_CHUNK_BYTES + 1
            num_chunks = max(num_processes * chunks_per_process, by_size)
            boundaries = find_chunk_boundaries(f, num_chunks, specials)
    else:
        boundaries = [0, os.path.getsize(path)]

    tasks = [(path, start, end, specials) for start, end in zip(boundaries[:-1], boundaries[1:])]

    total: Counter[bytes] = Counter()
    if num_processes == 1 or len(tasks) <= 1:
        # Straight to the counting: this process may already be under a
        # profiler, and two cProfile instances cannot be active at once.
        for task in tasks:
            total.update(_count_pretokens(task))
    else:
        # Spawned workers are direct children of this process, so their peak
        # memory shows up in its RUSAGE_CHILDREN on every platform. (Linux
        # would otherwise use a forkserver, whose children are not ours.)
        with get_context("spawn").Pool(num_processes) as pool:
            for counts in pool.imap_unordered(_count_chunk, tasks):
                total.update(counts)
    return total


# --------------------------------------------------------------------------
# Merging (incremental)
# --------------------------------------------------------------------------
def _merge_loop(
    vocab: dict[int, bytes],
    counts: Counter[bytes],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Steps 3 and 4 of training: build the caches, then merge until the vocabulary is full."""
    # Each distinct pre-token is stored once as a list of token IDs. Byte values
    # are the IDs of the single-byte tokens, so list(b"th") == [116, 104].
    words: list[list[int]] = [list(token) for token in counts]
    freqs: list[int] = list(counts.values())
    del counts

    # 3. Build the caches, once.
    #    pair_counts:   how often each adjacent pair occurs in the corpus
    #    pair_to_words: which words contain each pair (lazy: may hold stale entries)
    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_to_words: dict[Pair, set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        freq = freqs[i]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(i)

    # Max-heap on (count, pair bytes), built from a min-heap by negating the
    # count and reversing the bytes order. Entries are never updated in place:
    # a changed count pushes a fresh entry, and stale ones are skipped on pop.
    heap = [(-n, _Desc((vocab[a], vocab[b])), (a, b)) for (a, b), n in pair_counts.items()]
    heapq.heapify(heap)

    # 4. Merge until the vocabulary is full.
    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        # Highest count wins; ties go to the lexicographically greater pair.
        best: Pair | None = None
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg_count:  # still current?
                best = pair
                break
        if best is None:  # every pre-token is already a single token
            break

        a, b = best
        c = len(vocab)  # ID of the merged token
        vocab[c] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        # Visit every word that may contain the pair and apply the merge,
        # recording how the counts of the neighbouring pairs change.
        delta: dict[Pair, int] = defaultdict(int)
        for i in pair_to_words.pop(best, ()):
            word = words[i]
            if a not in word or b not in word:  # stale index entry (C-level scans, cheap)
                continue
            freq = freqs[i]
            n = len(word)
            out: list[int] = []
            just_merged = False  # was out[-1] placed by this merge?
            j = 0
            while j < n:
                t = word[j]
                if t == a and j + 1 < n and word[j + 1] == b:
                    if out:
                        if just_merged:
                            # Two merges back to back: the pair between them is
                            # new, and the old (b, a) there was already removed
                            # as the right-hand pair of the previous merge.
                            delta[(c, c)] += freq
                            pair_to_words[(c, c)].add(i)
                        else:
                            left = out[-1]
                            delta[(left, a)] -= freq
                            delta[(left, c)] += freq
                            pair_to_words[(left, c)].add(i)
                    if j + 2 < n:
                        delta[(b, word[j + 2])] -= freq
                    out.append(c)
                    just_merged = True
                    j += 2
                else:
                    if just_merged:
                        delta[(c, t)] += freq
                        pair_to_words[(c, t)].add(i)
                        just_merged = False
                    out.append(t)
                    j += 1
            if len(out) != n:
                words[i] = out

        # The merged pair itself is gone from every word.
        del pair_counts[best]
        delta.pop(best, None)  # can appear when a == b, e.g. (a, a) in [a, a, a]

        # Apply the net changes; pairs the merge did not touch never enter delta.
        for pair, change in delta.items():
            if change == 0:
                continue
            n = pair_counts.get(pair, 0) + change
            if n > 0:
                pair_counts[pair] = n
                heapq.heappush(heap, (-n, _Desc((vocab[pair[0]], vocab[pair[1]])), pair))
            else:
                pair_counts.pop(pair, None)

    return vocab, merges


class _Desc:
    """
    Wraps a pair of bytes so that a min-heap pops the lexicographically greatest first.

    No __eq__ on purpose: tuple comparison then falls straight through to __lt__
    when counts tie, which is the common case.
    """

    __slots__ = ("key",)

    def __init__(self, key: tuple[bytes, bytes]) -> None:
        self.key = key

    def __lt__(self, other: "_Desc") -> bool:
        return self.key > other.key


# The guard is required: worker processes re-import this file, and anything
# outside it would run again in every worker.
if __name__ == "__main__":
    import sys

    vocab, merges = train_bpe(sys.argv[1], vocab_size=10_000, special_tokens=["<|endoftext|>"])
    print(f"{len(vocab)} tokens, {len(merges)} merges")
    print(merges[:10])
