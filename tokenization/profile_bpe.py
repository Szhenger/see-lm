"""
Profile the BPE trainers.

Layout this script expects:

    project/
        data/                <- corpora (DATA_DIR below)
        tokenization/
            profile_bpe.py   <- this file
            naive_bpe.py     <- each *_bpe.py exposes train_bpe()
            optimal_bpe.py

Examples (from the project root):
    uv run python tokenization/profile_bpe.py                            # first 2 GB of data/owt_train.txt
    uv run python tokenization/profile_bpe.py --no-profile --processes 8
    uv run python tokenization/profile_bpe.py --sample-mb 100            # a smaller slice, for quick feedback
    uv run python tokenization/profile_bpe.py --sample-mb 0              # the whole file
    uv run python tokenization/profile_bpe.py --impl naive --sample-mb 20 --vocab-size 500
    uv run python tokenization/profile_bpe.py --out train.prof           # then: uvx snakeviz train.prof
    uv run python tokenization/profile_bpe.py other_corpus.txt           # a different file in data/

A bare file name is looked up in the data folder. A path works too.
--impl NAME profiles NAME_bpe.py, so new trainers are picked up automatically.
"""

import argparse
import cProfile
import importlib
import inspect
import os
import pstats
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Where the corpora live: the data/ folder next to tokenization/.
# Change this line, or pass --data-dir, if yours is somewhere else.
DATA_DIR = HERE.parent / "data"

# Profiled when no corpus is given on the command line.
DEFAULT_CORPUS = "owt_train.txt"

# How much of the corpus to use by default, in megabytes. 0 means the whole file.
DEFAULT_SAMPLE_MB = 2000

COPY_BLOCK = 64 * 1024 * 1024  # bytes copied at a time when writing a sample
BOUNDARY_WINDOW = 16 * 1024 * 1024  # how far back to look for a document boundary


def main() -> None:
    implementations = available_implementations()
    default_impl = "optimal" if "optimal" in implementations else implementations[0]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "corpus",
        nargs="?",
        default=DEFAULT_CORPUS,
        help=f"a file name inside the data folder, or a path to a text file (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--sample-mb",
        type=int,
        default=DEFAULT_SAMPLE_MB,
        help=f"use only the first N megabytes, saved once as a sample file next to the corpus; "
        f"0 means the whole file (default: {DEFAULT_SAMPLE_MB})",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder holding the corpora (default: {DATA_DIR})")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--impl", choices=implementations, default=default_impl, help="NAME profiles NAME_bpe.py")
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="worker processes (default 1, so the profiler can see pre-tokenization)",
    )
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"], help="special tokens")
    parser.add_argument("--sort", choices=["tottime", "cumtime", "ncalls"], default="tottime")
    parser.add_argument("--top", type=int, default=15, help="number of rows to print")
    parser.add_argument("--out", help="save the raw profile to this file")
    parser.add_argument("--no-profile", action="store_true", help="only time the run, with no profiler overhead")
    args = parser.parse_args()

    corpus = find_corpus(args.corpus, args.data_dir)
    if args.sample_mb:
        corpus = make_sample(corpus, args.sample_mb, args.special)
    train_bpe = load_trainer(args.impl)

    # Pass num_processes only to trainers that accept it.
    kwargs = {}
    if "num_processes" in inspect.signature(train_bpe).parameters:
        kwargs["num_processes"] = args.processes
        processes = f"{args.processes} process(es)"
    else:
        processes = "single process (this trainer has no num_processes)"

    size_mb = corpus.stat().st_size / 1e6
    print(f"{args.impl}_bpe.train_bpe | {corpus.name} ({size_mb:.1f} MB) | vocab {args.vocab_size} | {processes}")
    if size_mb > 1000:
        print("note: large corpus, expect a long run; a smaller --sample-mb gives faster feedback")

    profiler = None if args.no_profile else cProfile.Profile()
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    vocab, merges = train_bpe(corpus, args.vocab_size, args.special, **kwargs)
    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - start

    note = "" if args.no_profile else " (includes profiler overhead)"
    print(f"{len(vocab)} tokens, {len(merges)} merges in {elapsed:.2f} s{note}")
    own, worker = peak_memory_mb()
    workers = f", {worker:.0f} MB largest worker" if kwargs.get("num_processes", 1) > 1 else ""
    print(f"peak memory: {own:.0f} MB main process{workers}")

    if profiler:
        if args.out:
            profiler.dump_stats(args.out)
            print(f"profile saved to {args.out}")
        print()
        pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(args.top)


def available_implementations() -> list[str]:
    """Every NAME_bpe.py next to this script is an implementation called NAME."""
    this_file = Path(__file__).resolve()
    names = sorted(
        path.stem.removesuffix("_bpe")
        for path in HERE.glob("*_bpe.py")
        if path.resolve() != this_file  # this script matches the glob too
    )
    if not names:
        sys.exit(f"No *_bpe.py files found in {HERE}")
    return names


def load_trainer(name: str):
    """Return train_bpe from NAME_bpe.py."""
    module = importlib.import_module(f"{name}_bpe")
    if not hasattr(module, "train_bpe"):
        sys.exit(f"{name}_bpe.py has no train_bpe() function")
    return module.train_bpe


def find_corpus(arg: str, data_dir: Path) -> Path:
    """Accept a path, or a bare file name to look up in the data folder."""
    folders = [Path.cwd(), data_dir, HERE]
    for folder in folders:
        candidate = folder / arg
        if candidate.is_file():
            return candidate.resolve()
    looked = "\n  ".join(str(folder) for folder in dict.fromkeys(folders))
    sys.exit(f"Corpus not found: {arg}\nLooked in:\n  {looked}")


def make_sample(corpus: Path, megabytes: int, special_tokens: list[str]) -> Path:
    """Copy the first `megabytes` MB of the corpus to a sibling file, once, and return it."""
    target = megabytes * 1_000_000
    if target >= corpus.stat().st_size:
        return corpus  # the whole file is smaller than the requested sample

    sample = corpus.with_name(f"{corpus.stem}.first{megabytes}mb{corpus.suffix}")
    if sample.exists():
        return sample

    # Copy in blocks so the sample never has to fit in memory. Write to a
    # temporary name first, so an interrupted copy is never mistaken for a sample.
    partial = sample.with_name(sample.name + ".partial")
    with open(corpus, "rb") as source, open(partial, "wb") as destination:
        remaining = target
        while remaining > 0:
            block = source.read(min(COPY_BLOCK, remaining))
            if not block:
                break
            destination.write(block)
            remaining -= len(block)

    # End on a document boundary, so no document is cut in half.
    size = partial.stat().st_size
    window = min(BOUNDARY_WINDOW, size)
    with open(partial, "rb") as f:
        f.seek(size - window)
        tail = f.read(window)
    cut = tail.rfind(special_tokens[0].encode("utf-8")) if special_tokens else -1
    if cut > 0 or (cut == 0 and size > window):
        os.truncate(partial, size - window + cut)
    else:
        # No boundary nearby: at least drop a multi-byte character split at the end.
        os.truncate(partial, size - _incomplete_utf8_tail(tail))

    partial.rename(sample)
    print(f"wrote sample: {sample} ({sample.stat().st_size / 1e6:.1f} MB)")
    return sample


def _incomplete_utf8_tail(data: bytes) -> int:
    """How many trailing bytes belong to a UTF-8 character that was cut off."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0b1100_0000 == 0b1000_0000:  # continuation byte: keep looking for the lead byte
            continue
        if byte < 0b1000_0000:  # ASCII: nothing was cut
            return 0
        needed = 2 if byte >> 5 == 0b110 else 3 if byte >> 4 == 0b1110 else 4
        return back if back < needed else 0
    return 0


def peak_memory_mb() -> tuple[float, float]:
    """Peak resident memory of this process and of its largest worker, in MB."""
    scale = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux kilobytes
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e6
    worker = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale / 1e6
    return own, worker


# The guard matters here too: worker processes re-import this file.
if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
    except BrokenPipeError:
        # Output was piped into something like `head`, which closed early.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)