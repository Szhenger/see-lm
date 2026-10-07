"""
Train a BPE tokenizer on a corpus, save the result, and optionally profile the run.

By default the whole corpus is used, training runs on all cores, and the
vocabulary, merges and a summary (time, memory, longest token) are written to
tokenization/outputs/. With --profile, the run is also traced with cProfile.

Layout this script expects:

    project/
        data/                <- corpora (DATA_DIR below)
        tokenization/
            train_bpe.py   <- this file
            naive_bpe.py     <- each *_bpe.py exposes train_bpe()
            optimal_bpe.py
            outputs/         <- created on first run

Examples (from the project root):
    uv run python tokenization/train_bpe.py                                   # all of TinyStories, vocab 10,000
    uv run python tokenization/train_bpe.py --name tinystories                # same, outputs named tinystories_*
    uv run python tokenization/train_bpe.py owt_train.txt --vocab-size 32000  # all of OpenWebText
    uv run python tokenization/train_bpe.py --sample-mb 100                   # only the first 100 MB, for quick feedback
    uv run python tokenization/train_bpe.py --processes 4
    uv run python tokenization/train_bpe.py --profile                         # cProfile; one process, so pre-tokenization is visible
    uv run python tokenization/train_bpe.py --profile --out train.prof        # then: uvx snakeviz train.prof
    uv run python tokenization/train_bpe.py --impl naive --sample-mb 20 --vocab-size 500
    uv run python tokenization/train_bpe.py other_corpus.txt                  # a different file in data/

A bare file name is looked up in the data folder. A path works too.
--impl NAME uses NAME_bpe.py, so new trainers are picked up automatically.

Files written to tokenization/outputs/ (change with --out-dir). NAME comes from
--name and defaults to the trained file's name, lowercased, without extension;
a sample therefore gets its own name and never overwrites a full-corpus run:
    NAME_vocab.pkl      exact, for loading into the tokenizer later
    NAME_merges.pkl     exact, for loading into the tokenizer later
    NAME_vocab.txt      readable: one "ID<TAB>token" per line
    NAME_merges.txt     readable: one "left<TAB>right" per line, in merge order
    NAME_summary.json   time, memory and longest token, for the writeup
"""

import argparse
import cProfile
import importlib
import inspect
import json
import os
import pickle
import pstats
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Where the corpora live: the data/ folder next to tokenization/.
# Change this line, or pass --data-dir, if yours is somewhere else.
DATA_DIR = HERE.parent / "data"

# Where the trained vocabulary, merges and summary go. Change with --out-dir.
OUT_DIR = HERE / "outputs"

# Trained on when no corpus is given on the command line: the full TinyStories
# training set, as named by the course's download command.
DEFAULT_CORPUS = "TinyStoriesV2-GPT4-train.txt"

# How much of the corpus to use by default, in megabytes. 0 means the whole file.
DEFAULT_SAMPLE_MB = 0

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
        help="use only the first N megabytes, saved once as a sample file next to the corpus; "
        f"0 means the whole file (default: {DEFAULT_SAMPLE_MB})",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder holding the corpora (default: {DATA_DIR})")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--impl", choices=implementations, default=default_impl, help="NAME uses NAME_bpe.py")
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="worker processes (default: all cores; 1 with --profile, so the profiler can see pre-tokenization)",
    )
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"], help="special tokens")
    parser.add_argument("--name", help="prefix of the output files (default: the trained file's name, lowercased)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"folder for the output files (default: {OUT_DIR})")
    parser.add_argument("--no-save", action="store_true", help="do not write the vocabulary, merges or summary")
    parser.add_argument("--profile", action="store_true", help="trace the run with cProfile and print the top functions")
    parser.add_argument("--sort", choices=["tottime", "cumtime", "ncalls"], default="tottime", help="order of the profile table")
    parser.add_argument("--top", type=int, default=15, help="number of profile rows to print")
    parser.add_argument("--out", help="save the raw profile to this file (implies --profile)")
    args = parser.parse_args()
    if args.out:
        args.profile = True

    corpus = find_corpus(args.corpus, args.data_dir)
    if args.sample_mb:
        corpus = make_sample(corpus, args.sample_mb, args.special)
    train_bpe = load_trainer(args.impl)
    name = args.name or corpus.stem.lower()

    # Pass num_processes only to trainers that accept it.
    kwargs = {}
    if "num_processes" in inspect.signature(train_bpe).parameters:
        processes = args.processes or (1 if args.profile else os.cpu_count() or 1)
        kwargs["num_processes"] = processes
        processes_note = f"{processes} process(es)"
    else:
        processes = 1
        processes_note = "single process (this trainer has no num_processes)"

    size_mb = corpus.stat().st_size / 1e6
    print(f"{args.impl}_bpe.train_bpe | {corpus.name} ({size_mb:.1f} MB) | vocab {args.vocab_size} | special {args.special} | {processes_note}")
    if size_mb > 1000 and (args.profile or processes == 1):
        print("note: large corpus on one process, expect a long run; a smaller --sample-mb gives faster feedback")

    profiler = cProfile.Profile() if args.profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    vocab, merges = train_bpe(corpus, args.vocab_size, args.special, **kwargs)
    if profiler:
        profiler.disable()
    elapsed = time.perf_counter() - start

    summary = summarize(vocab, merges, corpus, args, processes, elapsed)
    note = " (includes profiler overhead)" if args.profile else ""
    print(f"{summary['tokens']} tokens, {summary['merges']} merges")
    print(f"time: {elapsed:.1f} s = {elapsed / 60:.2f} min = {elapsed / 3600:.4f} h{note}")
    workers = f", {summary['peak_memory_mb_largest_worker']} MB largest worker" if processes > 1 else ""
    print(f"peak memory: {summary['peak_memory_mb_main']} MB main process{workers}")
    print(f"longest learned token: {summary['longest_token_bytes']} bytes, {summary['longest_token']}")

    if not args.no_save:
        for path in save(vocab, merges, summary, args.out_dir, name):
            print(f"saved: {path}")

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
    similar = sorted(path.name for path in data_dir.glob("*.txt")) if data_dir.is_dir() else []
    hint = f"\nText files in the data folder: {', '.join(similar)}" if similar else ""
    sys.exit(f"Corpus not found: {arg}\nLooked in:\n  {looked}{hint}")


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


def summarize(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    corpus: Path,
    args: argparse.Namespace,
    processes: int,
    elapsed: float,
) -> dict:
    """The numbers the assignment asks about: time, memory, and the longest token."""
    own, worker = peak_memory_mb()

    # The special tokens are given, not learned, so leave them out here.
    specials = {token.encode("utf-8") for token in args.special}
    longest = max((token for token in vocab.values() if token not in specials), key=len, default=b"")

    return {
        "corpus": corpus.name,
        "corpus_mb": round(corpus.stat().st_size / 1e6, 1),
        "sample_mb": args.sample_mb,  # 0 means the whole file
        "implementation": f"{args.impl}_bpe",
        "vocab_size_requested": args.vocab_size,
        "special_tokens": args.special,
        "tokens": len(vocab),
        "merges": len(merges),
        "processes": processes,
        "profiled": args.profile,  # if true, the time includes cProfile overhead
        "seconds": round(elapsed, 2),
        "hours": round(elapsed / 3600, 4),
        "peak_memory_mb_main": round(own),
        "peak_memory_mb_largest_worker": round(worker),
        "longest_token": repr(longest),
        "longest_token_bytes": len(longest),
    }


def save(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    summary: dict,
    out_dir: Path,
    name: str,
) -> list[Path]:
    """
    Write the vocabulary and merges, each in an exact and a readable form, plus the summary.

    The .pkl files keep the bytes exactly and are what the tokenizer should
    load. The .txt files are for reading: tokens are written the way Python
    prints bytes, such as b' the', separated by a tab.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_pkl = out_dir / f"{name}_vocab.pkl"
    merges_pkl = out_dir / f"{name}_merges.pkl"
    vocab_txt = out_dir / f"{name}_vocab.txt"
    merges_txt = out_dir / f"{name}_merges.txt"
    summary_json = out_dir / f"{name}_summary.json"

    with open(vocab_pkl, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_pkl, "wb") as f:
        pickle.dump(merges, f)
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token_id in sorted(vocab):
            f.write(f"{token_id}\t{vocab[token_id]!r}\n")
    with open(merges_txt, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left!r}\t{right!r}\n")
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return [vocab_pkl, merges_pkl, vocab_txt, merges_txt, summary_json]


# The guard matters here too: worker processes re-import this file.
if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
    except BrokenPipeError:
        # Output was piped into something like `head`, which closed early.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(1)
