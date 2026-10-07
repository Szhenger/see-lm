"""
Train a BPE tokenizer on a corpus, save the result, and optionally profile the run.

By default the whole corpus is used, training runs on all cores, and the
vocabulary, merges and a summary (time, memory, longest token) are written to
tokenization/trainer/artifacts/. With --profile, the run is also traced with cProfile:
the main process directly, and each worker process through a hook in the
trainer, so parallel pre-tokenization shows up too.

Layout this script expects:

    project/
        data/                <- corpora (DATA_DIR below)
        tokenization/
            trainer/
                train_bpe.py     <- this file
                naive_bpe.py     <- each *_bpe.py exposes train_bpe()
                optimal_bpe.py
                artifacts/       <- created on first run
            tokenizer/

Examples (from the project root):
    uv run python tokenization/trainer/train_bpe.py                                   # all of TinyStories, vocab 10,000 -> tinystories_*
    uv run python tokenization/trainer/train_bpe.py owt_train.txt --vocab-size 32000  # all of OpenWebText -> owt_*
    uv run python tokenization/trainer/train_bpe.py --sample-mb 100                   # only the first 100 MB, for quick feedback
    uv run python tokenization/trainer/train_bpe.py --processes 4
    uv run python tokenization/trainer/train_bpe.py --profile                         # cProfile tables for the main process and the workers
    uv run python tokenization/trainer/train_bpe.py --profile --out train.prof        # then: uvx snakeviz train.prof (workers: train.prof.workers.prof)
    uv run python tokenization/trainer/train_bpe.py --impl naive --sample-mb 20 --vocab-size 500
    uv run python tokenization/trainer/train_bpe.py other_corpus.txt                  # a different file in data/

A bare file name is looked up in the data folder. A path works too.
--impl NAME uses NAME_bpe.py, so new trainers are picked up automatically.

Files written to tokenization/trainer/artifacts/ (change with --out-dir). NAME is
--name, or tinystories or owt for the course corpora, or else the trained file's
name, lowercased. A sample run appends .firstNmb to NAME and a profiled run
appends .profiled, so neither overwrites a clean full-corpus run:
    NAME_vocab.pkl      exact, for loading into the tokenizer later
    NAME_merges.pkl     exact, for loading into the tokenizer later
    NAME_vocab.txt      readable: one "ID<TAB>token" per line
    NAME_merges.txt     readable: one "left<TAB>right" per line, in merge order
    NAME_summary.json   time, memory and longest token, for the writeup
"""

import argparse
import contextlib
import cProfile
import importlib
import inspect
import json
import os
import pickle
import pstats
import resource
import sys
import tempfile
import time
from pathlib import Path

from tokenization.common import ARTIFACTS_DIR, DATA_DIR, END_OF_TEXT, non_negative_int, positive_int

HERE = Path(__file__).resolve().parent

# Short output names for the course corpora; anything else is named after its file.
SHORT_NAMES = {"tinystoriesv2-gpt4-train": "tinystories", "owt_train": "owt"}

# Where the trained vocabulary, merges and summary go. Change with --out-dir.
# The data folder and this one are defined once, in tokenization/common.py.
OUT_DIR = ARTIFACTS_DIR

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
        type=non_negative_int,
        default=DEFAULT_SAMPLE_MB,
        help="use only the first N megabytes, saved once as a sample file next to the corpus; "
        f"0 means the whole file (default: {DEFAULT_SAMPLE_MB})",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder holding the corpora (default: {DATA_DIR})")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--impl", choices=implementations, default=default_impl, help="NAME uses NAME_bpe.py")
    parser.add_argument("--processes", type=positive_int, default=None, help="worker processes (default: all cores)")
    parser.add_argument("--special", nargs="*", default=[END_OF_TEXT], help="special tokens")
    parser.add_argument(
        "--name",
        help="prefix of the output files (default: tinystories, owt, or the trained file's name); a sample run appends .firstNmb",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"folder for the output files (default: {OUT_DIR})")
    parser.add_argument("--no-save", action="store_true", help="do not write the vocabulary, merges or summary")
    parser.add_argument("--profile", action="store_true", help="trace the run with cProfile and print the top functions")
    parser.add_argument("--sort", choices=["tottime", "cumtime", "ncalls"], default="tottime", help="order of the profile table")
    parser.add_argument("--top", type=int, default=15, help="number of profile rows to print")
    parser.add_argument(
        "--out",
        help="save the raw profile to this file (implies --profile); with workers, their combined profile goes to FILE.workers.prof",
    )
    args = parser.parse_args()
    if args.out:
        args.profile = True

    whole = find_corpus(args.corpus, args.data_dir)
    corpus = make_sample(whole, args.sample_mb, args.special) if args.sample_mb else whole
    sample_mb = args.sample_mb if corpus != whole else 0  # 0: the whole file was used
    name = args.name or short_name(whole)
    if sample_mb:
        name += f".first{sample_mb}mb"  # a sample never overwrites a full-corpus run
    if args.profile:
        name += ".profiled"  # profiler overhead inflates the timings, so keep them apart from a clean run
    trainer = load_trainer(args.impl)
    train_bpe = trainer.train_bpe
    # Trainers that define this name make each pool worker profile itself and
    # write worker-<pid>.prof into the folder the variable names.
    profile_var = getattr(trainer, "PROFILE_DIR_VAR", None)

    # Pass num_processes only to trainers that accept it.
    kwargs = {}
    if "num_processes" in inspect.signature(train_bpe).parameters:
        processes = args.processes or os.cpu_count() or 1
        kwargs["num_processes"] = processes
        processes_note = f"{processes} process(es)"
    else:
        processes = 1
        processes_note = "single process (this trainer has no num_processes)"

    size_mb = corpus.stat().st_size / 1e6
    print(f"{args.impl}_bpe.train_bpe | {corpus.name} ({size_mb:.1f} MB) | vocab {args.vocab_size} | special {args.special} | {processes_note}")
    if size_mb > 1000 and processes == 1:
        print("note: large corpus on one process, expect a long run; a smaller --sample-mb gives faster feedback")

    profiler = cProfile.Profile() if args.profile else None
    profile_workers = bool(profiler and processes > 1 and profile_var)
    if profile_var:
        os.environ.pop(profile_var, None)  # a value left in the shell must not profile a plain run
    worker_folder = tempfile.TemporaryDirectory(prefix="bpe-profile-") if profile_workers else contextlib.nullcontext()
    with worker_folder as profile_dir:
        if profile_workers:
            os.environ[profile_var] = profile_dir  # inherited by the spawned workers
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        vocab, merges = train_bpe(corpus, args.vocab_size, args.special, **kwargs)
        if profiler:
            profiler.disable()
        elapsed = time.perf_counter() - start
        if profile_workers:
            os.environ.pop(profile_var, None)
            worker_stats, worker_count = load_worker_stats(Path(profile_dir))  # before the folder is removed
        else:
            worker_stats, worker_count = None, 0

    summary = summarize(vocab, merges, corpus, args, processes, sample_mb, elapsed)
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
            if worker_stats:
                workers_out = f"{args.out}.workers.prof"
                worker_stats.dump_stats(workers_out)
                print(f"worker profile saved to {workers_out}")
        if profile_workers and not worker_stats:
            print("\nnote: no worker profiles came back; if the trainer used its pool, pre-tokenization is missing below")
        print(f"\n=== main process{' (merge loop; pre-tokenization ran in the workers)' if worker_stats else ''}")
        pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(args.top)
        if worker_stats:
            # cProfile measures each worker's wall time, not CPU time.
            print(f"=== {worker_count} worker process(es), combined: each worker's own seconds, summed, so more than the wall time")
            worker_stats.strip_dirs().sort_stats(args.sort).print_stats(args.top)


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
    """Return the module NAME_bpe.py, checked to have a train_bpe() function."""
    module = importlib.import_module(f"{name}_bpe")
    if not hasattr(module, "train_bpe"):
        sys.exit(f"{name}_bpe.py has no train_bpe() function")
    return module


def short_name(corpus: Path) -> str:
    """Prefix for the output files: tinystories, owt, or the file's own name."""
    stem = corpus.stem.lower()
    return SHORT_NAMES.get(stem, stem)


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


def load_worker_stats(profile_dir: Path) -> tuple[pstats.Stats | None, int]:
    """Combine the worker-<pid>.prof files the trainer's workers wrote, and count them.

    A file that cannot be read is reported and skipped: the profile is a side
    channel, and the trained vocabulary must still be saved after this.
    """
    stats = None
    loaded = 0
    for file in sorted(profile_dir.glob("worker-*.prof")):
        try:
            if stats is None:
                stats = pstats.Stats(str(file))
            else:
                stats.add(str(file))
        except Exception as error:  # noqa: BLE001 - anything wrong with the file
            print(f"note: skipped unreadable worker profile {file.name}: {error}")
            continue
        loaded += 1
    if stats is not None:
        stats.files = []  # otherwise print_stats lists these temporary paths as a header
    return stats, loaded


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
    sample_mb: int,
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
        "sample_mb": sample_mb,  # 0 means the whole file was used
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
