"""
Encode the training and validation corpora into token IDs for language model training.

Each corpus is encoded with the tokenizer trained on it: TinyStories with the
10K TinyStories tokenizer, OpenWebText with the 32K OpenWebText tokenizer.
The IDs are written as a 1-D NumPy array of dtype uint16 (every ID fits), one
.npy file per corpus, which loads with numpy.load(path, mmap_mode="r").

Jobs (NAME -> DATA_DIR/tokens/NAME.npy), validation sets first so a problem
shows up in seconds rather than after a multi-gigabyte training set:
    tinystories_valid   TinyStoriesV2-GPT4-valid.txt   with tinystories_vocab.pkl / tinystories_merges.pkl
    owt_valid           owt_valid.txt                  with owt_vocab.pkl / owt_merges.pkl
    tinystories_train   TinyStoriesV2-GPT4-train.txt   with the TinyStories tokenizer
    owt_train           owt_train.txt                  with the OpenWebText tokenizer

A job writes NAME.npy.partial and renames it when complete, so a file called
NAME.npy is always a whole array. Next to each array, NAME.json records what
produced it: the source, the vocabulary and merges files and a hash of the
vocabulary, the vocabulary size, the largest ID and the end-of-text ID. A job
is skipped when its array exists and its sidecar records the same vocabulary
hash as the tokenizer about to be used; an array made with another
vocabulary is encoded again, because its IDs would be meaningless under
this one. --force encodes again regardless.

Examples (from the project root):
    uv run python tokenization/tokenizer/encode_datasets.py
    uv run python tokenization/tokenizer/encode_datasets.py owt_valid tinystories_valid
    uv run python tokenization/tokenizer/encode_datasets.py --owt-vocab path/to/owt_vocab.pkl --owt-merges path/to/owt_merges.pkl
    uv run python tokenization/tokenizer/encode_datasets.py --processes 4 --force
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from tokenization.common import DATA_DIR, END_OF_TEXT, add_tokenizer_arguments, load_tokenizer, positive_int, tokenizer_paths
from tokenization.tokenizer.optimal_tokenizer import OptimalTokenizer

# job name -> (corpus file, tokenizer name)
JOBS = {
    "tinystories_valid": ("TinyStoriesV2-GPT4-valid.txt", "tinystories"),
    "owt_valid": ("owt_valid.txt", "owt"),
    "tinystories_train": ("TinyStoriesV2-GPT4-train.txt", "tinystories"),
    "owt_train": ("owt_train.txt", "owt"),
}

MAX_UINT16 = (1 << 16) - 1


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def recorded_vocab_sha256(sidecar: Path) -> str | None:
    """The vocabulary hash a sidecar records, or None if there is no readable sidecar."""
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")).get("vocab_sha256")
    except (OSError, ValueError):
        return None


def encode_job(
    name: str,
    source: Path,
    tokenizer: OptimalTokenizer,
    vocab_path: Path,
    merges_path: Path,
    vocab_sha256: str,
    out_dir: Path,
    processes: int | None,
) -> dict:
    """Encode one corpus to out_dir/NAME.npy and write out_dir/NAME.json; return the summary."""
    output = out_dir / f"{name}.npy"
    partial = out_dir / f"{name}.npy.partial"
    started = time.time()
    try:
        count = tokenizer.encode_file(source, partial, num_processes=processes)
        ids = np.load(partial, mmap_mode="r")
        assert ids.dtype == np.uint16, f"{partial} has dtype {ids.dtype}, expected uint16"
        assert ids.shape == (count,), f"{partial} has shape {ids.shape}, expected ({count},)"
        del ids
        os.replace(partial, output)  # only a complete, checked array ever carries the final name
    finally:
        partial.unlink(missing_ok=True)
    seconds = time.time() - started

    source_bytes = source.stat().st_size
    summary = {
        "source": str(source),
        "source_bytes": source_bytes,
        "output": str(output),
        "dtype": "uint16",
        "tokens": count,
        "bytes_per_token": round(source_bytes / count, 3),
        "seconds": round(seconds, 1),
        "vocab": str(vocab_path),
        "merges": str(merges_path),
        "vocab_sha256": vocab_sha256,
        "vocab_size": len(tokenizer.vocab),
        "max_id": max(tokenizer.vocab),
        "special_tokens": {END_OF_TEXT: tokenizer.id_of[END_OF_TEXT.encode("utf-8")]},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="*", choices=list(JOBS), default=list(JOBS), help="which corpora to encode (default: all)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder holding the corpora (default: {DATA_DIR})")
    parser.add_argument("--out-dir", type=Path, default=None, help="folder for the arrays (default: DATA_DIR/tokens)")
    add_tokenizer_arguments(parser)
    parser.add_argument("--processes", type=positive_int, default=None, help="worker processes (default: all cores)")
    parser.add_argument("--force", action="store_true", help="encode again even if a matching output exists")
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each tokenizer once, up front, so a wrong path or size fails before any encoding.
    needed = dict.fromkeys(JOBS[name][1] for name in args.jobs)
    tokenizers: dict[str, OptimalTokenizer] = {}
    hashes: dict[str, str] = {}
    for tok_name in needed:
        tokenizers[tok_name] = load_tokenizer(args, tok_name)
        vocab_path, _ = tokenizer_paths(args, tok_name)
        hashes[tok_name] = sha256_of(vocab_path)
        largest = max(tokenizers[tok_name].vocab)
        if largest > MAX_UINT16:
            raise SystemExit(f"{tok_name} tokenizer has IDs up to {largest}, which do not fit uint16")
        print(f"{tok_name} tokenizer: {len(tokenizers[tok_name].vocab):,} tokens from {vocab_path}")

    for name in args.jobs:
        corpus, tok_name = JOBS[name]
        source = args.data_dir / corpus
        output = out_dir / f"{name}.npy"
        if output.exists() and not args.force:
            recorded = recorded_vocab_sha256(out_dir / f"{name}.json")
            if recorded == hashes[tok_name]:
                print(f"{name}: {output} exists and was made with this vocabulary, skipped (--force encodes again)")
                continue
            why = "has no sidecar" if recorded is None else "was made with a different vocabulary"
            print(f"{name}: {output} exists but {why}, encoding again")
        if not source.is_file():
            print(f"{name}: {source} not found, skipped")
            continue
        if source.stat().st_size == 0:
            print(f"{name}: {source} is empty, skipped")
            continue
        print(f"{name}: encoding {source} ({source.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        summary = encode_job(name, source, tokenizers[tok_name], *tokenizer_paths(args, tok_name), hashes[tok_name], out_dir, args.processes)
        print(f"{name}: {summary['tokens']:,} tokens, {summary['bytes_per_token']} bytes/token, {summary['seconds']} s -> {output}", flush=True)


if __name__ == "__main__":
    main()
