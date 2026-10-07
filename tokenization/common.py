"""
What the tokenization scripts share: where things are, what they are called,
and a few command-line helpers.

The trainer (trainer/train_bpe.py), the corpus encoder (tokenizer/encode_datasets.py)
and the experiments (tokenizer/experiments/) all import from here, so the data
folder, the artifacts folder, the document separator and the trained
tokenizers' names and sizes are defined once.
"""

import argparse
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # tokenization/ -> project root

# Where the corpora live: the data/ folder at the project root. Set the
# DATA_DIR environment variable (download_data.sh honors the same one) or pass
# --data-dir if yours is somewhere else.
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT / "data"))

# Where the trainer writes the vocabulary, merges and summary of a run, and
# where the other scripts load the trained tokenizers from.
ARTIFACTS_DIR = PROJECT / "tokenization" / "trainer" / "artifacts"

# Separates documents in the corpora, and is the one special token of the
# trained tokenizers.
END_OF_TEXT = "<|endoftext|>"

# The trained tokenizers: name -> vocabulary size. NAME_vocab.pkl and
# NAME_merges.pkl in the artifacts folder are the trainer's outputs for the
# corpus of that name.
TOKENIZERS = {"tinystories": 10_000, "owt": 32_000}


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def add_tokenizer_arguments(parser: argparse.ArgumentParser) -> None:
    """--NAME-vocab and --NAME-merges for every trained tokenizer, defaulting to the artifacts folder."""
    for name, size in TOKENIZERS.items():
        parser.add_argument(
            f"--{name}-vocab",
            type=Path,
            default=ARTIFACTS_DIR / f"{name}_vocab.pkl",
            help=f"vocabulary of the {size:,}-token {name} tokenizer (default: artifacts/{name}_vocab.pkl)",
        )
        parser.add_argument(
            f"--{name}-merges",
            type=Path,
            default=ARTIFACTS_DIR / f"{name}_merges.pkl",
            help=f"merges of the {name} tokenizer (default: artifacts/{name}_merges.pkl)",
        )


def tokenizer_paths(args: argparse.Namespace, name: str) -> tuple[Path, Path]:
    """The vocabulary and merges files given for the tokenizer called name."""
    return getattr(args, f"{name}_vocab"), getattr(args, f"{name}_merges")


def load_tokenizer(args: argparse.Namespace, name: str):
    """
    The tokenizer called name, loaded from the files given on the command line.

    Its vocabulary must have the size TOKENIZERS promises. A tokenizer of
    another size, e.g. an artifacts folder overwritten by a run with a
    different --vocab-size, would otherwise be used silently, and anything
    encoded with it would not match what the scripts' documentation says.
    """
    from tokenization.tokenizer.optimal_tokenizer import OptimalTokenizer  # keeps the trainer free of this import

    vocab_path, merges_path = tokenizer_paths(args, name)
    tokenizer = OptimalTokenizer.from_files(vocab_path, merges_path, [END_OF_TEXT])
    expected = TOKENIZERS[name]
    if len(tokenizer.vocab) != expected:
        raise SystemExit(
            f"{vocab_path} holds {len(tokenizer.vocab):,} tokens, but the {name} tokenizer has {expected:,}. "
            f"Point --{name}-vocab and --{name}-merges at the right run, or change TOKENIZERS in tokenization/common.py."
        )
    return tokenizer
