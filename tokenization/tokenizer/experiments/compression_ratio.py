"""
Compression ratio of the trained tokenizers, in bytes per token.

Samples documents from the TinyStories and OpenWebText validation sets,
encodes each with the TinyStories tokenizer (10K vocabulary) and the
OpenWebText tokenizer (32K vocabulary), and reports how many bytes of text
each token stands for. A tokenizer is also run on the other corpus, to show
how much is lost when the training and target text differ.

Documents are separated by <|endoftext|> in the corpora. Each sampled
document is encoded without that separator, and its size is the length of
its UTF-8 encoding.

Examples (from the project root):
    uv run python tokenization/tokenizer/experiments/compression_ratio.py
    uv run python tokenization/tokenizer/experiments/compression_ratio.py --docs 50 --seed 1
    uv run python tokenization/tokenizer/experiments/compression_ratio.py --owt-vocab path/to/owt_vocab.pkl --owt-merges path/to/owt_merges.pkl

Writes compression_ratio.json next to this script (change with --out).
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from tokenization.common import DATA_DIR, END_OF_TEXT, PROJECT, TOKENIZERS, add_tokenizer_arguments, load_tokenizer, positive_int, tokenizer_paths
from tokenization.tokenizer.optimal_tokenizer import OptimalTokenizer

HERE = Path(__file__).resolve().parent

# tokenizer name -> the validation set of the corpus it was trained on
CORPORA = {"tinystories": "TinyStoriesV2-GPT4-valid.txt", "owt": "owt_valid.txt"}


def sample_documents(path: Path, count: int, rng: random.Random) -> list[str]:
    """`count` documents chosen uniformly from the file, which separates documents with END_OF_TEXT."""
    documents = [d for d in path.read_bytes().split(END_OF_TEXT.encode("utf-8")) if d]
    if count > len(documents):
        raise SystemExit(f"{path} has {len(documents):,} documents, cannot sample {count:,}")
    return [documents[i].decode("utf-8") for i in sorted(rng.sample(range(len(documents)), count))]


def measure(tokenizer: OptimalTokenizer, documents: list[str]) -> dict:
    """Bytes and tokens per document, plus two aggregates of bytes per token."""
    per_document = []
    for text in documents:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text, "round trip failed"
        n_bytes, n_tokens = len(text.encode("utf-8")), len(ids)
        per_document.append({"bytes": n_bytes, "tokens": n_tokens, "bytes_per_token": round(n_bytes / n_tokens, 3)})
    total_bytes = sum(d["bytes"] for d in per_document)
    total_tokens = sum(d["tokens"] for d in per_document)
    return {
        # Headline: every byte counts the same, so long documents weigh more.
        "bytes_per_token": round(total_bytes / total_tokens, 3),
        # Every document counts the same, whatever its length.
        "mean_of_document_ratios": round(statistics.mean(d["bytes_per_token"] for d in per_document), 3),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "documents": per_document,
    }


def relative(path: Path) -> str:
    """A path relative to the project root when inside it, so the results do not depend on the machine."""
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=positive_int, default=10, help="documents to sample from each corpus (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the sampling (default: 0)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"folder holding the corpora (default: {DATA_DIR})")
    add_tokenizer_arguments(parser)
    parser.add_argument("--out", type=Path, default=HERE / "compression_ratio.json", help="where to write the results")
    args = parser.parse_args()

    tokenizers = {name: load_tokenizer(args, name) for name in TOKENIZERS}
    rng = random.Random(args.seed)
    samples = {name: sample_documents(args.data_dir / file, args.docs, rng) for name, file in CORPORA.items()}

    results = {
        "seed": args.seed,
        "documents_per_corpus": args.docs,
        "corpora": {name: relative(args.data_dir / file) for name, file in CORPORA.items()},
        "tokenizers": {
            name: {"vocab_size": len(t.vocab), "vocab": relative(tokenizer_paths(args, name)[0]), "merges": relative(tokenizer_paths(args, name)[1])}
            for name, t in tokenizers.items()
        },
        # Each tokenizer on the corpus it was trained on.
        "own_corpus": {name: measure(tokenizers[name], samples[name]) for name in tokenizers},
        # Each tokenizer on the other corpus.
        "other_corpus": {
            f"{tok}_tokenizer_on_{corpus}": measure(tokenizers[tok], samples[corpus])
            for tok in tokenizers
            for corpus in samples
            if tok != corpus
        },
    }
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"{args.docs} documents per corpus, seed {args.seed}\n")
    print(f"{'tokenizer':<14}{'vocab':>7}  {'corpus':<12}{'bytes/token':>12}{'mean of doc ratios':>20}{'tokens':>9}")
    for name in tokenizers:
        r = results["own_corpus"][name]
        print(f"{name:<14}{len(tokenizers[name].vocab):>7}  {name:<12}{r['bytes_per_token']:>12.3f}{r['mean_of_document_ratios']:>20.3f}{r['total_tokens']:>9}")
    print("\non the other corpus:")
    for key, r in results["other_corpus"].items():
        tok, corpus = key.split("_tokenizer_on_")
        print(f"{tok:<14}{len(tokenizers[tok].vocab):>7}  {corpus:<12}{r['bytes_per_token']:>12.3f}{r['mean_of_document_ratios']:>20.3f}{r['total_tokens']:>9}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
