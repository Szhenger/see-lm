"""
Parameter and FLOPs accounting for the Transformer language model in see_lm.py.

Nearly all of a forward pass is matrix multiplies, and A (m x n) times B (n x p)
costs 2mnp FLOPs, so the count is the list of matmuls of one pass over a sequence
of L tokens, with d = d_model, h = num_heads, f = d_ff and V = vocab_size:

    per layer   qkv_proj      (L x d) @ (d x 3d)              6 L d^2
                scores        (L x d/h) @ (d/h x L), h heads  2 L^2 d
                weights @ V   (L x L) @ (L x d/h), h heads    2 L^2 d
                output_proj   (L x d) @ (d x d)               2 L d^2
                ffn           w1, w3: (L x d) @ (d x f)       6 L d f
                              w2:     (L x f) @ (f x d)
    once        lm_head       (L x d) @ (d x V)               2 L d V

The causal mask does not reduce the count: the full score matrix is computed and
then masked. Softmax, RMSNorm, RoPE, SiLU, the gate product and the residual adds
are elementwise or reductions and are not counted, and the embedding is a gather.

Parameters: token_embeddings V d; per layer 4 d^2 (qkv_proj and output_proj),
3 d f (w1, w2, w3) and 2 d (the ln1 and ln2 gains); ln_final d; lm_head d V,
untied from the embedding. There are no biases and RoPE has no parameters.

Both counts are checked against the model itself on the meta device, where
nothing is allocated or computed: the parameters by instantiating TransformerLM,
the FLOPs by tracing a forward pass with torch.utils.flop_counter.

Examples (from the project root):
    uv run python architecture/transformer/transformer_accounting.py
    uv run python architecture/transformer/transformer_accounting.py --models small xl
    uv run python architecture/transformer/transformer_accounting.py --vocab-size 32000 --context-length 256

Writes transformer_accounting.json next to this script (change with --out).
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from architecture.layers.positionwise_feedforward import SwiGLU
from architecture.transformer.see_lm import TransformerLM

HERE = Path(__file__).resolve().parent

# GPT-2 sizes: name -> (num_layers, d_model, num_heads). Vocabulary and context
# length are shared, and d_ff follows the SwiGLU default (8/3 d_model rounded up
# to a multiple of 64), which gives the 4,288 of the XL configuration.
GPT2 = {"small": (12, 768, 12), "medium": (24, 1024, 16), "large": (36, 1280, 20), "xl": (48, 1600, 25)}
VOCAB_SIZE = 50_257
CONTEXT_LENGTH = 1_024
ROPE_THETA = 10_000.0

BYTES_PER_FP32 = 4


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


def gpt2_config(name: str, vocab_size: int, context_length: int) -> ModelConfig:
    num_layers, d_model, num_heads = GPT2[name]
    d_ff = SwiGLU(d_model, device=torch.device("meta")).d_ff
    return ModelConfig(vocab_size, context_length, num_layers, d_model, num_heads, d_ff)


def count_parameters(cfg: ModelConfig) -> dict[str, int]:
    """Trainable parameters by component, summed over all layers, plus the total."""
    d, f, V, n = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.num_layers
    counts = {
        "token_embeddings": V * d,
        "attention": n * 4 * d * d,
        "feed_forward": n * 3 * d * f,
        "rms_norm": (2 * n + 1) * d,
        "lm_head": d * V,
    }
    counts["total"] = sum(counts.values())
    return counts


def count_flops(cfg: ModelConfig, seq_len: int) -> dict[str, int]:
    """Matmul FLOPs of one forward pass over seq_len tokens, by matmul, summed over all layers, plus the total."""
    L, d, f, V, n = seq_len, cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.num_layers
    counts = {
        "qkv_proj": n * 6 * L * d * d,
        "scores": n * 2 * L * L * d,
        "weights_v": n * 2 * L * L * d,
        "output_proj": n * 2 * L * d * d,
        "feed_forward": n * 6 * L * d * f,
        "lm_head": 2 * L * d * V,
    }
    counts["total"] = sum(counts.values())
    return counts


# How the matmuls are grouped when comparing models; the L^2 terms are the
# only ones that grow faster than linearly with the sequence length.
GROUPS = {
    "attn projections": ("qkv_proj", "output_proj"),
    "attn scores (L^2)": ("scores", "weights_v"),
    "feed-forward": ("feed_forward",),
    "lm_head": ("lm_head",),
}


def flop_shares(flops: dict[str, int]) -> dict[str, float]:
    return {group: sum(flops[k] for k in keys) / flops["total"] for group, keys in GROUPS.items()}


def verify(cfg: ModelConfig, parameters: int, flops: int) -> None:
    """Check the formulas against TransformerLM on the meta device, where the model costs nothing to build or run."""
    meta = torch.device("meta")
    model = TransformerLM(**asdict(cfg), rope_theta=ROPE_THETA, device=meta)
    actual = sum(p.numel() for p in model.parameters())
    if actual != parameters:
        raise AssertionError(f"model has {actual:,} parameters, accounting says {parameters:,}")
    # Only mm and bmm occur here and both have counter formulas, so the totals must agree exactly.
    with FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, cfg.context_length, dtype=torch.long, device=meta))
    if counter.get_total_flops() != flops:
        raise AssertionError(f"forward pass is {counter.get_total_flops():,} FLOPs, accounting says {flops:,}")


def account(cfg: ModelConfig) -> dict:
    parameters = count_parameters(cfg)
    flops = count_flops(cfg, cfg.context_length)
    verify(cfg, parameters["total"], flops["total"])
    return {
        "config": asdict(cfg),
        "parameters": parameters,
        "fp32_bytes": parameters["total"] * BYTES_PER_FP32,
        "forward_flops": flops,
        "forward_flop_shares": {group: round(share, 4) for group, share in flop_shares(flops).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(GPT2), default=list(GPT2), help="GPT-2 sizes to account for (default: all)")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help=f"vocabulary size (default: {VOCAB_SIZE:,})")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH, help=f"context length, also the length of the input sequence (default: {CONTEXT_LENGTH:,})")
    parser.add_argument("--out", type=Path, default=HERE / "transformer_accounting.json", help="where to write the results")
    args = parser.parse_args()

    results = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "flops_per_matmul": "2 m n p for (m x n) @ (n x p); elementwise work is not counted",
        "bytes_per_parameter": BYTES_PER_FP32,
        "models": {name: account(gpt2_config(name, args.vocab_size, args.context_length)) for name in args.models},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"vocab_size {args.vocab_size:,}, context_length {args.context_length:,}, one forward pass over {args.context_length:,} tokens\n")
    print(f"{'model':<8}{'layers':>7}{'d_model':>9}{'heads':>7}{'d_ff':>7}{'parameters':>16}{'fp32 GB':>9}{'fp32 GiB':>10}{'forward TFLOPs':>16}")
    for name, r in results["models"].items():
        c = r["config"]
        print(
            f"{name:<8}{c['num_layers']:>7}{c['d_model']:>9}{c['num_heads']:>7}{c['d_ff']:>7}"
            f"{r['parameters']['total']:>16,}{r['fp32_bytes'] / 1e9:>9.2f}{r['fp32_bytes'] / 2**30:>10.2f}"
            f"{r['forward_flops']['total'] / 1e12:>16.3f}"
        )

    print("\nshare of forward FLOPs:")
    print(f"{'model':<8}" + "".join(f"{group:>20}" for group in GROUPS))
    for name, r in results["models"].items():
        print(f"{name:<8}" + "".join(f"{100 * share:>19.1f}%" for share in r["forward_flop_shares"].values()))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
