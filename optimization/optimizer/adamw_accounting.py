"""
Memory and FLOPs accounting for one AdamW training step of the Transformer
language model in see_lm.py, with every tensor in float32 (4 bytes).

With P = parameters, B = batch_size, L = context_length, d = d_model,
h = num_heads, f = d_ff (8/3 d, which SwiGLU rounds up to a multiple of 64)
and V = vocab_size, the resident tensors of one step are

    parameters       P
    gradients        P
    optimizer state  2 P                AdamW's first and second moments
    activations      per layer   8 B L d    ln1, ln2, Q, K, V, attention output, output_proj, w2
                                 4 B L f    w1, w3, SiLU(w1), SiLU(w1) * w3
                                 2 B h L^2  QK^T and its softmax
                     once        B L d      final RMSNorm
                                 2 B L V    logits and the cross-entropy on them

so peak memory is 4 (4 P + A) bytes with A the activation elements, which is
linear in B: a * B + b with a = 4 A(B=1) and b = 16 P. Only the tensors listed
are counted as activations; the residual stream, the embedding output, RoPE
tables and the causal mask are not. What autograd actually saves for backward
is a different set and is reported beside the count, not checked against it.

FLOPs of one step: a forward pass over B sequences of L tokens (the matmul
count of transformer_accounting, B times the single-sequence count since the
attention terms are B L^2, not (B L)^2), a backward pass at twice the forward
(Kaplan et al., Hoffmann et al.), and the AdamW update, about 12 elementwise
FLOPs per parameter. Training time at a given MFU is
steps * step_flops / (mfu * peak_flops_per_second).

Parameters and FLOPs are checked against TransformerLM on the meta device as
in transformer_accounting, over a batch of three sequences so that the batch
scaling of the FLOPs is checked as well.

Examples (from the project root):
    uv run python optimization/optimizer/adamw_accounting.py
    uv run python optimization/optimizer/adamw_accounting.py --models small xl --batch-size 32
    uv run python optimization/optimizer/adamw_accounting.py --memory-gb 40 --mfu 0.4 --steps 100000

Writes adamw_accounting.json next to this script (change with --out).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from architecture.transformer.see_lm import TransformerLM
from architecture.transformer.transformer_accounting import (
    BYTES_PER_FP32,
    ModelConfig,
    add_model_arguments,
    count_flops,
    count_parameters,
    model_configs,
    verify,
)
from optimization.learner.cross_entropy import cross_entropy

HERE = Path(__file__).resolve().parent

# About a dozen elementwise FLOPs per parameter in AdamW.step: the decay
# multiply, the two moment updates, sqrt, + eps, the division and the scaled add.
OPTIMIZER_FLOPS_PER_PARAM = 12
BACKWARD_TO_FORWARD = 2

# Defaults for (b) and (d): an 80 GB accelerator, and an H100's fp32 (TF32) peak.
MEMORY_GB = 80
PEAK_TFLOPS = 495
MFU = 0.5
STEPS = 400_000
BATCH_SIZE = 1_024

SECONDS_PER_HOUR = 3_600


def count_activations(cfg: ModelConfig, batch_size: int) -> dict[str, int]:
    """Activation elements of one forward pass at batch_size, by tensor, summed over all layers, plus the total."""
    B, L, d, h, f, V, n = batch_size, cfg.context_length, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.vocab_size, cfg.num_layers
    counts = {
        "rms_norms": n * 2 * B * L * d,
        "qkv_proj": n * 3 * B * L * d,
        "scores": n * B * h * L * L,
        "softmax": n * B * h * L * L,
        "weights_v": n * B * L * d,
        "output_proj": n * B * L * d,
        "w1": n * B * L * f,
        "w3": n * B * L * f,
        "silu": n * B * L * f,
        "gate_product": n * B * L * f,
        "w2": n * B * L * d,
        "final_rms_norm": B * L * d,
        "logits": B * L * V,
        "cross_entropy": B * L * V,
    }
    counts["total"] = sum(counts.values())
    return counts


def memory_bytes(cfg: ModelConfig, batch_size: int) -> dict[str, int]:
    """Peak memory of one step at batch_size, by kind, plus the total."""
    P = count_parameters(cfg)["total"]
    counts = {
        "parameters": P * BYTES_PER_FP32,
        "gradients": P * BYTES_PER_FP32,
        "optimizer_state": 2 * P * BYTES_PER_FP32,
        "activations": count_activations(cfg, batch_size)["total"] * BYTES_PER_FP32,
    }
    counts["total"] = sum(counts.values())
    return counts


def memory_line(cfg: ModelConfig) -> tuple[int, int]:
    """(a, b) of peak bytes = a * batch_size + b: the per-sample activations and the batch-independent rest."""
    return memory_bytes(cfg, 1)["activations"], memory_bytes(cfg, 0)["total"]


def max_batch_size(a: int, b: int, memory_budget: int) -> int:
    """Largest batch_size with a * batch_size + b within memory_budget."""
    return max(0, (memory_budget - b) // a)


def step_flops(cfg: ModelConfig, batch_size: int) -> dict[str, int]:
    """FLOPs of one AdamW step at batch_size: forward, backward and the update, plus the total."""
    forward = batch_size * count_flops(cfg, cfg.context_length)["total"]
    counts = {
        "forward": forward,
        "backward": BACKWARD_TO_FORWARD * forward,
        "optimizer": OPTIMIZER_FLOPS_PER_PARAM * count_parameters(cfg)["total"],
    }
    counts["total"] = sum(counts.values())
    return counts


def training_hours(cfg: ModelConfig, batch_size: int, steps: int, mfu: float, peak_flops: float) -> dict[str, float]:
    """Wall-clock hours for `steps` steps at `mfu` of `peak_flops` per second, with the 6 P D estimate beside it."""
    total = steps * step_flops(cfg, batch_size)["total"]
    tokens = steps * batch_size * cfg.context_length
    six_pd = 6 * count_parameters(cfg)["total"] * tokens
    sustained = mfu * peak_flops
    return {
        "total_flops": total,
        "tokens": tokens,
        "six_pd_flops": six_pd,
        "hours": total / sustained / SECONDS_PER_HOUR,
        "six_pd_hours": six_pd / sustained / SECONDS_PER_HOUR,
    }


def autograd_saved_bytes(model: TransformerLM, batch_size: int) -> int:
    """Bytes autograd saves for the backward of one step beyond the model's parameters and buffers, traced on the meta device."""
    resident = {id(t) for t in model.parameters()} | {id(t) for t in model.buffers()}
    saved: dict[int, torch.Tensor] = {}  # held by id so a freed tensor's id is not reused

    def pack(t: torch.Tensor) -> torch.Tensor:
        # A view (a weight's .mT, the 2-d fold of a matmul input) keeps its whole base alive, so the base is what is counted, once.
        root = t if t._base is None else t._base
        if id(root) not in resident:
            saved[id(root)] = root
        return t

    meta = torch.device("meta")
    ids = torch.zeros(batch_size, model.context_length, dtype=torch.long, device=meta)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        cross_entropy(model(ids), ids)
    return sum(t.numel() * t.element_size() for t in saved.values())


def account(cfg: ModelConfig, args: argparse.Namespace) -> dict:
    parameters = count_parameters(cfg)
    model = verify(cfg, batch_size=3)

    a, b = memory_line(cfg)
    budget = int(args.memory_gb * 1e9)
    return {
        "config": asdict(cfg),
        "d_ff_exact": 8 * cfg.d_model / 3,
        "parameters": parameters,
        "memory": {
            "per_sample_activation_bytes": a,
            "batch_independent_bytes": b,
            "activations_per_sample": count_activations(cfg, 1),
            "at_batch_size": memory_bytes(cfg, args.batch_size),
            "autograd_saved_bytes_per_sample": autograd_saved_bytes(model, 1),
            "budget_bytes": budget,
            "max_batch_size": max_batch_size(a, b, budget),
        },
        "step_flops": step_flops(cfg, args.batch_size),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "mfu": args.mfu,
            "peak_tflops": args.peak_tflops,
            **training_hours(cfg, args.batch_size, args.steps, args.mfu, args.peak_tflops * 1e12),
        },
    }


def gb(n: float) -> str:
    return f"{n / 1e9:,.2f} GB"


def report(name: str, r: dict) -> None:
    c, m, t = r["config"], r["memory"], r["training"]
    n, d, h, f, V, L = c["num_layers"], c["d_model"], c["num_heads"], c["d_ff"], c["vocab_size"], c["context_length"]
    P = r["parameters"]["total"]
    print(f"=== {name}: {n} layers, d_model {d}, {h} heads, d_ff {f} (8/3 d_model = {r['d_ff_exact']:.2f}), vocab {V:,}, context {L:,} ===\n")

    print("(a) peak memory of one AdamW step, float32, as elements x 4 bytes")
    print("    parameters       P = V d + n (4 d^2 + 3 d f + 2 d) + d + d V" + f" = {P:,}")
    print("    gradients        P")
    print("    optimizer state  2 P")
    print("    activations      A = B [ n (8 L d + 4 L f + 2 h L^2) + L d + 2 L V ]"
          "  (with f = 8/3 d: n (56/3 L d + 2 h L^2) + L d + 2 L V)")
    print("    total            4 (4 P + A) bytes")
    print(f"\n    per-sample activations (B = 1), {m['activations_per_sample']['total']:,} elements:")
    for key, elements in m["activations_per_sample"].items():
        if key != "total":
            print(f"      {key:<16}{elements:>18,}")
    print(f"    autograd saves {gb(m['autograd_saved_bytes_per_sample'])} per sample beyond parameters and buffers (a different set; not the count above)")
    at = m["at_batch_size"]
    print(f"\n    at batch_size {t['batch_size']:,}: parameters {gb(at['parameters'])}, gradients {gb(at['gradients'])}, "
          f"optimizer state {gb(at['optimizer_state'])}, activations {gb(at['activations'])}, total {gb(at['total'])}\n")

    print("(b) peak bytes = a * batch_size + b")
    print(f"    a = {m['per_sample_activation_bytes']:,} bytes ({gb(m['per_sample_activation_bytes'])}) per sample of activations")
    print(f"    b = {m['batch_independent_bytes']:,} bytes ({gb(m['batch_independent_bytes'])}) = 16 P for parameters, gradients and both moments")
    print(f"    max batch_size within {gb(m['budget_bytes'])}: {m['max_batch_size']}\n")

    s = r["step_flops"]
    print(f"(c) FLOPs of one AdamW step at batch_size {t['batch_size']:,}")
    print("    forward   B x matmul FLOPs of one sequence: B [ n (8 L d^2 + 4 L^2 d + 6 L d f) + 2 L d V ]" + f" = {s['forward']:.3e}")
    print(f"    backward  {BACKWARD_TO_FORWARD} x forward = {s['backward']:.3e}")
    print(f"    optimizer {OPTIMIZER_FLOPS_PER_PARAM} P elementwise = {s['optimizer']:.3e} ({100 * s['optimizer'] / s['total']:.4f}% of the step)")
    print(f"    total     {s['total']:.3e}\n")

    print(f"(d) {t['steps']:,} steps at batch_size {t['batch_size']:,} on {t['peak_tflops']} TFLOP/s at {100 * t['mfu']:.0f}% MFU")
    print(f"    {t['tokens']:.3e} tokens, {t['total_flops']:.3e} FLOPs (6 P D estimate: {t['six_pd_flops']:.3e})")
    print(f"    {t['hours']:,.1f} hours = {t['hours'] / 24:,.1f} days (6 P D estimate: {t['six_pd_hours']:,.1f} hours)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser, default_models=["xl"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"batch size for (c) and (d) (default: {BATCH_SIZE:,})")
    parser.add_argument("--memory-gb", type=float, default=MEMORY_GB, help=f"memory budget for (b), in GB of 1e9 bytes (default: {MEMORY_GB})")
    parser.add_argument("--steps", type=int, default=STEPS, help=f"training steps for (d) (default: {STEPS:,})")
    parser.add_argument("--mfu", type=float, default=MFU, help=f"model FLOPs utilization for (d) (default: {MFU})")
    parser.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS, help=f"hardware peak in TFLOP/s for (d) (default: {PEAK_TFLOPS})")
    parser.add_argument("--out", type=Path, default=HERE / "adamw_accounting.json", help="where to write the results")
    args = parser.parse_args()

    results = {
        "bytes_per_element": BYTES_PER_FP32,
        "backward_to_forward": BACKWARD_TO_FORWARD,
        "optimizer_flops_per_parameter": OPTIMIZER_FLOPS_PER_PARAM,
        "models": {name: account(cfg, args) for name, cfg in model_configs(args).items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    for name, r in results["models"].items():
        report(name, r)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
