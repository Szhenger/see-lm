"""
Sweep the peak learning rate of the TinyStories model and report what it does
to the loss: the learning curves of every rate on one figure, a table of their
final and best losses, the rates that diverged, and where the best rate sits
against the edge of divergence.

Each rate is one run of tiny_stories.py, named ts-lr<rate>, with the cosine
floor at a tenth of the peak; the runs go one at a time, as two runs share
one accelerator and each gets half of it. A run has diverged when a logged loss
is not finite, or the training loss at some point rose half again above both
its first logged value and ln(vocab_size), the loss of guessing uniformly (the
optimizer blew up, whether or not clipping later pulled it back), or the last
training loss is more than half again above the run's best (the optimizer
undid its own progress). A run that exists already is kept, so a sweep can be
extended or re-reported without training again.

Writes, under training/experiments/learning_rate/ (change with --out):
    curves.png       training and validation loss against steps, one line per rate
    report.md        the table and the edge-of-stability analysis

Examples (from the project root):
    uv run python training/experiments/learning_rate.py
    uv run python training/experiments/learning_rate.py --lrs 1e-3 3e-3 1e-2 3e-2 --iterations 1000 --warmup-iters 50
    uv run python training/experiments/learning_rate.py --report
"""

import argparse
import math
from pathlib import Path

import numpy as np

from training.common import RUNS_DIR
from training.experiments.experiment_logger import Run, best, load_run, plot
from training.experiments.tiny_stories import TINY_STORIES
from training.infrastructure.training_together import parse_args as parse_train_args
from training.infrastructure.training_together import train

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "learning_rate"
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DIVERGENCE = 1.5  # a loss this many times above where it started, or the final loss this far above the best, is a divergence


def run_name(lr: float) -> str:
    return f"ts-lr{lr:g}"


def sweep(lrs: list[float], iterations: int, warmup_iters: int, runs_dir: Path, extra: list[str]) -> list[Run]:
    """Train one run per rate that has no run folder yet, then load them all."""
    runs = []
    for lr in lrs:
        name = run_name(lr)
        run_dir = runs_dir / name
        if not (run_dir.exists() and any(run_dir.iterdir())):  # the trainer's own test: it refuses any non-empty folder
            print(f"\n=== {name}: peak learning rate {lr:g}", flush=True)
            defaults = {**TINY_STORIES, "lr": lr, "min_lr": lr / 10, "iterations": iterations, "warmup_iters": warmup_iters}
            train(parse_train_args(["--name", name, "--out-dir", str(runs_dir), *extra], defaults=defaults))
        runs.append(load_run(runs_dir / name))
    return runs


def diverged(run: Run) -> bool:
    losses, valid = run.values("loss"), run.values("valid_loss")
    if len(losses) == 0:
        return False
    if not np.isfinite(losses).all() or not np.isfinite(valid).all():
        return True
    blew_up = losses.max() > DIVERGENCE * max(losses[0], math.log(run.config["vocab_size"]))
    return blew_up or losses[-1] > DIVERGENCE * losses.min()


def report(runs: list[Run], out_dir: Path) -> str:
    """Write curves.png and report.md for the runs, in order of their peak learning rate, and return the report."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("the report needs the matplotlib package: uv add --dev matplotlib") from None

    runs = sorted(runs, key=lambda r: r.config["lr"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("loss", "valid_loss")):
        plot(runs, metric, ax=ax)
        ax.set_yscale("log")
        ax.set_title(f"{metric} by peak learning rate")
    fig.savefig(out_dir / "curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = ["| run | peak lr | steps | peak loss | final loss | final valid_loss | best valid_loss | diverged |", "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    stable, unstable = [], []
    best_valid = {}  # run name -> its best validation loss, non-finite counted as worst
    for run in runs:
        lr, losses, valid = run.config["lr"], run.values("loss"), run.values("valid_loss")
        peak_loss, final_loss = (f"{losses.max():.2f}", f"{losses[-1]:.4f}") if len(losses) else ("-", "-")
        if len(valid):
            best_valid[run.name] = best(valid, "valid_loss")
        final_valid, top_valid = (f"{valid[-1]:.4f}", f"{best_valid[run.name]:.4f}") if len(valid) else ("-", "-")
        bad = diverged(run)
        if len(losses):  # a run that never logged a loss is in the table but settles nothing
            (unstable if bad else stable).append(run)
        lines.append(f"| {run.name} | {lr:g} | {run.steps()} | {peak_loss} | {final_loss} | {final_valid} | {top_valid} | {'yes' if bad else ''} |")
    table = "\n".join(lines)

    analysis = []
    with_valid = [r for r in stable if np.isfinite(best_valid.get(r.name, np.inf))]
    if not with_valid:
        analysis.append("No stable run with a validation loss: every rate tried diverged." if runs and not stable else "No run logged a validation loss, so there is no best rate.")
    else:
        top = min(with_valid, key=lambda r: best_valid[r.name])
        analysis.append(f"Best run: {top.name} at peak learning rate {top.config['lr']:g}, validation loss {best_valid[top.name]:.4f}.")
        edge = min((r.config["lr"] for r in unstable), default=None)
        last_stable = max(r.config["lr"] for r in stable)
        if edge is None:
            analysis.append(f"No run diverged: the largest rate tried, {last_stable:g}, is still stable, so the edge of stability lies above it.")
        elif last_stable > edge:
            analysis.append(f"Divergence is not monotone in the rate: {edge:g} diverged but {last_stable:g} did not, so the edge of stability is not resolved by this sweep.")
        else:
            analysis.append(
                f"Divergence first appears at {edge:g}; the largest stable rate is {last_stable:g}. "
                f"The best rate is {edge / top.config['lr']:.3g}x below the first divergent one"
                + (", at the edge of stability." if top.config["lr"] == last_stable else f" and {last_stable / top.config['lr']:.3g}x below the largest stable one.")
            )
    text = f"# Learning rate sweep\n\n{table}\n\n" + "\n".join(analysis) + "\n\n![curves](curves.png)\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lrs", type=float, nargs="+", default=list(LEARNING_RATES), help=f"peak learning rates to sweep (default: {' '.join(f'{lr:g}' for lr in LEARNING_RATES)})")
    parser.add_argument("--iterations", type=int, default=TINY_STORIES["iterations"], help=f"steps per run (default: {TINY_STORIES['iterations']})")
    parser.add_argument("--warmup-iters", type=int, default=TINY_STORIES["warmup_iters"], help=f"warmup steps per run (default: {TINY_STORIES['warmup_iters']})")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=f"folder holding the runs (default: {RUNS_DIR})")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help=f"folder for curves.png and report.md (default: {OUT_DIR})")
    parser.add_argument("--report", action="store_true", help="only report the runs that exist; train nothing")
    parser.add_argument("extra", nargs="*", help="further trainer flags, after --, such as -- --no-compile --device cpu")
    args = parser.parse_args()
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):
        parser.error("--lrs must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.report:
        runs = [load_run(args.runs_dir / run_name(lr)) for lr in args.lrs if (args.runs_dir / run_name(lr) / "log.jsonl").is_file()]
        if not runs:
            raise SystemExit(f"no runs named {run_name(args.lrs[0])} and so on under {args.runs_dir}")
    else:
        runs = sweep(args.lrs, args.iterations, args.warmup_iters, args.runs_dir, args.extra)
    print(report(runs, args.out), end="")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
