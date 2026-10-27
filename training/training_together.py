"""
Train the Transformer language model on an encoded corpus.

Every hyperparameter of the model, the optimizer and the schedule is a
command-line argument, so runs with different settings are one command each.
The training and validation corpora are the uint16 .npy arrays written by
encode_datasets.py, opened as memory maps so a corpus larger than memory costs
nothing to open. One step is: learning rate from the cosine schedule, a batch
of random windows, forward, cross-entropy, backward, gradient clipping, AdamW.

A run owns one folder, training/runs/NAME (change with --out-dir), holding
    config.json      the arguments the run was started with
    log.jsonl        one JSON line per logged step and per evaluation, each stamped with the wall-clock time
    checkpoint.pt    the latest model, optimizer and iteration, written atomically

Training loss, learning rate, gradient norm and throughput are logged every
--log-interval steps; the validation loss, the mean over --eval-batches random
batches, every --eval-interval steps and at the end. The checkpoint is written
every --checkpoint-interval steps, at the end, and on Ctrl-C. --resume reloads
it and continues from its iteration under the same arguments, with the log's
wall-clock continuing from where it stopped. --wandb PROJECT mirrors the log to
Weights & Biases, rewinding its run to the checkpoint's step on a resume.
experimentation/experiment_logger.py reads the logs back as curves and tables.

Examples (from the project root):
    uv run python training/training_together.py --train data/tokens/tinystories_train.npy --valid data/tokens/tinystories_valid.npy
    uv run python training/training_together.py --train data/tokens/tinystories_train.npy --valid data/tokens/tinystories_valid.npy --name ts-lr3e-3 --lr 3e-3
    uv run python training/training_together.py --train data/tokens/owt_train.npy --valid data/tokens/owt_valid.npy --d-model 768 --num-layers 12 --num-heads 12 --iterations 20000
    uv run python training/training_together.py --train data/tokens/tinystories_train.npy --name ts-lr3e-3 --resume
    uv run python training/training_together.py --train data/tokens/tinystories_valid.npy --iterations 50 --warmup-iters 5 --batch-size 4 --d-model 64 --num-layers 1 --num-heads 2 --device cpu
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from architecture.transformer.see_lm import TransformerLM
from experimentation.experiment_logger import ExperimentLogger
from optimization.gradients.gradient_clipping import clip_grad_norm_
from optimization.learning.cross_entropy import cross_entropy
from optimization.learning.learning_schedule import cosine_lr
from optimization.optimizer.adamw_optimizer import AdamW
from training.common import default_d_ff, default_device
from training.loop_utils import get_batch, load_checkpoint, load_tokens, save_checkpoint

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--train", type=Path, required=True, help="training token array (.npy)")
    data.add_argument("--valid", type=Path, default=None, help="validation token array (.npy); without it there is no evaluation")
    data.add_argument("--vocab-size", type=int, default=None, help="vocabulary size (default: read from the sidecar .json next to --train)")

    model = parser.add_argument_group("model")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=None, help="feed-forward width (default: 8/3 d_model rounded up to a multiple of 64)")
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    optim = parser.add_argument_group("optimizer and schedule")
    optim.add_argument("--iterations", type=int, default=5_000, help="training steps")
    optim.add_argument("--batch-size", type=int, default=32)
    optim.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    optim.add_argument("--min-lr", type=float, default=1e-4, help="learning rate at the end of the cosine")
    optim.add_argument("--warmup-iters", type=int, default=100)
    optim.add_argument("--cosine-iters", type=int, default=None, help="iteration at which the cosine reaches --min-lr (default: --iterations)")
    optim.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("BETA1", "BETA2"))
    optim.add_argument("--eps", type=float, default=1e-8)
    optim.add_argument("--weight-decay", type=float, default=0.1)
    optim.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping threshold; 0 disables clipping")

    run = parser.add_argument_group("run")
    run.add_argument("--name", default=None, help="run folder name under --out-dir (default: a timestamp)")
    run.add_argument("--out-dir", type=Path, default=RUNS_DIR, help=f"folder holding the runs (default: {RUNS_DIR})")
    run.add_argument("--resume", action="store_true", help="continue from the run's checkpoint")
    run.add_argument("--device", default=None, help="cuda, mps or cpu (default: the machine's accelerator)")
    run.add_argument("--bf16", action="store_true", help="run the forward and backward under bfloat16 autocast")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--log-interval", type=int, default=10)
    run.add_argument("--eval-interval", type=int, default=250)
    run.add_argument("--eval-batches", type=int, default=20)
    run.add_argument("--checkpoint-interval", type=int, default=500)
    run.add_argument("--wandb", default=None, metavar="PROJECT", help="mirror the log to this Weights & Biases project; a resume rewinds the W&B run to the checkpoint's step")

    args = parser.parse_args()
    if args.cosine_iters is None:
        args.cosine_iters = args.iterations
    if args.name is None:
        args.name = time.strftime("%Y%m%d-%H%M%S")
    for flag in ("iterations", "batch_size", "log_interval", "eval_interval", "eval_batches", "checkpoint_interval"):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.max_grad_norm < 0:
        parser.error("--max-grad-norm must be 0 or more")
    if args.warmup_iters < 0 or args.cosine_iters < args.warmup_iters:
        parser.error(f"need 0 <= --warmup-iters <= --cosine-iters, got {args.warmup_iters} and {args.cosine_iters}")
    recorded = sidecar_vocab_size(args.train)
    if args.vocab_size is None:
        if recorded is None:
            parser.error(f"--vocab-size is needed: no sidecar .json with a vocab_size next to {args.train}")
        args.vocab_size = recorded
    elif recorded is not None and args.vocab_size < recorded:
        parser.error(f"--vocab-size {args.vocab_size} is smaller than the {recorded} the sidecar next to {args.train} records")
    return args


def sidecar(tokens: Path) -> dict:
    """What encode_datasets.py recorded next to a token array, or {} if there is no readable sidecar."""
    try:
        return json.loads(tokens.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sidecar_vocab_size(tokens: Path) -> int | None:
    try:
        return int(sidecar(tokens)["vocab_size"])
    except (KeyError, TypeError, ValueError):
        return None


@torch.no_grad()
def evaluate(model: TransformerLM, data: np.ndarray, args: argparse.Namespace, device: torch.device) -> float:
    """Mean cross-entropy over --eval-batches random batches of data, the same batches at every evaluation."""
    generator = torch.Generator().manual_seed(args.seed + 1)
    model.eval()
    total = torch.zeros((), device=device)
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, device, generator)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            total += cross_entropy(model(x), y)
    model.train()
    return total.item() / args.eval_batches


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else default_device()
    run_dir = args.out_dir / args.name
    checkpoint = run_dir / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        sys.exit(f"--resume: no checkpoint at {checkpoint}")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        sys.exit(f"{run_dir} already holds a run; pick another --name or pass --resume")
    train_hash, valid_hash = sidecar(args.train).get("vocab_sha256"), sidecar(args.valid).get("vocab_sha256") if args.valid else None
    if train_hash and valid_hash and train_hash != valid_hash:
        sys.exit(f"{args.train} and {args.valid} were encoded with different vocabularies")
    train_data = load_tokens(args.train)
    valid_data = load_tokens(args.valid) if args.valid else None

    torch.manual_seed(args.seed)
    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads,
        args.d_ff if args.d_ff is not None else default_d_ff(args.d_model),
        args.rope_theta, device=device,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)
    start = load_checkpoint(checkpoint, model, optimizer) if args.resume else 0

    # Batches come from the CPU generator whatever the device; a resumed run is
    # reseeded by its iteration so it does not replay the batches it already saw.
    generator = torch.Generator().manual_seed(args.seed + start)

    # The logger creates the run folder, so a failed start above does not leave
    # a folder that refuses the next attempt. A fresh run writes config.json; a
    # resumed one logs the arguments it ran with and continues the wall-clock.
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    logger = ExperimentLogger(run_dir, config, resume=args.resume, step=start, wandb_project=args.wandb)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters on {device}, {len(train_data):,} training tokens, run folder {run_dir}", flush=True)
    if start:
        print(f"resumed at iteration {start}", flush=True)

    model.train()
    running_loss = torch.zeros((), device=device)
    running_norm = torch.zeros((), device=device)
    since_log = 0
    last_log = time.perf_counter()
    completed = start  # steps whose optimizer update has been applied

    # Ctrl-C is deferred to the end of the step in progress, so the checkpoint
    # never holds half-updated weights; a second Ctrl-C stops at once, unsaved.
    interrupted = False

    def on_interrupt(signum, frame) -> None:
        nonlocal interrupted
        interrupted = True
        signal.signal(signal.SIGINT, previous_handler)

    previous_handler = signal.signal(signal.SIGINT, on_interrupt)
    try:
        for iteration in range(start, args.iterations):
            lr = cosine_lr(iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.context_length, device, generator)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm:
                running_norm += clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            step = completed = iteration + 1

            # Accumulated on the device; .item() syncs only at log time.
            running_loss += loss.detach()
            since_log += 1
            if step % args.log_interval == 0 or step == args.iterations:
                now = time.perf_counter()
                logger.log(
                    step,
                    loss=running_loss.item() / since_log,
                    lr=lr,
                    grad_norm=running_norm.item() / since_log,
                    tokens_per_s=since_log * args.batch_size * args.context_length / (now - last_log),
                )
                running_loss.zero_()
                running_norm.zero_()
                since_log = 0
                last_log = now
            if valid_data is not None and (step % args.eval_interval == 0 or step == args.iterations):
                eval_start = time.perf_counter()
                logger.log(step, valid_loss=evaluate(model, valid_data, args, device))
                last_log += time.perf_counter() - eval_start  # evaluation time does not count toward throughput
            if step % args.checkpoint_interval == 0 or step == args.iterations:
                save_checkpoint(model, optimizer, step, checkpoint)
            if interrupted:
                save_checkpoint(model, optimizer, completed, checkpoint)
                print(f"\ninterrupted after iteration {completed}, checkpoint written to {checkpoint}", flush=True)
                sys.exit(130)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        logger.finish(exit_code=1 if sys.exc_info()[0] is not None else None)  # a crash or interrupt marks the W&B run failed
    print(f"done: {args.iterations} iterations, checkpoint at {checkpoint}", flush=True)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
