"""
Train the TinyStories language model: training_together.py at the settings of
the TinyStories experiment, every one of which its flags can still override.

Model (22.7M parameters, 12.5M outside the embedding and output tables): the 10,000-token TinyStories BPE
vocabulary, context 256, d_model 512, d_ff 1344, 4 layers of 16 heads, RoPE
theta 10,000. Budget: 40,960,000 tokens as 5,000 steps of 32 x 256, the GPU
budget of 327,680,000 scaled down for this machine (an M5 with 16 GB), about an
hour on mps with bf16 autocast and the compiled model. AdamW at a peak learning
rate of 1e-3 after 200 warmup steps, cosine-decayed to 1e-4 at the last step,
betas (0.9, 0.95), weight decay 0.1, gradients clipped at norm 1; of 1e-3, 3e-3
and 6e-3, the first had the lowest validation loss after 300 steps on this machine.

Examples (from the project root):
    uv run python training/experiments/tiny_stories.py --name ts-base
    uv run python training/experiments/tiny_stories.py --name ts-lr3e-3 --lr 3e-3 --min-lr 3e-4
    uv run python training/experiments/tiny_stories.py --name ts-base --resume
    uv run python training/experiments/tiny_stories.py --name ts-full --iterations 40000 --wandb seelm
    uv run python training/infrastructure/softmax_decoder.py --run ts-base --prompt "Once upon a time"
"""

from tokenization.common import DATA_DIR
from training.infrastructure.training_together import parse_args, train

TOKENS = DATA_DIR / "tokens"  # where encode_datasets.py writes the arrays

TINY_STORIES = {
    "train": TOKENS / "tinystories_train.npy",
    "valid": TOKENS / "tinystories_valid.npy",
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "num_layers": 4,
    "num_heads": 16,
    "rope_theta": 10_000.0,
    "iterations": 5_000,
    "batch_size": 32,
    "lr": 1e-3,
    "min_lr": 1e-4,
    "warmup_iters": 200,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "bf16": True,
    "compile": True,
}


def main() -> None:
    train(parse_args(defaults=TINY_STORIES, description=__doc__))


if __name__ == "__main__":
    main()
