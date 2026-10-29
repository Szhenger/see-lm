"""
Sample a completion from the Transformer language model.

Given a prompt x_{1..t}, `decode` draws one token at a time from the model's
next-token distribution and appends it, until <|endoftext|> (or any eos_id) is
drawn or max_new_tokens have been generated. Each step feeds the model the last
context_length tokens, so a conversation longer than the context slides.

    p = softmax(logits / temperature)            temperature > 0; 0 means argmax
    top-p: keep the smallest set of tokens whose mass reaches top_p, renormalize

`generate` wraps it with a tokenizer, text in and text out. The command line
loads a run written by training_together.py (its config.json and checkpoint.pt)
and the tokenizer the run's corpus was encoded with, read from the sidecar .json
next to that corpus.

Examples (from the project root):
    uv run python training/infrastructure/softmax_decoder.py --run ts-lr3e-3 --prompt "Once upon a time"
    uv run python training/infrastructure/softmax_decoder.py --run ts-lr3e-3 --prompt "Once upon a time" --temperature 0.8 --top-p 0.9 --max-new-tokens 200
    uv run python training/infrastructure/softmax_decoder.py --run ts-lr3e-3 --prompt "The cat" --temperature 0 --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from architecture.layers.softmax_function import softmax
from architecture.transformer.see_lm import TransformerLM
from tokenization.common import END_OF_TEXT
from tokenization.tokenizer.optimal_tokenizer import OptimalTokenizer
from training.common import RUNS_DIR, default_d_ff, default_device


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One token id drawn from logits of shape (..., vocab_size), returned with shape (...)."""
    if temperature < 0:
        raise ValueError(f"sample_next expects temperature >= 0, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"sample_next expects 0 < top_p <= 1, got {top_p}")
    if temperature == 0:
        return logits.argmax(dim=-1)

    # The max comes off before the division so a tiny temperature cannot overflow to inf.
    logits = logits.float()
    probs = softmax((logits - logits.amax(dim=-1, keepdim=True)) / temperature, dim=-1)
    if top_p < 1:
        # Sorted descending, a token is kept while the mass before it is still below top_p,
        # so the top token always survives and the kept set is the smallest that reaches top_p.
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(mass_before >= top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    # multinomial wants its generator on the tensor's device, so the draw happens on the
    # generator's: a seeded CPU generator then gives the same tokens whatever device the model is on.
    draw_device = generator.device if generator is not None else probs.device
    flat = probs.reshape(-1, probs.shape[-1]).to(draw_device)
    # (..., vocab_size) -> (..., 1) -> (...)
    return torch.multinomial(flat, 1, generator=generator).view(probs.shape[:-1]).to(logits.device)


@torch.no_grad()
def decode(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    vocab_limit: int | None = None,
) -> torch.Tensor:
    """
    The completion of prompt, a 1-d tensor of token ids: the sampled ids in order,
    without the prompt and without the eos_id that ended it. Empty if the first draw is eos.
    Only ids below vocab_limit are drawn: a model whose vocab_size was padded past the
    tokenizer's can otherwise sample an id no token decodes.
    """
    if prompt.dim() != 1:
        raise ValueError(f"decode expects a 1-d prompt of token ids, got shape {tuple(prompt.shape)}")
    if prompt.numel() == 0:
        raise ValueError("decode expects a non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError(f"decode expects max_new_tokens >= 0, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"decode expects temperature >= 0, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"decode expects 0 < top_p <= 1, got {top_p}")
    if vocab_limit is not None and not 0 < vocab_limit <= model.vocab_size:
        raise ValueError(f"decode expects 0 < vocab_limit <= vocab_size={model.vocab_size}, got {vocab_limit}")
    device = next(model.parameters()).device
    ids = prompt.to(device=device, dtype=torch.int64)
    generated: list[torch.Tensor] = []

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            # The last context_length ids: the model rejects a longer sequence.
            logits = model(ids[-model.context_length :])[-1, :vocab_limit]
            token = sample_next(logits, temperature, top_p, generator)
            if eos_id is not None and token.item() == eos_id:
                break
            generated.append(token)
            ids = torch.cat((ids, token.view(1)))
    finally:
        model.train(was_training)
    return torch.stack(generated) if generated else ids.new_empty(0)


def generate(
    model: TransformerLM,
    tokenizer: OptimalTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> str:
    """The text the model continues prompt with, stopped at the tokenizer's end-of-text token if it has one."""
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
    if ids.numel() == 0:
        raise ValueError("generate expects a prompt that encodes to at least one token")
    eos_id = tokenizer.id_of.get(END_OF_TEXT.encode("utf-8"))
    vocab_limit = min(max(tokenizer.vocab) + 1, model.vocab_size)
    completion = decode(model, ids, max_new_tokens, temperature, top_p, eos_id, generator, vocab_limit)
    return tokenizer.decode(completion.tolist())


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def load_run(run_dir: Path, device: torch.device) -> tuple[TransformerLM, OptimalTokenizer, dict]:
    """The model of a training run, with its checkpoint weights, the tokenizer its corpus was encoded with, and its config."""
    config_path, checkpoint_path = run_dir / "config.json", run_dir / "checkpoint.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        sys.exit(f"{run_dir} is not a training run: it needs config.json and checkpoint.pt")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        sidecar = json.loads(Path(config["train"]).with_suffix(".json").read_text(encoding="utf-8"))
        tokenizer = OptimalTokenizer.from_files(sidecar["vocab"], sidecar["merges"], [END_OF_TEXT])
    except (OSError, KeyError, ValueError) as error:
        sys.exit(f"cannot load the tokenizer of {run_dir} from the sidecar next to {config.get('train')}: {error}")

    d_model = config["d_model"]
    d_ff = config["d_ff"] if config.get("d_ff") is not None else default_d_ff(d_model)
    model = TransformerLM(
        config["vocab_size"], config["context_length"], d_model, config["num_layers"], config["num_heads"],
        d_ff, config["rope_theta"], device=device,
    )
    # Onto the CPU first, then load_state_dict copies each tensor to the model's device.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model, tokenizer, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder name under --out-dir, or a path to a run folder")
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR, help=f"folder holding the runs (default: {RUNS_DIR})")
    parser.add_argument("--prompt", default="", help=f"text to complete (default: start a new document after {END_OF_TEXT})")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature; 0 picks the most likely token")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling threshold in (0, 1]; 1 samples from the full distribution")
    parser.add_argument("--seed", type=int, default=None, help="seed for the sampler (default: unseeded)")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: the machine's accelerator)")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be 0 or more")
    if args.temperature < 0:
        parser.error("--temperature must be 0 or more")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else default_device()
    run_dir = Path(args.run) if Path(args.run).is_dir() else args.out_dir / args.run
    model, tokenizer, _ = load_run(run_dir, device)
    # A CPU generator, as in the trainer: the seed then reproduces the same text on any device.
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    # An empty prompt starts a fresh document, as the corpus separates them with the end-of-text token.
    prompt = args.prompt or END_OF_TEXT
    completion = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p, generator)
    print(args.prompt + completion, flush=True)


if __name__ == "__main__":
    main()
