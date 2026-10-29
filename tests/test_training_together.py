import json
from pathlib import Path

import pytest

from training.experiments.tiny_stories import TINY_STORIES
from training.infrastructure.training_together import parse_args


@pytest.fixture
def tokens(tmp_path):
    """A token array path whose sidecar records a 10,000-token vocabulary; the array itself is never opened by the parser."""
    (tmp_path / "train.json").write_text(json.dumps({"vocab_size": 10_000}))
    return tmp_path / "train.npy"


def test_train_is_required_without_defaults(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--valid", "v.npy"])
    assert "--train" in capsys.readouterr().err


def test_defaults_fill_in_and_the_command_line_overrides_them(tokens):
    defaults = {"train": tokens, "d_ff": 1344, "lr": 3e-3, "compile": True, "bf16": True}
    args = parse_args([], defaults=defaults)
    assert (args.train, args.d_ff, args.lr, args.compile, args.bf16) == (tokens, 1344, 3e-3, True, True)
    args = parse_args(["--lr", "1e-3", "--no-compile", "--d-ff", "1408"], defaults=defaults)
    assert (args.train, args.d_ff, args.lr, args.compile, args.bf16) == (tokens, 1408, 1e-3, False, True)


def test_a_default_naming_no_argument_is_rejected(tokens):
    with pytest.raises(ValueError, match="warmup_iter"):
        parse_args([], defaults={"train": tokens, "warmup_iter": 200})


def test_derived_arguments(tokens):
    args = parse_args(["--train", str(tokens), "--iterations", "300"])
    assert args.cosine_iters == 300
    assert args.name
    args = parse_args(["--train", str(tokens), "--iterations", "300", "--cosine-iters", "5000", "--name", "probe"])
    assert (args.cosine_iters, args.name) == (5000, "probe")


def test_vocab_size_comes_from_the_sidecar(tokens, tmp_path):
    assert parse_args(["--train", str(tokens)]).vocab_size == 10_000
    assert parse_args(["--train", str(tokens), "--vocab-size", "10048"]).vocab_size == 10_048
    with pytest.raises(SystemExit):
        parse_args(["--train", str(tokens), "--vocab-size", "9000"])
    with pytest.raises(SystemExit):  # no sidecar and no --vocab-size
        parse_args(["--train", str(tmp_path / "other.npy")])


def test_invalid_values_are_rejected(tokens):
    for flags in (["--iterations", "0"], ["--max-grad-norm", "-1"], ["--warmup-iters", "10", "--cosine-iters", "5"]):
        with pytest.raises(SystemExit):
            parse_args(["--train", str(tokens), *flags])


def test_tiny_stories_defaults_parse(tokens):
    args = parse_args([], defaults={**TINY_STORIES, "train": tokens})
    assert (args.vocab_size, args.d_ff, args.iterations, args.cosine_iters) == (10_000, 1344, 5000, 5000)
    assert Path(args.valid) == TINY_STORIES["valid"]
