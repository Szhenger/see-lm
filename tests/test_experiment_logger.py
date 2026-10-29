import json

import numpy as np
import pytest

from training.experiments.experiment_logger import (
    ExperimentLogger,
    Run,
    changes,
    curve,
    load_run,
    load_runs,
    plot,
    read_log,
    summarize,
)


def records_of(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def test_fresh_run_writes_config_and_stamped_records(tmp_path):
    run_dir = tmp_path / "run"
    config = {"lr": 1e-3, "name": "run"}
    with ExperimentLogger(run_dir, config) as logger:
        logger.log(10, loss=2.5, lr=1e-3)
        logger.log(10, valid_loss=2.4)
        logger.log(20, loss=2.0, lr=1e-3)

    assert json.loads((run_dir / "config.json").read_text()) == config
    records = records_of(run_dir)
    assert [r["step"] for r in records] == [10, 10, 20]
    assert records[0]["loss"] == 2.5 and records[1]["valid_loss"] == 2.4
    for record in records:
        assert set(record) >= {"step", "wall_time", "timestamp"}
        assert record["wall_time"] >= 0
    assert records[0]["wall_time"] <= records[1]["wall_time"] <= records[2]["wall_time"]


def test_resume_continues_the_clock_and_keeps_config(tmp_path):
    run_dir = tmp_path / "run"
    with ExperimentLogger(run_dir, {"lr": 1e-3}) as logger:
        logger.log(10, loss=2.5)
    # A session that stopped after 100 s of wall-clock.
    with open(run_dir / "log.jsonl", "a") as f:
        f.write(json.dumps({"step": 20, "wall_time": 100.0, "timestamp": 0.0, "loss": 2.0}) + "\n")

    with ExperimentLogger(run_dir, {"lr": 5e-4}, resume=True, step=20) as logger:
        assert logger.elapsed >= 100.0
        logger.log(30, loss=1.5)

    assert json.loads((run_dir / "config.json").read_text()) == {"lr": 1e-3}
    records = records_of(run_dir)
    assert records[2] == {"event": "resume", "step": 20, "wall_time": 100.0, "lr": 5e-4}
    assert records[3]["step"] == 30 and records[3]["wall_time"] >= 100.0


def test_resume_supersedes_the_steps_past_its_checkpoint():
    records = [
        {"step": 10, "wall_time": 1.0, "loss": 3.0},
        {"step": 20, "wall_time": 2.0, "loss": 2.5},
        {"step": 30, "wall_time": 3.0, "loss": 2.4},
        {"event": "resume", "step": 10, "wall_time": 3.0, "lr": 1e-3},
        {"step": 20, "wall_time": 4.0, "loss": 2.6},
    ]
    steps, times, values = curve(records, "loss")
    np.testing.assert_array_equal(steps, [10, 20])
    np.testing.assert_array_equal(values, [3.0, 2.6])
    assert Run("r", {}, records).steps() == 20
    assert curve(records, "lr")[2].size == 0  # the marker's arguments are not metrics


def test_resume_without_a_log_starts_at_zero(tmp_path):
    with ExperimentLogger(tmp_path / "run", resume=True) as logger:
        assert logger.elapsed < 10.0


def test_read_log_skips_a_line_cut_short(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"step": 1, "wall_time": 0.5, "loss": 3.0}\n{"step": 2, "wall_ti')
    assert read_log(path) == [{"step": 1, "wall_time": 0.5, "loss": 3.0}]
    assert read_log(tmp_path / "missing.jsonl") == []


def test_curve_takes_only_records_holding_the_metric():
    records = [
        {"event": "resume", "wall_time": 0.0, "lr": 1e-3},
        {"step": 10, "wall_time": 1.0, "loss": 3.0},
        {"step": 10, "wall_time": 1.5, "valid_loss": 2.9},
        {"step": 20, "wall_time": 2.0, "loss": 2.5},
        {"step": 20, "wall_time": 2.5, "valid_loss": 2.4},
    ]
    steps, times, values = curve(records, "valid_loss")
    np.testing.assert_array_equal(steps, [10, 20])
    np.testing.assert_array_equal(times, [1.5, 2.5])
    np.testing.assert_array_equal(values, [2.9, 2.4])
    assert steps.dtype == np.int64 and values.dtype == np.float64
    empty = curve(records, "grad_norm")
    assert all(len(a) == 0 for a in empty)


def test_changes_ignores_run_naming_keys():
    base = {"lr": 1e-3, "num_layers": 4, "name": "base", "out_dir": "a", "resume": False}
    other = {"lr": 3e-3, "num_layers": 4, "name": "other", "out_dir": "b", "resume": True, "bf16": True}
    assert changes(other, base) == {"lr": 3e-3, "bf16": True}
    assert changes(base, base) == {}


def make_run(root, name, config, valid_losses):
    with ExperimentLogger(root / name, config) as logger:
        for i, loss in enumerate(valid_losses, start=1):
            logger.log(100 * i, loss=loss + 0.1)
            logger.log(100 * i, valid_loss=loss)


def test_load_runs_and_summarize(tmp_path, monkeypatch):
    make_run(tmp_path, "base", {"lr": 1e-3, "num_layers": 4, "name": "base"}, [3.0, 2.5, 2.0])
    make_run(tmp_path, "deep", {"lr": 1e-3, "num_layers": 8, "name": "deep"}, [3.1, 2.2, 2.3])
    (tmp_path / "not-a-run").mkdir()

    runs = load_runs(tmp_path)
    assert [r.name for r in runs] == ["base", "deep"]
    assert runs[1].config["num_layers"] == 8 and runs[1].steps() == 300
    assert [r.name for r in load_runs(tmp_path, ["deep", "base"])] == ["deep", "base"]
    monkeypatch.chdir(tmp_path.parent)
    (tmp_path.parent / "base").mkdir(exist_ok=True)  # a cwd folder named like a run does not hijack it
    assert load_runs(tmp_path, ["base"])[0].steps() == 300
    assert load_runs(tmp_path / "missing") == []
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path, ["deep", "typo"])

    table = summarize(runs)
    rows = table.splitlines()
    assert rows[0].startswith("| run | changes vs base |")
    assert rows[2] == "| base | baseline | 300 | 0:00:00 | 2.0000 | 2.0000 |"
    assert rows[3] == "| deep | num_layers=8 | 300 | 0:00:00 | 2.3000 | 2.2000 |"
    assert "| base | num_layers=4 |" in summarize(runs, baseline="deep")
    assert summarize([]) == "no runs\n"
    with pytest.raises(ValueError):
        summarize(runs, baseline="typo")


def test_summarize_without_the_metric(tmp_path):
    run = load_run(tmp_path / "empty")
    assert run == Run("empty", {}, [])
    assert "| empty | baseline | 0 | 0:00:00 | - | - |" in summarize([run])


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    make_run(tmp_path, "base", {"lr": 1e-3}, [3.0, 2.5])
    make_run(tmp_path, "fast", {"lr": 3e-3}, [2.9, 2.3])
    out = tmp_path / "curves.png"
    ax = plot(load_runs(tmp_path), "valid_loss", "wall_time", out)
    assert out.stat().st_size > 0
    assert ax.get_xlabel() == "wall-clock time (min)" and ax.get_ylabel() == "valid_loss"
    assert [line.get_label() for line in ax.get_lines()] == ["base", "fast"]
    with pytest.raises(ValueError):
        plot([], "valid_loss", "epoch")


def test_best_follows_the_metric_direction():
    from training.experiments.experiment_logger import (
        best,
        lower_is_better,
        worse,
        worst,
    )

    assert lower_is_better("loss") and lower_is_better("valid_loss") and lower_is_better("grad_norm")
    assert not lower_is_better("tokens_per_s") and not lower_is_better("accuracy")
    assert best(np.array([3.0, 2.0, 2.5]), "loss") == 2.0
    assert best(np.array([3.0e4, 4.0e4, 3.5e4]), "tokens_per_s") == 4.0e4
    assert best(np.array([3.0, np.nan, 2.5, -np.inf]), "loss") == 2.5
    assert best(np.array([np.nan, np.inf]), "loss") == np.inf and worst("loss") == np.inf
    assert best(np.array([np.nan, 1.0]), "tokens_per_s") == 1.0 and worst("tokens_per_s") == -np.inf
    assert worse(3.0, 2.0, "loss") and not worse(2.0, 3.0, "loss") and worse(np.nan, 1e9, "loss")
    assert worse(2.0, 3.0, "tokens_per_s") and worse(np.nan, 0.0, "tokens_per_s")


def test_curve_keeps_the_worst_of_a_step_measured_twice():
    # Two sessions without a resume marker between them, so neither supersedes the other.
    records = [
        {"step": 10, "wall_time": 1.0, "loss": 3.0, "tokens_per_s": 100.0},
        {"step": 20, "wall_time": 2.0, "loss": 2.0},
        {"step": 10, "wall_time": 3.0, "loss": 2.5, "tokens_per_s": 90.0},
        {"step": 20, "wall_time": 4.0, "loss": float("nan")},
    ]
    steps, times, values = curve(records, "loss")
    np.testing.assert_array_equal(steps, [10, 20])
    np.testing.assert_array_equal(times, [3.0, 4.0])
    assert values[0] == 3.0 and np.isnan(values[1])
    steps, times, values = curve(records, "tokens_per_s")
    np.testing.assert_array_equal(values, [90.0])


def test_summarize_best_for_a_higher_is_better_metric(tmp_path):
    with ExperimentLogger(tmp_path / "fast", {"lr": 1e-3}) as logger:
        logger.log(10, tokens_per_s=3.0e4)
        logger.log(20, tokens_per_s=4.0e4)
        logger.log(30, tokens_per_s=3.5e4)
    assert "| fast | baseline | 30 | 0:00:00 | 35000.0000 | 40000.0000 |" in summarize(load_runs(tmp_path), metric="tokens_per_s")


class _FakeWandbRun:
    def __init__(self, kwargs):
        self.kwargs, self.logged, self.defined, self.finished, self.exit_code = kwargs, [], [], False, None

    def define_metric(self, name, summary):
        self.defined.append((name, summary))

    def log(self, data, step):
        self.logged.append((step, data))

    def finish(self, exit_code=None):
        self.finished, self.exit_code = True, exit_code


def test_wandb_mirrors_fresh_and_rewinds_on_resume(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    calls, runs = [], []

    def fake_init(**kwargs):
        calls.append(kwargs)
        runs.append(_FakeWandbRun(kwargs))
        return runs[-1]

    monkeypatch.setattr(wandb, "init", fake_init)
    run_dir = tmp_path / "runs" / "exp"
    with ExperimentLogger(run_dir, {"lr": 1e-3}, wandb_project="seelm") as logger:
        logger.log(10, loss=2.0, tokens_per_s=1.0e4)
        logger.log(20, loss=1.5, tokens_per_s=1.1e4)
        mirror = logger._wandb
    assert calls[0] == {"project": "seelm", "name": "exp", "config": {"lr": 1e-3}, "dir": str(tmp_path / "runs"), "id": "exp", "resume": "never"}
    assert mirror.defined == [("loss", "min"), ("tokens_per_s", "max")] or mirror.defined == [("tokens_per_s", "max"), ("loss", "min")]
    assert [s for s, _ in mirror.logged] == [10, 20] and mirror.logged[1][1]["loss"] == 1.5 and "wall_time" in mirror.logged[1][1]
    assert mirror.finished and mirror.exit_code is None

    with ExperimentLogger(run_dir, {"lr": 5e-4}, resume=True, step=10, wandb_project="seelm") as logger:
        logger.log(20, loss=1.7)
    assert calls[1] == {"project": "seelm", "name": "exp", "config": {"lr": 5e-4}, "dir": str(tmp_path / "runs"), "resume_from": "exp?_step=10"}
    assert sorted(p.name for p in run_dir.iterdir()) == ["config.json", "log.jsonl"]

    with pytest.raises(RuntimeError):
        with ExperimentLogger(run_dir, resume=True, step=20, wandb_project="seelm") as logger:
            raise RuntimeError("crash")
    assert logger._wandb is None and calls[2]["resume_from"] == "exp?_step=20"
    assert runs[2].finished and runs[2].exit_code == 1  # a crash marks the W&B run failed, not finished


def test_wandb_resume_falls_back_when_there_is_nothing_to_rewind(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        if "resume_from" in kwargs:
            raise wandb.Error("no such run")
        return _FakeWandbRun(kwargs)

    monkeypatch.setattr(wandb, "init", fake_init)
    with pytest.warns(UserWarning, match="could not rewind W&B run exp to step 10"):
        with ExperimentLogger(tmp_path / "exp", resume=True, step=10, wandb_project="seelm"):
            pass
    assert [("resume_from" in c, c.get("resume")) for c in calls] == [(True, None), (False, "allow")]
    assert calls[1]["id"] == "exp"


def test_wandb_offline_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    run_dir = tmp_path / "runs" / "offline"
    with ExperimentLogger(run_dir, {"lr": 1e-3}, wandb_project="seelm-test") as logger:
        logger.log(10, loss=2.0)
        logger.log(10, valid_loss=1.9)
    assert sorted(p.name for p in run_dir.iterdir()) == ["config.json", "log.jsonl"]
    assert (tmp_path / "runs" / "wandb").is_dir()
