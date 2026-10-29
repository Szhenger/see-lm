"""
Track a training run: its configuration, its metrics against gradient steps
and wall-clock time, and the curves and tables that compare runs.

A run owns one folder (training/runs/NAME for the trainer), holding
    config.json      the arguments the run was started with
    log.jsonl        one JSON line per logged step and per evaluation

Every metric record carries the gradient step, the seconds of wall-clock time
since the run started and the time of day, so any metric can be plotted against
steps or against time. A resumed run continues the clock from its last record
rather than restarting it, so a curve that spans several sessions stays
continuous. A record with an "event" key is a marker, not a metric: a resume
writes one holding the step it restarted from and the arguments it ran under,
and the readers drop the records past that step from the session it replaced.

`ExperimentLogger` is the writing side: log(step, **metrics) prints the record
and appends it, and mirrors it to a Weights & Biases project when one is given;
a resume rewinds the W&B run to the checkpoint's step, so both records agree.
Where a choice would otherwise be arbitrary the reading side takes the worst
case: a step measured twice keeps its worse value and later time, and a
non-finite value counts as the worst possible when picking a run's best.
`load_run`, `curve`, `summarize` and `plot` are the reading side: one run's
config and records, one metric as arrays of steps, times and values, a Markdown
table of a folder's runs with their settings against a baseline, and learning
curves of several runs on one figure.

Examples (from the project root):
    uv run python training/experiments/experiment_logger.py summary
    uv run python training/experiments/experiment_logger.py summary --baseline ts-base --out training/experiments/experiment_log.md
    uv run python training/experiments/experiment_logger.py plot --metric valid_loss --out curves.png
    uv run python training/experiments/experiment_logger.py plot ts-base ts-lr3e-3 --metric loss --x wall_time --out curves.png
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.common import RUNS_DIR

CONFIG_FILE = "config.json"
LOG_FILE = "log.jsonl"

# Arguments that name a run rather than configure it; left out of the summary's "changes" column.
_RUN_KEYS = {"name", "out_dir", "resume", "wandb"}
# Metrics where higher is better, by name fragment; every other metric is loss-like and lower is better.
_HIGHER_IS_BETTER = ("tokens_per_s", "throughput", "accuracy")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
class ExperimentLogger:
    """Writes a run's config.json and log.jsonl, stamping every record with its step and wall-clock time."""
    def __init__(self, run_dir: str | Path, config: dict | None = None, resume: bool = False, step: int = 0, wandb_project: str | None = None) -> None:
        """A resumed run gives the step it restarts from; its records past that step are superseded."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / LOG_FILE
        self._wandb = None
        self._defined: set[str] = set()
        if wandb_project:  # before config.json and the log are written, so a failed init leaves the run folder empty
            self._wandb = _wandb_init(wandb_project, self.run_dir, config, step if resume else None)
        elapsed = 0.0
        if resume:
            # The clock continues from the last record of the whole log, superseded ones included: that time was
            # spent, so the worst case is the true one. Changed arguments are logged, not rewritten.
            elapsed = next((r["wall_time"] for r in reversed(read_log(self.path)) if "wall_time" in r), 0.0)
            self.write(event="resume", step=step, wall_time=elapsed, **(config or {}))
        elif config is not None:
            (self.run_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        self._start = time.perf_counter() - elapsed
 
    @property
    def elapsed(self) -> float:
        """Seconds of wall-clock time since the run started, across resumes."""
        return time.perf_counter() - self._start

    def log(self, step: int, **metrics) -> None:
        """Print the metrics at step and append them to the log with the step and the wall-clock time."""
        record = {"step": step, "wall_time": round(self.elapsed, 3), "timestamp": time.time(), **metrics}
        shown = {"step": step, "time": _hms(record["wall_time"]), **metrics}
        print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in shown.items()), flush=True)
        self.write(**record)
        if self._wandb is not None:
            for name in metrics.keys() - self._defined:  # W&B's summary picks the best under the same direction as summarize
                self._wandb.define_metric(name, summary="min" if lower_is_better(name) else "max")
                self._defined.add(name)
            self._wandb.log({"wall_time": record["wall_time"], **metrics}, step=step)

    def write(self, **record) -> None:
        """Append record as one JSON line."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finish(self, exit_code: int | None = None) -> None:
        """Close the W&B run; a non-zero exit_code marks it failed rather than finished."""
        if self._wandb is not None:
            self._wandb.finish(exit_code=exit_code)
            self._wandb = None

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        self.finish(exit_code=1 if exc_type is not None else None)


def _wandb_init(project: str, run_dir: Path, config: dict | None, step: int | None):
    """The W&B run mirroring run_dir, its files beside the run folder; a resume at step rewinds the W&B run to it."""
    import wandb  # the second of import time is paid only by runs that mirror

    name = run_dir.name
    common = dict(project=project, name=name, config=config, dir=str(run_dir.parent))
    if step is None:  # a fresh run never continues a W&B run of the same name; a reused name is an error
        return wandb.init(id=name, resume="never", **common)
    try:
        return wandb.init(resume_from=f"{name}?_step={step}", **common)
    except wandb.Error as error:  # nothing to rewind: the run was not mirrored before this session
        warnings.warn(f"could not rewind W&B run {name} to step {step} ({error}); continuing it instead", stacklevel=3)
        return wandb.init(id=name, resume="allow", **common)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    config: dict
    records: list[dict]

    def curve(self, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return curve(self.records, metric)

    def values(self, metric: str) -> np.ndarray:
        return self.curve(metric)[2]

    def steps(self) -> int:
        """The last step logged, 0 when nothing was."""
        return max((r["step"] for r in stamped(self.records)), default=0)

    def wall_time(self) -> float:
        return max((r["wall_time"] for r in stamped(self.records)), default=0.0)


def read_log(path: str | Path) -> list[dict]:
    """The records in a log.jsonl; a missing file is empty and a line cut short by a crash is skipped."""
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    try:
        config = json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = {}
    return Run(run_dir.name, config, read_log(run_dir / LOG_FILE))


def load_runs(runs_dir: str | Path = RUNS_DIR, names: list[str] | None = None) -> list[Run]:
    """The runs under runs_dir, by name; names picks and orders them, else every folder with a log is taken."""
    runs_dir = Path(runs_dir)
    if names:
        dirs = [Path(name) if (Path(name) / LOG_FILE).is_file() and not (runs_dir / name / LOG_FILE).is_file() else runs_dir / name for name in names]
        missing = [str(d) for d in dirs if not (d / LOG_FILE).is_file()]
        if missing:
            raise FileNotFoundError(f"no {LOG_FILE} in {', '.join(missing)}")
        return [load_run(d) for d in dirs]
    return [load_run(d) for d in sorted(runs_dir.iterdir()) if (d / LOG_FILE).is_file()] if runs_dir.is_dir() else []


def stamped(records: list[dict]) -> list[dict]:
    """The stamped metric records, in log order, without the markers and without the records a resume superseded."""
    kept = []
    for r in records:
        if "event" in r:
            if r["event"] == "resume" and "step" in r:  # the session before it went past the checkpoint it restarted from
                kept = [k for k in kept if k["step"] <= r["step"]]
        elif "step" in r and "wall_time" in r:
            kept.append(r)
    return kept


def curve(records: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(steps, wall_times, values) of the records that hold metric, by step; a step measured twice keeps its worse value and later time."""
    by_step: dict[int, tuple[float, float]] = {}
    for r in stamped(records):
        if metric not in r:
            continue
        step, t, v = r["step"], float(r["wall_time"]), float(r[metric])
        if step in by_step:
            t0, v0 = by_step[step]
            t, v = max(t, t0), (v if worse(v, v0, metric) else v0)
        by_step[step] = (t, v)
    steps = sorted(by_step)
    times, values = zip(*(by_step[s] for s in steps)) if steps else ((), ())
    return np.array(steps, dtype=np.int64), np.array(times, dtype=np.float64), np.array(values, dtype=np.float64)


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------
def lower_is_better(metric: str) -> bool:
    return not any(word in metric for word in _HIGHER_IS_BETTER)


def worst(metric: str) -> float:
    """The worst possible value of metric, which a non-finite measurement counts as."""
    return np.inf if lower_is_better(metric) else -np.inf


def worse(a: float, b: float, metric: str) -> bool:
    """Whether a is a worse value of metric than b."""
    a, b = (x if np.isfinite(x) else worst(metric) for x in (a, b))
    return a > b if lower_is_better(metric) else a < b


def best(values: np.ndarray, metric: str) -> float:
    """The best value of metric among values under its direction; non-finite values count as the worst possible."""
    w = worst(metric)
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=w, posinf=w, neginf=w)
    return float(values.min() if lower_is_better(metric) else values.max())


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------
def changes(config: dict, baseline: dict) -> dict:
    """The settings in config that differ from baseline, ignoring the ones that only name the run."""
    keys = [k for k in [*baseline, *config] if k not in _RUN_KEYS]
    return {k: config.get(k) for k in dict.fromkeys(keys) if config.get(k) != baseline.get(k)}


def summarize(runs: list[Run], baseline: str | None = None, metric: str = "valid_loss") -> str:
    """A Markdown table of the runs: their changes against the baseline (default: the first run), steps, time, and final and best metric."""
    if not runs:
        return "no runs\n"
    if baseline is not None and not any(r.name == baseline for r in runs):
        raise ValueError(f"no run named {baseline!r} among {', '.join(r.name for r in runs)}")
    base = next((r for r in runs if r.name == baseline), runs[0])
    lines = [
        f"| run | changes vs {base.name} | steps | wall-clock | final {metric} | best {metric} |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        diff = changes(run.config, base.config)
        shown = "baseline" if run is base else ", ".join(f"{k}={v}" for k, v in diff.items()) or "same"
        values = run.values(metric)
        final, top = (f"{values[-1]:.4f}", f"{best(values, metric):.4f}") if len(values) else ("-", "-")
        lines.append(f"| {run.name} | {shown} | {run.steps()} | {_hms(run.wall_time())} | {final} | {top} |")
    return "\n".join(lines) + "\n"


def plot(runs: list[Run], metric: str = "valid_loss", x: str = "step", out: str | Path | None = None, ax=None):
    """One line per run of metric against x ("step" or "wall_time"); saved to out when given, else shown."""
    if x not in ("step", "wall_time"):
        raise ValueError(f"plot expects x to be 'step' or 'wall_time', got {x!r}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plotting needs the matplotlib package: uv add --dev matplotlib") from None
    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        steps, times, values = run.curve(metric)
        if len(values):
            ax.plot(steps if x == "step" else times / 60, values, label=run.name)
    ax.set_xlabel("gradient steps" if x == "step" else "wall-clock time (min)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    if runs:
        ax.legend()
    if out is not None:
        ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    elif owns_figure and plt.get_backend().lower() != "agg":  # a caller's figure is theirs to show; a headless process has no window
        plt.show()
    return ax


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("summary", "a Markdown table of the runs"), ("plot", "learning curves of the runs")):
        command = commands.add_parser(name, help=text)
        command.add_argument("runs", nargs="*", help="run names under --runs-dir, or run folders (default: every run there)")
        command.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=f"folder holding the runs (default: {RUNS_DIR})")
        command.add_argument("--metric", default="valid_loss", help="the logged metric to report (default: valid_loss)")
        command.add_argument("--out", type=Path, default=None, help="file to write (default: the terminal, or a window)")
    commands.choices["summary"].add_argument("--baseline", default=None, help="run whose settings the others are compared with (default: the first)")
    commands.choices["plot"].add_argument("--x", choices=("step", "wall_time"), default="step", help="horizontal axis (default: step)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        runs = load_runs(args.runs_dir, args.runs)
        if not runs:
            sys.exit(f"no runs under {args.runs_dir}")
        if args.command == "summary":
            table = summarize(runs, args.baseline, args.metric)
            if args.out is None:
                print(table, end="")
            else:
                args.out.write_text(table, encoding="utf-8")
                print(f"{len(runs)} runs summarized in {args.out}")
        else:
            plot(runs, args.metric, args.x, args.out)
            if args.out is not None:
                print(f"{len(runs)} runs plotted to {args.out}")
    except (FileNotFoundError, ImportError, ValueError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()