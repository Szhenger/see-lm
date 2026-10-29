# Learning rate sweep: what the 300-step probe shows

Nine peak learning rates from 1e-4 to 1, each a 300-step run of the TinyStories
model (warmup 100, cosine to a tenth of the peak, AdamW with betas 0.9 and 0.95,
weight decay 0.1, gradients clipped at norm 1) on an M5 with 16 GB, one run at a
time. Runs are under training/runs/lr-probe; the table and curves are in
probe-300/report.md and probe-300/curves.png.

## The table, read top to bottom

Validation loss after 300 steps falls from 4.02 at 1e-4 to 2.57 at 3e-3, then
climbs again: 2.94 at 1e-2, 3.82 at 3e-2, and 4.4 to 4.6 at 1e-1 and 3e-1. The
three rates from 3e-2 up end worse than 1e-4, the slowest rate tried. Only the
run at 1 is flagged as diverged: its training loss rose from 16 at the first log
to a peak of 89, ten times the loss of guessing uniformly, before clipping and
the cosine decay brought it back to 8.7, still worse than the untrained model.

## Two edges, not one

With gradient clipping at norm 1 and Adam's per-parameter normalisation, the
optimizer cannot run away until the rate is absurd. The hard edge, where the
loss exceeds uniform guessing and the run is lost, sits between 0.3 and 1. The
soft edge, where a larger rate stops helping and starts hurting, sits between
3e-3 and 1e-2: convergence is fastest at 3e-3 and visibly worse one step up.

The peak gradient norms mark the instability even where clipping hides it in the
loss. The runs at 3e-3 and below never exceed a norm of 2.4. The norm reaches 27
at 1e-1, 13 at 3e-1 and 67,646 at 1. From 1e-2 upward the curves also flatten
early: the loss drops fast in the first 100 steps and then stalls, the signature
of steps too large to settle into the minimum they found.

The folk wisdom that the best rate sits at the edge of stability holds for the
soft edge: the best rate is one to three times below the point where larger
rates degrade. It does not hold for the hard edge, which is more than 100 times
above the best rate; clipping moves that edge far away without moving the
optimum. A sweep with clipping off, passing `--max-grad-norm 0` through the
script, would bring the hard edge down toward the best rate and make the
comparison sharper.

## Caveats

This is a 300-step probe, not the final-loss sweep the specification asks for.
Short runs favour larger rates, and this probe anneals fully within its 300
steps. An earlier 300-step probe that held the rate near its peak (cosine
stretched to 5,000 steps) ranked 1e-3 above 3e-3 by 0.04; this annealed probe
ranks 3e-3 above 1e-3 by 0.16. The TinyStories default of 1e-3 came from the
first probe, and the full 5,000-step run at that rate (ts-base) reached a
validation loss of 1.673, under the 2.00 the specification requires. A full run
at 3e-3 against ts-base is the test that settles which rate is better at the
real budget. The full sweep over the rates worth running at that budget is

    uv run python training/experiments/learning_rate.py --lrs 3e-4 1e-3 3e-3 1e-2

about 55 minutes per rate on this machine, with 1e-4, 1e-1, 3e-1 and 1 dropped
as too slow, stalled, or already shown to diverge.
