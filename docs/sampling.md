# The sampled layer: a profiler's time-sampled counters, exported one value a sample

The stamps give where a window starts, which role and phase each stretch is, who released whom. They cannot give the
occupancy of the machine between stamps. A profiler can: while the UNTRACED kernel runs, it reads hardware counters at
a fixed interval and keeps every reading. warpline reads that series, finds the window's period in it, folds it onto
one window, and lays the fold under the stamps. Nothing here is recorded in the kernel.

## How the file is made

The profiler's sections `PmSampling` and `PmSampling_WarpStates` collect the series: on the card used here, fifteen
warp-stall reasons (`smsp__warps_issue_stalled_<reason>.avg`, in warps: how many warps wait on a barrier, a long-latency
operand, the memory pipe, ...), a reading every sampling interval (the profiler chooses it unless told; 1.5 us in the
runs read so far, set with `--pm-sampling-interval`). The profiler replays the kernel once for each counter group, so
each series comes from its own PASS of the kernel. A kernel run of one CTA is one REPORT; the report is exported:

    ncu --import <report> --page raw --print-metric-instances values --csv > run.csv

The file has three rows: the metric names, their units, their values. A sampled metric's name starts `pmsampling:` and
its value is `<aggregate> (<v0>; <v1>; ...)`, one value a sample. `profiler__pmsampler_interval_time` is the sampling
interval in microseconds, in the same form (one value a pass). Other columns are ignored. One file is one RUN; give
`warpline fold` several runs of one kernel and it averages them.

## What a sample is

A sample is a TOTAL over its interval, not an instant. The folded profile is therefore the true profile smoothed by a
box one interval wide: a share of a window equal to the interval over the window's period (0.38 on the runs read so far). Everything
sharper than that is gone, and the fold says so (`THE BLUR`).

## The steps, each computed from the series alone

1. START. A series' first live sample is its kernel's start, to within a sample. Series with the same sample count were
   collected together, so a series starts with its pass (a sparse series' own first nonzero sample is not its start).
2. STEADY SPAN. Past `--skip-windows` windows (default 4) from the start, to `--tail` samples (default 3) before the
   live end.
3. PERIOD, in samples: the period whose fold has the largest between-bin variance. Scanned over 2..6 samples at a step
   of 0.002, the best refined to 1e-4; a shorter period (a divisor of the best, within 0.7%) whose score is within 90% of
   the best is taken instead, so a multiple of the period is not mistaken for it. Its SCORE is the share of the series'
   variance the fold explains. A window under two samples cannot be told from its mirror image (it aliases): a period
   found within 15% of two is flagged.
4. FLAT. A series whose best fold explains under 25% of its variance (`--min-score`) has too little variation to give a
   period; it is reported flat and folded on the agreed period.
5. PASSES AGREE. The reference is the series that the most other series' periods lie within the tolerance (default 1%) of. A series that
   does not agree is flagged and left out of the fold.
6. STEADY. The period of the steady span's first half and of its second half agree within the tolerance, else DRIFT is
   flagged. A half is half the samples, so this check resolves a period no better than the blur over the windows in the
   half (about 1-2% on 75 samples); the report prints the spread it found.
7. FOLD. Every series on the agreed period into N bins (`--bins`, default 8; an N that leaves under 10 samples a bin on
   every series is refused): each bin's mean, standard error and count.
8. RUNS. The runs' periods (in microseconds) agree within the tolerance or the run is flagged and left out. Each run is
   aligned to the first by the rotation of one series' fold (`barrier`, else the most-varying) that maximizes the
   correlation; a run whose alignment is weak (correlation under 0.7) or ambiguous (a different rotation within 0.05) is
   flagged and left out; the rest are averaged bin by bin (weighted by count).

## Laid under the stamps

`warpline overlay` takes a stamp record that has barrier waits, builds from its average window the mean number of lanes
inside a `barrier.*.wait` region at each window position (smoothed by the same box), and finds the rotation of the
samples' `barrier` fold that correlates best with it. The result is REFUSED when the correlation is under 0.7 or when
the runner-up rotation (the best at a rotation a blur-width or more away) is within 0.05. See the README's assumptions.
