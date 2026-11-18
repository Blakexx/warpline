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

## The profiler must sample the launch whose stamps are recorded

The join of the samples to the stamps is made on the launch's two edges and on its CYCLES, so the two recordings must be
OF THE SAME LAUNCH, or of launches of one kernel whose cycles agree: the stamp record is the launch the profiler sampled
(the profiler replays the kernel for each counter group, and the stamps are written by that replayed launch; one
`launch.begin` / `launch.end` pair, tied, in each). WHY:

1. A profiler's time sampling STRETCHES the kernel in real time (14-18% on the kernel measured: 198.6 us unsampled, 226 to
   231 us sampled) and leaves its CYCLES unchanged (341,245 cycles unsampled; 337K to 343K sampled): the cycle counter does
   not count the time the sampling takes. The stated interval (1.5 us) is right in real time; in cycles a sample is 2,214 to
   2,233, the same for every variant of the kernel. A stamp record of an UNSAMPLED launch has its launch in a time that no
   sampled series spans: only its cycles carry over.
2. The stretch is not even at the scale of a phase (ties give 1,336 to 1,609 cycles a microsecond in single phases) and is
   even over a window and over the launch (to a percent). So the samples are placed by cycles, not by time, and the position
   inside a phase is only as good as that.
3. A pass's length varies from run to run in samples (159 to 165 live samples against a record's one launch of 161.3), and a
   window's period jitters (5,462 cycles, standard deviation 319), though not in the samples' rhythm (see THE FRAME): a
   record's one length is never compared to a pass to the sample.

A stamp record of a launch the profiler did not sample (a different run, a different clock, the profiler off) is a record
of a kernel whose window is not the sampled window: its windows can be used as fractions of its launch only as far as the
cost of stamping leaves them there (`set` checks it: an edges variant against a stamped one).

## Laid under the stamps: a pass is scaled by its own two edges, on the kernel's cycles

`warpline overlay` takes a stamp record whose lanes carry the launch's two edges (`launch.begin`, `launch.end`, each with a
tie: [format.md](format.md)) and REFUSES a record without them, naming what the kernel must stamp.

1. THE LAUNCH from the record: T in CYCLES (the earliest `launch.begin` to the latest `launch.end`), and the steady window
   starts as FRACTIONS of the launch.
2. A PASS (the series that were collected together; LIVE: any series of the pass over a threshold, default any nonzero
   value): its first live sample i0 and its last i1. The kernel's entry is at sample-time i0 + a and its exit at i1 + b,
   a and b in [0, 1) unknown. Sample j's centre is then at the fraction f = ((j + 0.5) - (i0 + a)) / ((i1 + b) - (i0 + a))
   of the launch, the cycle f * T past the entry; its WINDOW POSITION is read from a frame of the record's window
   anchors (f falls between two consecutive window starts: the fraction of that window). STATED ASSUMPTION: over a
   pass the profiler's time is linear in the kernel's cycles (even over a window and a launch, not within a phase).
3. THE CHECK that replaces the length check: a pass's CYCLES A SAMPLE (T over its live span) against the median of its
   variant's passes; one more than `--cps-tolerance` (default 3%) away is flagged and left out. Ties are no longer needed to
   place samples; they still give the rate and the launch's time.
4. THE FRAME FOLLOWS THE SAMPLES. The anchors' jitter is not the samples' rhythm on the real kernel: the intervals of
   `phase.window.begin` on one lane have a lag-1 autocorrelation of -0.5 (the signature of independent noise on each
   anchor: an early window is followed by a late one), and the pooled sharpness of every pass is higher on a mean period
   than on the raw anchors, and rises with the smoothing. The default frame is a LOCAL fit of the anchors (a line fitted at
   each anchor with Gaussian weights of standard deviation h windows: it keeps a slow drift of the window's length and drops
   a wobble shorter than about 2 pi h windows); h is chosen among the raw anchors, 2, 4, 8, 16, 32 windows and the mean
   period (the limit) by the pooled sharpness summed over the variant's passes, the smoothest within one noise standard
   deviation of the best. `--frame anchors|mean` and `--frame-smooth H` fix it; the sharpness of every pass on every
   candidate is printed with the frame chosen.
5. THE JOINT FIT of a pass. Its runs (each a pass of its own) are fitted at once to one pooled profile: each run's entry
   offset a and scale d = b - a inside what its two edges allow, maximizing the pooled fold's sharpness (the share of the
   pooled samples' variance in the first three harmonics, summed over the pass's series; a sample is a total over an
   interval, which leaves little above the third, and a harmonic fit has no bin edges). No pairwise rotations, no run left
   out for a weak fold. A common shift of every run only rotates the pooled profile and leaves the sharpness unchanged: the
   absolute offset is the edges' alone. Each run's scale is known to the plateau within one noise sigma
   (sqrt(2 Q / N)) of its best, and at each scale of the plateau the run's entry and exit must stay inside their live
   samples: the intersection over runs of the common shifts that allow is the BOX, and the placement its middle.
6. THE UNCERTAINTY IS MEASURED. (a) The runs are resampled (a bootstrap over runs, and every leave-one-run-out), each fit
   from scratch, and the spread of each series' profile phase is the larger of the two standard deviations; the series'
   half-width is the box's half-width and two spreads in quadrature. With fewer than three runs the spread is not measurable
   and the half-width is the box alone (said). (b) THE ALIAS CHECK: the pooled objective at the placement against its value
   with each run alone, and every run together, shifted by one whole sample and by one window, the margin twice the
   resampled standard deviation of the difference. A run shifted alone is told apart by the others (with one run it is
   not: ambiguous). Every run shifted together is an exact rotation of the pooled profile: the objective is the same, and
   the edges decide it; it is listed so that the reader sees it. A series whose placement moves by more than half a bin
   under a candidate within the margin, or whose resampled fits land a sample away in a tenth of the cases, is named
   PLACEMENT AMBIGUOUS BY A SAMPLE with both candidates. (c) Where the objective would be higher by more than the margin
   with a run's entry outside its first live sample, FIT AND EDGES DISAGREE is named (the placement keeps the edges' bound).
   The scale of a run at the end of the range its edges allow is the reason the box can be empty or narrow; it is shown.
   No period is fitted here, so none of the period checks of `fold` can drop a series.
7. THE CROSS-CHECK: where the record has barrier waits, the placed `barrier` fold against the waits' profile (smoothed by
   the samples' box): the rotation in bins and the correlation; they agree within the placement's half-width plus a bin, or
   a named finding.

THE EDGES VARIANT: a variant whose record has the launch events and no window anchors is the plain kernel with its two
edges, at no cost a window. Its samples are placed by ITS OWN edges on its own cycles; its windows are a named stamped
variant's (manifest `windows_from`) on ITS cycles. STATED ASSUMPTION: the stamps add cycles to the windows and none to the
setup, so the code before the first window (the skipped windows are windows) is the same cycles in both, and the rest of the
edges variant's launch is the stamped variant's windows in their relative lengths, each shortened by the factor
(T_edges - setup) / (T_stamped - setup) (`Launch.fractions_on`). THE CHECK: the edges variant's placed fold against the
stamped variant's own, series by series: the rotation (to a fraction of a bin) and the correlation; a rotation over half a
bin or a correlation under 0.7 is a named finding; the rotation is also set against the two placements' measured spreads
(two each, in quadrature) and a bin. A plain variant with no stamps is placed through a stamped variant by rotation (the
shape check), and its fold is also compared with the edges variant's.

For a set of variants the plain kernel has no stamps: its samples are placed through a stamped variant (README, `set`).

### The table

A row a window bin; a column a ROLE (the phase most of the role's lanes are in, each lane read in its OWN window: a role
whose window is shifted is not smeared across the scope's), a column a varying stall reason (its folded value with its
standard error), a footer with each series' placement half-width. `--json` and `--perfetto` as before.
