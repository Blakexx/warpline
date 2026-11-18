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
   window's period jitters (5,462 cycles, standard deviation 319): a record's one length is never compared to a pass to the
   sample, and a sample is never placed by one mean period.

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
   of the launch, the cycle f * T past the entry; its WINDOW POSITION is read from the record's own window anchors (f falls
   between two consecutive anchors: the fraction of that window), never from one mean period. STATED ASSUMPTION: over a
   pass the profiler's time is linear in the kernel's cycles (even over a window and a launch, not within a phase).
3. THE CHECK that replaces the length check: a pass's CYCLES A SAMPLE (T over its live span) against the median of its
   variant's passes; one more than `--cps-tolerance` (default 3%) away is flagged and left out. Ties are no longer needed to
   place samples; they still give the rate and the launch's time.
4. THE SCALE b - a is fixed by the fold's sharpness: the share of the pass's variance in the first three harmonics of its
   fold, summed over its series, scanned over the range the edges allow; the plateau within one standard deviation of the
   score's own noise is the scale's uncertainty. A run whose sharpest scale is at the edge of the range, or beyond it, says
   so. THE OFFSET a is not seen by one pass: runs of the SAME series are the same profile rotated by their own a; each
   run's rotation against the first is the phase of the first harmonics of the samples at their placed positions (to a
   fraction of a bin, with a bootstrap standard deviation), and the runs' intervals for a are intersected through them. Per
   series, the half-width (the entry's plus the scale's) is printed in windows and bins with the runs that fed it; over half
   a bin it is LOOSE. No period is fitted here, so none of the period checks of `fold` can drop a series (they did: see
   the passes' series disagreeing with their reference by 1%, or flat in a run, which left no series common to every run).
5. THE FRAME: each pass prints its sharpness on the record's window anchors beside its sharpness on a mean period over the
   same span; where a mean period is sharper by more than the noise the samples do NOT follow the anchors' jitter (the
   anchor stamp's wobble is not the kernel's rhythm), the finding `ANCHORS NOT FOLLOWED` is named, and `--frame mean` places
   on the mean period.
6. THE CROSS-CHECK: where the record has barrier waits, the placed `barrier` fold against the waits' profile (smoothed by
   the samples' box): the rotation in bins and the correlation; they agree within the placement's half-width plus a bin, or
   a named finding.

THE EDGES VARIANT: a variant whose record has the launch events and no window anchors is the plain kernel with its two
edges, at no cost a window. Its samples are placed by ITS OWN edges on its own cycles; its windows are a named stamped
variant's as fractions of that variant's launch (manifest `windows_from`). STATED ASSUMPTION: the stamped variant's
windows fall at the same fractions of the launch as the edges variant's (the stamps cost cycles in the windows and not in
the setup, so the setup's share of the launch differs: the check below measures what that does). THE CHECK: the edges
variant's placed fold against the stamped variant's own, series by series: the rotation (to a fraction of a bin) and the
correlation; a rotation over half a bin or a correlation under 0.7 is a named finding. A plain variant with no stamps is
placed through a stamped variant by rotation (the shape check), and its fold is also compared with the edges variant's.

For a set of variants the plain kernel has no stamps: its samples are placed through a stamped variant (README, `set`).

### The table

A row a window bin; a column a ROLE (the phase most of the role's lanes are in, each lane read in its OWN window: a role
whose window is shifted is not smeared across the scope's), a column a varying stall reason (its folded value with its
standard error), a footer with each series' placement half-width. `--json` and `--perfetto` as before.
