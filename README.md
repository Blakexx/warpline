# warpline

A kernel's timeline, read out of stamps, and a checker of the compiled kernel. warpline is ONE repository of TWO MODULES
that share only the one command ([the two modules](docs/modules.md)): the **recorder** (`warpline/recorder/`: the stamps
and the Python that reads their records) and the **checker** (`warpline/checker/`: the vendor tools' outputs, the kernel
model, the rules, facts and declarations). Read the recorder's half for a kernel's timeline, the checker's for a compiled
kernel's code. The recorder has TWO HALVES:

- **in the kernel**: `include/warpline/stamp.cuh`, a few free functions (`stamp`, `mark`, a cursor, `dump`) that write
  a clock value and a name where YOU say, switched at compile time by YOUR condition; `include/warpline/host.hpp`, which
  turns the rows copied back into a record. The pieces and [the guide](docs/guide.md) (what a stamp costs, where stamps
  go, how to spend a budget) are this half; [examples/](examples/README.md) has a complete kernel.
- **on the record**: `python -m warpline.recorder` (or `warpline`), below: the checks, the average window, the merge of runs, the sampled layer.
  It reads any record in the [format](docs/format.md), whoever wrote the stamps.

For any GPU kernel: at each cycle of a window, what is each warp doing, what is each resource carrying and for whom,
what is each warp waiting on, and what ended the wait. An agent or a person reads it to choose the next change.

The **checker** (`warpline/checker/`, the second module): given a compiled kernel it says what is wrong with it, with no
GPU and no stamps: [the checker](docs/checker.md), [its model](docs/checker-model.md), [writing a
rule](docs/checker-rules.md), [the facts](docs/checker-facts.md), [declaring](docs/checker-declarations.md), [the notice ledger](docs/checker-notices.md), [profiler rules](docs/checker-profile.md). `python -m warpline.checker check <binary> --declare <file>` (or `warpline check ...`).

## If you are an agent, start here

Three commands, in this order, each answers in one screen (at most 40 lines) so that you never read a record's raw
stamps (a real kernel's record holds 8,000 to 31,000):

1. `python -m warpline summary <record>`: the record's health in a line, the launch, the window, and per ROLE its
   phases in order (cycles, time, share of the window), its waits by barrier, and THE LONGEST CHAIN (the role whose work
   is the largest share of the window). A role whose waits the record does not stamp (a detail run stamps one barrier's)
   shows "waits: not stamped in this record", no work, and is never the chain; the chain is named only when every role's
   waits are stamped. Several records of one kernel, one detail a run, read together:
   `python -m warpline summary A.json B.json C.json`.
2. `python -m warpline diff <before> <after>`: after you change the kernel, what moved: a row a role and phase, each
   change marked `*` (outside the spread) or `~` (not resolved). It compares one kernel's runs and refuses two different
   kernels. The spread is within each run: diff two runs of the SAME kernel first to see how much runs differ.
3. `python -m warpline sql <record> "<query>"`: the question nobody anticipated, over tables of the record
   (`--schema` lists them; `--named` lists ready queries with their SQL; rows are capped, `--limit N`).

`--json` on `summary` and `diff` gives the same as data. [docs/commands.md](docs/commands.md) is the reference.

## The one primitive

A **stamp**: a name and the clock's value, written by a warp at a point of its code. Nothing else is recorded.
Regions, waits, who released whom, what an asynchronous resource carried and the critical path are all computed by
the reader afterwards. See [docs/format.md](docs/format.md).

## The naming convention (the reader's only knowledge of meaning)

| name | meaning |
|---|---|
| `X.begin` / `X.end` | region `X` |
| `barrier.B.arrive`, `barrier.B.wait.begin` / `.end` | arrivals at, and waits on, barrier `B` |
| `op.R.issue` / `op.R.done` | an operation on resource `R`; the payload is its id |
| `write.F`, `poll.F.begin` / `.end` | the stamped write a polled wait watches, and the poll |

Facts declared once about a kernel (a barrier's arrival count, any pairing the names do not give) travel in the record.

## Commands (Python 3 standard library only)

    python -m warpline summary  <record> [<record> ...] [--cta C] [--tolerance PCT] [--json]   ONE SCREEN: health, launch, window, roles' phases and waits
    python -m warpline diff     <before> <after> [--json]     ONE SCREEN: what changed, outside the spread or not
    python -m warpline sql      <record> "<query>" [--csv|--json] [--limit N] | --schema | --named [NAME]
                                                         the record as SQLite tables, for an ad hoc question
    python -m warpline verify   <record> [--strict] [--all]   tier-0 logical checks of the record (findings by kind, the worst case)
    python -m warpline path     <record> [--cta C] [--lane L] [--steady REGION --skip N | --from T --to T]
                                                         critical path, the blocking waits with their releasers, slack
    python -m warpline step     <record> --at T [--cta C]  what every lane and resource is doing at time T
    python -m warpline window   <record> [--anchor NAME] [--skip N] [--cta C] [--json] [-o out]
                                                         the average window: each stamp's offset in the steady windows
    python -m warpline rates    <record> [--anchor NAME] [--skip N] [--cta C] [--tolerance PCT] [--all] [--json]
                                                         the rate between the cycle counter and time, from the ties
    python -m warpline merge    <record> <record> ... -o out [--tolerance PCT] [--anchor NAME]
                                                         several runs overlaid on their anchors
    python -m warpline perfetto <record> -o out.json     Chrome trace-event JSON for Perfetto's UI
    python -m warpline fold     <csv> [<csv> ...] [--bins N] [--json]
                                                         a profiler's sampled series: the window's period, the
                                                         checks, the fold onto one window, the blur
    python -m warpline overlay  <record> <csv> [<csv> ...] [--bins N] [--cta C] [--windows-from RECORD] [--frame local|anchors|mean]
                                [--frame-smooth H] [--boot N] [--cps-tolerance PCT] [--json] [--perfetto out.json]
                                                         the samples placed under a stamp record, on cycles, each pass
                                                         scaled by its own two launch edges
    python -m warpline set      <manifest.json> [--cost PCT] [--merge-unchecked] [-o merged.json] [--json]
                                                         variants the user promises are one kernel: checked, merged

`check`, `rates`, `merge`, `overlay` and `set` print findings grouped by kind with a count and the worst case; `--all`
prints every one (`--json` is unchanged). `rates` prints a line a role, not a line a lane, unless `--all`.

`path` prints the slack of every wait off the critical path: how much later its releaser could have arrived before
the wait's lane would have joined the path (the path's time at the lane's next dependency on the path, minus the time
the lane reached it). Where the record cannot say, the slack is "unknown", with the reason; it is never the wait's length.

### The average window

A stamp costs, so a kernel traced with every stamp runs slower than the kernel. The method: every run carries the same
few ANCHORS (a phase's start, say) and, inside a small stamp budget, one group of detail; the runs are overlaid on the
anchors. `window` is the first half. A window is the span between consecutive `begin` stamps of the anchor region on a
scope's first lane carrying it (default: the region whose begin recurs most regularly there; the command says which).
Over the steady windows (the first `--skip`, default 2, and the last partial one dropped) it gives each lane's each
stamp: its offset from the window's start (mean), its spread (min, max, standard deviation) and how many windows held
it. A lane's stamps are reported as they fall in the anchor lane's window, so a lane whose own window is shifted shows
it. THE TABLE IS BY ROLE, EACH IN ITS OWN WINDOW: a role's window is taken from the anchor event of the role's own lanes
where it has one (else the scope's, said), so a role whose window is shifted is not smeared across the scope's. Per role a
table of PHASES: the phase, its start within the role's own window, its DURATION (begin to end), the spread across lanes
and across windows, in cycles, and the role's own period; the durations of a role's phases sum to its period, and the
table says when they do not. `--json` prints an average-window record.

### The two clocks: ties and `rates`

A stamp's clock is a processor's cycle counter; a profiler's samples are on time, and the counter's rate against time
is not one number (it depends on what the kernel executes and whether the device's clock is locked). So the ANCHORS
(a window's start, a phase's start) can be taken on both clocks: a record's `ties` are `[lane, cycles, nanoseconds]`,
one moment read twice on one lane ([docs/format.md](docs/format.md)). The importer reads an event kind 6, "time", that
immediately follows a stamp of the same site, as that stamp's tie (it is no stamp).

`rates` reads, per lane, the rate between consecutive ties (cycles over microseconds); per REGION (one whose begin
stamp carries a tie and whose end, or the next tied stamp, carries one) its cycles, nanoseconds and rate, averaged over
the steady windows with the spread; by role (the spread across lanes shown) a table a region and the window's own
period in cycles, in nanoseconds and its rate. THE GRAIN, the time counter's step (the greatest common divisor of the
readings' differences; the smallest nonzero difference is stated beside it), gives every rate's uncertainty: the grain
over the interval's nanoseconds. A CHECK: the lanes of a scope share one processor's counter, so their rates over the
same window agree within `--tolerance` (default 0.5%); a lane that does not is a named finding (`lane_rate_disagrees`).

Where a variant's stamp record has ties, `set` reads the variant's window in time from them (its own measurement) and
reports it beside the period its samples give; they must agree within `--ties-tolerance` (default 2%), else the
finding `time_bases_disagree`; the variant's clock is reported from the ties a region. `overlay` does NOT use the ties
to place samples: a profiler's time sampling stretches the kernel in time and leaves its cycles unchanged, so the samples
are joined on CYCLES (a pass scaled by its own two edges, below); the ties give the rate and the launch's time. A record
without ties says so: time comes from the samples alone.

### The merge

`merge` is the second half. Each input is one run (its own binary: a stamp is switched at compile time). The ANCHORS
are the stamp names present in every run. Per anchor and role, the mean offset must agree across runs within
`--tolerance` (default 3, a percent of the window's period), and so must the periods; a run that disagrees is flagged
with the anchor and the sizes and left out of the merged timeline, its detail reported as untrusted. A run whose record
carries `twin` is flagged when recorded/plain exceeds 1 + 2 x tolerance; one without is reported as unchecked. The
merged record holds the anchors (mean over the accepted runs) and every other stamp from the run that carried it, each
tagged with its runs (`facts.runs`, `facts.window.stamps[i].runs`). The reader runs on it; a wait and its arrivals come
from one run, and the join refuses (`cross_run_join_refused`) to match a wait from one run with an arrival of another.

Perfetto output: a track a lane (regions as nested slices, waits as slices named by their barrier), a track a resource
(its operations as async slices), and flow arrows from the stamp that released a wait to the wait it ended.

`python -m warpline.recorder.importers.stamped_clocks <input.json> -o <dir> [--launch-begin NAME] [--launch-end NAME]` reads a
record of per-lane (clock, event) stamps with named kinds and writes warpline records. A phase-kind stamp whose site's
legend name is the launch's begin or end (defaults `entry` and `exit`) becomes `launch.begin` / `launch.end`, opens no phase
region, and `launch.end` closes the lane's open phase region; an input without them imports as it always did.

### The sampled layer

A profiler can read hardware counters while the untraced kernel runs: a series of values, one a sampling interval (see
[docs/sampling.md](docs/sampling.md) for how the file is made). The window repeats, so the samples FOLD onto one window.

`fold` reads one export a run: for each series the window's PERIOD in samples (the scan that maximizes the folded
profile's between-bin variance, 2..6 samples at 0.002, refined to 1e-4) with its score; then the checks below; then
every series folded on the agreed period into `--bins` bins (default 8; fewer than 10 samples a bin is refused), each
bin's mean, standard error and count, and the BLUR stated. Several runs of one kernel are aligned to the first by the
rotation of their `barrier` fold, then averaged; each run's shift and correlation are printed.

`overlay` takes a stamp record and places the samples under it, JOINED ON CYCLES, each pass SCALED BY ITS OWN TWO EDGES. The
kernel carries two forced, highly coarse marks, one at its ENTRY and one at its EXIT, each on both clocks (a stamp with a tie),
a pair a lane a launch, outside every loop, so they cost a window nothing: the instant events `launch.begin` and
`launch.end` ([docs/format.md](docs/format.md)). Both sides see those two moments: the stamps as the marks, the samples as the
series going live and going quiet. The overlay is REFUSED without them, naming what the kernel must stamp.

A profiler's time sampling STRETCHES the kernel in real time and leaves its cycles unchanged, so the join is on the cycle
counter. For a pass (the series collected together) with first live sample i0 and last i1, the entry is at sample-time i0 + a
and the exit at i1 + b, a and b in [0, 1) unknown; sample j's centre is at the fraction f = ((j + 0.5) - (i0 + a)) /
((i1 + b) - (i0 + a)) of the launch, the cycle f * T past the entry (T the record's launch in cycles), and its WINDOW POSITION
is read from a FRAME of the record's window anchors (f falls between two consecutive window starts: the fraction of that
window). STATED ASSUMPTION: over a pass the profiler's time is linear in the kernel's cycles (even over a window and
over the launch, not within a phase: fact of the kernel measured, said in the output). A pass's length varies from run to
run, so the old length check is gone; in its place a pass's CYCLES A SAMPLE (T over its live span) is reported, and a pass
more than `--cps-tolerance` (default 3%) from the median of its variant's passes is flagged and left out.

THE FRAME FOLLOWS THE SAMPLES, not one lane's anchors: the anchors' window-to-window jitter is not the samples' rhythm
(on the kernel measured the intervals' lag-1 autocorrelation is -0.5: an early window is followed by a late one, and a
mean period folds sharper in every pass). The default frame (`--frame local`) is a LOCAL fit of the anchors: at each anchor
a line fitted to the anchors with Gaussian weights of standard deviation h windows, so a slow drift of the window's length is
kept and a wobble shorter than about 2 pi h windows is not. The bandwidth h is CHOSEN, from the raw anchors (h = 0), 2, 4, 8,
16, 32 windows and the mean period (h -> infinity), by the pooled sharpness summed over the variant's passes: the smoothest
frame within one noise standard deviation of the sharpest. So the anchors are followed where the samples follow them and
not where they do not. `--frame anchors` and `--frame mean` are stated alternatives, `--frame-smooth H` fixes h, and the
sharpness of every pass on every candidate frame is printed with the one chosen, never silent.

THE FIT of a pass is JOINT. All its runs are fitted at once to one pooled profile: each run has an entry offset a in [0, 1)
sample and a scale d = b - a (the exit offset b in [0, 1)), and the pooled fold's SHARPNESS is maximized (the share of the
pooled samples' variance that the first three harmonics of the window explain, summed over the pass's series; a sample is a
total over an interval, which leaves little above the third, and a harmonic fit has no bin edges). No run is left out for
a weak fold. The objective is blind to a COMMON shift of every run (it only rotates the pooled profile), so the absolute
placement comes from the launch's edges alone: the BOX is the interval of common shifts that keep every run's entry inside
its first live sample and its exit inside its last (each run's scale known only to the plateau within one noise standard
deviation of its best), and the placement is the box's middle. THE UNCERTAINTY IS MEASURED: the runs are resampled (a
bootstrap over runs and every leave-one-run-out, each refitted from scratch) and the spread of each series' profile phase
is reported in windows and bins; the series' half-width is the box's half-width and two spreads in quadrature (over half a bin
it is LOOSE). THE ALIAS CHECK evaluates the pooled objective at the placement against its value with each run alone, and
every run together, shifted by one whole sample and by one window, with a margin of twice the resampled standard deviation of the
difference; a series whose placement moves by more than half a bin under a candidate within the margin, or whose
resampled fits land a sample away in a tenth of the cases, is named PLACEMENT AMBIGUOUS BY A SAMPLE, with both candidates.
Where the fit would move a run outside its first live sample by more than the margin it names FIT AND EDGES DISAGREE.
The table: a row a window bin; a column a ROLE (the phase most of its lanes are in, each lane in its own window), a column a
varying stall reason (its folded value with its standard error), a footer with each series' placement half-width. Where the
record has barrier waits the placed `barrier` fold is CROSS-CHECKED against them.

A record that has the two launch edges and NO window anchors (the plain kernel with its two edges: an EDGES variant) is placed
by its own edges on its own cycles, its windows another record's as fractions of that record's launch (`--windows-from
RECORD`; in a manifest `"windows_from": "<variant name>"`).

`--json` writes it as data and `--perfetto` the stamp lanes with one counter track a series over one average window.

`set` takes a MANIFEST of variants of one kernel (a stamp is switched at compile time, so each variant is its own
binary; warpline never switches one, it is told they are one kernel):

    {"kernel": "...", "variants": [
      {"name": "plain",  "plain": true,  "stamps": null,        "samples": ["a.csv", "b.csv"]},
      {"name": "coarse", "plain": false, "stamps": "rec.json",  "samples": []}]}

Exactly one variant is plain (paths are relative to the manifest). A stamped variant may omit its `name` when its record
carries a `variant` (`Record::variant("coarse")` on the host), which then names it. A variant may carry `"windows_from": "<name>"`: it is an
EDGES variant, its record the launch's two edges and nothing else, its windows the named stamped variant's as fractions of
that variant's launch; its samples are placed by its own edges and own cycles, and its fold is checked against the stamped
variant's own placed fold series by series (the rotation to a fraction of a bin and the correlation; over half a bin or under
0.7 is a named finding). The plain variant's fold is also compared with the edges variant's. For each variant: its window in microseconds from its
samples; where it has stamps, its window in cycles, and so the CLOCK during that run (cycles an us: a measurement of that
run, not a setting). THE PROMISE is checked: a variant's window over the plain one's is its COST; beyond `--cost` (default
6%, either direction) the variant is named and left out of the merge. A variant that has stamps and no samples cannot be
checked against a plain one that has samples only: it is UNCHECKED, never reported as passing (the report says what to
record: a sampling export of that variant's run, or a stamp record of the plain one), and is not merged unless
`--merge-unchecked`. The variants that pass and carry stamps are merged with the merge above, each stamp tagged by its
variant (`facts.variants`), and the PLAIN variant's samples are placed THROUGH a stamped variant, since the plain kernel has no stamps: (a) the stamped
variant's own samples are placed on cycles by its own launch edges (the cycles-a-sample check applies in full);
(b) the plain variant's fold of each varying series is rotated onto that variant's placed fold of the same series by
correlation, THE SHAPE CHECK of the promise (a correlation under 0.7, or a runner-up within 0.05, is a named finding for
that series: the variants' sampled profiles differ, or cannot be told apart by rotation; the series is not overlaid);
(c) the stamped variant's phases are shown with the plain variant's stall reasons placed by (a)+(b), the stamped variant's
own samples' placement printed beside them (and beside the edges variant's). Each placed variant reports each pass's cycles a
sample, shown even when it passes. A set whose stamped variants carry no launch edges is reported as before and the overlay
is REFUSED, naming them.

#### Assumptions of the sampled layer

CARRIED IN THE OUTPUT: each series' pass, each pass's cycles a sample, each run's fitted entry offset and scale (with the
scale's plateau), the pooled sharpness and the edges' box, the measured spread of each series' profile phase (bootstrap and
leave-one-run-out, in windows and bins), the alias check's objective values and margins, the placement's half-width a
series, the blur, the sharpness of every pass on every candidate frame and the frame chosen.

CHECKED, and flagged when it fails (a flag is a result; nothing is loosened to remove it):
- the passes behave alike: each series was collected in its own pass of the kernel, so each series' period must agree
  with the others' within `--tolerance` (default 1%); a series that does not is left out of the fold; a flat series is
  folded on the agreed period;
- the kernel is steady: the period from the first half of a series' steady span and from the second agree (the check
  resolves no better than the blur over the windows in a half: the report prints the spread it found);
- runs of one kernel agree in period, and each aligns to the first (correlation of 0.7 or more, no equal runner-up);
- a window under about two samples is flagged: it aliases;
- the promise of a variant set: window against the plain variant's, within `--cost`;
- a pass's cycles a sample (T over its live span) within `--cps-tolerance` (default 3%) of its variant's median (it replaces
  the length check, which refused most real passes: a pass's length varies); the stamps' barrier waits and the placed fold
  agree (the cross-check); an edges variant's placed fold has the stamped variant's own up to half a bin and a correlation of
  0.7 (and the rotation is also set against the two placements' measured spreads plus a bin); the plain variant's fold of a
  series has the stamped variant's shape up to a rotation (the shape check); a placement is not AMBIGUOUS BY A SAMPLE and the
  fit does not disagree with the edges (named where it does).

CAN ONLY BE STATED (nothing in one recording checks them):
- THE TWO CLOCKS: a stamp's clock is a cycle counter, and its rate against time depends on what runs and on the
  device's clock lock, so a cycle count becomes time only by its own record's ties. The anchors are read on both
  clocks (each an exact tie); a stamp between two ties is placed by interpolation, exact only to the ties' spacing
  (and exact where the rate is constant between them); no lane's clock is assumed another's: ties are used on their
  own lane (the lanes of a scope are checked against each other by `rates`, not assumed to agree), and nothing is joined
  across scopes by cycle count; the rate carries the uncertainty of the time counter's grain;
- a sample is a total over its interval, so the fold is the true profile smoothed by a box that wide (the interval over
  the period); detail narrower than the box is not there;
- the fold is an average window, never one window; steady windows only;
- a launch of one CTA stands for the grid (a full grid has a different window, by an amount that is measured, not
  assumed);
- SAMPLING STRETCHES TIME AND NOT CYCLES: a profiler's time sampling lengthens the kernel in real time (14 to 18% on the kernel
  measured) and leaves its cycles unchanged; the join is on cycles, and a sample's time is never taken as the stamps' time;
- A PASS IS SCALED BY ITS OWN TWO EDGES: its entry falls in its first live sample and its exit in its last (live: any series of
  the pass over the threshold); a profiler that reads zero in the first interval, or a kernel whose tail runs past its exit
  mark, puts an edge in a neighbouring sample, and the scale then reaches the end of the range the edges allow (the output
  says when it does, and when the sharpest scale lies beyond it);
- LINEAR OVER A PASS: over a pass the profiler's time is linear in the kernel's cycles (even over a window and over the launch
  to a percent, not within a phase): a sample inside a phase is placed as well as that;
- EACH PASS HAS ITS OWN START: the series of one report were not measured in one execution (one pass each), so each pass is
  placed by its own entry positions (one a run, fitted jointly); nothing is carried from one pass to another;
- THE EDGES FIX THE ABSOLUTE OFFSET, AND NOTHING ELSE DOES: the sharpness is the same when every run's profile is shifted
  together (by a sample, or by any amount), so the placement rests on the first and last live sample being the sample
  of the kernel's entry and exit; a profiler that reads zero where the kernel already runs moves an edge a sample and the
  placement with it, and no resampling of the runs sees it (a common error): the alias check lists the shift of every
  run together as a rotation the objective cannot tell, and says the edges decide it;
- THE SPREAD IS THE RUNS': the resampled spread is the spread of the estimator over the runs of the recording; with fewer
  than three runs it cannot be measured and the half-width is the edges' box alone (said);
- THE PLAIN VARIANT'S SETUP AND PROFILE ARE THE STAMPED VARIANT'S, up to the rotation the shape check finds: the plain
  kernel has no stamps and is placed through a stamped variant; the rotation is known to a bin, and the shape check alone
  says the profiles are alike (a rotation of a profile that has a different cause is not told apart by it);
- sampling may perturb the kernel (the window with sampling on against off is not checked here);
- a sample's position in the window is taken at the CENTRE of its interval (the sample is a total over it): if the
  profiler stamps a sample at the end of its interval, the placement is late by half an interval;
- the stamps and the samples must be of the same launch (or of launches of one kernel whose cycles agree): see
  [docs/sampling.md](docs/sampling.md); the implied clock (stamped cycles over sampled microseconds) is printed so it can be
  judged;
- AN EDGES VARIANT'S WINDOWS ARE THE STAMPED VARIANT'S ON ITS OWN CYCLES: the stamps add cycles to the windows and none to the
  setup, so the code before the first window (the skipped windows are windows) is taken as the stamped variant's, in
  cycles, and the rest of the edges variant's launch as the stamped variant's windows in their relative lengths, each
  shortened alike (`Launch.fractions_on`); the check of the two placed folds measures what is left.

## Tests

    bazel test //...                               # run before a push: it includes the mirror's declarations check (docs/mirror.md)
    bazel run //:launch_set -- /tmp/set_synth     # a synthetic launch: a plain and a stamped variant, three runs each

## Not built yet

The in-kernel stamp (a pinned clock read and a floating store, conditional at compile time) and its service (a
shared-memory ring drained in the background, dropping and counting when full) do not exist yet. Today the reader
works on records written by other recorders through an importer. Also not built: a run's twin measured (the merge reads a `twin` it is given), a card's
capacity descriptor, the inferred basis for the work between stamps, a simulator.

## License

Apache-2.0, Copyright 2026 Blake Bottum.

## Building and installing

Everything is built by Bazel; there is no other packaging file.

    bazel test //...                       # the tests
    bazel run //:cli -- verify <record>    # the command (both modules' verbs), from a checkout
    bazel build //:wheel                   # the wheel: pip install bazel-bin/warpline-0.1.0-py3-none-any.whl

Another Bazel workspace depends on it as a module (`bazel_dep(name = "warpline")` at a commit), and on ONE of its modules by
the target: `@warpline//:recorder` (the reader), `@warpline//:checker` (the checker; `checker.bzl`'s macros use it alone),
`@warpline//:warpline` (both, with the one command). The Python import is `warpline.recorder.<module>` or
`warpline.checker.<module>`; the headers are `@warpline//:stamp` and `@warpline//:host`.
