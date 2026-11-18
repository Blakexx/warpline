# warpline

A kernel's timeline, read out of stamps.

For any GPU kernel: at each cycle of a window, what is each warp doing, what is each resource carrying and for whom,
what is each warp waiting on, and what ended the wait. An agent or a person reads it to choose the next change.

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

    python -m warpline check    <record> [--strict]     tier-0 logical checks
    python -m warpline path     <record> [--cta C] [--lane L] [--steady REGION --skip N | --from T --to T]
                                                         critical path, the blocking waits with their releasers, slack
    python -m warpline step     <record> --at T [--cta C]  what every lane and resource is doing at time T
    python -m warpline window   <record> [--anchor NAME] [--skip N] [--cta C] [--json] [-o out]
                                                         the average window: each stamp's offset in the steady windows
    python -m warpline rates    <record> [--anchor NAME] [--skip N] [--cta C] [--tolerance PCT] [--json]
                                                         the rate between the cycle counter and time, from the ties
    python -m warpline merge    <record> <record> ... -o out [--tolerance PCT] [--anchor NAME]
                                                         several runs overlaid on their anchors
    python -m warpline perfetto <record> -o out.json     Chrome trace-event JSON for Perfetto's UI
    python -m warpline fold     <csv> [<csv> ...] [--bins N] [--json]
                                                         a profiler's sampled series: the window's period, the
                                                         checks, the fold onto one window, the blur
    python -m warpline overlay  <record> <csv> [<csv> ...] [--bins N] [--cta C] [--windows-from RECORD] [--frame anchors|mean]
                                [--cps-tolerance PCT] [--json] [--perfetto out.json]
                                                         the samples placed under a stamp record, on cycles, each pass
                                                         scaled by its own two launch edges
    python -m warpline set      <manifest.json> [--cost PCT] [--merge-unchecked] [-o merged.json] [--json]
                                                         variants the user promises are one kernel: checked, merged

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

`python -m warpline.importers.stamped_clocks <input.json> -o <dir> [--launch-begin NAME] [--launch-end NAME]` reads a
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
is read from the record's own window anchors (f falls between two consecutive anchors: the fraction of that window), never
from one mean period. STATED ASSUMPTION: over a pass the profiler's time is linear in the kernel's cycles (even over a window
and over the launch, not within a phase: fact of the kernel measured, said in the output). A pass's length varies from run to
run, so the old length check (a record's one length against the live span, to a sample) is gone; in its place a pass's CYCLES
A SAMPLE (T over its live span) is reported, and a pass more than `--cps-tolerance` (default 3%) from the median of its
variant's passes is flagged and left out. The scale b - a is fixed by the fold's sharpness within the range the edges
allow; the offset a by the runs of one pass, whose folds are the same profile rotated by their own a (the rotation to a
fraction of a bin, the runs' intervals intersected). Per series the half-width (the entry's plus the scale's) is printed in
windows and bins with the runs that fed it; over half a bin it is LOOSE. A pass also prints its sharpness on the anchors
against a mean period, and names `ANCHORS NOT FOLLOWED` where the samples do not follow the anchors' jitter (`--frame mean`).
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

Exactly one variant is plain (paths are relative to the manifest). A variant may carry `"windows_from": "<name>"`: it is an
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

CARRIED IN THE OUTPUT: each series' pass, each pass's cycles a sample, the scale and its plateau, the entry offset's interval,
each run's rotation (to a fraction of a bin, with its bootstrap spread), the placement's half-width a series, the blur, the
sharpness on the anchors against a mean period.

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
  the length check, which refused most real passes: a pass's length varies); the runs' entry intervals intersect; the
  stamps' barrier waits and the placed fold agree (the cross-check); an edges variant's placed fold has the stamped
  variant's own up to half a bin and a correlation of 0.7; the plain variant's fold of a series has the stamped
  variant's shape up to a rotation (the shape check); the samples follow the window anchors' jitter (else
  `ANCHORS NOT FOLLOWED`).

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
  placed by its own entry position (an interval, narrowed by every run); nothing is carried from one pass to another;
- THE PLAIN VARIANT'S SETUP AND PROFILE ARE THE STAMPED VARIANT'S, up to the rotation the shape check finds: the plain
  kernel has no stamps and is placed through a stamped variant; the rotation is known to a bin, and the shape check alone
  says the profiles are alike (a rotation of a profile that has a different cause is not told apart by it);
- a run's rotation against the first is known to its bootstrap spread (a fraction of a bin), which the intersection carries
  as two standard deviations; a rotation of weak strength (the folds' first harmonics barely agree) is printed as such;
- sampling may perturb the kernel (the window with sampling on against off is not checked here);
- a sample's position in the window is taken at the CENTRE of its interval (the sample is a total over it): if the
  profiler stamps a sample at the end of its interval, the placement is late by half an interval;
- the stamps and the samples must be of the same launch (or of launches of one kernel whose cycles agree): see
  [docs/sampling.md](docs/sampling.md); the implied clock (stamped cycles over sampled microseconds) is printed so it can be
  judged;
- AN EDGES VARIANT'S WINDOWS ARE THE STAMPED VARIANT'S, as fractions of the launch: the stamps cost cycles in the windows
  and not in the setup, so the first window's fraction differs between the two; the check of the two placed folds measures
  what that does.

## Tests

    bazel test //...
    bazel run //:launch_set -- build/set_synth     # a synthetic launch: a plain and a stamped variant, three runs each

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
    bazel run //:cli -- check <record>     # the command, from a checkout
    bazel build //:wheel                   # the wheel: pip install bazel-bin/warpline-0.1.0-py3-none-any.whl

Another Bazel workspace depends on it as a module (`bazel_dep(name = "warpline")` at a commit).
