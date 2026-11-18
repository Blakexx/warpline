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
    python -m warpline merge    <record> <record> ... -o out [--tolerance PCT] [--anchor NAME]
                                                         several runs overlaid on their anchors
    python -m warpline perfetto <record> -o out.json     Chrome trace-event JSON for Perfetto's UI
    python -m warpline fold     <csv> [<csv> ...] [--bins N] [--json]
                                                         a profiler's sampled series: the window's period, the
                                                         checks, the fold onto one window, the blur
    python -m warpline overlay  <record> <csv> [<csv> ...] [--bins N] [--cta C] [--json] [--perfetto out.json]
                                                         the fold laid under a stamp record's average window
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
it. The table is by role (lanes of a role averaged, their spread shown); `--json` prints an average-window record.

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

`python -m warpline.importers.stamped_clocks <input.json> -o <dir>` reads a record of per-lane (clock, event) stamps
with named kinds and writes warpline records.

### The sampled layer

A profiler can read hardware counters while the untraced kernel runs: a series of values, one a sampling interval (see
[docs/sampling.md](docs/sampling.md) for how the file is made). The window repeats, so the samples FOLD onto one window.

`fold` reads one export a run: for each series the window's PERIOD in samples (the scan that maximizes the folded
profile's between-bin variance, 2..6 samples at 0.002, refined to 1e-4) with its score; then the checks below; then
every series folded on the agreed period into `--bins` bins (default 8; fewer than 10 samples a bin is refused), each
bin's mean, standard error and count, and the BLUR stated. Several runs of one kernel are aligned to the first by the
rotation of their `barrier` fold, then averaged; each run's shift and correlation are printed.

`overlay` takes a stamp record whose lanes carry `barrier.*.wait` stamps (a fine record; without waits it says so and
stops), builds from its average window the mean number of lanes inside a wait at each window position, smooths it by the
samples' blur, and finds the SHIFT: the rotation of the samples' `barrier` fold with the largest correlation with it. It
prints the shift, the correlation and the runner-up's, and REFUSES the alignment (a named finding, no overlay) below a
correlation of 0.7 or when the runner-up is within 0.05: an alignment that could be another is none. Accepted, it prints
for each window position each role's phase (the phase most of the role's lanes are in) beside each varying stall
reason's value; `--json` writes it as data and `--perfetto` the stamp lanes with one counter track a reason over one
average window.

`set` takes a MANIFEST of variants of one kernel (a stamp is switched at compile time, so each variant is its own
binary; warpline never switches one, it is told they are one kernel):

    {"kernel": "...", "variants": [
      {"name": "plain",  "plain": true,  "stamps": null,        "samples": ["a.csv", "b.csv"]},
      {"name": "coarse", "plain": false, "stamps": "rec.json",  "samples": []}]}

Exactly one variant is plain (paths are relative to the manifest). For each variant: its window in microseconds from its
samples; where it has stamps, its window in cycles, and so the CLOCK during that run (cycles an us: a measurement of that
run, not a setting). THE PROMISE is checked: a variant's window over the plain one's is its COST; beyond `--cost` (default
6%, either direction) the variant is named and left out of the merge. A variant that has stamps and no samples cannot be
checked against a plain one that has samples only: it is UNCHECKED, never reported as passing (the report says what to
record: a sampling export of that variant's run, or a stamp record of the plain one), and is not merged unless
`--merge-unchecked`. The variants that pass and carry stamps are merged with the merge above, each stamp tagged by its
variant (`facts.variants`), and the plain variant's samples are laid on the merged timeline when a variant with waits
allows the alignment.

#### Assumptions of the sampled layer

CARRIED IN THE OUTPUT: each series' pass, the interval, the period found and its score, each run's shift and
correlation, the blur, the alignment's shift, correlation and runner-up.

CHECKED, and flagged when it fails (a flag is a result; nothing is loosened to remove it):
- the passes behave alike: each series was collected in its own pass of the kernel, so each series' period must agree
  with the others' within `--tolerance` (default 1%); a series that does not is left out of the fold; a flat series is
  folded on the agreed period;
- the kernel is steady: the period from the first half of a series' steady span and from the second agree (the check
  resolves no better than the blur over the windows in a half: the report prints the spread it found);
- runs of one kernel agree in period, and each aligns to the first (correlation of 0.7 or more, no equal runner-up);
- a window under about two samples is flagged: it aliases;
- the promise of a variant set: window against the plain variant's, within `--cost`.

CAN ONLY BE STATED (nothing in one recording checks them):
- a sample is a total over its interval, so the fold is the true profile smoothed by a box that wide (the interval over
  the period); detail narrower than the box is not there;
- the fold is an average window, never one window; steady windows only;
- a launch of one CTA stands for the grid (a full grid has a different window, by an amount that is measured, not
  assumed);
- the series of one report were not measured in one execution (one pass each): their relative phase is known to a sample
  (a share of a window equal to the blur), not better; an alignment found on the `barrier` series is applied to all;
- sampling may perturb the kernel (the window with sampling on against off is not checked here);
- a sample's position in the window is taken at the START of its interval: if the profiler stamps a sample at the end of
  its interval, the reported origin is late by one blur;
- the stamps and the samples may come from different launches: the overlay uses window position alone, and the
  implied clock (stamped cycles over sampled microseconds) is printed so it can be judged.

## Tests

    python3 -m unittest discover -s tests -v

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
