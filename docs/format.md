# The record format, `warpline/0`

One JSON document. The kernel records one primitive, a STAMP: a name and the clock's value, written by a warp at a
point of its code. Everything else (regions, waits, who released whom, what a resource carried, the critical path)
is computed by the reader from the stamps, their names and a few facts declared once.

```json
{
  "format": "warpline/0",
  "clock":  {"unit": "cycles", "hz": null},
  "names":  {"0": "fold.begin", "1": "barrier.full.arrive"},
  "lanes":  [{"id": "cta0/warp3", "role": "fold", "group": "box0", "issued": 1976}],
  "stamps": [["cta0/warp3", 0, 351], ["cta0/warp3", 1, 575, 12]],
  "dropped": {"cta0/warp3": 0},
  "facts":  {"barriers": {"full": {"count": 8}}, "operations": {}},
  "twin":   null
}
```

A kernel can write this record without a script in between: `include/warpline/host.hpp` (plain C++17) turns the rows
copied from the device into this document, with the stamps named, the launch's edges and ties, a phase's begin and end,
and the 32-bit clock words unwrapped ([guide](guide.md)). The format needed nothing added for it.

## Fields

| field | meaning |
|---|---|
| `format` | `"warpline/0"`. A reader refuses any other. |
| `clock` | `unit`: `"cycles"`. `hz`: the clock's frequency (a number) or `null`. Times are integers in this unit. With `ties` (below): `time_unit`: `"ns"` and `time_origin`, a free string saying where the nanosecond axis starts (the record's own origin). |
| `names` | id (a string of an integer) -> name. Names are the user's; two ids never share a name. |
| `lanes` | A list. Each: `id` (a string; `scope/name`, e.g. `cta0/warp3`), `role` (a string: "a fold role"), optional `group`, optional `issued` (how many stamps the warp issued; with `dropped` it lets a reader account for every stamp). |
| `stamps` | A list of `[lane, name_id, time, payload?]`. `time` is an integer; `payload` an optional integer. A lane's stamps are listed in the order the warp wrote them; their times do not go back. |
| `dropped` | lane -> how many stamps were lost (a full ring drops the newest). |
| `facts` | Declared once about the kernel; see below. |
| `twin` | Optional: the same kernel with every stamp compiled out, measured in the same session. `{"plain": cycles, "recorded": cycles}`: the window's time untraced and traced (registers and spills may be added). `merge` flags a run whose recorded/plain exceeds 1 + 2 x tolerance, and reports a run with no `twin` as unchecked. |
| `ties` | Optional: a list of `[lane, cycles, nanoseconds]`, each ONE moment read on both clocks on one lane (see TIES below). Absent or empty is valid. |
| `source` | Optional free object: where the record came from. Not read. |

A **scope** is the part of a lane id before `/` (a CTA). Lanes of one scope share a clock; the reader joins stamps
only within a scope, because clocks of different scopes are unrelated.

## Ties: the two clocks

A stamp's clock is a processor's cycle counter. A profiler's samples, and anything outside that processor, are on
time. The counter's rate against time is not one number: it depends on what the kernel executes and on whether the
device's clock is locked. So some stamps, the ANCHORS (a window's start, a phase's start), are taken on both clocks. A
TIE, `[lane, cycles, nanoseconds]`, is one moment's cycle count (on the stamps' own axis: the same origin and unit as the
`stamps`' times) and its nanosecond reading (an integer on the record's own time axis: `clock.time_unit` = `"ns"`,
`clock.time_origin` says where it starts), on one lane. A tie is an exact point between the two axes; two ties give the
rate between them; a stamp between two ties is placed in time by interpolation, exact only to the ties' spacing, and a
stamp outside a lane's tied span is not placed. A lane's ties are used on that lane: no lane's clock is assumed another's.

Loading checks that a tie's lane exists, that its coordinates are integers and that a record with ties states
`clock.time_unit`. That a lane's ties do not go backwards, in cycles or in nanoseconds, is a FINDING of `check`
(`tie_backwards`: a wrapped or mis-decoded reading), and a record whose ties go backwards places nothing by them.
`python -m warpline rates <record>` reads the rates (README).

## The naming convention

The reader's only knowledge of meaning. A name is matched by its shape:

| name | meaning |
|---|---|
| `X.begin`, `X.end` | brackets region `X` on the lane. Regions nest (a stack). `X` may contain dots. |
| `barrier.B.arrive` | an arrival at barrier `B`. |
| `barrier.B.wait.begin`, `barrier.B.wait.end` | the lane waits on `B`, from begin to end. |
| `op.R.issue`, `op.R.done` | an asynchronous operation on resource `R`; the payload is the operation's id. `done` is where a warp learned of completion: an upper bound. |
| `write.F` | the stamped write that a polled wait watches (field `F`). |
| `poll.F.begin`, `poll.F.end` | a polled wait on `F`. |
| `launch.begin`, `launch.end` | the launch's two edges: instant events, one of each a lane at most, each with a tie (see LAUNCH below). Never a region `launch`. |
| anything else | a mark: kept, drawn as a point, joined to nothing. |

## The launch's two edges: `launch.begin`, `launch.end`

The overlay of a profiler's time-sampled counters ([sampling.md](sampling.md)) is PLACED BY THESE TWO MARKS and refused
without them. The kernel stamps one at its ENTRY and one at its EXIT, each forced (always on, not a stamp that a budget
drops) and highly coarse, each on both clocks: a stamp WITH A TIE (`ties`, above), a pair a lane a launch, outside every
loop, so the pair costs a window nothing. Instant events, one of each a lane at most:

- `launch.begin`, `launch.end` are not a region: the reader draws them as points and never pairs them as `launch`;
- a lane carries both or neither; each has a tie (a tie whose cycle count is the stamp's); no stamp of the lane falls
  before its `launch.begin` or after its `launch.end`; a lane's open phase region is closed by its `launch.end`;
- the launch of a scope is from the EARLIEST lane's `launch.begin` to the LATEST lane's `launch.end`, in nanoseconds AND
  in CYCLES (`T_cycles`: the earliest lane's stamp to the latest lane's, the lanes of a scope sharing one counter). The
  samples are joined to the stamps on cycles (below).
- a record that has the two edges and NO window anchors is the plain kernel with its two edges (an EDGES variant, see
  `windows_from` below): its launch is read without windows, and its windows are a named stamped record's, as fractions
  of that record's launch.

WHY. The samples carry no mark of their own: a window's phase is not in a series of counters, only the series going live
(the kernel's entry, to within a sample) and going quiet (its exit). Placing the samples by correlating a stall series with
the stamps' barrier waits needs a record that stamps every wait, and such a record distorts the kernel it describes. Two
marks outside every loop do not. Both sides then see the same two moments: the stamps as the marks, the samples as the
first and last live samples; the entry's position inside its sample is an interval (narrowed by every run), never a point.
THE CYCLES: a profiler's time sampling STRETCHES the kernel in real time (14-18% on the kernel measured) and leaves its
cycles unchanged; so the launch is taken in cycles, and a sample's place in the launch is the fraction of the pass's own two
edges, times T in cycles (docs/sampling.md). A pass's length varies from run to run (159 to 165 live samples against one
record's 161.3): a record's one length is never compared with a pass's live span to the sample; the cycles a sample (T over
the live span) is reported a pass and checked against the variant's median.
`check` names, as findings (`launch_edges`): `launch_edge_missing` (a lane with one and not the other),
`launch_edge_untied` (an edge without a tie), `launch_edge_repeated` (more than one of a kind on a lane) and
`stamp_outside_launch`. A record with no edge at all holds, and imports as it always did; the overlay refuses it.

## Facts

```json
"facts": {
  "barriers": {
    "B": {"count": 4,                 // arrivals that complete a phase
          "stamped": 4,               // of them, how many are stamped (default: count)
          "issue": "R"},              // for an unstamped arrival: the resource whose `op.R.issue` is named as its cause
    "S": {"count": 0}                 // no arrivals: the lane's own work ends the wait
  },
  "operations": {
    "R": {"done": "barrier.B.wait.end"}   // an operation's completion is a wait end (instead of `op.R.done`)
  }
}
```

A barrier with `stamped < count` has an arrival no warp stamps (a bulk copy's landing). The reader says a wait on
it was released by something unstamped and names the nearest `op.R.issue` at or before the wait's end.
`operations[R].done` pairs each issue with the issuing lane's next such wait end; if that lane never waits on the
barrier, with the earliest wait end on it in the scope at or after the issue.

## Facts written by `window` and `merge`

`window --json` writes an AVERAGE WINDOW record: the same format, its `stamps` at mean offsets (rounded to cycles) from
the window's start, per lane in time order, and in `facts`:

```json
"window": {
  "scopes": {"cta0": {"period": 6731.4, "period_std": 146.8, "period_min": 6286, "period_max": 7126,
                      "anchor": "phase.window.begin", "anchor_lane": "cta0/warp0", "chosen": "...",
                      "windows": 61, "skip": 2}},
  "stamps": [{"lane": "cta0/warp8", "name": "phase.adding.begin", "occ": 0, "mean": 1720.9, "std": 131.3,
              "min": 1500, "max": 2100, "n": 61}]
}
```

`facts.window.stamps[i]` is the spread of `stamps[i]`: `mean` the offset, `occ` which time in the window the lane
stamped the name (0, 1, ...), `n` the windows that held it. A lane's offsets are as they fall in the ANCHOR lane's
window: a lane whose own window starts earlier shows its first stamps late in the previous window.

`merge` writes a merged record: an average window whose anchors are the mean over the accepted runs and whose other
stamps come from the run that carried them. Besides `window` it carries:

```json
"runs":  [{"index": 0, "source": "a.json", "accepted": true, "reference": true, "periods": {"cta0": 6731.4},
           "twin": {"plain": 6500, "recorded": 6620}, "twin_status": "ok", "twin_ratio": 1.018, "flags": []}],
"merge": {"tolerance": 0.03, "anchors": ["phase.window.begin"], "findings": ["..."], "untrusted": {"1": ["..."]}}
```

`twin_status` is `ok`, `over` or `unchecked` (no `twin`). In a merged record `facts.window.stamps[i]` also holds
`runs` (the accepted runs that carried the stamp; an anchor carries all of them) and `anchor`. The reader never
joins a wait to an arrival whose `runs` do not meet the wait's: the join finds `cross_run_join_refused`.

## What the reader computes, and how each result is known

SLACK of a wait off the critical path is how much later its releaser could have arrived before the wait's lane would
have joined the path: the path's time at the lane's next dependency on the path, minus the time the lane reached it.
The next dependency is the lane's first later stamp that is an arrival at a barrier phase a wait on the path waits for
(the path's time is that phase's last arrival) or a stamp inside a stretch of the path where the lane itself works
(slack 0). Where no later stamp is either, the slack is unknown, with the reason; never the wait's length.

Every result carries a basis: `measured` (an interval between two stamps) or `upper_bound` (a completion, or a
flight ended by an unstamped arrival). A reader never reports a quantity the stamps do not bound.

Loading validates the schema and raises a named error: `bad_json`, `format_unsupported`, `missing_field`,
`bad_type`, `bad_clock`, `bad_name_id`, `duplicate_name`, `bad_lane`, `duplicate_lane`, `bad_stamp`, `unknown_lane`,
`unknown_name_id`, `bad_time`, `bad_payload`, `bad_dropped`, `bad_tie`, `bad_facts` (also: `facts.window.stamps` not one entry a stamp). Strict loading (`load(path, strict=True)`)
also runs the tier-0 checks and raises `StrictError` with the findings.

## The sampled layer

A profiler's time-sampled counters are not part of the stamp format; `fold`, `overlay` and `set` read them from the
profiler's export (docs/sampling.md) and write their results as JSON (`--json`).

`fold --json`: `period_samples`, `period_us`, `interval_us`, `blur` (`fraction_of_window`, `bins`), `flags`, `scan`
(the range, step, refinement, harmonic rule and flat bound used), `runs[]` (each run's `period_samples`, `period_us`,
`series[]` with `pass`, `start`, `status` = ok | flat | dead | disagrees | short, `period`, `score`, `halves`, `drift`,
`bins`), `run_shifts[]` (`status`, and the `shift` with `bins`, `corr`, `runner_up_shift`, `runner_up_corr`) and
`series{}`: the average fold of each series, `bins[]` of `{mean, se, n}` over the window in equal parts.

`overlay --json`: `launch` (begin, end, length in ns and in cycles, `window_fractions`: the steady window starts as fractions
of the launch), `windows_from`, `cycles_a_sample` (the median, the tolerance and per pass and run `live`,
`cycles_a_sample`, `ratio`, `flagged`), `assumption` (the join's linearity), `passes` (per pass: its series, `a_lo`/`a_hi`
the entry offset in samples, `half_a_windows`, `half_d_windows`, `runs_fed`, the `narrowing` as runs were intersected, per
run its `live`, `d` (the scale b - a), `a`, `rotation_windows`, `findings`), `series` (per series: its pass,
`half_windows`, `half_bins`, `half_entry_windows`, `half_scale_windows`, `varying`, `runs_fed`, `loose`, and its `bins`
folded at its placement: bin b is window position b/N .. (b+1)/N), `cross_check` (the stamps' barrier waits against the
placed fold, where the record has waits), `findings`, `profile` and `overlay`: per bin, `phase` of each ROLE (each lane in
its own window) with its `share` of lanes, and each reason's `mean`, `se`, `n`. A record without the launch's edges is
refused (no output, exit 2).

The Perfetto file of `overlay --perfetto` is `perfetto`'s output for the average window plus a process of counter tracks
(`ph: "C"`), one a series, a value at the start of each bin over one average window.

The manifest's variant carries `windows_from`: the name of a stamped variant, for a variant whose record has the launch's
edges and no window anchors. `set --json`: per variant `windows_from`, `launch`, `placement` (as `overlay --json`) and
`check_against_windows_from` (per series: `rotation_bins`, `corr`, `finding`); and per variant `status` = plain | ok | over | unchecked, `period_us`,
`period_cycles`, `clock_cycles_per_us`, `ratio_to_plain` and its `ratio_basis` (us | cycles); `overlay`: the note that
refuses it (a stamped variant without launch edges), or, per stamped variant it places the plain samples through, its
`placement` (as `overlay --json`) and per plain series the shape check's `finding`, `rotation_bins`, `corr`,
`half_windows` and `bins`. A merged record written by
`set -o` carries `facts.variants` (run index to variant name) beside the merge's `facts.runs`.
