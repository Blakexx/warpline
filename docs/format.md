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

## Fields

| field | meaning |
|---|---|
| `format` | `"warpline/0"`. A reader refuses any other. |
| `clock` | `unit`: `"cycles"`. `hz`: the clock's frequency (a number) or `null`. Times are integers in this unit. |
| `names` | id (a string of an integer) -> name. Names are the user's; two ids never share a name. |
| `lanes` | A list. Each: `id` (a string; `scope/name`, e.g. `cta0/warp3`), `role` (a string: "a fold role"), optional `group`, optional `issued` (how many stamps the warp issued; with `dropped` it lets a reader account for every stamp). |
| `stamps` | A list of `[lane, name_id, time, payload?]`. `time` is an integer; `payload` an optional integer. A lane's stamps are listed in the order the warp wrote them; their times do not go back. |
| `dropped` | lane -> how many stamps were lost (a full ring drops the newest). |
| `facts` | Declared once about the kernel; see below. |
| `twin` | Optional: the same kernel with every stamp compiled out, measured in the same session. `{"plain": cycles, "recorded": cycles}`: the window's time untraced and traced (registers and spills may be added). `merge` flags a run whose recorded/plain exceeds 1 + 2 x tolerance, and reports a run with no `twin` as unchecked. |
| `source` | Optional free object: where the record came from. Not read. |

A **scope** is the part of a lane id before `/` (a CTA). Lanes of one scope share a clock; the reader joins stamps
only within a scope, because clocks of different scopes are unrelated.

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
| anything else | a mark: kept, drawn as a point, joined to nothing. |

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
`unknown_name_id`, `bad_time`, `bad_payload`, `bad_dropped`, `bad_facts` (also: `facts.window.stamps` not one entry a stamp). Strict loading (`load(path, strict=True)`)
also runs the tier-0 checks and raises `StrictError` with the findings.

## The sampled layer

A profiler's time-sampled counters are not part of the stamp format; `fold`, `overlay` and `set` read them from the
profiler's export (docs/sampling.md) and write their results as JSON (`--json`).

`fold --json`: `period_samples`, `period_us`, `interval_us`, `blur` (`fraction_of_window`, `bins`), `flags`, `scan`
(the range, step, refinement, harmonic rule and flat bound used), `runs[]` (each run's `period_samples`, `period_us`,
`series[]` with `pass`, `start`, `status` = ok | flat | dead | disagrees | short, `period`, `score`, `halves`, `drift`,
`bins`), `run_shifts[]` (`status`, and the `shift` with `bins`, `corr`, `runner_up_shift`, `runner_up_corr`) and
`series{}`: the average fold of each series, `bins[]` of `{mean, se, n}` over the window in equal parts.

`overlay --json`: `stamps` (the window in cycles), `samples`, `profile` (mean lanes in a wait a bin, smoothed), `fold`,
`alignment` (`shift`, `corr`, `runner_up_shift`, `runner_up_corr`, `accepted`, `finding`, `corrs` for every shift) and
`overlay` (null when refused): per bin, `phase` of each role with its `share` of lanes, and each reason's `mean`, `se`,
`n`, rotated onto the window position. The shift is the rotation of the samples' fold: `rotated[b] = fold[(b + shift)
mod bins]`.

The Perfetto file of `overlay --perfetto` is `perfetto`'s output for the average window plus a process of counter tracks
(`ph: "C"`), one a varying reason, a value at the start of each bin over one average window; `otherData.sampled_layer`
carries the blur and the alignment.

The set's manifest (README) and `set --json`: per variant `status` = plain | ok | over | unchecked, `period_us`,
`period_cycles`, `clock_cycles_per_us`, `ratio_to_plain` and its `ratio_basis` (us | cycles). A merged record written by
`set -o` carries `facts.variants` (run index to variant name) beside the merge's `facts.runs`.
