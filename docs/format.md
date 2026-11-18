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
| `twin` | Optional: the window's time, and registers and spills, recorded against the same kernel with every stamp compiled out. |
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

## What the reader computes, and how each result is known

Every result carries a basis: `measured` (an interval between two stamps) or `upper_bound` (a completion, or a
flight ended by an unstamped arrival). A reader never reports a quantity the stamps do not bound.

Loading validates the schema and raises a named error: `bad_json`, `format_unsupported`, `missing_field`,
`bad_type`, `bad_clock`, `bad_name_id`, `duplicate_name`, `bad_lane`, `duplicate_lane`, `bad_stamp`, `unknown_lane`,
`unknown_name_id`, `bad_time`, `bad_payload`, `bad_dropped`, `bad_facts`. Strict loading (`load(path, strict=True)`)
also runs the tier-0 checks and raises `StrictError` with the findings.
