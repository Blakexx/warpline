# The three commands an agent iterates with: `summary`, `diff`, `sql`

A record of a real kernel holds 8,000 to 31,000 stamps; an agent must not read them. These three answer what an agent
asks again and again, each in one screen: what does this kernel's window look like (`summary`), what changed between two
runs (`diff`), and an ad hoc question (`sql`). Text output is at most 40 lines; where something does not fit it says what
was cut.

## `python -m warpline summary <record> [<record> ...] [--cta C] [--anchor NAME] [--skip N] [--tolerance PCT] [--json]`

One line of health (the record's `variant` when it names one, lanes, roles, stamps, dropped, ties, findings BY KIND with
counts: `verify` lists them, and the lanes OPEN AT THE END, a count, not findings: a lane that stamps no `launch.end` has
its open regions closed at its last stamp, and `verify` prints the same count), the launch
(cycles, time, cycles an us; needs `launch.begin`/`launch.end` with ties, else said), the window (anchor, period in
cycles and time, spread over the steady windows), then per ROLE: its lanes, its own period, its WORK (period less its
stamped waits, as a share of the window; none where the role's waits are not stamped), its waits (a window: count, total cycles and share, by barrier group) and its
phases in order with duration in cycles, share of the window, its own time from the ties (the cycle counter's rate changes
between phases, so a phase's time is read from its own ties, not from one rate), and the window-to-window spread; a phase
a window does not always hold shows `held N%`.

WHAT A RECORD CAN SAY ABOUT A ROLE'S WAITS, from its stamps alone, one of three (`waits_state`): `stamped` (a lane of the
role stamps a wait or a poll begin: its waits are measured); `known_unstamped` (no wait stamped, but a lane arrives at a
barrier: the role takes part in barriers whose waits this record does not stamp, named in the line); `unknown` (no barrier
stamped on it: the record cannot say). A detail run stamps one barrier's waits, so its other roles are the second or the
third. For those `summary` prints "waits: not stamped in this record" in place of 0, shows no work share (a period less
nothing is not work) and never names the role the longest chain. The format declares no role at a barrier (`facts.barriers`
holds counts), so a role that waits at a barrier and stamps no arrival there cannot be seen from the record; the smallest
addition would be a list of the waiting roles on a declared barrier.

THE LONGEST CHAIN: the role whose work is the largest share of the window, the role the window is waiting on as far as
the record can say. It is named only among roles whose waits are stamped, and only when EVERY role's are; otherwise the
line says `not named` and which roles' waits are not stamped. Even where named, a wait the record does not stamp counts
as work (a role's waits at a barrier the record does not stamp are not seen), and a role that waits on another's work is
shown by its waits, not by what it waits for. When a role has more phases than the screen holds, its smallest are left
out (`--json` has all).

SEVERAL RECORDS of one kernel (`summary A.json B.json C.json`; one detail a run): the launch, window and phases are the
first record's; each other record's period and each phase must agree with it within the larger of two combined standard
errors and `--tolerance` percent (default 3, the merge's) of the window, else the record is named, with what differs, and
its waits are not taken. Each role's waits are taken from the record(s) that stamp them, a barrier from the first that
stamps it (never twice); the output says they come from different runs, each a few percent from the plain kernel (its
stamps cost cycles), and which run gave which. The chain is named once every role's waits are stamped by some agreeing
record. Records of different kernels are refused. It reuses the merge's common anchor and tolerance; it does not use the
merge itself, which overlays stamps and drops a run on its anchors' offsets, where this asks whether the runs' numbers
agree. A detail run's own stamps stretch its window, so a coarse record and a detail run can disagree at the default:
raise `--tolerance` to take the run, knowing how much it stretches.

`--json` keys (stable; new keys are added, none renamed): `variant` (a string or null), `scope`, `scopes`, `health`
{`lanes`, `roles`, `stamps`, `dropped`, `ties`, `findings` {code: count}, `open_at_end` {`lanes`, `regions`}}, `launch` {`cycles`, `ns`, `cycles_per_us`, `setup_cycles`} or null with
`launch_note`, `window` {`anchor`, `anchor_lane`, `windows`, `period`, `period_std`, `period_se`, `period_ns`, `skip`} or
null with `window_note`, `waits_stamped` (any role), `roles` [{`role`, `lanes`, `waits_state`, `arrives_at`, `period`, `period_std`, `windows`, `own_window`,
`work`, `work_share`, `waits` {`count`, `cycles`, `share`, `std`, `se`, `by_barrier` [{`barrier`, `count`, `cycles`,
`share`}]} or null, `phases` [{`name`, `occ`, `label`, `start`, `cycles`, `ns`, `cycles_per_us`, `share`, `jitter`,
`se`, `held`, `windows`}]}], `longest_chain` {`named`, `role`, `work_share`, `unstamped_roles`, `next_role`, `next_work_share`, `note`}. `waits`, `work` and
`work_share` are null for a role whose waits are not stamped. With several records: `runs` [{`name`, `variant`, `base`,
`period`, `period_change`, `agrees`, `compared`, `disagree`, `twin_ratio`, `stamps_waits_of`, `arrives_only`}], `anchor_common`,
`tolerance`, and each role's `waits_runs` (the records its waits come from; a barrier's entry names its `run`).
Cycles are the record's clock; `ns` is null where the record has no ties; a wait's count and cycles are a window, a lane
averaged over its role's lanes.

## `python -m warpline diff <before> <after> [--cta C] [--anchor NAME] [--json]`

The same structure, a row a role and phase: before, after, the change in cycles and percent, and a mark. The launch, the
window's period and each role's waits (cycles a window) likewise, where the role's waits are stamped in BOTH records: a role
whose waits are not stamped in one or both is listed as "not stamped", not as 0, and a change from not stamped to a number
is not reported as a change; phases in only one record are listed. Two records of
different kernels (different lanes or roles) are REFUSED, naming what differs and, when either record carries one, the
two variants (exit 2).

THE TEST. Each record's own window-to-window standard deviation of a quantity, over the square root of the windows it
holds, is that quantity's standard error `se`. A change under TWO combined standard errors, `2 * sqrt(se_before^2 +
se_after^2)`, is `~` (not resolved); over it `*` (outside the spread). The launch is one value a record, with no spread:
no mark. This is the spread WITHIN each run, not between runs: two runs of one kernel can differ by more (the clocks,
what else ran, placement). To see how much, diff two runs of the same kernel first, and read a change no larger than
what they show as noise. Rows are ordered resolved first, then by size; rows past the screen are counted.

`--json` keys: `test`, `caveat`, `launch` {`before`, `after`, `change`, `percent`, `ns_before`, `ns_after`, `note`} or
null, `window` and `rows` [{`kind` (`window`, `phase`, `waits`), `role`, `name`, `before`, `after`, `change`, `percent`,
`se`, `resolved`}], `only_before`, `only_after`, `anchors`, `waits_stamped`, `waits_not_compared` [{`role`, `before`, `after` (a `waits_state`), `before_cycles`, `after_cycles`}].

## `python -m warpline sql <record> "<query>" [--csv|--json] [--limit N] [--cta C]`

The record is loaded into an in-memory SQLite database (read-only once loaded) and one query is run. Times are INTEGER
CYCLES. Output is capped at 50 rows (`--limit N`; the output says when rows were cut). `sql <record> --schema` prints the
tables and columns; `--named` lists the named queries with their SQL (they teach the schema), `--named NAME` runs one:
`phases-by-role`, `waits-by-role`, `longest-waits`, `waits-by-barrier`, `window-jitter`, `rate-by-phase`, `stamps-by-name`,
`region-spread`. `--json` prints `{columns, rows, truncated, limit, row_count}`.

| table | one row | columns |
|---|---|---|
| `lanes` | a lane | `lane`, `scope`, `role`, `grp`, `stamps`, `dropped` |
| `stamps` | a stamp | `idx`, `lane`, `t`, `name`, `kind`, `key`, `payload` |
| `regions` | a region of a lane | `lane`, `name`, `depth`, `t_begin`, `t_end`, `cycles`, `window` |
| `windows` | a steady window | `scope`, `role` (`''` the scope's; else the role's own), `idx`, `t_begin`, `t_end`, `cycles` |
| `ties` | a tie | `lane`, `cycles`, `ns` |
| `roles` | a role of the scope | `role`, `lanes`, `waits_state`, `arrives_at` |
| `waits` | a wait | `lane`, `role`, `kind`, `barrier`, `barrier_group`, `t_begin`, `t_end`, `cycles`, `window`, `released_by`, `released_at`, `release_kind` |
| `phases` | a role's phase in its average window | `role`, `name`, `occurrence`, `start`, `cycles`, `ns`, `share`, `jitter`, `held`, `lanes`, `waits_state` |
| `findings` | a check's finding | `code`, `lane`, `t`, `message` |

This format has no `site` on a stamp, so `stamps` has `kind` and `key` (the name's meaning by the naming convention)
and the payload instead. `windows` and `regions.window` count from 0 at the first steady window of the scope's anchor
lane; `phases` are in each role's own window. A role with no rows in `waits` has ZERO waits only where `roles.waits_state` is
`stamped`; otherwise the record does not stamp its waits (`waits-by-role` gives NULL, never 0, for it, and `phases` carries
the role's `waits_state`). `waits.released_by` is where the reader's join found the releasing stamp.

## Bounded output of the older commands

`verify`, `rates`, `merge`, `overlay` and `set` group findings by kind: the kind, its count and the worst case (or the
first), `--all` for every one. `rates` prints a line a role for the ties and omits the per-lane region lines and a lane
a line for the counter check unless `--all`. `--json` is unchanged.

## The checker's `manifest`

`python -m warpline manifest <binary | stem --fixture> [-o FILE]` writes the manifest (JSON); with `--text [--kernel REGEX]` it prints each
kernel's fields, one `path = value` a line, each loop's source line as a note above it: the names a `bound`, a `hash` or the coverage check of the declarations may use
([the manifest](checker-manifest.md)). Plain text, stable order.

## The checker's `held`

`python -m warpline held <binary | stem --fixture> [--kernel REGEX] [--lines LISTING]` belongs to the checker module (it reads a compiled
kernel, not a record) and is listed with `check`, `manifest` and `compare`: for each kernel and loop, what is live at the loop's top,
by the source line that made it ([the checker](checker.md#what-a-loop-holds)). Plain text, stable order, at most a line a place.
