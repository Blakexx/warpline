# The three commands an agent iterates with: `summary`, `diff`, `sql`

A record of a real kernel holds 8,000 to 31,000 stamps; an agent must not read them. These three answer what an agent
asks again and again, each in one screen: what does this kernel's window look like (`summary`), what changed between two
runs (`diff`), and an ad hoc question (`sql`). Text output is at most 40 lines; where something does not fit it says what
was cut.

## `python -m warpline summary <record> [--cta C] [--anchor NAME] [--skip N] [--json]`

One line of health (lanes, roles, stamps, dropped, ties, findings BY KIND with counts: `check` lists them), the launch
(cycles, time, cycles an us; needs `launch.begin`/`launch.end` with ties, else said), the window (anchor, period in
cycles and time, spread over the steady windows), then per ROLE: its lanes, its own period, its WORK (period less its
stamped waits, as a share of the window), its waits (a window: count, total cycles and share, by barrier group) and its
phases in order with duration in cycles, share of the window, its own time from the ties (the cycle counter's rate changes
between phases, so a phase's time is read from its own ties, not from one rate), and the window-to-window spread; a phase
a window does not always hold shows `held N%`.

THE LONGEST CHAIN: the role whose work is the largest share of the window, the role the window is waiting on as far as
the record can say. A record that stamps no barrier waits (a coarse record) cannot tell: the line says `not named`, and
shows no work. Even where named, a wait the record does not stamp counts as work, and a role that waits on another's work
is shown by its waits, not by what it waits for. When a role has more phases than the screen holds, its smallest are
left out (`--json` has all).

`--json` keys (stable; new keys are added, none renamed): `scope`, `scopes`, `health` {`lanes`, `roles`, `stamps`,
`dropped`, `ties`, `findings` {code: count}}, `launch` {`cycles`, `ns`, `cycles_per_us`, `setup_cycles`} or null with
`launch_note`, `window` {`anchor`, `anchor_lane`, `windows`, `period`, `period_std`, `period_se`, `period_ns`, `skip`} or
null with `window_note`, `waits_stamped`, `roles` [{`role`, `lanes`, `period`, `period_std`, `windows`, `own_window`,
`work`, `work_share`, `waits` {`count`, `cycles`, `share`, `std`, `se`, `by_barrier` [{`barrier`, `count`, `cycles`,
`share`}]} or null, `phases` [{`name`, `occ`, `label`, `start`, `cycles`, `ns`, `cycles_per_us`, `share`, `jitter`,
`se`, `held`, `windows`}]}], `longest_chain` {`named`, `role`, `work_share`, `next_role`, `next_work_share`, `note`}.
Cycles are the record's clock; `ns` is null where the record has no ties; a wait's count and cycles are a window, a lane
averaged over its role's lanes.

## `python -m warpline diff <before> <after> [--cta C] [--anchor NAME] [--json]`

The same structure, a row a role and phase: before, after, the change in cycles and percent, and a mark. The launch, the
window's period and each role's waits (cycles a window) likewise; phases in only one record are listed. Two records of
different kernels (different lanes or roles) are REFUSED, naming what differs (exit 2).

THE TEST. Each record's own window-to-window standard deviation of a quantity, over the square root of the windows it
holds, is that quantity's standard error `se`. A change under TWO combined standard errors, `2 * sqrt(se_before^2 +
se_after^2)`, is `~` (not resolved); over it `*` (outside the spread). The launch is one value a record, with no spread:
no mark. This is the spread WITHIN each run, not between runs: two runs of one kernel can differ by more (the clocks,
what else ran, placement). To see how much, diff two runs of the same kernel first, and read a change no larger than
what they show as noise. Rows are ordered resolved first, then by size; rows past the screen are counted.

`--json` keys: `test`, `caveat`, `launch` {`before`, `after`, `change`, `percent`, `ns_before`, `ns_after`, `note`} or
null, `window` and `rows` [{`kind` (`window`, `phase`, `waits`), `role`, `name`, `before`, `after`, `change`, `percent`,
`se`, `resolved`}], `only_before`, `only_after`, `anchors`, `waits_stamped`.

## `python -m warpline sql <record> "<query>" [--csv|--json] [--limit N] [--cta C]`

The record is loaded into an in-memory SQLite database (read-only once loaded) and one query is run. Times are INTEGER
CYCLES. Output is capped at 50 rows (`--limit N`; the output says when rows were cut). `sql <record> --schema` prints the
tables and columns; `--named` lists the named queries with their SQL (they teach the schema), `--named NAME` runs one:
`phases-by-role`, `longest-waits`, `waits-by-barrier`, `window-jitter`, `rate-by-phase`, `stamps-by-name`,
`region-spread`. `--json` prints `{columns, rows, truncated, limit, row_count}`.

| table | one row | columns |
|---|---|---|
| `lanes` | a lane | `lane`, `scope`, `role`, `grp`, `stamps`, `dropped` |
| `stamps` | a stamp | `idx`, `lane`, `t`, `name`, `kind`, `key`, `payload` |
| `regions` | a region of a lane | `lane`, `name`, `depth`, `t_begin`, `t_end`, `cycles`, `window` |
| `windows` | a steady window | `scope`, `role` (`''` the scope's; else the role's own), `idx`, `t_begin`, `t_end`, `cycles` |
| `ties` | a tie | `lane`, `cycles`, `ns` |
| `waits` | a wait | `lane`, `role`, `kind`, `barrier`, `barrier_group`, `t_begin`, `t_end`, `cycles`, `window`, `released_by`, `released_at`, `release_kind` |
| `phases` | a role's phase in its average window | `role`, `name`, `occurrence`, `start`, `cycles`, `ns`, `share`, `jitter`, `held`, `lanes` |
| `findings` | a check's finding | `code`, `lane`, `t`, `message` |

This format has no `site` on a stamp, so `stamps` has `kind` and `key` (the name's meaning by the naming convention)
and the payload instead. `windows` and `regions.window` count from 0 at the first steady window of the scope's anchor
lane; `phases` are in each role's own window. `waits.released_by` is where the reader's join found the releasing stamp.

## Bounded output of the older commands

`check`, `rates`, `merge`, `overlay` and `set` group findings by kind: the kind, its count and the worst case (or the
first), `--all` for every one. `rates` prints a line a role for the ties and omits the per-lane region lines and a lane
a line for the counter check unless `--all`. `--json` is unchanged.
