# The run manifest: what warpline learned from one run of a kernel

The static [manifest](checker-manifest.md) is a build output: what the compiled code of each kernel is and what the rules found
in it, with no GPU. Some facts exist only after the kernel has run: its timeline, read out of a record of its stamps (the
recorder), and a profiler's counters and verdicts on it (the checker's [profile rules](checker-profile.md)). A RUN MANIFEST is
one JSON file a run of a kernel that holds them, in the same family as the static manifest and judged by the same
[checkers](checker-manifest.md#the-checkers-in-the-declarations). It is the one form a consumer keeps of what warpline learned
from a run, whether the run was stamped, profiled or both.

    python -m warpline run --stamps --profile SourceCounters --place NAME -o DIR -- ./task   # a run made and written: DIR/run.manifest.json
    python -m warpline run-manifest [--record R]... [--profile CSV]... --place NAME -o run.manifest.json
    python -m warpline run-manifest ... --place NAME --text          # every field, a line each, then the lines said
    python -m warpline compare run.manifest.json --declare declarations.json
    python -m warpline check <binary> --declare declarations.json --run run.manifest.json   # the binary and its run, judged as one

`warpline run` ([the run layer](run-layer.md)) starts the task, collects its records and the profiler's exports into one
directory and writes the run manifest there with this command's code, from those files; `run-manifest` takes files made any
other way.

- `--record R` (the recorder's): a record the stamps wrote ([the format](format.md)); repeat it for several records of one run,
  a detail each, read together as `summary A B C` reads them ([summary](commands.md)).
- `--profile CSV` (the checker's): a Nsight Compute export, the details page or the source page ([the exports](checker-profile.md#the-exports-ncupy));
  repeat it for both.
- `--place NAME`: the kernel the run is of. A record does not name its kernel, so the caller gives it: the static manifest's
  `place`, the demangled kernel name (`sample::tile()`). A profile's kernel is matched to it by the name before the parameter
  list (`warpline.checker.manifest.same_kernel`, the rule a profile is matched to a binary's kernel by); a lone profile is paired
  with the place whatever it is called, and a profile of another kernel is a place of its own, said to match no place.

At least one input is given. A stamped-only run gives records alone, a profiled-only run exports alone, a run with both gives
both. `run-manifest` exits 0, and 2 when an input is not usable or not given. The command is assembled from the modules present
([the two modules](modules.md)): each module gives what it can read, and an input whose module is absent is refused by name
(`--profile needs the checker module`). A tree with only the recorder writes and judges a run manifest of its figures; one with
only the checker, of its profile rules.

## The file

    {"format": "warpline-run-manifest/1",
     "said": ["note: waits not stamped for role consumer (it arrives at req): no waits, no work"],
     "places": [
      {"place": "sample::tile()", "kernel": true,
       "figures": {
        "health": {"lanes": 2, "roles": 2, "stamps": 248, "dropped": 0, "findings": {}, "open_at_end": {"lanes": 0, "regions": 0}},
        "launch": {"cycles": 12150, "ns": 12150, "grid_ns": 12400, "scopes": 2},
        "window": {"anchor": "tick.begin", "period": 1000.0, "period_std": 0.0, "period_se": 0.0, "windows": 10},
        "roles": {
         "producer": {"period": 1000.0, "work": 710.0, "work_share": 0.71,
                      "waits": {"count": 1.0, "cycles": 290.0, "cycles_se": 3.16, "share": 0.29,
                                "by_barrier": {"ack": {"count": 1.0, "cycles": 290.0, "share": 0.29}}},
                      "phases": {"load": {"cycles": 300.0, "cycles_se": 3.16, "share": 0.3, "jitter": 10.0, "held": 1.0}, ...}},
         "consumer": {"period": 1000.0, "phases": {...}}},
        "counters": {"dram__bytes_read.sum": 110016, "sm__inst_executed.sum": 6301, ...}},
       "rules": {
        "shared_wavefronts_excess": {"count": 98304, "bound": false, "default": 0, "declarable": true,
                                     "message": "...", "fix": "...",
                                     "sites": [{"where": "0x3b90", "count": 32768, "what": "LDS.U16 R20, [R0]: ..."}]}}}]}

The same shape as the static manifest wherever the meaning is the same: `format`, `said`, `places`; a place has `place`,
`kernel`, `figures` and `rules`. The named place is the kernel (`kernel: true`) and always has `figures` (`{}` where no record was
given); a place a profile names that is not it has `rules` and no `figures`. A run manifest has no `notices` and no mangled
`name`: those are the build's.

- `figures`: the run's figures as the recorder's `summary` computes them (`summary.digest`, `digest_many` for several records,
  with `summary`'s defaults: the first scope, its anchor, its windows skipped), read from the records by the recorder itself:
  no saved `summary --json` in between. The parts, in the file's order:
  - `health`: `lanes`, `roles` (how many), `stamps`, `dropped`, `findings` (a count by kind, the record's own checks, `verify`),
    `open_at_end` (`lanes`, `regions`: closed at a lane's last stamp).
  - `launch`: `cycles`, `ns` (the first scope's two edges with ties); `grid_ns` and `scopes`, THE GRID'S SPAN: the earliest
    tied `launch.begin` to the latest tied `launch.end` over every scope that stamps both, on the record's one nanosecond axis
    (the ties are joined across scopes, the cycle counters never: `warpline.recorder.launch.grid_span`), and how many scopes it is over (a line
    of `said` where some scope stamps no tied edges). A floor's bound is on `grid_ns` ([the floor](floor.md)): the work a
    project declares is the launch's, every CTA's.
  - `window`: `anchor` (the anchor's name: a note, never a field), `period` (cycles), `period_std` and `period_se` (its
    window-to-window standard deviation and its standard error), `windows` (the steady windows the period rests on).
  - `roles/<role>`: `period` (its own window's), `work` and `work_share` (the period less its stamped waits, as a share of the
    window), `waits` (`count`, `cycles` and its standard error `cycles_se`, `share`, and `by_barrier/<barrier>`: `count`,
    `cycles`, `share`), `phases/<phase>` (`cycles` and its standard error `cycles_se`, `share` of the window, `jitter`, the
    window-to-window spread, and `held`, the share of windows that hold it). A phase is named as `summary` labels it
    (`name`, or `name#occurrence` where a role's window holds it more than once).
  - `counters` (the checker's, from a profile's details page): each raw metric the profiler collected on the kernel's
    launch, by its name (`dram__bytes_read.sum`, `sm__inst_executed.sum`; a section's figures named in words, `DRAM
    Throughput`, are not counters), what [the floor's counters](floor.md#the-counters) are judged on
    (`counters/<metric>`). They are one launch's: where the pages hold several launches of the kernel, none is written and a
    line of `said` says so (`warpline.checker.manifest.run_counters`).

  Where `summary` gives a standard error it is kept beside its figure (`period_se`, `cycles_se`). Where `summary` gives null,
  the figure is ABSENT and a line of `said` says why, never 0 in place of not measured: a role whose waits the record does not
  stamp has no `waits`, `work` or `work_share`; a record with no launch edges with ties has no `launch`; one with no steady
  window has no `window` and no role's period or phases; waits taken from several records have no standard error.
- `rules`: the profile rules' findings, written exactly as the static manifest writes a rule's entry (`count`, `bound`,
  `default`, `declarable`, `message`, `fix`, `sites`). A site's `where` is the instruction's offset in the kernel (its address less
  the first row's: the same axis as a static rule's site), or the profiler's rule name (`profiler_verdicts`).
- `said`: what was not run or not measured: `note: no record given: the run's figures are not measured`, `note: profile rules not
  run (no --profile)`, `note: profile rule X not run on 'K': the profile has no details page`, `note: waits not stamped for role X
  (...)`, a record of several that disagrees with the first, the scopes not measured; and, in a manifest `warpline run` wrote,
  `note: profiled with Nsight Compute <version>` after the profile's lines.

The file has no time, path, host or run identity (a record of several is named by its place among them), and its keys are in a
fixed order: the same input files write the same bytes. The profiler's version, which `warpline run` says, is the tool's that
made the exports, not the run's identity. Which commit, card and moment a run was is the storing project's to
record beside the file, never warpline's. `warpline.manifest.loads(text)` reads and validates either format (`ManifestError`:
the format, the shape, a figure that is not a number, a structural figure that is not a count, a part no run has, a count that
is not its sites' sum, a kernel with no figures); `warpline.manifest.dumps(m)` writes it; `warpline.manifest.run_manifest(place,
parts)` assembles one from what the modules give (`manifest.RunPart`: the recorder's `summary.run_figures(digest)`, the
checker's `manifest.run_rules(profiles, place, profile_rules)`).

A FIELD is a number of the figures or a rule's count, at its path: `health/lanes`, `window/period`,
`roles/producer/phases/load/cycles`, `roles/producer/waits/by_barrier/ack/share`, `rules/shared_wavefronts_excess`. `--text`
prints each, `path = value`, and `warpline.manifest.fields(place)` returns them.

## Judging values that vary

`warpline compare` takes a run manifest and judges it with the checkers of the static manifest, from the same declarations:

| checker | on a run manifest |
|---|---|
| `violations` | the profile rules' counts, as for any rule: `allow`, else the rule's default |
| `bound` | any run figure, with `at_most` or `at_least` (a number: `0.95` of a share, `1100` cycles) |
| `bound` `exactly` | refused at load (exit 2): a run's figure varies from run to run. The STRUCTURAL figures take it: `health/lanes`, `health/roles`, `health/dropped`, `launch/scopes` |
| `idle` | the run's role phases against the phases a binding stream is busy in ([the floor](floor.md#the-bound-and-the-idle-check)) |
| `hash` | refused at load (exit 2): a hash holds values that do not move, and a run's do |
| coverage, `waive` | as for the static manifest: every field covered or waived |

The structural figures are integers the launch fixes, not measurements: how many lanes and roles the kernel was launched with,
how many stamps a ring dropped (a project that reads a run only when nothing was dropped holds `exactly: 0`), and how many
scopes stamped the launch's tied edges (a project that reads the grid's span only over every CTA holds its grid). Every other
run figure is a measurement of one run and moves with the clocks, what else ran and where the work was placed.

    {"declarations": [
      {"kernel": "^sample::tile\\(", "rule": "shared_wavefronts_excess", "allow": 98304, "reason": "over the profiled window"},
      {"kernel": "^sample::tile\\(", "check": "bound", "figure": "window/period", "at_most": 1100, "reason": "the window's budget"},
      {"kernel": "^sample::tile\\(", "check": "bound", "figure": "window/windows", "at_least": 10,
       "reason": "the steady windows the period rests on"},
      {"kernel": "^sample::tile\\(", "check": "bound", "figure": "health/dropped", "exactly": 0,
       "reason": "a run whose ring dropped stamps is not read"}]}

A project bounds how much evidence a figure rests on with an ordinary bound on the window count (`window/windows`, `at_least`).
A bound on a figure the run did not measure (`roles/consumer/waits/cycles` where the consumer's waits are not stamped) names
no field and is refused, as a bound on a renamed loop is. No statistics are made here: whether two runs differ is the
recorder's `diff` and its two-standard-error test ([diff](commands.md)).

## With the static manifest: one declarations file

A project that has both keeps one declarations file and judges the binary and its kernels' runs as one: `check <binary>
--declare FILE --run run.manifest.json` (`--run` repeated, a run manifest a kernel), or `compare build.manifest.json
run.manifest.json --declare FILE`. Several manifests are judged as one: a kernel is its place name, and its fields are those of
every file that holds it, so a bound on `kernel/registers` and a bound on `window/period` of one kernel are both judged and
neither names no field. `check --run` refuses a file that is not a run manifest.

## In the build graph

`checker.bzl` judges a run manifest in a consumer's graph the way `warpline_check_test` judges a binary:

    load("@warpline//:checker.bzl", "warpline_check_test", "warpline_compare_test")
    warpline_compare_test(name = "tile_run", manifests = ["tile.run.manifest.json"], declare = "tile.declarations.json")
    warpline_check_test(name = "tile_check", binary = ":tile.cubin", declare = "tile.declarations.json",
                        runs = ["tile.run.manifest.json"])

The run manifest is a file the project keeps (a run on a GPU wrote it, `warpline run`); the tests read it and run no GPU. `warpline_check`'s
comparison action takes `runs` as well. `//:run_manifest_example` writes one in this repository's graph from a record the
recorder's test helpers make and a profile fixture, and `//:run_manifest_compare_test` and `//:run_manifest_check_test` judge it
against `tests/run_manifest.declarations.json`.
