# The floor: a kernel's lower bound, from its declared work and a card's measured rates

A FLOOR is what a kernel's work cannot take less than on a card: each stream's work over the rate the card reached on it,
and the stream that binds. It predicts nothing: it is a lower bound, and a project's target is a factor over it. Three
inputs, each a JSON file of its own format:

- the kernel's WORK, declared by the project, by phase (`warpline-work/1`);
- the card's RATES, written by the probes on the card (`warpline-rates/1`, [the probes](#the-probes));
- the PARAMETERS the work's expressions read (a problem's sizes: `--param NAME=VALUE`, or a JSON object's numeric fields).

```
python -m warpline floor kernel.work.json --rates card.rates.json --param ctas=132 [-o FILE] # the floor, by phase
python -m warpline floor kernel.work.json --rates card.rates.json --parameters cell.json \
    --bound kernel.bound.json --factor 1.15 [--kernel REGEX] [--figure launch/grid_ns]      # and the bound declarations
python -m warpline compare run.manifest.json --declare kernel.bound.json                    # a run judged by them
```

The code is the common part's `warpline/floor.py` ([the two modules](modules.md)); it reads neither module. The bound it
writes is judged by the common checkers (`warpline/checkers.py`) over a run manifest ([the run manifest](run-manifest.md)), so
the gate is FLOOR, then BOUND, then CHECK against a run's manifest.

## The work: `warpline-work/1`

```json
{"format": "warpline-work/1", "kernel": "^my::kernel<", "parameters": ["tokens", "ctas"], "sms": "min(ctas, 132)",
 "phases": [
  {"role": "producer", "phase": "phase.load", "work": {"hbm": {"read": "tokens * 4096"},
                                                       "l2": [{"pattern": "shared", "footprint": 4194304, "bytes": "ctas * 4194304"}]}},
  {"role": "consumer", "phase": "phase.mma", "work": {"tensor": [{"shape": "m16n8k16", "dtype": "f16.f32", "count": "tokens * 64"}],
                                                      "shared": {"read": "tokens * 16384"},
                                                      "issue": {"integer_address": "tokens * 40", "float": "tokens * 16"}}},
  {"role": "consumer", "phase": "phase.store", "work": {"hbm": {"write": "tokens * 256"}}}],
 "overlap": [["producer/phase.load", "consumer/phase.mma"]],
 "serial": {"consumer/phase.store": "why no overlapped form exists within the resources"}}
```

- A PHASE is a role's phase as the [run manifest](run-manifest.md) names it (`roles/<role>/phases/<phase>`: a phase stamped
  at a site named `load` is `phase.load`); its key is `role/phase`. Its WORK is the whole launch's, by kind:
  - `hbm`: `read`, `write`, bytes;
  - `l2`: a list of `{"pattern": "own", "bytes": B}` (lines each SM reads alone, past L1) and `{"pattern": "shared",
    "footprint": F, "bytes": B}` (lines every CTA reads, from a table of F bytes);
  - `shared`: `read`, `write`, bytes;
  - `tensor`: a list of `{"shape", "dtype", "count"}`, warp-level MMA instructions;
  - `issue`: warp instructions by class: the checker's instruction classes (`warpline.checker.model.INSTRUCTION_CLASSES`)
    spelled as a loop's `class/<name>` figures are ([the manifest](checker-manifest.md#a-kernels-figures)): `float`,
    `integer_address`, `shared_load`, ...
- A value is a number or an EXPRESSION over the declared `parameters`: numbers, the names, `+ - * / // % **`, parentheses,
  `min`, `max`, `ceil`, `floor`. A name not declared is refused at load; a declared one not given is refused at evaluation.
- `sms`, an expression: the SMs the launch occupies (the table's count where absent); its work spreads evenly over them.
- `overlap`: sets of phases of different roles that run at once (a role's own phases run in sequence).
- `serial`: a phase that runs alone, with the reason no overlapped form exists. It is priced like any other; the
  [idle check](#the-bound-and-the-idle-check) admits it.
- `kernel`: the pattern the bound names (`--kernel` overrides it); `about`: a note.

Declaring less work than the kernel does lowers its floor and never makes it wrong; declaring work no rate row prices is
refused (a stream with no probe is a gap, never a zero).

## The rates: `warpline-rates/1`

```json
{"format": "warpline-rates/1", "device": "NVIDIA H100 80GB HBM3", "sms": 132, "clock_ghz": 1.98,
 "rows": [{"stream": "hbm.read", "rate": 13.1, "sms": 132, "clock_ghz": 1.755, "probe": "...", "check": "exact: ...",
           "counter": {"metrics": ["dram__bytes_read.sum"], "per": 1}}, ...]}
```

A ROW prices one STREAM: `rate` is the stream's work a cycle an SM (bytes, MMA instructions, warp instructions), measured on
`sms` SMs at `clock_ghz` (each the table's where absent). The streams and the work each covers:

| stream | covers | key |
|---|---|---|
| `hbm.read`, `hbm.write` | the bytes read, written | |
| `hbm` | both (a copy's rate) | |
| `l2.own` | `l2` own bytes | |
| `l2.shared` | `l2` shared bytes, at a footprint | `footprint` |
| `shared.read`, `shared.write`, `shared` | the bytes read, written, both | |
| `tensor` | the MMAs of one shape and dtype | `shape`, `dtype` |
| `issue` | every class's warp instructions | |
| `issue.<class>` | the class's | |

- A shared L2 quantity at footprint F is priced by the `l2.shared` row of the largest footprint at most F (a smaller
  footprint reads no slower: the probes' rows show it). The rates depend on the pattern and the footprint: the own lines of
  each SM past L1 read slower than one table every SM shares.
- A row that covers two quantities (`hbm`, `shared`) prices their sum at the best of its own rate and its parts': a copy
  streams slower than a read (measured, [the examples](#the-examples)), and a read priced at the copy's rate would be bounded
  too high.
- A row is used only for a launch on the SMs it was measured on: the per-SM rate of a device stream depends on how many SMs
  share it (on an H100, an own slice past L1 measured about 31 bytes a cycle an SM on 132 SMs and about 52 on 66).
- `counter`: the profiler's metrics that count the row's work (summed, times `per`), for [the counters' check](#the-fixtures).
- `clock_ghz` of the table is the highest clock any row ran at.

## The floor

- A phase's floor, on each row that prices its work: the work over the rate times the SMs, in cycles, and in nanoseconds.
  TWO KINDS OF STREAM: an SM's OWN (`shared`, `tensor`, `issue`) is in nanoseconds at the table's clock; a DEVICE stream
  (`hbm`, `l2`), shared by every SM, at its row's own clock (its rate times its clock is the bytes a nanosecond the probe
  measured). The row largest in nanoseconds BINDS the phase.
- Each SM runs the phases IN SEQUENCE, in the declaration's order; an OVERLAP set runs at once, its floor the largest, over the
  rows, of its members' summed work on the row (the largest member's where their streams differ; more where two share one).
- THE LAUNCH FLOOR is the larger of two terms: the SMs' CHAIN, the groups' floors on the SMs' own streams summed, and the
  DEVICE's, each device stream's work over the whole launch at its rate. The CTAs need not pass a phase together, so a device
  stream one SM leaves idle in its compute phase another SM may fill in its load phase: a sum of phases on HBM is no lower
  bound (measured on the `serial` example, [below](#the-examples), on an RTX 3080 Ti: its grid ran in 821 us, its phases'
  sum on HBM and the float pipe is 1.07 ms). The binding row is the larger term's.
- The nanoseconds hold for a run whose SMs' clock is at most the table's: an SM stream runs faster at a higher clock.

`floor` prints a phase a line, each overlap set, the chain, the device and the launch, and which phases the binding stream's
family is busy in (IDLE where neither its own work nor a declared overlap is, nor a `serial`); `--json` gives the same.

## The bound and the idle check

`--bound FILE --factor F` writes the declarations ([the manifest's checkers](checker-manifest.md)) a run manifest is judged by:

| declaration | holds |
|---|---|
| `present`, `at_least` 1 | the bound's kernel is in the manifests judged (a bound on no kernel would judge nothing) |
| `bound` `at_most` floor x F | the target, on `--figure` |
| `bound` `at_least` floor | a run under its floor refutes the rates or the work declared |
| `idle` | every measured phase is one the binding stream's family is busy in, or declared serial |

`--figure` is `launch/grid_ns` (default: every scope's launch at once, the span the work declared is the launch's),
`launch/ns` (the first scope's) or `launch/cycles` (the first scope's, against the floor in cycles). The values are rounded at
six significant digits away from refusing the floor's own value.

THE IDLE CHECK (`check: "idle"`, `warpline/checkers.py`) reads a run manifest's role phases
(`roles/<role>/phases/<phase>/cycles`, `phase#2` as `phase`) and refuses a measured phase that is neither `busy` nor `serial`,
a declared phase no run manifest measures, and a kernel whose run manifests measure no phase:

    my::kernel(...): idle: phase consumer/phase.store (3210 cycles a window) idles hbm: no work on it is declared there, nor in
      a phase declared to run beside it (...)
      fix: overlap it with a phase that moves hbm (an overlap set of the work declared), or declare it serial with the reason ...

WHAT IT DOES NOT MEASURE: whether two roles' phases declared to overlap ran at once. A run manifest's phases carry their
cycles, share, jitter and hold, each in its own role's window, and no start on one frame (the recorder's digest has a
phase's `start` in its role's own window only), so the overlap is the declaration's. A run manifest that gave each phase's
start on the scope's one window (`roles/<role>/phases/<phase>/start`, the roles' windows aligned on one anchor) would let the
check refuse a declared overlap that did not happen.

## In the build graph

`floor.bzl`'s `warpline_floor_bound` is the floor as a build action: the work, the rates, the parameters (`params`, a name to a
number's text), the project's `factor` and the `figure`; its default output is the declarations file, and its output group
`floor` the floor's text. A project's gate is then the action and `checker.bzl`'s `warpline_compare_test` over the kernel's run
manifests, with no number copied by hand:

    load("@warpline//:floor.bzl", "warpline_floor_bound")
    load("@warpline//:checker.bzl", "warpline_compare_test")
    warpline_floor_bound(name = "tile_bound", work = "tile.work.json", rates = "h100.rates.json", params = {"n": "10"},
                         factor = "1.15")
    warpline_compare_test(name = "tile_floor", manifests = ["tile.run.manifest.json"], declare = ":tile_bound")

`//:tile_floor_bound` and `//:run_manifest_floor_test` are that pair in this repository, over the graph's run of `sample::tile()`
(`tests/floor/tile.work.json` at `tests/floor/tile.rates.json`).

## The probes

`probes/rates.cu` is one program that measures every row on the card it runs on and writes the table:

    bazel build //probes:rates && .bazel/bin/probes/rates.bin rates.json      # on a machine with a GPU

A CTA an SM, 512 threads; each launch records a CTA's nanosecond timer and cycle counter at its start and end, the span the
last end less the first start, the clock the CTAs' mean cycles over nanoseconds, and a row's rate the work over SMs x span x
clock, the best of five launches after a warm-up (every run's rate is written beside it). Every launch's result is checked
EXACTLY against the host (each thread's sum of the words it read, every word written, every accumulator, every chain), and a
row whose check fails is not written. The rows: `hbm.read`, `hbm.write`, `hbm` (a copy), `l2.own` (a CTA its own slice past
L1, every SM's slice in the L2), `l2.shared` at 128 KB, 1, 4 and 16 MB (every CTA one table past L1; a footprint the L2 does
not hold is no row), `shared.read`, `shared.write` (conflict-free 16-byte accesses), `tensor` (`mma.sync` m16n8k16 f16 into
f32), `issue` (FFMA and LOP3 chains on two pipes) and `issue.float`.

THE WORK A TRIP IS THE SASS'S: the program prices a trip at constants (`kGlobalTrip`, `kSharedTrip`, `kMmaTrip`, `kMixTrip`,
`kFloatTrip`) and `//probes:rates_sm_80_check` and `rates_sm_90a_check` hold each probe loop's class counts and instructions
to them exactly (`probes/rates.declarations.json`), on every build. They found two things the assembler does: it folds a
chain of adds of a constant into one add (so the integer chains are LOP3s each read by two later steps), and it hoists a plain
`ld.shared` out of a loop (so the shared probes use `ld.volatile`).

## The examples

`examples/floor/floor_examples.cu` holds four kernels, each with its work declared beside it (`<name>.work.json`): `stream`
(HBM binds), `shared` (shared memory binds), `mma` (the tensor pipe binds) and `serial` (an HBM load phase, then, behind a
barrier, an FFMA phase that moves no byte: declared `serial`, the idle check's refusal when it is not). Each is stamped (the
launch's edges and each window's phases, by warp 0 of each CTA) and runs its stamped launch inside the run layer's region.

## The fixtures

The tests hold the floor against runs recorded once on a card: `tests/floor/fixtures/<device>/` holds the probes' table and,
an example each, a stamped run's manifest, a profiled run's counters (the details page) and the launch's parameters.

    bazel run //:record_floor_fixtures            # on a card; writes the checkout's tests/floor/fixtures/<device>/

It needs a GPU, the profiler's counters granted to the user (Nsight Compute: `NVreg_RestrictProfilingToAdminUsers=0`, or
root) and the pinned Nsight Compute (`@nsight_compute`, fetched by the target). It runs the probes once, then each example
twice through the run layer (`warpline run --stamps`, and `--profile` with every row's metrics, the SMs' elapsed cycles, the
launch's time and the `SpeedOfLight` section, whose rules give the details page the columns its reader requires). Under a
minute of card (12 s on an RTX 3080 Ti, the build done); the tables are 1 GiB each, so 2 GiB of device memory and 1 GiB of host
memory.

`//:floor_fixtures_<device>` (`tests/test_floor_fixtures.py`) is generated for each recorded device, and none exists until one
is recorded. For each example it holds that the floor at the card's rates is never above its measured `launch/grid_ns` (the
floor's own `present`, `at_least` and `idle` declarations pass on its run manifest), that the stream the floor names is the one
the counters show (the binding row's family leads the counters' shares of the rows' rates over the run's elapsed cycles: a
counter tells a family's rows apart no better than its metrics, every L2 row reading the L2's bytes), and that the `serial`
example without its declaration is refused by the idle check.
