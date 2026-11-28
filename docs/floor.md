# The floor: a kernel's lower bound, from its work and a card's rates

A FLOOR is what a kernel's work cannot take less than on a card: each stream's work over the rate the card reached on it,
and the stream that binds. It predicts nothing: it is a lower bound, and a project's target is a factor over it. Its inputs:

- the kernel's WORK, a function of the CELL (the sizes and names that pick the kernel's instance) to its resource use, by
  phase and by stream (`warpline-work/1`, [the work function](#the-work-function));
- the card's RATES, a descriptor (`warpline-rates/1`): measured by the probes on the card ([the probes](#the-probes)), one
  of the descriptors warpline ships ([the rates](#the-rates-warpline-rates1)), or hand-written from a spec sheet;
- the CELL itself: `--parameters cell.json` (a JSON object) and `--param NAME=VALUE` (a number where it reads as one).

```
python -m warpline floor my.kernels:work --rates h100-80gb-hbm3 --parameters cell.json [-o FILE]   # the floor, by phase
python -m warpline floor kernel.work.json --rates card.rates.json --param ctas=132 \
    --bound kernel.bound.json --factor 1.15 [--tolerance 0.05] [--kernel REGEX] [--figure launch/grid_ns]  # and its bounds
python -m warpline compare stamped.run.manifest.json profiled.run.manifest.json --declare kernel.bound.json  # a run judged
```

The code is the common part's `warpline/floor.py` ([the two modules](modules.md)); it reads neither module. The bounds it
writes are judged by the common checkers (`warpline/checkers.py`) over a run's manifests ([the run manifest](run-manifest.md)):
the time on a stamped run's, the counters on a profiled run's. The gate is FLOOR, then BOUND, then CHECK, and it closes a
loop: the floor is priced from the work, and the work is held against what the hardware counted, so a work function that
is wrong is refused on every profiled run, not only a kernel that is slow.

## The work function

THE WORK IS THE DESIGN'S MODEL OF A CTA AS A BLACK BOX: what enters and leaves it (the bytes at each memory level) and the
compute the design requires of it, as a function of the cell. What happens inside the CTA is the implementation's free
choice, bounded only by the rules a project sets on how it must behave; the implementation's objective is a CTA whose
counters match the work's expected counters and whose time comes within the factor of the floor.

A WORK FUNCTION is a Python callable from the cell (a dict) to the work (a `warpline-work/1` object, below), named
`module.path:function`: `floor` imports it (`warpline.floor.work_function`) and calls it with the cell, and the object it
returns is validated as a file is (`warpline.floor.work_of`). Its amounts are numbers it computed; the format's expressions
are there for the file form. The FILE FORM, for a project that writes no Python, is the same object as JSON, its amounts
numbers or expressions over its declared `parameters`, the cell's numbers. `tests/tile_work.py` is a work function and
`tests/floor/tile.work.json` the same work as a file.

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
    `integer_address`, `shared_load`, ... The MMAs are the `tensor` kind, never an issue class `tensor` (refused: the
    issue stream counts them already, an MMA being a warp instruction issued).
- A value is a number or an EXPRESSION over the declared `parameters`: numbers, the names, `+ - * / // % **`, parentheses,
  `min`, `max`, `ceil`, `floor`. A name not declared is refused at load; a declared one not given is refused at evaluation.
- `sms`, an expression: the SMs the launch occupies (the table's count where absent); its work spreads evenly over them.
- `overlap`: sets of phases of different roles that run at once (a role's own phases run in sequence).
- `serial`: a phase that runs alone, with the reason no overlapped form exists. It is priced like any other; the
  [idle check](#the-bound-and-the-idle-check) admits it.
- `kernel`: the pattern the bound names (`--kernel` overrides it); `about`: a note.

Declaring less work than the kernel does lowers its floor; [the counters](#the-counters) refuse it on a profiled run.
Declaring work no rate row prices is refused (a stream with no probe is a gap, never a zero).

## The rates: `warpline-rates/1`

```json
{"format": "warpline-rates/1", "device": "NVIDIA H100 80GB HBM3", "sms": 132, "clock_ghz": 2.04,
 "source": {"kind": "measured", "card": "NVIDIA H100 80GB HBM3", "probe": "f219fd8212a2d31340c645da0ab8ef3fd087e800"},
 "rows": [{"stream": "hbm.read", "rate": 12.4152, "sms": 132, "clock_ghz": 1.9634, "probe": "...", "check": "exact: ..."}, ...]}
```

THE SOURCE says where the rates come from, and a floor prints it: `measured`, with the `card` they were measured on and the
`probe`'s commit (the warpline commit the probe was built from), or `spec`, `{"kind": "spec", "from": "<the spec sheet>"}`,
hand-written from a card's published peaks: a valid lower bound, looser than a measured one (a stream reaches less than its
peak). A table without a source is refused.

THE SHIPPED DESCRIPTORS: warpline ships the measured tables of well-known cards as package data, `warpline/rates/<name>.json`,
the name the device's past the vendor's word (`warpline.floor.slug`): `h100-80gb-hbm3`, the first. `--rates` takes a file or
a shipped descriptor's name or its device's (`--rates h100-80gb-hbm3`, `--rates "NVIDIA H100 80GB HBM3"`;
`warpline.floor.rates_of`). A user keeps their own descriptors, the probe's tables of their cards, as files.

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
| `issue` | every class's warp instructions, and the MMAs | |
| `issue.<class>` | the class's | |

- A shared L2 quantity at footprint F is priced by the `l2.shared` row of the largest footprint at most F (a smaller
  footprint reads no slower: the probes' rows show it). The rates depend on the pattern and the footprint: the own lines of
  each SM past L1 read slower than one table every SM shares.
- A row that covers two quantities (`hbm`, `shared`) prices their sum at the best of its own rate and its parts': a copy
  streams slower than a read (measured, [the examples](#the-examples)), and a read priced at the copy's rate would be bounded
  too high.
- A row is used only for a launch on the SMs it was measured on: the per-SM rate of a device stream depends on how many SMs
  share it (on an H100, an own slice past L1 measured about 31 bytes a cycle an SM on 132 SMs and about 52 on 66).
- An MMA is priced by its `tensor` row (which must exist) and by the `issue` row, whose slot it takes.
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

`floor` prints the rates' source, a phase a line, each overlap set, the chain, the device and the launch, which phases the
binding stream's family is busy in (IDLE where neither its own work nor a declared overlap is, nor a `serial`), each
counted stream's expected count and band, and the metrics to profile with; `--json` gives the same.

## The counters

Each COUNTED STREAM is a profiler metric (Nsight Compute's names, the same on Ampere and Hopper) that counts what crosses a
CTA at one level, or one pipe's instructions (`warpline.floor.COUNTERS`):

| stream | metric | x | counts the work's |
|---|---|---|---|
| `hbm.read`, `hbm.write` | `dram__bytes_read.sum`, `dram__bytes_write.sum` | 1 | `hbm` read, write bytes |
| `l2` | `lts__t_sectors_srcunit_tex.sum` | 32 | every byte an SM moves past its L1: `hbm` read and write, `l2` own and shared |
| `shared.read`, `shared.write` | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`, `..._op_st.sum` | 128 | `shared` read, write bytes (a wavefront 128 bytes: a bank conflict counts more) |
| `tensor.hmma`, `tensor.imma` | `sm__inst_executed_pipe_tensor_op_hmma.sum`, `..._imma.sum` | 1 | the `tensor` MMAs of float, integer inputs |
| `tensor.gmma`, `tensor.dmma` | `sm__inst_executed_pipe_tensor_op_gmma.sum`, `..._dmma.sum` | 1 | the warpgroup (`m64...`), double MMAs: Hopper's and the A100's, held where declared |
| `issue` | `sm__inst_executed.sum` | 1 | every warp instruction: the `issue` classes and the MMAs |

A stream every Ampere and Hopper card counts is held even where the work declares nothing on it, at none: a work that
leaves a stream out is refused as surely as one that undercounts it. THE BAND about a stream's expected count (the launch's
work it counts) is `--tolerance` (0.05 by default) of the launch floor's nanoseconds at the slowest row that prices the
stream: the work that stream moves in that share of the floor. A count inside it is a work function right to within that
share of the floor on that stream; a stream no row prices (an integer MMA on a card with no such probe) is held at its count
exactly. The bounds are written on the profiled run manifest's counters (`counters/<metric>`, [the run
manifest](run-manifest.md)):

    my::kernel(...): bound: counters/dram__bytes_read.sum is 1107417088, at most 581331000 declared (hbm.read: the work
      declares 5.53648e+08 bytes (dram__bytes_read.sum x 1); a count above is work the kernel does that its work does not
      declare: UNDER-DECLARED (the band: 0.05 of the floor's 172067 ns at hbm.read))

and OVER-DECLARED for a count below. A run profiled without a counted metric is refused by name (the bound names no field):
profile with the metrics `floor` prints (`warpline.floor.metrics`), `warpline run --profile <them>`. The HMMA pipe counts the
compiler's `HFMA2.MMA` moves too, a few a kernel, within the band.

WHAT IT DOES NOT COUNT: the issue classes one by one (the profiler's pipes are not the checker's classes: `IMAD` runs on the
FMA pipe), and, on Hopper, a warpgroup MMA's operand reads from shared memory and a bulk copy's writes into it, which are not
the LSU's wavefronts the `shared` streams count (`..._op_gmma` counts the first): a Hopper kernel whose shared traffic is
those is not yet held on it.

## The bound and the idle check

`--bound FILE --factor F` writes the declarations ([the manifest's checkers](checker-manifest.md)) a run's manifests are
judged by:

| declaration | holds |
|---|---|
| `present`, `at_least` 1 | the bound's kernel is in the manifests judged (a bound on no kernel would judge nothing) |
| `bound` `at_most` floor x F | the target, on `--figure` |
| `bound` `at_least` floor | a run under its floor refutes the rates or the work declared |
| `idle` | every measured phase is one the binding stream's family is busy in, or declared serial |
| `bound` on `counters/<metric>`, `at_most` and `at_least` | each counted stream within its band ([the counters](#the-counters)) |

`--figure` is `launch/grid_ns` (default: every scope's launch at once, the span the work declared is the launch's),
`launch/ns` (the first scope's) or `launch/cycles` (the first scope's, against the floor in cycles). The values are rounded at
six significant digits away from refusing the floor's own value. A stamped run's manifest and a profiled run's are judged as
one (a kernel's fields are those of every manifest that holds it).

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

`floor.bzl`'s `warpline_floor_bound` is the floor as a build action: the work (`work`, a declaration's file, or
`work_function`, `module.path:function`, with the `deps` its module is in), the rates (a file: the project's own, or a shipped
descriptor, `@warpline//:warpline/rates/h100-80gb-hbm3.json`), the cell (`parameters`, a JSON file, and `params`, a name to
a value's text), the project's `factor`, the counters' `tolerance` and the `figure`; its default output is the declarations
file, and its output group `floor` the floor's text. A project's gate is then the action and `checker.bzl`'s
`warpline_compare_test` over the kernel's run manifests, stamped and profiled, with no number copied by hand:

    load("@warpline//:floor.bzl", "warpline_floor_bound")
    load("@warpline//:checker.bzl", "warpline_compare_test")
    warpline_floor_bound(name = "tile_bound", work_function = "my.kernels:tile_work", deps = [":kernels"],
                         rates = "@warpline//:warpline/rates/h100-80gb-hbm3.json", params = {"n": "10"}, factor = "1.15")
    warpline_compare_test(name = "tile_floor", manifests = ["tile.stamped.run.manifest.json",
                                                           "tile.profiled.run.manifest.json"], declare = ":tile_bound")

`//:tile_floor_bound` and `//:tile_floor_function_bound` (the file and the function, `tests/tile_work.py`), judged by
`//:run_manifest_floor_test` and `//:run_manifest_floor_function_test`, are that pair in this repository, over the graph's
stamped run of `sample::tile()` and a profiled one (`tests/floor/tile.counters.csv`), at `tests/floor/tile.rates.json`.

## The probes

`probes/rates.cu` is one program that measures every row on the card it runs on and writes the table, its source the card
and the commit given:

    bazel run @warpline//probes:rates -- rates.json COMMIT      # on a machine with a GPU; COMMIT, the warpline pin's

`//probes:rates` is PUBLIC and built by the building workspace's own CUDA toolchain (`probes/probe.bzl`): a workspace that
depends on warpline registers a `warpline_cuda_toolchain` (its nvcc and the toolkit's files) for
`@warpline//probes:cuda_toolchain_type`, and nvcc's host compiler is its own C++ toolchain:

    load("@warpline//probes:probe.bzl", "warpline_cuda_toolchain")
    warpline_cuda_toolchain(name = "cuda", nvcc = "@my_cuda//:bin/nvcc", files = ["@my_cuda//:toolkit"])
    toolchain(name = "cuda_toolchain", toolchain = ":cuda", toolchain_type = "@warpline//probes:cuda_toolchain_type")
    # MODULE.bazel: register_toolchains("//:cuda_toolchain")

warpline's own build registers its pinned toolkit (`//toolchain:cuda_probe_toolchain`, a dev dependency).

A CTA an SM, 512 threads; each launch records a CTA's nanosecond timer and cycle counter at its start and end, the span the
last end less the first start, the clock the CTAs' mean cycles over nanoseconds, and a row's rate the work over SMs x span x
clock, the best of five launches after a warm-up (every run's rate is written beside it). Every launch's result is checked
EXACTLY against the host (each thread's sum of the words it read, every word written, every accumulator, every chain), and a
row whose check fails is not written. The rows: `hbm.read`, `hbm.write`, `hbm` (a copy), `l2.own` (a CTA its own slice past
L1, every SM's slice in the L2), `l2.shared` at 128 KB, 1, 4 and 16 MB (every CTA one table past L1; a footprint the L2 does
not hold is no row), `shared.read`, `shared.write` (conflict-free 16-byte accesses), `tensor` (`mma.sync` m16n8k16 f16 into
f32), `issue` (FFMA and LOP3 chains on two pipes) and `issue.float`.

THE WORK A TRIP IS THE SASS'S: the program prices a trip at constants (`kGlobalTrip`, `kSharedTrip`, `kMmaTrip`, `kMixTrip`,
`kFloatTrip`) and `//probes:rates_sm_80_check` and `rates_sm_90a_check` (warpline's own, on its pinned toolkit) hold each
probe loop's class counts and instructions to them exactly (`probes/rates.declarations.json`), on every build. They found two
things the assembler does: it folds a chain of adds of a constant into one add (so the integer chains are LOP3s each read by
two later steps), and it hoists a plain `ld.shared` out of a loop (so the shared probes use `ld.volatile`).

## The examples

`examples/floor/floor_examples.cu` holds four kernels, each with its work declared beside it (`<name>.work.json`): `stream`
(HBM binds), `shared` (shared memory binds), `mma` (the tensor pipe binds) and `serial` (an HBM load phase, then, behind a
barrier, an FFMA phase that moves no byte: declared `serial`, the idle check's refusal when it is not). Each is stamped (the
launch's edges and each window's phases, by warp 0 of each CTA) and runs its stamped launch inside the run layer's region.
Each declares its warp instructions by class as the sm_90a SASS's loops hold them, which is what the issue counter holds them
to. Dev only, as the fixtures are: a workspace that depends on warpline never builds them.

## The fixtures

The tests hold the floor against runs recorded once on a card: `tests/floor/fixtures/<device>/` holds, an example each, a
stamped run's manifest, a profiled run's counters (the details page) and the launch's parameters; the card's probes' table is
its shipped descriptor, `warpline/rates/<device>.json`.

    bazel run //:record_floor_fixtures            # on a card; writes the checkout's descriptor and fixtures

It needs a GPU, the profiler's counters granted to the user (Nsight Compute: `NVreg_RestrictProfilingToAdminUsers=0`, or
root) and the pinned Nsight Compute (`@nsight_compute`, fetched by the target). It runs the probes once (the probe's commit
the checkout's HEAD, `-dirty` where `probes/` differs from it), then each example twice through the run layer (`warpline run
--stamps`, and `--profile` with the counters' metrics, the SMs' elapsed cycles, the launch's time and the `SpeedOfLight`
section, whose rules give the details page the columns its reader requires). Under a minute of card (12 s on an RTX 3080 Ti,
the build done); the tables are 1 GiB each, so 2 GiB of device memory and 1 GiB of host memory.

`//:floor_fixtures_<device>` (`tests/test_floor_fixtures.py`) is generated for each recorded device, and none exists until one
is recorded. For each example it holds that the floor at the card's descriptor is never above its measured `launch/grid_ns`
and every counter its work expects is in its band (the floor's own `present`, `at_least`, `idle` and counters' declarations
pass on its run manifests, the profiled one written from the details page as `warpline run` writes it), that the stream the
floor names is the one the counters show (the binding row's family leads the counters' shares of the rows' rates over the
run's elapsed cycles), that a deliberately wrong work (HBM's read halved or doubled, the issue left out, the shared read
doubled) is refused naming its stream, and that the `serial` example without its declaration is refused by the idle check.
The H100's runs predate two counted metrics (`lts__t_sectors_srcunit_tex.sum`, `sm__inst_executed_pipe_tensor_op_imma.sum`):
the test holds that exactly those are missing until the card is recorded again.
