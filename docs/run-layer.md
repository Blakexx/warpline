# The run layer: an instrumented run of your own task

warpline's readers run nothing: they read a record the stamps wrote and a profiler's exports. The RUN LAYER is what makes
those files from a run of YOUR task: two host pieces you put around your own launch, and a runner, `warpline run`, that starts
your command once, collects what it wrote and writes the [run manifest](run-manifest.md). warpline still manages nothing in
the kernel ([the guide](guide.md)): the stamps, their switch and which variant you run are yours; the runner is handed the
command, built plain or stamped as you chose.

## What you write in the task

`include/warpline/run.hpp` (host C++17 over the CUDA runtime: `cuda_runtime.h`, `cuda_profiler_api.h`, linked with cudart,
as any CUDA host code; Bazel `@warpline//:run`) has two pieces, and `host.hpp`'s `Record` writes the run's record:

- `warpline::run::Rows<CAP>`, THE STAMP BUFFER: `n` rows of `kHead + CAP` entries on the device, cleared. `rows.device()` is
  what your kernel takes, and `Cursor<CAP>::bind(rows, i)` binds row `i` to a warp. `rows.collect(rec, lane_of)` copies the
  rows back and adds each to a `host::Record`, row `i` named by `lane_of(i)`, a `host::Lane{scope, lane, role, group}`.
  `rows.clear()` zeroes it again before another launch whose record you take.
- `warpline::run::Region`, THE MARKED REGION: its start synchronizes the device and calls `cudaProfilerStart`, its end (`end()`,
  or leaving its scope) synchronizes and calls `cudaProfilerStop`. Under `warpline run --profile` the profiler captures the
  kernels launched inside a region and none other; outside a profiler the two calls do nothing.
- `Record::write_to_run(name)` (`host.hpp`): under `warpline run --stamps`, the record is written into the run as
  `<name>.record.json` and the call returns true; run any other way it writes nothing and returns false. One record a name a run:
  a name the run already has is refused.

```cpp
#include "warpline/run.hpp"

warpline::run::Rows<kCap> rows(n_warps);                    // allocated and cleared before the region
{
  warpline::run::Region region;                             // what the profiler captures
  my_kernel<true><<<grid, block>>>(rows.device(), args);    // your stamped variant
}                                                           // synchronized; the capture stops
warpline::host::Record rec;                                 // the legend and the barriers (host.hpp)
rec.name(kFill, "fill");
rec.barrier("full", 1);
rows.collect(rec, [](uint32_t i) {
  return warpline::host::Lane{"cta" + std::to_string(i / 2), "warp" + std::to_string(i % 2), i % 2 ? "consumer" : "producer"};
});
rec.write_to_run("tile");                                   // DIR/records/tile.record.json under `warpline run --stamps`
```

`examples/two_roles.cu` is a whole program built so; outside the runner it writes the record to the path it is given.

THE CONTRACT, for a task that is not C++: the runner gives a stamped run's task the variable `WARPLINE_RECORD_DIR`, a directory
it writes each record into as `<name>.record.json` ([the format](format.md)); and a profiled run captures what the task launches
between the CUDA runtime's `cudaProfilerStart` and `cudaProfilerStop`. Any task that does those two things is run as this one
is. warpline has no Python host piece: its Python reads records and exports.

## The runner

    warpline run [--stamps] [--profile SPEC]... [--place NAME] -o DIR -- <command> [<argument>...]

- `--stamps` (the recorder's): the task is told where its records go (`WARPLINE_RECORD_DIR=DIR/records`), and after it every
  `DIR/records/*.record.json` is collected, each loaded; several are read together, in the order of their names, as
  `summary A B C` reads them.
- `--profile SPEC` (the checker's): the task runs under Nsight Compute, `ncu --profile-from-start off --export DIR/profile`,
  with the sections and metrics of SPEC: a comma list, repeated to add; an item with `__` is a metric (Nsight Compute's names
  are `unit__counter...`), any other a section identifier. After it the report's pages are exported with the flags the
  readers read, declared once in `warpline/checker/ncu.py` (`PAGES`): `DIR/details.csv`, and, where the section
  `SourceCounters` was collected (the source page's columns are its metrics), `DIR/source.<I>.csv` for each result I
  (`--launch-skip I --launch-count 1`). The profiler's version (`ncu --version`) is written to `DIR/ncu-version.txt` and said
  in the run manifest: `note: profiled with Nsight Compute 2025.3.0.0 (build ...)`.
- The profiler is the toolkit's: `ncu` is found from `--toolkit DIR`, `CUDA_HOME`, then `PATH`, as the checker's other tools are.
  This repository pins it with the toolkit (the `nsight_compute` component of the release manifest, at the toolkit's root:
  `@cuda_tools//:ncu`), so a run here names one Nsight Compute version.
- `--place NAME`: the kernel the run is of, as `run-manifest` takes it. With it the run manifest is written to
  `DIR/run.manifest.json` from the collected files by `run-manifest`'s code (the same bytes as `run-manifest` on those files,
  with the profiler's version said after the profile's lines); without it the files are collected and the `run-manifest`
  command that writes it is printed.
- `-o DIR`: the run's directory, made; one that exists and holds anything is refused, so nothing of another run is collected.
- `--`, then the command and its arguments, given to the task untouched. The command is the task's own; it runs in the
  directory the runner was started in.

At least one of `--stamps` and `--profile` is given; both make one run, stamped and profiled. The runner exits 0, and 2 when it
is refused: no instrument, no command, a directory not empty, an instrument whose module the installation has not got (named:
`--profile needs the checker module`), a task that exited nonzero (a failed task is not a run, and nothing is collected), a
stamped run that wrote no record, a record that is not one, a profiled run that captured nothing (no kernel ran inside a
region), and an export a reader refuses (a column it reads is missing: [the exports](checker-profile.md#the-exports-ncupy)).

A stamped run that is also profiled is timed under the profiler's conditions: Nsight Compute locks the clocks to base by
default (`--clock-control base`) and replays a kernel once a pass of its metrics, so its stamps describe the kernel so run.
Take a timeline to compare with the plain kernel's from a run without `--profile`.

## Where it lives

The runner is in the COMMON part (`warpline/run.py`, the verb in `warpline/cli.py`), by the modules' rule ([the two
modules](modules.md)): it drives both modules' instruments and writes the run manifest, which is common code, and neither
module may import the other, so neither can hold a verb that uses both. Each module gives what it does with a run through its
`cli`, as it gives `run-manifest` its part: `RUN_TAKES`, a flag to a `run.Takes` (`start` before the task: a command prefix and
environment variables; `collect` after it: the files, read as the `run-manifest` input `gives`, and the lines said). The
recorder takes `--stamps` (`recorder/cli.py`), the checker `--profile` (`checker/cli.py`, which runs the profiler through
`checker/runner.py`, the checker's one module that starts a process). A tree with one module's directory runs `warpline run`
with that module's instrument and refuses the other's by name.

## On a card

The tests run the runner on a stub task and a stub `ncu` (`tests/test_run.py`, `tests/run_stub.py`), with no GPU. On a machine
with a GPU and Nsight Compute, the example is the check that the pieces and the profiler agree:

    bazel build //examples:two_roles
    bazel run //:cli -- run --stamps --profile SourceCounters,SpeedOfLight \
      --place 'void two_roles<true, true>(warpline::Entry*, float*)' -o /tmp/two_roles_run -- "$PWD/bazel-bin/examples/two_roles.bin"

It exits 0; the directory holds `records/two_roles.record.json`, `profile.ncu-rep`, `details.csv` with one result (the region's
one launch of the program's seventeen), `source.0.csv`, `ncu-version.txt` and `run.manifest.json`, whose place has `figures`
(`health/lanes` 2 and `health/roles` 2: the first scope's, `cta0`'s two warps) and `rules` (`shared_wavefronts_excess`, `profiler_verdicts`), and whose `said` holds the
profiler's version.
