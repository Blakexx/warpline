# The two modules: the recorder and the checker

One repository, one package (`warpline`), two modules that share no code, and a common part both import:

| module | directory | what it reads | start with | Bazel target |
| --- | --- | --- | --- | --- |
| the RECORDER | `warpline/recorder/` (and `include/warpline/`, the in-kernel stamps it reads and the run layer's host pieces) | a record: the stamps a kernel wrote | [the README](../README.md), [the guide](guide.md), [the commands](commands.md), [the format](format.md) | `//:recorder` |
| the CHECKER | `warpline/checker/` | a compiled kernel: the vendor tools' outputs (SASS, control-flow graph, resource usage, the build's notices), and a profiler's exports | [the checker](checker.md) | `//:checker` |

Read the recorder's half to find where a stamped kernel's time goes; read the checker's to find what the compiled code of a
kernel does that a rule forbids, with no GPU and no stamps. Neither needs the other.

## What they share

The common part (target `//:shared`) holds the version, the command's assembly, and what is about manifests and not about where
they came from:

- `warpline/__init__.py` (the version) and `warpline/command.py` (assembles the one command from the modules present: the parser,
  each module's verbs under its heading in `--help`, the common verbs after them, a refusal of a verb two modules would share);
- `warpline/manifest.py`: both manifest formats, the static `warpline-manifest/1` ([the manifest](checker-manifest.md)) and the
  `warpline-run-manifest/1` of a run ([the run manifest](run-manifest.md)): their form, reading, writing, validation and fields;
- `warpline/checkers.py` and `warpline/declare.py`: the checkers (`violations`, `bound`, `hash`, `present`, `idle`, coverage
  and `waive`), the declarations and the comparison, over one manifest or several;
- `warpline/run.py`: the runner, which starts a user's command once under the modules' instruments and collects what they
  made ([the run layer](run-layer.md));
- `warpline/floor.py`: a kernel's floor from its declared work and a card's measured rates, and the bound declarations written
  from it ([the floor](floor.md)): what judges a run manifest's time, read from neither module;
- `warpline/cli.py`: the common verbs, `compare`, `run-manifest`, `run` and `floor`.

The common part imports neither module. The checker writes the static manifest and gives a run manifest its `rules` (the profile
rules on a profiler's exports); the recorder gives a run manifest its `figures` (its `summary` of the records). Each gives what it
can through its `cli` (`RUN_GIVES`, and the checker's `rule_names`), and `run-manifest` asks the modules the command was assembled
from: an input whose module is absent is refused by name (`--profile needs the checker module`). The runner is common for the
same reason: it drives both modules' instruments and writes the run manifest, and neither module may import the other. Each
module gives what it does with a run through its `cli` (`RUN_TAKES`: the recorder `--stamps`, the checker `--profile`), and
`run` refuses an instrument whose module is absent by name (`--stamps needs the recorder module`).

## The command

`warpline <verb>` (`python -m warpline`, `bazel run //:cli --`) has both modules' verbs and the common ones, listed by module in
`--help`:

- recorder: `summary`, `diff`, `sql`, `verify` (the record's own tier-0 checks), `path`, `step`, `perfetto`, `window`, `rates`, `merge`, `fold`, `overlay`, `set`
- checker: `check` (a kernel against the rules and the declarations, with its run manifests), `manifest` (what the rules found and a kernel's figures; `--text` lists every field), `held` (what each loop of a kernel holds at its top, by where it was made)
- common: `compare` (manifests against the declarations), `run-manifest` (a run's manifest from its records and a profiler's exports), `run` (a run of the user's task, stamped, profiled or both, collected with its run manifest), `floor` (a kernel's floor from its declared work and a card's rates, and the bound declarations)

A verb is in one module or in the common part. Each module's command alone is `python -m warpline.recorder` /
`python -m warpline.checker` (`//:recorder_cli`, `//:checker_cli`), with the common verbs.

## Depending on one alone

- Bazel: depend on `@warpline//:checker` or `@warpline//:recorder` (each brings `:shared`, never the other module). `checker.bzl`
  (`warpline_check`, `warpline_check_test`, `warpline_compare_test`, and `warpline_check_actions` for a rule of the project's own)
  depends on the checker alone.
- Python: import `warpline.checker.*` or `warpline.recorder.*`, and the common `warpline.manifest`, `warpline.checkers`,
  `warpline.declare`; a source tree that has only one module's directory (and the common files) runs the command of what it has,
  and writes and judges a run manifest of its own part.
- The module's tests are `tests/recorder/` and `tests/checker/`; `tests/test_command.py` holds that neither module imports the
  other and that the common files import neither; `tests/test_run_manifest.py` runs a tree with one module's directory;
  `tests/test_run.py` runs the runner, and refuses each module's instrument where only the other module is present.
