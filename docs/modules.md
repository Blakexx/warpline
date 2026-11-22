# The two modules: the recorder and the checker

One repository, one package (`warpline`), two modules that share no code:

| module | directory | what it reads | start with | Bazel target |
| --- | --- | --- | --- | --- |
| the RECORDER | `warpline/recorder/` (and `include/warpline/`, the in-kernel stamps it reads) | a record: the stamps a kernel wrote | [the README](../README.md), [the guide](guide.md), [the commands](commands.md), [the format](format.md) | `//:recorder` |
| the CHECKER | `warpline/checker/` | a compiled kernel: the vendor tools' outputs (SASS, control-flow graph, resource usage, the build's notices, a profiler's exports) | [the checker](checker.md) | `//:checker` |

Read the recorder's half to find where a stamped kernel's time goes; read the checker's to find what the compiled code of a
kernel does that a rule forbids, with no GPU and no stamps. Neither needs the other.

## What they share

Only `warpline/__init__.py` (the version) and `warpline/command.py` (assembles the one command from the modules present:
the parser, each module's verbs under its heading in `--help`, a refusal of a verb two modules would share). Target `//:shared`.

## The command

`warpline <verb>` (`python -m warpline`, `bazel run //:cli --`) has both modules' verbs, listed by module in `--help`:

- recorder: `summary`, `diff`, `sql`, `verify` (the record's own tier-0 checks), `path`, `step`, `perfetto`, `window`, `rates`, `merge`, `fold`, `overlay`, `set`
- checker: `check` (a kernel against the rules and the declarations), `facts`, `compare`

A verb is in one module. Each module's command alone is `python -m warpline.recorder` / `python -m warpline.checker`
(`//:recorder_cli`, `//:checker_cli`).

## Depending on one alone

- Bazel: depend on `@warpline//:checker` or `@warpline//:recorder` (each brings `:shared`, never the other module). `checker.bzl`
  (`warpline_check`, `warpline_check_test`) depends on the checker alone.
- Python: import `warpline.checker.*` or `warpline.recorder.*`; a source tree that has only one module's directory (and the two
  shared files) runs the command of what it has.
- The module's tests are `tests/recorder/` and `tests/checker/`; `tests/test_command.py` holds that neither module imports the other.
