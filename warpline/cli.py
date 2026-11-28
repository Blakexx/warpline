"""THE COMMON VERBS, in every command (`command.py`), whichever modules are present. Exit 0 when nothing is refused, 1 on a
refusal, 2 when an input, the declarations, notes, ledger or a manifest file is not usable.

- `warpline compare <manifest> [<manifest> ...] --declare FILE`: manifests, static or run, against the project's declarations
  (`declare.compare`): the one verdict. Several are judged as one (a binary's manifest and its kernels' run manifests).
- `warpline run-manifest [--record R]... [--profile CSV]... --place NAME [-o FILE] [--text]`: the RUN MANIFEST of one run of
  the kernel NAME (docs/run-manifest.md), from what the modules present give: the recorder the figures of the record(s), the
  checker the profile rules' findings on a profiler's exports. At least one input; an input whose module is absent is refused
  by name.
- `warpline run [--stamps] [--profile SPEC]... [--place NAME] -o DIR -- <command>`: the user's command run once, instrumented
  (`run.py`, docs/run-layer.md): its records and the profiler's exports collected into DIR, and, with `--place`, the run
  manifest written there from them (`DIR/run.manifest.json`, by `run-manifest`'s code). At least one instrument; an
  instrument whose module is absent is refused by name.
- `warpline floor <work> --rates FILE [--param NAME=VALUE]... [--parameters FILE] [--json] [-o FILE] [--bound FILE --factor F
  [--kernel REGEX] [--figure launch/grid_ns|launch/ns|launch/cycles] [--reason TEXT]]`: a kernel's FLOOR (`floor.py`, docs/floor.md), its
  declared work over a device's measured rates, by phase, and the stream that binds it; with `--bound`, the declarations a
  run manifest is judged by (`compare`), written to FILE.

A module gives what it can through its `cli`: `RUN_GIVES`, a flag of `run-manifest` to a function `(paths, place) ->
manifest.RunPart`; `RUN_TAKES`, a flag of `run` to what it does with a run (`run.Takes`); and `rule_names(ledger)`, the rule
names a declaration may name (with the project's ledger, `--ledger`). This file imports neither module: it asks the modules the
command was assembled from (`args.modules`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import checkers, declare, floor as floor_mod, manifest as manifest_mod, run as run_mod

HEADING = "common: a run of a task, its manifest, a kernel's floor, and manifests judged against the project's declarations"
#: the inputs of the common verbs, each read by one module: a flag of an absent module is refused by name
NEEDS = {"--record": "recorder", "--profile": "checker", "--ledger": "checker", "--stamps": "recorder"}
#: where `run` writes the run manifest in the run's directory
RUN_MANIFEST = "run.manifest.json"
#: what a run manifest says of an input not given
NOT_GIVEN = {"--record": "note: no record given: the run's figures are not measured",
             "--profile": "note: profile rules not run (no --profile)"}
_USER_ERRORS = (declare.DeclarationError, checkers.CheckerError, manifest_mod.ManifestError, manifest_mod.InputError,
                run_mod.RunError, floor_mod.FloorError, OSError)


class _Refused(Exception):
    """An input the command cannot take: an absent module's, or none at all."""


def _gathered(modules, name: str) -> dict:
    out = {}
    for m in modules:
        out.update(getattr(m, name, {}))
    return out


def _needs(flag: str) -> _Refused:
    return _Refused(f"{flag} needs the {NEEDS[flag]} module, which this installation does not have")


def _known_rules(a, manifests) -> list[str]:
    """The rule names a declaration may name: those of the modules present (with the project's ledger) and those the manifests
    hold."""
    found = [m.rule_names for m in a.modules if hasattr(m, "rule_names")]
    if getattr(a, "ledger", None) and not found:
        raise _needs("--ledger")
    return sorted({*declare.manifest_rules(manifests), *(n for names in found for n in names(getattr(a, "ledger", None)))})


def judge(a, manifests: list[manifest_mod.Manifest]) -> int:
    """Compare the manifests with the declarations (`a.declare`) and notes, print the result, and return the exit status."""
    names = _known_rules(a, manifests)
    text = Path(a.declare).read_text() if a.declare else '{"declarations": []}'
    declarations = declare.load_declarations(text, names)
    notes = declare.load_notes(text, names)
    if getattr(a, "notes", None):
        notes.update(declare.load_notes(Path(a.notes).read_text(), names))
    result = declare.compare(manifests, declarations, notes, checkers.load(text))
    for refusal in result.refusals:
        print("\n".join(refusal.lines()))
    for stale in result.stale:
        print(f"note: the declaration for {stale.rule} on {stale.kernel.pattern!r} names no kernel")
    if result.figures:
        print(f"{len(result.figures)} figure(s) of a bound rule, with no bound declared:")
        print("\n".join(result.figures))
    for line in result.stale_checks:
        print(line)
    for line in result.said:
        print(line)
    kernels = len({p.place for m in manifests for p in m.places if p.kernel})
    rules = len({name for m in manifests for p in m.places for name in p.rules})
    counted = [m.notices for m in manifests if m.notices is not None]
    extra = f", {sum(counted)} notice(s)" if counted else ""
    print(f"{kernels} kernel(s), {rules} rule(s){extra}: {len(result.refusals)} refusal(s)")
    return 1 if result.refusals else 0


def _fail(e: Exception) -> int:
    print(f"warpline: {e}", file=sys.stderr)
    return 2


def compare_manifests(a) -> int:
    """`compare`: manifest files against the declarations."""
    try:
        return judge(a, [manifest_mod.loads(Path(p).read_text()) for p in a.manifest_file])
    except (_Refused, *_USER_ERRORS) as e:
        return _fail(e)


def made_run_manifest(modules, given: dict[str, list[str]], place: str,
                      said: dict[str, tuple[str, ...]] | None = None) -> manifest_mod.Manifest:
    """The run manifest of the kernel `place` from the inputs given (`run-manifest`'s flags to their files), each read by the
    module that gives it, with the lines `said` of an input after its part (the profiler's version, from `run`)."""
    if not any(given.values()):
        raise _Refused("a run manifest needs at least one input: --record (the stamps' record) or --profile "
                       "(a profiler's export)")
    gives = _gathered(modules, "RUN_GIVES")
    for flag, paths in given.items():
        if paths and flag not in gives:
            raise _needs(flag)
    parts: list = []
    for flag in NOT_GIVEN:
        paths = given.get(flag) or []
        parts += [gives[flag](paths, place) if paths else NOT_GIVEN[flag], *(said or {}).get(flag, ())]
    return manifest_mod.run_manifest(place, parts)


def write_run_manifest(a) -> int:
    """`run-manifest`: the run manifest of the kernel `--place` from the inputs given, written to `-o FILE` (else stdout), or
    with `--text` every field as a line."""
    try:
        made = made_run_manifest(a.modules, {"--record": a.record or [], "--profile": a.profile or []}, a.place)
        text = (manifest_mod.report(made) or "no kernel\n") if a.text else manifest_mod.dumps(made)
        if a.out:
            Path(a.out).write_text(text)
        else:
            sys.stdout.write(text)
    except (_Refused, *_USER_ERRORS) as e:
        return _fail(e)
    return 0


def run_task(a) -> int:
    """`run`: the command run once under the instruments asked for, their files collected into `-o DIR`, and with `--place`
    the run manifest written there."""
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    asked = [(flag, values) for flag, values in (("--stamps", [] if a.stamps else None), ("--profile", a.profile))
             if values is not None]
    try:
        if not asked:
            raise _Refused("a run takes an instrument: --stamps (the task's records), --profile SPEC (the profiler), or both")
        takes = _gathered(a.modules, "RUN_TAKES")
        for flag, _ in asked:
            if flag not in takes:
                raise _needs(flag)
        out, collected = run_mod.run(a.out, [(flag, takes[flag], values) for flag, values in asked], command)
        for flag, got in collected.items():
            print(f"{flag}: {', '.join(os.path.relpath(p, out) for p in (*got.paths, *got.beside))}")
        given = {takes[flag].gives: list(got.paths) for flag, got in collected.items()}
        said = {takes[flag].gives: got.said for flag, got in collected.items()}
        if a.place is None:
            inputs = " ".join(f"{flag} {p}" for flag, paths in given.items() for p in paths)
            print(f"no --place: no run manifest written; it is `warpline run-manifest {inputs} --place NAME`")
            return 0
        path = out / RUN_MANIFEST
        path.write_text(manifest_mod.dumps(made_run_manifest(a.modules, given, a.place, said)))
        print(f"wrote {path}")
    except (_Refused, *_USER_ERRORS) as e:
        return _fail(e)
    return 0


def _given(a) -> dict:
    """The parameters given: a file's numeric top-level fields (`--parameters`, a cell's description), then each `--param`."""
    given: dict = {}
    if a.parameters:
        try:
            doc = json.loads(Path(a.parameters).read_text())
        except ValueError as e:
            raise floor_mod.FloorError(f"{a.parameters}: not JSON ({e})") from e
        if not isinstance(doc, dict):
            raise floor_mod.FloorError(f"{a.parameters}: the parameters are a JSON object")
        given.update({k: v for k, v in doc.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for item in a.param or []:
        name, sep, value = item.partition("=")
        try:
            given[name] = float(value) if sep else None
        except ValueError:
            given[name] = None
        if given[name] is None:
            raise floor_mod.FloorError(f"--param {item!r}: NAME=VALUE, the value a number")
    return given


def write_floor(a) -> int:
    """`floor`: the floor of the work declared at the rates, printed (or `--json`); with `--bound`, the declarations written."""
    try:
        work = floor_mod.load_work(Path(a.work).read_text())
        result = floor_mod.floor(work, floor_mod.load_rates(Path(a.rates).read_text()), _given(a))
        if a.bound:
            kernel = a.kernel or work.kernel
            if not kernel:
                raise floor_mod.FloorError("--bound needs the kernel's pattern: --kernel, or the declaration's `kernel`")
            if a.factor is None:
                raise floor_mod.FloorError("--bound needs --factor, the project's (1.15: within 15% of the floor)")
            decl = floor_mod.declarations(result, kernel, a.factor, a.figure, a.reason or "")
            Path(a.bound).write_text(json.dumps(decl, indent=1) + "\n")
        text = json.dumps(floor_mod.to_data(result), indent=1) + "\n" if a.json else floor_mod.report(result)
        if a.out:
            Path(a.out).write_text(text)
        else:
            sys.stdout.write(text)
    except (_Refused, *_USER_ERRORS) as e:
        return _fail(e)
    return 0


def execute(a) -> int:
    return a.fn(a)


def register(sub) -> None:
    cp = sub.add_parser("compare", help="MANIFEST files (a static one, run ones) against the project's declarations: the "
                                        "refusals (exit 1)")
    cp.add_argument("manifest_file", metavar="manifest", nargs="+",
                    help="a manifest file (`warpline manifest`, `warpline run-manifest`); several are judged as one")
    cp.add_argument("--declare", metavar="FILE", required=True, help="the project's declarations (JSON)")
    cp.add_argument("--notes", metavar="FILE", help="the project's notes per rule, printed after a refusal's fix")
    cp.add_argument("--ledger", metavar="FILE", help="the project's notice ledger, for the rule names its rows count under "
                                                     "(the checker's)")
    cp.set_defaults(fn=compare_manifests)
    rm = sub.add_parser("run-manifest", help="the RUN MANIFEST of one run of a kernel: the record's figures and the profile "
                                             "rules' findings, as JSON (--text: every field, a line each)")
    rm.add_argument("--record", metavar="R", action="append",
                    help="a record the stamps wrote (the recorder's); repeat for several records of one run (a detail each)")
    rm.add_argument("--profile", metavar="CSV", action="append",
                    help="a Nsight Compute export (`--page details --csv`, or `--page source --print-source sass --csv`; "
                         "the checker's); repeat for both pages")
    rm.add_argument("--place", metavar="NAME", required=True,
                    help="the kernel the run is of: its demangled name, the static manifest's place (`sample::tile()`)")
    rm.add_argument("-o", "--out", metavar="FILE", help="write the run manifest here (default: stdout)")
    rm.add_argument("--text", action="store_true", help="print every field as `path = value` (what a bound or coverage names)")
    rm.set_defaults(fn=write_run_manifest)
    rn = sub.add_parser("run", help="a RUN of the task: the command run once, stamped, profiled or both, its records and the "
                                    "profiler's exports collected into a directory, with the run manifest (--place)")
    rn.add_argument("--stamps", action="store_true",
                    help="collect the records the task writes (the recorder's): `Record::write_to_run` in the task")
    rn.add_argument("--profile", metavar="SPEC", action="append",
                    help="run the task under Nsight Compute, limited to its marked regions (the checker's): a comma list of "
                         "section identifiers and metrics (an item with `__` is a metric); repeat to add")
    rn.add_argument("--place", metavar="NAME",
                    help="the kernel the run is of (its demangled name): write DIR/run.manifest.json")
    rn.add_argument("-o", "--out", metavar="DIR", required=True, help="the run's directory: made, refused when not empty")
    rn.add_argument("command", nargs=argparse.REMAINDER, help="after `--`: the task's command and its arguments")
    rn.set_defaults(fn=run_task)
    fl = sub.add_parser("floor", help="a kernel's FLOOR: its declared work over a device's measured rates, by phase, the "
                                      "stream that binds (--bound: the declarations a run manifest is judged by)")
    fl.add_argument("work", help="the kernel's work declaration (`warpline-work/1`, docs/floor.md)")
    fl.add_argument("--rates", metavar="FILE", required=True, help="the device's rates (`warpline-rates/1`: the probes' table)")
    fl.add_argument("--param", metavar="NAME=VALUE", action="append", help="a parameter the declaration's expressions read")
    fl.add_argument("--parameters", metavar="FILE", help="a JSON object whose numeric top-level fields are parameters")
    fl.add_argument("--json", action="store_true", help="the floor as data")
    fl.add_argument("-o", "--out", metavar="FILE", help="write the floor (text, or --json) here (default: stdout)")
    fl.add_argument("--bound", metavar="FILE", help="write the bound declarations here (`compare` judges a run manifest by them)")
    fl.add_argument("--factor", type=float, help="the bound's factor over the floor, the project's (1.15)")
    fl.add_argument("--kernel", metavar="REGEX", help="the kernel the bound names (default: the declaration's `kernel`)")
    fl.add_argument("--figure", default="launch/grid_ns", choices=floor_mod.FIGURES,
                    help="the run manifest's figure the bound is on (default launch/grid_ns: every scope's launch at once)")
    fl.add_argument("--reason", metavar="TEXT", help="the project's words, written after the floor's in each bound's reason")
    fl.set_defaults(fn=write_floor)
