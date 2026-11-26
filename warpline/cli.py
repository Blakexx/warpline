"""THE COMMON VERBS, in every command (`command.py`), whichever modules are present. Exit 0 when nothing is refused, 1 on a
refusal, 2 when an input, the declarations, notes, ledger or a manifest file is not usable.

- `warpline compare <manifest> [<manifest> ...] --declare FILE`: manifests, static or run, against the project's declarations
  (`declare.compare`): the one verdict. Several are judged as one (a binary's manifest and its kernels' run manifests).
- `warpline run-manifest [--record R]... [--profile CSV]... --place NAME [-o FILE] [--text]`: the RUN MANIFEST of one run of
  the kernel NAME (docs/run-manifest.md), from what the modules present give: the recorder the figures of the record(s), the
  checker the profile rules' findings on a profiler's exports. At least one input; an input whose module is absent is refused
  by name.

A module gives what it can through its `cli`: `RUN_GIVES`, a flag of `run-manifest` to a function `(paths, place) ->
manifest.RunPart`, and `rule_names(ledger)`, the rule names a declaration may name (with the project's ledger, `--ledger`).
This file imports neither module: it asks the modules the command was assembled from (`args.modules`).
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import checkers, declare, manifest as manifest_mod

HEADING = "common: a run's manifest, and manifests judged against the project's declarations"
#: the inputs of the common verbs, each read by one module: a flag of an absent module is refused by name
NEEDS = {"--record": "recorder", "--profile": "checker", "--ledger": "checker"}
#: what a run manifest says of an input not given
NOT_GIVEN = {"--record": "note: no record given: the run's figures are not measured",
             "--profile": "note: profile rules not run (no --profile)"}
_USER_ERRORS = (declare.DeclarationError, checkers.CheckerError, manifest_mod.ManifestError, manifest_mod.InputError, OSError)


class _Refused(Exception):
    """An input the command cannot take: an absent module's, or none at all."""


def _gives(modules) -> dict:
    out = {}
    for m in modules:
        out.update(getattr(m, "RUN_GIVES", {}))
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


def write_run_manifest(a) -> int:
    """`run-manifest`: the run manifest of the kernel `--place` from the inputs given, written to `-o FILE` (else stdout), or
    with `--text` every field as a line."""
    given = {"--record": a.record or [], "--profile": a.profile or []}
    try:
        if not any(given.values()):
            raise _Refused("a run manifest needs at least one input: --record (the stamps' record) or --profile "
                           "(a profiler's export)")
        gives = _gives(a.modules)
        for flag, paths in given.items():
            if paths and flag not in gives:
                raise _needs(flag)
        parts = [gives[flag](paths, a.place) if paths else NOT_GIVEN[flag] for flag, paths in given.items()]
        made = manifest_mod.run_manifest(a.place, parts)
        text = (manifest_mod.report(made) or "no kernel\n") if a.text else manifest_mod.dumps(made)
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
