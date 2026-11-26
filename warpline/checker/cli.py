"""The checker's commands. Exit 0 when nothing is refused, 1 on a refusal, 2 when a tool, a text or the declaration,
ledger, notes or manifest file is not usable.

- `warpline manifest <binary> [-o FILE]`: the MANIFEST of a run (`manifest.py`), written as JSON: what each rule found and the figures of each kernel; `--text` prints every field as `path = value`.
- `warpline compare <manifest> --declare FILE`: the manifest against the project's declarations (`declare.compare`): the one verdict.
- `warpline check <binary> --declare FILE`: both in one command (`--manifest FILE` also writes the manifest).
- `warpline held <binary> [--kernel REGEX] [--lines FILE]`: what each loop of each kernel holds at its top, by where it was made
  (`held.py`), as text.

Optional inputs of a run: `--notices <build log>` (the toolkit's messages, read against the notice ledger: the module's base
ledger, with the rows of `--ledger <file>` overriding or adding), `--profile <csv>` (Nsight Compute's exports, any number), `--lines <listing>` (the disassembly with source-line marks, so a
finding names source lines).
Of a comparison: `--notes <file>` (the project's notes per rule, beside or instead of the declarations file's own) and
`--ledger` (to know the rules a project's own rows count under)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .. import command
from . import checkers, declare, held, manifest as manifest_mod, model, ncu, notices, readers, runner
from .rules import profile_registry, registry

_USER_ERRORS = (declare.DeclarationError, checkers.CheckerError, notices.LedgerError, manifest_mod.ManifestError, runner.ToolError, readers.ReaderError,
                model.ModelError, OSError)


def _ledger(a) -> list[notices.Row]:
    ledger = notices.base_ledger()
    if a.ledger:
        ledger = notices.merge(ledger, notices.load_ledger(Path(a.ledger).read_text()))
    return ledger


def kernels(a) -> dict[str, model.Kernel]:
    """The kernels of the binary (or fixture) `a.record`, with source lines where `a.lines` (or a fixture's `.lines`) is given."""
    lines = getattr(a, "lines", None)
    return runner.kernels_of_files(a.record, lines) if a.fixture else runner.kernels_of(a.record, a.arch, a.toolkit, lines)


def collect_manifest(a, ledger: list[notices.Row]) -> manifest_mod.Manifest:
    """The manifest of the binary (or fixture) `a.record`, with its notices and profiles when given."""
    found = kernels(a)
    read = notices.read_notices(Path(a.notices).read_text()) if a.notices else None
    profiles = ncu.profiles_of([Path(p).read_text() for p in a.profile or ()]) if a.profile else None
    said = [] if a.profile else [f"note: profile rules not run (no --profile): {', '.join(profile_registry())}"]
    return manifest_mod.collect(list(found.values()), registry(), notices=read, ledger=ledger, profiles=profiles,
                             profile_rules=profile_registry(), said=said)


def judge_manifest(a, manifest: manifest_mod.Manifest, ledger: list[notices.Row]) -> int:
    """Compare `manifest` with the declarations (`a.declare`) and notes, print the result, and return the exit status."""
    names = declare.rule_names(ledger)
    text = Path(a.declare).read_text() if a.declare else '{"declarations": []}'
    declarations = declare.load_declarations(text, registry(), names)
    notes = declare.load_notes(text, names)
    if a.notes:
        notes.update(declare.load_notes(Path(a.notes).read_text(), names))
    result = declare.compare(manifest, declarations, notes, checkers.load(text))
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
    kernels = sum(p.kernel for p in manifest.places)
    rules = len({name for p in manifest.places for name in p.rules})
    extra = f", {manifest.notices} notice(s)" if manifest.notices is not None else ""
    print(f"{kernels} kernel(s), {rules} rule(s){extra}: {len(result.refusals)} refusal(s)")
    return 1 if result.refusals else 0


def check_binary(a) -> int:
    """`check`: the manifest of a binary, compared with the declarations."""
    try:
        ledger = _ledger(a)
        manifest = collect_manifest(a, ledger)
        if getattr(a, "manifest", None):
            Path(a.manifest).write_text(manifest_mod.dumps(manifest))
        return judge_manifest(a, manifest, ledger)
    except _USER_ERRORS as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2


def manifest_binary(a) -> int:
    """`manifest`: write the manifest of a binary (to `-o FILE`, else to stdout), or with `--text` every field of each kernel
    (`manifest.report`) as lines."""
    try:
        manifest = collect_manifest(a, _ledger(a))
        if a.text:
            if a.kernel:
                re.compile(a.kernel)
            text = manifest_mod.report(manifest, a.kernel) or "no kernel matches\n"
        else:
            text = manifest_mod.dumps(manifest)
        if a.out:
            Path(a.out).write_text(text)
        else:
            sys.stdout.write(text)
    except (re.error, *_USER_ERRORS) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    return 0


def held_binary(a) -> int:
    """`held`: what each loop of each kernel holds at its top (`held.report`), as text."""
    try:
        found = list(kernels(a).values())
        text = held.report(found, a.kernel)
    except (re.error, *_USER_ERRORS) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    if getattr(a, "lines", None) and not any(k.lines for k in found):
        print("warpline: note: the listing with source lines names none of the kernels' functions", file=sys.stderr)
    sys.stdout.write(text or "no kernel matches\n")
    return 0


def compare_manifest(a) -> int:
    """`compare`: a manifest file against the declarations."""
    try:
        return judge_manifest(a, manifest_mod.loads(Path(a.manifest_file).read_text()), _ledger(a))
    except _USER_ERRORS as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2


HEADING = "checker: read a compiled kernel (the vendor tools' outputs) against the rules and the project's declarations"


def execute(a) -> int:
    return a.fn(a)


def main(argv=None) -> int:
    """`python -m warpline.checker`: the checker's verbs alone."""
    return command.main(argv, [sys.modules[__name__]], prog="python -m warpline.checker")


def register(sub) -> None:
    c = sub.add_parser("check", help="a compiled kernel against the rules and the project's declarations (exit 1 on a refusal)")
    c.add_argument("record", help="a binary: executable, library, fatbin or cubin (or, with --fixture, the stem of saved outputs)")
    c.add_argument("--declare", metavar="FILE",
                   help="check the binary's kernels against the rules, with the project's declarations (JSON)")
    c.add_argument("--notices", metavar="LOG",
                   help="the build's messages (ptxas/nvcc output saved to a file), read against the notice ledger")
    c.add_argument("--ledger", metavar="FILE",
                   help="the project's notice ledger (JSON): its rows override or add to the module's base ledger")
    c.add_argument("--profile", metavar="CSV", action="append",
                   help="a Nsight Compute export (`--page details --csv`, or `--page source --print-source sass --csv`); "
                        "repeat for both pages")
    c.add_argument("--notes", metavar="FILE", help="the project's notes per rule ({\"notes\": {rule: text}}), printed "
                   "after a refusal's fix; the declarations file may carry them too")
    c.add_argument("--manifest", metavar="FILE", help="also write the run's manifest (docs/checker-manifest.md) to FILE")
    c.add_argument("--fixture", action="store_true",
                   help="the argument is the stem of saved tool outputs (STEM.sass, .cfg.dot, .res), not a binary")
    c.add_argument("--arch", help="the binary's architecture to read (sm_90a); needed for a fat binary of several")
    c.add_argument("--toolkit", metavar="DIR", help="the CUDA toolkit's root (default: CUDA_HOME, then PATH)")
    c.add_argument("--lines", metavar="LISTING", help="the disassembly with source-line marks (`nvdisasm --print-line-info` "
                     "of a cubin built with `--generate-line-info`), so a finding names source lines; with --fixture, "
                     "`STEM.lines` is read when it exists")
    c.set_defaults(fn=check_binary)
    fa = sub.add_parser("manifest", help="the MANIFEST of a compiled kernel: what each rule found and the figures of its loops, as "
                                         "JSON (--text: every field, a line each)")
    fa.add_argument("record", help="a binary: executable, library, fatbin or cubin (or, with --fixture, the stem of saved outputs)")
    fa.add_argument("-o", "--out", metavar="FILE", help="write the manifest here (default: stdout)")
    fa.add_argument("--text", action="store_true", help="print every field as `path = value` (what a bound, a hash or coverage names)")
    fa.add_argument("--kernel", metavar="REGEX", help="with --text, only the kernels whose demangled name matches")
    fa.add_argument("--notices", metavar="LOG", help="the build's messages, counted under the notice ledger's rules")
    fa.add_argument("--ledger", metavar="FILE", help="the project's notice ledger (JSON), over the module's base ledger")
    fa.add_argument("--profile", metavar="CSV", action="append", help="a Nsight Compute export; repeat for both pages")
    fa.add_argument("--fixture", action="store_true", help="the argument is the stem of saved tool outputs")
    fa.add_argument("--arch", help="the binary's architecture to read (sm_90a)")
    fa.add_argument("--toolkit", metavar="DIR", help="the CUDA toolkit's root (default: CUDA_HOME, then PATH)")
    fa.add_argument("--lines", metavar="LISTING", help="the disassembly with source-line marks (`nvdisasm --print-line-info` "
                     "of a cubin built with `--generate-line-info`), so a finding names source lines; with --fixture, "
                     "`STEM.lines` is read when it exists")
    fa.set_defaults(fn=manifest_binary)
    h = sub.add_parser("held", help="what each loop of a compiled kernel holds at its top, by where it was made (live registers)")
    h.add_argument("record", help="a binary: executable, library, fatbin or cubin (or, with --fixture, the stem of saved outputs)")
    h.add_argument("--kernel", metavar="REGEX", help="only the kernels whose demangled name matches")
    h.add_argument("--lines", metavar="LISTING", help="the disassembly with source-line marks (`nvdisasm --print-line-info` "
                     "of a cubin built with `--generate-line-info`), so a finding names source lines; with --fixture, "
                     "`STEM.lines` is read when it exists")
    h.add_argument("--fixture", action="store_true", help="the argument is the stem of saved tool outputs")
    h.add_argument("--arch", help="the binary's architecture to read (sm_90a)")
    h.add_argument("--toolkit", metavar="DIR", help="the CUDA toolkit's root (default: CUDA_HOME, then PATH)")
    h.set_defaults(fn=held_binary)
    cp = sub.add_parser("compare", help="a MANIFEST file against the project's declarations: the refusals (exit 1)")
    cp.add_argument("manifest_file", metavar="manifest", help="a manifest file written by `warpline manifest`")
    cp.add_argument("--declare", metavar="FILE", required=True, help="the project's declarations (JSON)")
    cp.add_argument("--notes", metavar="FILE", help="the project's notes per rule, printed after a refusal's fix")
    cp.add_argument("--ledger", metavar="FILE", help="the project's notice ledger, for the rule names its rows count under")
    cp.set_defaults(fn=compare_manifest)
