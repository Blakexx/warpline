"""The checker's commands. Exit 0 when nothing is refused, 1 on a refusal, 2 when a tool, a text or the declaration,
ledger, notes or manifest file is not usable.

- `warpline manifest <binary> [-o FILE]`: the static MANIFEST of a binary (`manifest.py`), written as JSON: what each rule found
  and the figures of each kernel; `--text` prints every field as `path = value`.
- `warpline check <binary> --declare FILE`: the manifest judged in one command (`--manifest FILE` also writes it), with the
  kernels' run manifests (`--run FILE`, repeated) judged with it against the one declarations file.
- `warpline held <binary> [--kernel REGEX] [--lines FILE]`: what each loop of each kernel holds at its top, by where it was made
  (`held.py`), as text.

Optional inputs of a binary: `--notices <build log>` (the toolkit's messages, read against the notice ledger: the module's base
ledger, with the rows of `--ledger <file>` overriding or adding), `--lines <listing>` (the disassembly with source-line marks, so
a finding names source lines). Of a comparison: `--notes <file>` (the project's notes per rule, beside or instead of the
declarations file's own) and `--ledger` (to know the rules a project's own rows count under).

What the checker gives the common verbs (`warpline/cli.py`): a run manifest's rules from a profiler's exports
(`run-manifest --profile`, `RUN_GIVES`), the profiled run of a task (`run --profile`, `RUN_TAKES`: the task under the profiler,
then its exports and the profiler's version) and the rule names a declaration may name (`rule_names`)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .. import checkers, command, declare
from .. import cli as common
from .. import manifest as formats
from ..run import Collected, Started, Takes
from . import held, manifest as manifest_mod, model, ncu, notices, readers, runner
from .rules import profile_registry, registry

_USER_ERRORS = (declare.DeclarationError, checkers.CheckerError, notices.LedgerError, formats.ManifestError, formats.InputError,
                runner.ToolError, readers.ReaderError, model.ModelError, OSError)


def _ledger(a) -> list[notices.Row]:
    ledger = notices.base_ledger()
    if a.ledger:
        ledger = notices.merge(ledger, notices.load_ledger(Path(a.ledger).read_text()))
    return ledger


def kernels(a) -> dict[str, model.Kernel]:
    """The kernels of the binary (or fixture) `a.record`, with source lines where `a.lines` (or a fixture's `.lines`) is given."""
    lines = getattr(a, "lines", None)
    return runner.kernels_of_files(a.record, lines) if a.fixture else runner.kernels_of(a.record, a.arch, a.toolkit, lines)


def collect_manifest(a, ledger: list[notices.Row]) -> formats.Manifest:
    """The static manifest of the binary (or fixture) `a.record`, with its notices when given."""
    found = kernels(a)
    read = notices.read_notices(Path(a.notices).read_text()) if a.notices else None
    return manifest_mod.collect(list(found.values()), registry(), notices=read, ledger=ledger)


def rule_names(ledger: str | None) -> list[str]:
    """The rule names a declaration may name (`manifest.rule_names`), with the rows of the project's ledger file when given."""
    try:
        rows = notices.merge(notices.base_ledger(), notices.load_ledger(Path(ledger).read_text())) if ledger else None
    except notices.LedgerError as e:
        raise formats.InputError(str(e)) from e
    return manifest_mod.rule_names(rows)


def give_rules(paths: list[str], place: str) -> formats.RunPart:
    """`run-manifest --profile`: the profile rules on the exports, at the place (`manifest.run_rules`)."""
    try:
        profiles = ncu.profiles_of([Path(p).read_text() for p in paths])
    except readers.ReaderError as e:
        raise formats.InputError(str(e)) from e
    return manifest_mod.run_rules(profiles, place, profile_registry())


#: what the checker gives a run manifest (`warpline/cli.py`)
RUN_GIVES = {"--profile": give_rules}


def start_profile(out: Path, specs: list[str]) -> Started:
    """`run --profile`, before the task: the profiler it runs under (`runner.profile_command`)."""
    try:
        return Started(prefix=tuple(runner.profile_command(out, specs)))
    except runner.ToolError as e:
        raise formats.InputError(f"--profile: {e}") from e


def collect_profile(out: Path, specs: list[str]) -> Collected:
    """`run --profile`, after the task: the report's pages as files (`runner.export`), and the profiler's version, said in the
    run manifest."""
    try:
        pages, version = runner.export(out, specs)
    except (runner.ToolError, readers.ReaderError) as e:
        raise formats.InputError(f"--profile: {e}") from e
    return Collected(tuple(str(p) for p in pages), (str(out / (runner.REPORT + ".ncu-rep")), str(out / runner.VERSION)),
                     (f"note: profiled with Nsight Compute {version}",))


#: what the checker does with a run (`warpline/run.py`)
RUN_TAKES = {"--profile": Takes("--profile", start_profile, collect_profile)}


def _runs(a) -> list[formats.Manifest]:
    """The run manifests of `--run`, each of the run format."""
    found = []
    for path in a.run or ():
        m = formats.loads(Path(path).read_text())
        if m.format != formats.RUN_FORMAT:
            raise formats.ManifestError(f"--run {path}: not a run manifest (format {formats.RUN_FORMAT!r}); a static manifest "
                                        f"is the binary's own")
        found.append(m)
    return found


def check_binary(a) -> int:
    """`check`: the manifest of a binary (and the run manifests of `--run`), compared with the declarations."""
    try:
        ledger = _ledger(a)
        manifest = collect_manifest(a, ledger)
        runs = _runs(a)
        if getattr(a, "manifest", None):
            Path(a.manifest).write_text(formats.dumps(manifest))
        return common.judge(a, [manifest, *runs])
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
            text = formats.report(manifest, a.kernel) or "no kernel matches\n"
        else:
            text = formats.dumps(manifest)
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


HEADING = "checker: read a compiled kernel (the vendor tools' outputs) against the rules and the project's declarations"


def execute(a) -> int:
    return a.fn(a)


def main(argv=None) -> int:
    """`python -m warpline.checker`: the checker's verbs, with the common ones."""
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
    c.add_argument("--run", metavar="FILE", action="append",
                   help="a run manifest of a kernel of the binary (`warpline run-manifest`), judged with the binary's manifest "
                        "against the one declarations file; repeat for several")
    c.add_argument("--notes", metavar="FILE", help="the project's notes per rule ({\"notes\": {rule: text}}), printed "
                   "after a refusal's fix; the declarations file may carry them too")
    c.add_argument("--manifest", metavar="FILE", help="also write the binary's manifest (docs/checker-manifest.md) to FILE")
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
