"""The checker's commands. Exit 0 when nothing is refused, 1 on a refusal, 2 when a tool, a text or the declaration,
ledger, notes or facts file is not usable.

- `warpline facts <binary> [-o FILE]`: the FACTS of a run (`facts.py`), written as JSON: what each rule found, per kernel.
- `warpline compare <facts> --declare FILE`: the facts against the project's declarations (`declare.compare`): the one verdict.
- `warpline check <binary> --declare FILE`: both in one command (`--facts FILE` also writes the facts).

Optional inputs of a run: `--notices <build log>` (the toolkit's messages, read against the notice ledger: the module's base
ledger, with the rows of `--ledger <file>` overriding or adding), `--profile <csv>` (Nsight Compute's exports, any number).
Of a comparison: `--notes <file>` (the project's notes per rule, beside or instead of the declarations file's own) and
`--ledger` (to know the rules a project's own rows count under)."""
from __future__ import annotations

import sys
from pathlib import Path

from . import declare, facts as facts_mod, model, ncu, notices, readers, runner
from .rules import profile_registry, registry

_USER_ERRORS = (declare.DeclarationError, notices.LedgerError, facts_mod.FactsError, runner.ToolError, readers.ReaderError,
                model.ModelError, OSError)


def _ledger(a) -> list[notices.Row]:
    ledger = notices.base_ledger()
    if a.ledger:
        ledger = notices.merge(ledger, notices.load_ledger(Path(a.ledger).read_text()))
    return ledger


def collect_facts(a, ledger: list[notices.Row]) -> facts_mod.Facts:
    """The facts of the binary (or fixture) `a.record`, with its notices and profiles when given."""
    found = (runner.kernels_of_files(a.record) if a.fixture else runner.kernels_of(a.record, a.arch, a.toolkit))
    read = notices.read_notices(Path(a.notices).read_text()) if a.notices else None
    profiles = ncu.profiles_of([Path(p).read_text() for p in a.profile or ()]) if a.profile else None
    said = [] if a.profile else [f"note: profile rules not run (no --profile): {', '.join(profile_registry())}"]
    return facts_mod.collect(list(found.values()), registry(), notices=read, ledger=ledger, profiles=profiles,
                             profile_rules=profile_registry(), said=said)


def judge_facts(a, facts: facts_mod.Facts, ledger: list[notices.Row]) -> int:
    """Compare `facts` with the declarations (`a.declare`) and notes, print the result, and return the exit status."""
    names = declare.rule_names(ledger)
    text = Path(a.declare).read_text() if a.declare else '{"declarations": []}'
    declarations = declare.load_declarations(text, registry(), names)
    notes = declare.load_notes(text, names)
    if a.notes:
        notes.update(declare.load_notes(Path(a.notes).read_text(), names))
    result = declare.compare(facts, declarations, notes)
    for refusal in result.refusals:
        print("\n".join(refusal.lines()))
    for stale in result.stale:
        print(f"note: the declaration for {stale.rule} on {stale.kernel.pattern!r} names no kernel")
    if result.figures:
        print(f"{len(result.figures)} figure(s) of a bound rule, with no bound declared:")
        print("\n".join(result.figures))
    for line in result.said:
        print(line)
    kernels = sum(p.kernel for p in facts.places)
    rules = len({name for p in facts.places for name in p.rules})
    extra = f", {facts.notices} notice(s)" if facts.notices is not None else ""
    print(f"{kernels} kernel(s), {rules} rule(s){extra}: {len(result.refusals)} refusal(s)")
    return 1 if result.refusals else 0


def check_binary(a) -> int:
    """`check`: the facts of a binary, compared with the declarations."""
    try:
        ledger = _ledger(a)
        facts = collect_facts(a, ledger)
        if getattr(a, "facts", None):
            Path(a.facts).write_text(facts_mod.dumps(facts))
        return judge_facts(a, facts, ledger)
    except _USER_ERRORS as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2


def facts_binary(a) -> int:
    """`facts`: write the facts of a binary (to `-o FILE`, else to stdout)."""
    try:
        text = facts_mod.dumps(collect_facts(a, _ledger(a)))
        if a.out:
            Path(a.out).write_text(text)
        else:
            sys.stdout.write(text)
    except _USER_ERRORS as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    return 0


def compare_facts(a) -> int:
    """`compare`: a facts file against the declarations."""
    try:
        return judge_facts(a, facts_mod.loads(Path(a.facts_file).read_text()), _ledger(a))
    except _USER_ERRORS as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
