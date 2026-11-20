"""`warpline check <binary> --declare <file>`: the checker's command. Exit 0 when nothing is refused, 1 on a refusal,
2 when a tool, a text or the declaration, ledger or notes file is not usable.

Optional inputs: `--notices <build log>` (the toolkit's messages, read against the notice ledger: the module's base
ledger, with the rows of `--ledger <file>` overriding or adding), `--profile <csv>` (Nsight Compute's exports, any number)
and `--notes <file>` (the project's notes per rule, beside or instead of the declarations file's own)."""
from __future__ import annotations

import sys
from pathlib import Path

from . import declare, model, ncu, notices, readers, runner
from .rules import profile_registry, registry


def check_binary(a) -> int:
    rules, profile_rules = registry(), profile_registry()
    try:
        ledger = notices.base_ledger()
        if a.ledger:
            ledger = notices.merge(ledger, notices.load_ledger(Path(a.ledger).read_text()))
        names = declare.rule_names(ledger)
        text = Path(a.declare).read_text() if a.declare else '{"declarations": []}'
        declarations = declare.load_declarations(text, rules, names)
        notes = declare.load_notes(text, names)
        if a.notes:
            notes.update(declare.load_notes(Path(a.notes).read_text(), names))
        found = (runner.kernels_of_files(a.record) if a.fixture
                 else runner.kernels_of(a.record, a.arch, a.toolkit))
        read = notices.read_notices(Path(a.notices).read_text()) if a.notices else []
        profiles = ncu.profiles_of([Path(p).read_text() for p in a.profile or ()])
    except (declare.DeclarationError, notices.LedgerError, runner.ToolError, readers.ReaderError, model.ModelError,
            OSError) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    kernels = list(found.values())
    refusals = declare.check(kernels, rules, declarations, notes)
    pretty = {k.name: k.demangled for k in kernels}
    places = declare.notice_places(read, pretty)
    if a.notices:
        try:
            more, reports = declare.check_notices(read, ledger, declarations, pretty, notes)
        except notices.LedgerError as e:
            print(f"warpline: {e}", file=sys.stderr)
            return 2
        refusals += more
    else:
        reports = []
    said = []
    if a.profile:
        more, said = declare.check_profiles(profiles, kernels, profile_rules, declarations, notes)
        refusals += more
        places += [p.kernel for p in profiles]
    else:
        said = [f"note: profile rules not run (no --profile): {', '.join(profile_rules)}"]
    for refusal in refusals:
        print("\n".join(refusal.lines()))
    for stale in declare.unmatched(declarations, kernels, places):
        print(f"note: the declaration for {stale.rule} on {stale.kernel.pattern!r} names no kernel")
    if reports:
        print(f"{len(reports)} notice row(s) reported:")
        print("\n".join(reports))
    for line in said:
        print(line)
    count = len(rules) + (len(profile_rules) if a.profile else 0)
    extra = f", {len(read)} notice(s)" if a.notices else ""
    print(f"{len(kernels)} kernel(s), {count} rule(s){extra}: {len(refusals)} refusal(s)")
    return 1 if refusals else 0
