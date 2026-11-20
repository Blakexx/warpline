"""`warpline check <binary> --declare <file>`: the checker's command. Exit 0 when nothing is refused, 1 on a refusal,
2 when a tool, a text or the declaration file is not usable."""
from __future__ import annotations

import sys
from pathlib import Path

from . import declare, model, readers, runner
from .rules import registry


def check_binary(a) -> int:
    rules = registry()
    try:
        declarations = declare.load_declarations(Path(a.declare).read_text(), rules) if a.declare else []
        found = (runner.kernels_of_files(a.record) if a.fixture
                 else runner.kernels_of(a.record, a.arch, a.toolkit))
    except (declare.DeclarationError, runner.ToolError, readers.ReaderError, model.ModelError, OSError) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    kernels = list(found.values())
    refusals = declare.check(kernels, rules, declarations)
    for refusal in refusals:
        print("\n".join(refusal.lines()))
    for stale in declare.unmatched(declarations, kernels):
        print(f"note: the declaration for {stale.rule} on {stale.kernel.pattern!r} names no kernel")
    print(f"{len(kernels)} kernel(s), {len(rules)} rule(s): {len(refusals)} refusal(s)")
    return 1 if refusals else 0
