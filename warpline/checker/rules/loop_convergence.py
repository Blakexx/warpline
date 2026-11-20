"""loop_convergence: a convergence instruction (`BSSY`, `BSYNC`, `BRA.DIV`) inside a natural loop.

Each one ends the toolkit's scheduling block inside the loop, so the loop's work is not interleaved across the join, and
a `BRA.DIV` is the guard of a warp collective the toolkit could not prove converged. A loop is a NATURAL loop of the
graph (a back edge to a block that dominates its source), the blocks it holds wherever they are laid out; an
instruction in no loop, however far a backward branch in the listing reaches over it, is not flagged."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for ins in kernel.convergence:
        if (ins.base in ("BSSY", "BSYNC") or ins.base == "BRA") and (header := kernel.block(ins.block).loop):
            found.append(Finding(f"{ins.address:#06x}", 1, f"{ins.opcode} at {ins.address:#06x} in the loop headed at "
                                 f"{kernel.block(header).start:#06x}"))
    return found


RULE = Rule(
    name="loop_convergence",
    statement="a convergence instruction (BSSY, BSYNC, BRA.DIV) inside a natural loop",
    check=check,
    refusal="a convergence barrier or divergence guard sits inside a loop: each trip pays the join and the toolkit "
            "cannot schedule across it",
    fix="hoist the divergent region out of the loop or make the condition warp-uniform in the source (a uniform "
        "predicate, a role decided once outside the loop), so the toolkit emits no join in the loop body; for a collective, "
        "give it all lanes by computing it before the branch and selecting after",
    fixtures=(Fixture("in_loop", 2), Fixture("outside_loop", 0), Fixture("stub_return", 0)),
)
