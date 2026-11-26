"""step_branches (reads BOTH kinds of step): a branch inside a loop of tensor steps: a convergence barrier (`BSSY`) or a forward conditional `BRA`.

Each one ends the toolkit's scheduling block inside the step, so the step body is not one block the toolkit can order.
Both kinds of step count (a branch ends the scheduling block around an `HMMA` as around an `HGMMA`). The loop is an innermost natural loop that holds steps (a loop inside it that holds none is part of its body); the
conditional back edge, the `BRA.DIV` and `BRA.CONV` guards of a collective's cold fallback, and blocks a converged warp
does not reach are not counted. One shape is not counted: the loop's exit laid out as its LATCH, a forward conditional
branch out of the loop after its last step, followed by nothing but unpredicated register copies and the unconditional
back edge: it ends the block at the step's end, not inside it."""
from __future__ import annotations

import re

from ..model import Kernel
from ...manifest import Finding
from ..rule import Fixture, Rule

_COPY = re.compile(r"^(MOV|IMAD\.MOV(\.U32)?)$")


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for region in (r for r in kernel.regions if r.kind == "loop"):
        body = region.instructions
        latch = body[-1].predicate == ""
        for j, ins in enumerate(body[:-1]):
            forward = ins.opcode == "BRA" and ins.predicate and ins.target is not None and ins.target > ins.address
            copies = all(_COPY.match(i.opcode) and i.predicate == "" for i in body[j + 1:-1])
            if forward and latch and ins.target > body[-1].address and copies:
                continue
            if ins.base == "BSSY" or forward:
                found.append(Finding(f"{ins.address:#06x}", 1, f"{ins.predicate + ' ' if ins.predicate else ''}{ins.opcode} at "
                                     f"{ins.address:#06x} inside the loop of steps at {region.address:#06x}"))
    return found


RULE = Rule(
    name="step_branches",
    statement="a convergence barrier or a forward conditional branch inside a loop of tensor steps (either kind)",
    check=check,
    refusal="a branch sits inside a loop of tensor steps: it ends the toolkit's scheduling block inside the step",
    fix="make the step body one straight block: move the conditional out of the loop or fold it into a select, and let "
        "the loop's exit be its only branch",
    fixtures=(Fixture("in_step", 2), Fixture("latch_exit", 0), Fixture("straight_body", 0)),
)
