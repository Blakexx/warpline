"""step_global: a global-memory access inside a loop of tensor steps, other than a copy that stages operands in.

A step's operands are staged ahead: `cp.async` (`LDGSTS`), the bulk copy in (`UBLKCP.S.G`) and the tensor-map copy
(`UTMALDG`) issue the copy and return; a load, store, atomic or reduction to global memory in the loop's body
(`LDG`, `STG`, `RED`, `ATOMG`, a bulk copy or reduction out) stalls the step on memory or competes for the port; a
reduction to global memory belongs to a drain outside the steps. The loop is an innermost natural loop that holds steps."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule

_STAGED = ("LDGSTS", "UBLKCP.S.G", "UTMALDG")


def check(kernel: Kernel) -> list[Finding]:
    return [Finding(f"{i.address:#06x}", 1, f"{i.opcode} at {i.address:#06x} inside the loop of steps at {r.address:#06x}")
            for r in kernel.regions if r.kind == "loop" for i in r.instructions
            if i.klass == "global" and not i.opcode.startswith(_STAGED)]


RULE = Rule(
    name="step_global",
    statement="a global-memory access other than a staging copy inside a loop of tensor steps",
    check=check,
    refusal="a loop of tensor steps accesses global memory directly: the step waits on memory or the port",
    fix="stage the operands ahead with `cp.async` or the bulk copy, and move global reductions and stores to a drain "
        "outside the step loop",
    fixtures=(Fixture("in_loop", 2), Fixture("staged", 0), Fixture("outside_loop", 0)),
)
