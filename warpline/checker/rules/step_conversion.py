"""step_conversion: an instruction on the conversion unit (`F2I`, `I2F`, `F2F`, `MUFU`) inside a loop of tensor steps.

The conversion unit runs 16 lanes a cycle for an SM where the FMA pipe's forms (a magic-constant `FFMA`, a `PRMT`) run at
64, so a conversion in the step's own loop competes for issue slots at a quarter of the rate. The loop is an innermost
natural loop that holds steps; instructions in blocks a converged warp does not reach, and dead ones, are not counted."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    return [Finding(f"{i.address:#06x}", 1, f"{i.opcode} at {i.address:#06x} inside the loop of steps at {r.address:#06x}")
            for r in kernel.regions if r.kind == "loop" for i in r.instructions if i.klass == "conversion unit"]


RULE = Rule(
    name="step_conversion",
    statement="an instruction on the conversion unit (F2I, I2F, F2F, MUFU) inside a loop of tensor steps",
    check=check,
    refusal="the conversion unit runs inside a loop of tensor steps: its 16 lanes a cycle an SM set the step's rate",
    fix="convert on the FMA pipe (a magic-constant add, a PRMT) or hoist the conversion out of the step loop and stage "
        "its result",
    fixtures=(Fixture("in_loop", 2), Fixture("outside_loop", 0), Fixture("fma_form", 0)),
)
