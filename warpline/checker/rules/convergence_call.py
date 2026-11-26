"""convergence_call: a call into a convergence subroutine that a converged warp executes.

A warp collective under control flow the toolkit cannot prove converged is emitted as a call into a short subroutine
that syncs the warp, runs the collective and returns (`CALL.REL.NOINC`): a call and a return for each collective. The
toolkit guards the ones it emits: the call sits behind a `BRA.CONV` that a converged warp takes past it, or in the
fallback a `BRA.DIV` enters, so a converged warp runs the inline collective; a guarded call is cold code and is not
flagged. A call into an arithmetic slow path (a body longer than a subroutine) is not one either. A call is flagged when its
block is reachable from the entry without passing a guard (`Block.cold`) and its target is a subroutine (`Call.subroutine`)."""
from __future__ import annotations

from ..model import Kernel
from ...manifest import Finding
from ..rule import Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    return [Finding(f"{c.instruction.address:#06x}", 1, f"{c.instruction.opcode} at {c.instruction.address:#06x} into the "
                    f"convergence subroutine at {c.target:#06x}, executed by a converged warp")
            for c in kernel.calls if c.subroutine and c.hot]


RULE = Rule(
    name="convergence_call",
    statement="a call into a convergence subroutine that a converged warp executes (not a guarded one)",
    check=check,
    refusal="a warp collective is outlined into a convergence subroutine with no guard: every warp pays a call, a warp "
            "sync and a return for it",
    fix="make the control flow around the collective warp-uniform in the source (decide the role once, outside), so the "
        "toolkit can prove the warp converged and emits the collective inline",
    fixtures=(Fixture("hot_call", 1), Fixture("guarded_conv", 0), Fixture("guarded_div", 0), Fixture("slow_path", 0)),
)
