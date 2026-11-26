"""clock_uniform: a cycle-counter reading moved into a uniform register and summed there.

A uniform register holds one value for the whole warp. A phase clock accumulated in uniform registers
(`S2UR UR6, SR_CLOCKLO` then `UIADD3 UR4, ..., UR4, UR6`) is a warp's single reading, not each thread's count, and the
sum is no lap of any thread. Found by a forward flow: a uniform register holds a reading from its `S2UR` of
`SR_CLOCKLO`/`SR_CLOCKHI` until it is next written; a `UIADD3` that reads one counts."""
from __future__ import annotations

from .. import flow
from ..model import Instruction, Kernel
from ...manifest import Finding
from ..rule import Fixture, Rule


def _clock(ins: Instruction) -> bool:
    return ins.opcode == "S2UR" and ("SR_CLOCKLO" in ins.operands or "SR_CLOCKHI" in ins.operands)


def step(state: frozenset, ins: Instruction) -> frozenset:
    if ins.dead:
        return state
    return state | ins.writes if _clock(ins) else state - ins.writes


def check(kernel: Kernel) -> list[Finding]:
    return [Finding(f"{ins.address:#06x}", 1, f"{ins.opcode} at {ins.address:#06x} sums a clock reading held in "
                    f"{', '.join(sorted(ins.reads & state))}")
            for state, ins in flow.walk(kernel, step) if ins.base == "UIADD3" and not ins.dead and ins.reads & state]


RULE = Rule(
    name="clock_uniform",
    statement="a cycle-counter reading moved into a uniform register and summed there",
    check=check,
    refusal="a cycle-counter reading is summed in a uniform register: the sum is a warp's one value, not a thread's count",
    fix="read the counter into a per-thread register (`clock()` into a vector register) and accumulate there, or make the "
        "uniform sum explicit in the source as one warp's total",
    fixtures=(Fixture("summed", 1), Fixture("overwritten", 0), Fixture("vector_sum", 0)),
)
