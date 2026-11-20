"""step_rolled: a tensor step inside a natural loop that does not also hold the wait on its commit group.

Steps issued from a rolled loop whose wait lies outside it run at about half a straight block's rate, with no notice
from the toolkit. The loop meant is the INNERMOST natural loop around a step; a kernel's outer loop that holds its
steps and the wait on them is not flagged, nor is a step outside every loop. A wait anywhere in the loop's blocks,
an inner loop's included, is the loop holding the wait."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    waiting = {w.instruction.block for w in kernel.waits}
    steps: dict[str, int] = {}
    for step in kernel.steps:
        loop = kernel.loop_of(step.instruction)
        if loop is not None and not loop.blocks & waiting:
            steps[loop.header] = steps.get(loop.header, 0) + 1
    found = []
    for lp in kernel.loops:
        if lp.header in steps:
            found.append(Finding(f"{lp.address:#06x}", steps[lp.header],
                                 f"{steps[lp.header]} tensor step(s) in the loop headed at {lp.address:#06x} "
                                 f"({len(lp.blocks)} block(s)) and no tensor wait in it"))
    return found


RULE = Rule(
    name="step_rolled",
    statement="a tensor step inside a natural loop that does not also hold the wait on its commit group",
    check=check,
    refusal="tensor steps are issued from a loop that does not wait on them: they run at about half the rate of a "
            "straight block of steps",
    fix="unroll the loop so the steps and their wait are one straight block, or move the wait into the loop body",
    fixtures=(Fixture("rolled", 4), Fixture("straight", 0), Fixture("outer_wait", 0), Fixture("nested", 2)),
)
