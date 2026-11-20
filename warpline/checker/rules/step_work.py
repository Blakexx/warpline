"""step_work: the instructions a thread executes for one tensor step, over its region.

The region is an innermost loop that holds steps (its instructions divided over its steps), or an unrolled step's own run
since the previous step. Each class of instruction (`Instruction.klass`) is rounded up a step and the figure is their
sum: the work the thread issues between one step and the next, which a step's own latency has to hide. The rule reports the
largest figure over the kernel's regions, the classes of that region in the finding; a project's declaration is its bound."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule
from ._steps import figures, where


def check(kernel: Kernel) -> list[Finding]:
    rows = sorted((figures(r) for r in kernel.regions), key=lambda f: -f.instructions)
    if not rows:
        return []
    top = rows[0]
    by = ", ".join(f"{c} {n}" for c, n in sorted(top.work.items(), key=lambda cn: -cn[1]))
    return [Finding(f"{top.region.address:#06x}", top.instructions,
                    f"{top.instructions} instructions a step at {where(top.region)}: {by}")]


RULE = Rule(
    name="step_work",
    statement="the instructions a thread executes for one tensor step, by class, over its region",
    check=check,
    refusal="a tensor step's region issues more instructions a step than the declared bound: the thread's issue slots, "
            "not the tensor pipe, set the step's rate",
    fix="move the work out of the step loop (hoist address arithmetic, stage conversions ahead, unroll so the descriptors "
        "are immediates), or declare a larger bound with its measurement",
    fixtures=(Fixture("loop", 5), Fixture("unrolled_pair", 4), Fixture("no_steps", 0)),
    bound=True,
)
