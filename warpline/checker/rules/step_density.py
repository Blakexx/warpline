"""step_density: the kernel's static instructions for each tensor step.

The instruction count of the whole function (the dead instructions under `@!PT` left out) divided by its tensor steps (the
empty group the toolkit issues to hold a wait is no step), rounded up: a coarse ceiling on how much of the kernel is not
tensor work; above it the kernel is issue-bound whatever its steps do. A project's declaration is its ceiling."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    steps = sum(not s.null and not s.instruction.dead for s in kernel.steps)
    if not steps:
        return []
    total = sum(not i.dead for i in kernel.instructions)
    return [Finding("kernel", -(-total // steps), f"{total} instructions for {steps} tensor step(s)")]


RULE = Rule(
    name="step_density",
    statement="the kernel's static instructions for each tensor step, rounded up",
    check=check,
    refusal="the kernel has more static instructions a tensor step than the declared ceiling: it is issue-bound",
    fix="cut the non-tensor instructions (hoist or share address arithmetic, drop scalar work from the step path) or "
        "give each thread more steps; or declare the ceiling the kernel's design needs",
    fixtures=(Fixture("sparse", 12), Fixture("dense", 4), Fixture("no_steps", 0)),
    bound=True,
)
