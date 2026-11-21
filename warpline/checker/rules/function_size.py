"""function_size: the instructions of a function, padding excluded.

The figure is the count of instructions the model holds for the kernel: the listing up to the graph's last instruction,
the bodies the graph does not hold (a convergence subroutine a `CALL` reaches), and no padding (the self-branch and `NOP`s
the toolkit appends to fill the function to its alignment). An instruction under `@!PT` stays: it is code the function
has. The rule reports a figure and never a verdict on its own: a project's declaration is the bound."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    n = len(kernel.instructions)
    if not n:
        return []
    return [Finding("kernel", n, f"{n} instructions, padding excluded")]


RULE = Rule(
    name="function_size",
    statement="the instructions of a function, padding excluded (a bound: the project declares the size it holds)",
    check=check,
    refusal="the function holds more instructions than the declared bound",
    fix="shrink the code (fewer unrolled copies, hoist what repeats), or declare a larger bound with its reason",
    fixtures=(Fixture("small", 4), Fixture("padded", 4), Fixture("large", 11)),
    bound=True,
)
