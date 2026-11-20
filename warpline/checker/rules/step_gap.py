"""step_gap: the longest run of instructions without a tensor step in a loop of steps.

In-order issue makes a run the time a warp leaves the tensor pipe to the scheduler's other warps. For each innermost
natural loop that holds steps (the instructions a converged warp reaches, dead ones left out) the rule takes the longest
run between two steps of one trip and the run around the back edge (the instructions after the trip's last step and before
its first: a rolled loop's own gap, an unrolled one's fill and drain); the figure is the larger, over the kernel's loops. A
project's declaration is its bound."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule
from ._steps import where


def check(kernel: Kernel) -> list[Finding]:
    rows = []
    for region in (r for r in kernel.regions if r.kind == "loop"):
        steps = {t.instruction.address for t in region.tensor if not t.null}
        at = [k for k, i in enumerate(region.instructions) if i.address in steps]
        inside = max((b - a - 1 for a, b in zip(at, at[1:])), default=0)
        around = len(region.instructions) - 1 - (at[-1] - at[0])
        rows.append((max(inside, around), inside, around, region))
    if not rows:
        return []
    figure, inside, around, region = max(rows, key=lambda r: r[0])
    return [Finding(f"{region.address:#06x}", figure, f"{figure} instructions without a step at {where(region)}: "
                    f"{inside} between steps, {around} around the back edge")]


RULE = Rule(
    name="step_gap",
    statement="the longest run of instructions without a tensor step in a loop of steps, between steps and around the back edge",
    check=check,
    refusal="a loop of tensor steps leaves the tensor pipe for more instructions than the declared bound between its steps",
    fix="interleave the loop's other work between its steps (issue loads and arithmetic between the steps, not in a "
        "block before or after them), or unroll so the fill and drain are amortized; or declare the bound the design needs",
    fixtures=(Fixture("gapped", 5), Fixture("dense_steps", 4), Fixture("no_steps", 0)),
    bound=True,
)
