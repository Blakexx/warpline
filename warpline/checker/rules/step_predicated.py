"""step_predicated (reads BOTH kinds): a tensor step under a predicate.

The toolkit if-converts a step that the source guards with a condition into a predicated instruction (`@P0 HGMMA`)
rather than a branch around it; a predicated step is issued by the warpgroup whether or not its predicate holds for a
lane, and the warpgroup's steps are collective, so a step that may be absent belongs behind a branch the whole group
takes together, or in straight code. A synchronous step (`@P0 HMMA`) is warp-collective in the same way: `mma.sync` has every lane of the warp issue it, so a lane-dependent
condition that predicates it is the same refusal, and the toolkit does if-convert it. Both kinds are read: the property is the
predicate, not the commit group. Instructions under the always-false `@!PT` are dead code and are not flagged."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    return [Finding(f"{s.instruction.address:#06x}", 1, f"{s.instruction.predicate} {s.instruction.opcode} at "
                    f"{s.instruction.address:#06x}")
            for s in kernel.steps if s.instruction.predicate and not s.instruction.dead]


RULE = Rule(
    name="step_predicated",
    statement="a tensor step, async or sync, under a predicate (if-converted)",
    check=check,
    refusal="a tensor step is predicated: the toolkit if-converted a condition that decides whether the group issues it",
    fix="guard the whole warpgroup's steps with a branch every lane takes together (a warp-uniform condition, behind its "
        "own fence), or issue the step unconditionally and select its result",
    fixtures=(Fixture("predicated", 1), Fixture("plain", 0), Fixture("dead", 0)),
)
