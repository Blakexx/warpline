"""loop_held: the most registers live at the top of any loop of a function (a bound: the project declares what a loop may hold).

A value formed before a loop and used after it, or by every trip, is held in a register for the whole loop. Under a small
register count the held values that do not fit are spilled and reloaded (rule `stack_memory`). The figure is the largest
count, over the function's natural loops, of the vector registers live at the loop's top (`Kernel.live_in` at the header's
entry; `held.py`); the finding names that loop's header, the count, and the three places that made the most of the held values:
the source lines where the listing with source-line marks was given (`--lines`), the addresses of the instructions that last
made them where not. A function with no loop has no finding. The rule reports a figure and never a verdict on its own."""
from __future__ import annotations

from ..held import holds
from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    found = holds(kernel)
    if not found:
        return []
    top = max(found, key=lambda h: (h.live, -h.loop.address))
    if not top.live:
        return []
    made = ", ".join(f"{g.label} ({len(g.registers)})" for g in top.groups[:3])
    return [Finding(f"{top.loop.address:#06x}", top.live,
                    f"{top.live} registers live at the top of the loop at {top.loop.address:#06x}, made at {made}")]


RULE = Rule(
    name="loop_held",
    statement="the most registers live at the top of any loop of a function (a bound: the project declares what a loop may hold)",
    check=check,
    refusal="a loop holds more registers across its trips than the declared bound",
    fix="form the values after the loop or inside it, hold fewer of them (recompute a cheap one, narrow a wide one), or "
        "declare the bound the design needs with its reason",
    fixtures=(Fixture("few", 8), Fixture("many", 30), Fixture("none", 0)),
    bound=True,
)
