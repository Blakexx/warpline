"""store_unfenced (reads ASYNC steps only): in a kernel that issues async tensor steps, a barrier arrival that a shared store reaches with no
async-proxy fence between.

A synchronous step (`mma.sync`) reads registers, so a kernel with only those is not read. An async tensor step reads shared memory through the async proxy, and a thread's ordinary stores reach it only past a
`fence.proxy.async` (`FENCE.VIEW.ASYNC`); a barrier arrival that tells the reader the data is there must come after the
fence. Found by a forward flow over the graph, not the listing's order: a store that reaches an arrival along any path
(the next trip of a loop included) with no fence on that path is unfenced; an arrival clears the state."""
from __future__ import annotations

from .. import flow
from ..model import Instruction, Kernel
from ..rule import Finding, Fixture, Rule


def step(state: frozenset, ins: Instruction) -> frozenset:
    if ins.dead:
        return state
    if ins.base in ("STS", "STSM"):
        return frozenset({ins.address})
    if ins.opcode.startswith(("FENCE.VIEW.ASYNC", "SYNCS.ARRIVE")):
        return frozenset()
    return state


def check(kernel: Kernel) -> list[Finding]:
    if not kernel.async_steps:
        return []
    return [Finding(f"{ins.address:#06x}", 1, f"the arrival at {ins.address:#06x} follows the shared store at "
                    f"{', '.join(f'{a:#06x}' for a in sorted(state))} with no async-proxy fence between")
            for state, ins in flow.walk(kernel, step) if state and not ins.dead and ins.opcode.startswith("SYNCS.ARRIVE")]


RULE = Rule(
    name="store_unfenced",
    statement="in a kernel with async tensor steps, a barrier arrival that a shared store reaches with no async-proxy fence between (async steps only)",
    check=check,
    refusal="a shared store is announced by a barrier arrival with no async-proxy fence between: the tensor pipe may "
            "read the old data",
    fix="issue `fence.proxy.async.shared::cta` after the stores and before the `mbarrier.arrive` that announces them",
    fixtures=(Fixture("arrive_after_store", 1), Fixture("fenced", 0), Fixture("no_steps", 0), Fixture("loop_carried", 1)),
)
