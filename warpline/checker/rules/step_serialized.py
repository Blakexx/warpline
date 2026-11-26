"""step_serialized (reads ASYNC steps only): a run of tensor steps each followed by its own wait for every group in flight.

The toolkit serializes a run of steps (a `WARPGROUP.ARRIVE`, one step, `WARPGROUP.DEPBAR.LE gsb0, 0x0`, repeated) when it cannot keep
the groups in flight, with or without a notice (`C7511`, `C7514`, `C7520`, ...): each step waits for the one before it and the run
loses the overlap the steps could have had. The rule reads the listing, not the notice: a step is SERIALIZED when a wait for all
groups (`DEPBAR` 0) follows it before the next step; steps are taken in address order, and two steps are in one
run only if they are in the same innermost loop (or both in none) and no unconditional branch, `EXIT` or `RET` lies between them. A run is the
longest stretch of consecutive steps that starts and ends on a serialized step and in which at most one step in eight is not
serialized, and it is a finding when it has `MINIMUM` steps. The null group the toolkit issues to hold a wait is no step."""
from __future__ import annotations

from ..model import Instruction, Kernel, TensorStep
from ...manifest import Finding
from ..rule import Fixture, Rule

#: the fewest steps in a run that is reported: a pair of steps each waited on is a plausible tail, not a run the toolkit serialized
MINIMUM = 4
#: the waits a finding names before it counts the rest
NAMED = 6


def _leaves(ins: Instruction) -> bool:
    """An instruction after which the next instruction in address order is not where control goes."""
    return ins.base in ("EXIT", "RET", "JMP") or (ins.base == "BRA" and ins.predicate == "")


def _chains(kernel: Kernel):
    """The steps in address order, split where an unconditional transfer separates two of them; each step with the address
    of the wait for all groups that follows it before the next step (None if none does)."""
    steps = [s for s in kernel.async_steps if not s.null and not s.instruction.dead]
    at = {ins.address: k for k, ins in enumerate(kernel.instructions)}
    waits = {w.instruction.address for w in kernel.waits if w.left == 0}
    chain: list[tuple[TensorStep, int | None]] = []
    for k, s in enumerate(steps):
        nxt = at[steps[k + 1].instruction.address] if k + 1 < len(steps) else len(kernel.instructions)
        wait, cut = None, k + 1 == len(steps) or kernel.block(steps[k + 1].instruction.block).loop != kernel.block(s.instruction.block).loop
        for ins in kernel.instructions[at[s.instruction.address] + 1:nxt]:
            if ins.address in waits and wait is None:
                wait = ins.address
            if _leaves(ins):
                cut = True
                break
        chain.append((s, wait))
        if cut:
            yield chain
            chain = []
    if chain:
        yield chain


def _runs(chain):
    """The runs of a chain: from a serialized step, while at most one step in eight is not serialized; ending on a serialized one."""
    k = 0
    while k < len(chain):
        if chain[k][1] is None:
            k += 1
            continue
        end, missed, pending = k, 0, 0
        for j in range(k + 1, len(chain)):
            if chain[j][1] is None:
                pending += 1
            elif (missed + pending) * 8 <= j - k + 1:
                end, missed, pending = j, missed + pending, 0
            else:
                break
        yield chain[k:end + 1]
        k = end + 1


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for chain in _chains(kernel):
        for run in _runs(chain):
            if len(run) < MINIMUM:
                continue
            waits = [w for _, w in run if w is not None]
            named = ", ".join(f"{w:#06x}" for w in waits[:NAMED]) + (f" and {len(waits) - NAMED} more" if len(waits) > NAMED else "")
            first, last = run[0][0].instruction.address, run[-1][0].instruction.address
            found.append(Finding(f"{first:#06x}", 1, f"{len(run)} tensor steps from {first:#06x} to {last:#06x}, "
                                 f"{len(waits)} of them each followed by its own wait for every group, at {named}"))
    return found


RULE = Rule(
    name="step_serialized",
    statement="a run of async tensor steps in which (nearly) every step is followed by its own wait for every group in flight before the next step (async steps only: a sync step has no wait)",
    check=check,
    refusal="the toolkit serialized a run of tensor steps: each step waits for all groups before the next issues, so the steps do not overlap",
    fix="find the notice or the shape that made the toolkit serialize (a branch that each thread decides for itself inside the "
        "steps, a `wgmma.fence` outside the block the steps are in, accumulators moved between shapes, too few registers) and "
        "make the steps one straight block, with the fence and the commit group around them in it",
    fixtures=(Fixture("serialized", 2), Fixture("fenced", 0), Fixture("mixed_shapes", 1)),
)
