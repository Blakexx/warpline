"""wait_all_with_work (reads ASYNC steps only): a wait for every group in flight after which the instructions up to the next step read and write none of the newest group's accumulators.

A wait for fewer groups (`wait_group N` with N above 0) would let that work overlap the newest group. The wait is
`WARPGROUP.DEPBAR.LE gsb0, 0x0`; the NEWEST GROUP is the steps back from it to the previous commit (a step with `gsb0`) or the
previous wait, read through the wait's block and the unique predecessors above it, and its registers are the steps' destinations
(`_accumulators.destination`). The scan runs forward over every path from the wait to the next step (`HGMMA` and the others, the
sync ones too) or the next wait or the function's end; work is what `wait_after_issue` calls work of substance (a memory load,
store or atomic, a copy, a barrier or an arrival: not a branch, a register move, address or predicate arithmetic, or a
`WARPGROUP` control; a barrier is work, a wait for fewer groups would let it overlap). The wait is a finding when
a path holds work, a step follows, and on no path does an instruction read or write the group's registers (every operand widened
by `_accumulators.width`, so a wide access is not missed).

What it cannot decide, and says nothing about: a wait whose group it cannot read (no step behind it before a join of several
predecessors, or a step with no shape or accumulator type), a path that reaches a `CALL`, `JMP` or an indirect branch (the
callee or the target is unseen), and a path that reads an accumulator through a register no instruction names (a descriptor
that points at registers). Work in a block the paths reach only through a cold fallback counts as work like any other; so does
work on a path that leaves a loop and comes back to a step (the next trip's loads overlap too)."""
from __future__ import annotations

import re

from ..model import Instruction, Kernel
from ..rule import Finding, Fixture, Rule
from ._accumulators import destination, touched

#: work of substance, as `wait_after_issue` reads it (memory loads, stores and atomics, asynchronous copies, barriers and
#: arrivals): not a branch, a convergence instruction, a `WARPGROUP` control, a register move or address and predicate arithmetic
_SUBSTANCE = re.compile(r"^(?:LDG(?!DEPBAR)|LDS|LDL|LD\b|STG|STS|STL|ST\b|ATOM|RED|UBLK|UTMA|SYNCS\.ARRIVE|BAR)")
_UNSEEN = ("CALL", "JMP", "BRX", "RET")
_STEPS = ("HGMMA", "IGMMA", "QGMMA", "BGMMA", "HMMA", "IMMA", "DMMA")


def _group(kernel: Kernel, wait: Instruction, steps: dict[int, frozenset[str]], commits: set[int],
           preds: dict[str, list[str]]) -> frozenset[str] | None:
    """The registers of the newest group before `wait`, or None where the group cannot be read."""
    regs: set[str] = set()
    block, upto = kernel.block(wait.block), wait.address
    seen = set()
    while True:
        body = [i for i in block.instructions if i.address < upto]
        for ins in reversed(body):
            if ins.opcode.startswith("WARPGROUP.DEPBAR") or (ins.address in commits and regs):
                return frozenset(regs) if regs else None
            if ins.base in _STEPS[:4] and not ins.dead:
                if ins.address not in steps or not steps[ins.address]:
                    return None
                regs |= steps[ins.address]
        if block.id in seen:
            return frozenset(regs) if regs else None
        seen.add(block.id)
        above = preds[block.id]
        if not above:
            return frozenset(regs) if regs else None
        if len(above) != 1:
            return None
        block, upto = kernel.block(above[0]), 1 << 62


def check(kernel: Kernel) -> list[Finding]:
    steps = {s.instruction.address: destination(s) for s in kernel.async_steps}
    commits = {s.instruction.address for s in kernel.async_steps if s.commit}
    preds: dict[str, list[str]] = {b.id: [] for b in kernel.blocks}
    for b in kernel.blocks:
        for nxt in b.successors:
            preds[nxt].append(b.id)
    found = []
    for wait in kernel.waits:
        if wait.left != 0 or wait.instruction.dead:
            continue
        group = _group(kernel, wait.instruction, steps, commits, preds)
        if not group:
            continue
        verdict = _scan(kernel, wait.instruction, group)
        if verdict is not None:
            work, step = verdict
            found.append(Finding(f"{wait.instruction.address:#06x}", 1,
                                 f"the wait for every group at {wait.instruction.address:#06x} is followed by {work} instruction(s) of work up to "
                                 f"the step at {step:#06x} that touch none of {sorted(group, key=lambda r: int(r[1:]))[0]} to "
                                 f"{sorted(group, key=lambda r: int(r[1:]))[-1]}, the newest group's accumulators"))
    return found


def _scan(kernel: Kernel, wait: Instruction, group: frozenset[str]) -> tuple[int, int] | None:
    """(the instructions of work, the address of a step reached) if the paths from the wait hold work, reach a step and touch
    none of `group`; None when a path touches it or reads past what the model sees, or there is no work or no step."""
    work: set[int] = set()
    reached: list[int] = []
    block = kernel.block(wait.block)
    todo = [(block.id, [i.address for i in block.instructions].index(wait.address) + 1)]
    seen: set[tuple[str, int]] = set()
    while todo:
        bid, k = todo.pop()
        block = kernel.block(bid)
        if (bid, k) in seen:
            continue
        seen.add((bid, k))
        for ins in block.instructions[k:]:
            if ins.dead:
                continue
            if ins.base in _STEPS:
                reached.append(ins.address)
                break
            if ins.opcode.startswith("WARPGROUP.DEPBAR"):
                break
            if ins.base in _UNSEEN:
                return None
            if touched(ins) & group:
                return None
            if _SUBSTANCE.match(ins.opcode):
                work.add(ins.address)
            if ins.base == "EXIT" and ins.predicate == "":
                break
        else:
            todo.extend((nxt, 0) for nxt in block.successors)
    if not work or not reached:
        return None
    return len(work), min(reached)


RULE = Rule(
    name="wait_all_with_work",
    statement="a wait for every group in flight after which the instructions up to the next step read and write none of the newest group's accumulators (a wait for fewer groups would let that work overlap; async steps only)",
    check=check,
    refusal="a wait for every group in flight holds back work that does not need the newest group: the warp idles for that group's latency",
    fix="wait for fewer groups (`wait_group N` with N above 0) so the work that does not read the newest group's accumulators "
        "overlaps it, or move that work between the steps and the wait",
    fixtures=(Fixture("wait_all_work", 1), Fixture("wait_all_reads", 0), Fixture("serialized", 0)),
)
