"""wait_after_issue (reads ASYNC steps only): a tensor wait directly after its group's steps, with nothing between.

A wait for all groups in flight (`WARPGROUP.DEPBAR.LE gsb0, 0x0`) that follows the group's last step with no work of
substance between them leaves the warp idle for the steps' whole latency. Work of substance is what a warp can do
while its steps run: memory loads, stores and atomics, asynchronous copies and barrier arrivals; not the commit, the
`WARPGROUP.ARRIVE`, constant loads, register moves, or address and predicate arithmetic. A synchronous step (`mma.sync`, `HMMA`) has no wait, so it neither ends the scan as a group's step nor is flagged. The scan goes back through
the wait's own basic block; a branch, a barrier, another wait or the block's start first leaves it unflagged."""
from __future__ import annotations

import re

from ..model import Kernel
from ..rule import Finding, Fixture, Rule

#: work of substance, by opcode (the matrix forms and the copies included)
_SUBSTANCE = re.compile(r"^(?:LDG(?!DEPBAR)|LDS|LDL|LD\b|STG|STS|STL|ST\b|ATOM|RED|UBLK|UTMA|SYNCS\.ARRIVE|BAR\.ARV)")
_STOPS = ("BRA", "BSYNC", "BREAK", "JMP", "EXIT", "RET", "CALL", "BAR", "SYNCS", "WARPGROUP", "HGMMA", "IGMMA",
          "QGMMA", "BGMMA")


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for wait in kernel.waits:
        if wait.left:
            continue
        body = kernel.block(wait.instruction.block).instructions
        before = body[:body.index(wait.instruction)]
        for ins in reversed(before):
            if ins.base in ("HGMMA", "IGMMA", "QGMMA", "BGMMA"):
                found.append(Finding(f"{wait.instruction.address:#06x}", 1,
                                     f"the wait at {wait.instruction.address:#06x} follows the step at "
                                     f"{ins.address:#06x} with no work between"))
                break
            if _SUBSTANCE.match(ins.opcode) or ins.opcode.startswith(_STOPS):
                break
    return found


RULE = Rule(
    name="wait_after_issue",
    statement="a tensor wait for every group in flight directly after its group's steps, with nothing between (async steps only: a sync step has no wait)",
    check=check,
    refusal="a tensor wait follows its steps with no work between: the warp idles for the steps' whole latency",
    fix="issue independent work (loads, stores, copies, arrivals) between the last step and the wait, or wait for "
        "fewer groups (`wait_group N` with N above 0) so the next group's steps overlap",
    fixtures=(Fixture("bare_wait", 1), Fixture("work_between", 0), Fixture("partial_wait", 0)),
)
