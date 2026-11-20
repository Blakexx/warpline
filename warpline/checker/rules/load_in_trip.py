"""load_in_trip: a global load whose value an instruction of the same trip of an innermost loop reads.

That trip waits out the load's whole latency before it can go on; a load issued a trip ahead of its use (its value read
in the next trip, before the next load overwrites it) is hidden behind the trip's work. Read in address order over the
blocks of each innermost natural loop: the first later instruction that reads a register the load writes (a `.64` or
`.128` load writes a run of registers) counts the load once; a write of the register first ends the scan."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule


def trips(kernel: Kernel):
    """Each innermost natural loop with its instructions in address order, the dead ones left out."""
    for lp in kernel.innermost():
        yield lp, sorted((i for b in lp.blocks for i in kernel.block(b).instructions if not i.dead), key=lambda i: i.address)


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for lp, trip in trips(kernel):
        for k, ins in enumerate(trip):
            if ins.base != "LDG" or not ins.writes:
                continue
            for later in trip[k + 1:]:
                if later.reads & ins.writes:
                    found.append(Finding(f"{ins.address:#06x}", 1, f"the load at {ins.address:#06x} is read by the instruction "
                                         f"at {later.address:#06x} in the same trip (loop headed at {lp.address:#06x})"))
                    break
                if later.writes & ins.writes:
                    break
    return found


RULE = Rule(
    name="load_in_trip",
    statement="a global load whose value an instruction of the same trip of an innermost loop reads",
    check=check,
    refusal="a loop trip reads the value of a global load it issued itself: it waits out the load's latency each trip",
    fix="issue the load a trip (a stage) ahead: load the next trip's operand at the top of this trip into a second set of "
        "registers and consume the previous trip's, or stage it with `cp.async` (`LDGSTS`) or the bulk copy",
    fixtures=(Fixture("read_in_trip", 1), Fixture("wide_pair", 1), Fixture("a_stage_ahead", 0), Fixture("outside_loop", 0)),
)
