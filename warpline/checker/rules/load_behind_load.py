"""load_behind_load: a global load issued after the same trip has already waited for another global load it does not depend on.

A warp issues its instructions in order, and an instruction that reads a load's value waits until the value has arrived. A
second load issued after that instruction therefore starts a whole memory round trip late, although its address was known
before the wait: the trip pays the two latencies one after the other. Issued before the first value is read, the two loads
are in flight together and the trip pays one.

What is counted, along each innermost loop's trip (its instructions in address order, over blocks a converged warp
executes: `load_in_trip.trips`): a register is DERIVED when a global load of the trip wrote it or an instruction that read a
derived register wrote it; the trip has WAITED once an instruction has read a derived register. A global load issued after
the trip has waited, whose address reads no derived register, is a finding. What is not:
  - a load whose address is derived from an earlier load (a chain: it cannot be issued sooner);
  - every load issued before the first read of any load's value, however many;
  - a load in a cold block (`Block.cold`), and dead code.
A register written from registers that are not derived is no longer derived. The assembler moves a load down to shorten the
life of what it loads, so a kernel whose source loads everything first can still compile to this."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule
from .load_in_trip import trips


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for lp, trip in trips(kernel):
        derived: set[str] = set()
        waited = None
        for ins in trip:
            if kernel.block(ins.block).cold:
                continue
            reads_derived = bool(ins.reads & derived)
            load = ins.base == "LDG" and bool(ins.writes)
            if load and waited is not None and not reads_derived:
                found.append(Finding(f"{ins.address:#06x}", 1, f"the load at {ins.address:#06x} is issued after the instruction at "
                                     f"{waited:#06x} waited for another load's value (loop headed at {lp.address:#06x})"))
            if reads_derived and waited is None:
                waited = ins.address
            if load or reads_derived:
                derived |= ins.writes
            else:
                derived -= ins.writes
    return found


RULE = Rule(
    name="load_behind_load",
    statement="a global load issued after its trip has already waited for another global load's value, with an address that depends on no load",
    check=check,
    refusal="a loop trip issues a load only after waiting for another load it does not depend on: it pays the two latencies in turn",
    fix="issue every load of the trip before the first use of any of their values; where the assembler moved a load down, put "
        "the loads under a condition it cannot fold (a value only the launch knows), or issue them a trip ahead",
    fixtures=(Fixture("split", 1), Fixture("together", 0)),
)
