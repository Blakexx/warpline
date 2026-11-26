"""load_in_trip: a global load whose value an instruction of the same trip of an innermost loop reads.

That trip waits out the load's whole latency before it can go on; a load issued a trip ahead of its use (its value read
in the next trip, before the next load overwrites it) is hidden behind the trip's work.

"Read in the trip that issued it" is a statement about control flow, not about the listing's order. A reader counts when
it is reachable from the load, inside the loop's body, WITHOUT crossing the loop's back edge (an edge into the header), over
blocks a converged warp executes. What is and is not counted:
  - a read that PRECEDES the load in the trip (reached only through the back edge) is the next trip's: not counted, however
    the listing orders it;
  - a read only in a COLD block (`Block.cold`: the fallback a `BRA.DIV` enters, the code past a `BRA.CONV`, code under
    `@!PT`; the toolkit lays these out after the load) is no read of a converged warp: not counted, and not reported
    either, since a finding carries no notes (the cold block is a path, not a fact about the load);
  - a read on a path that leaves the loop, or on a path that only the other arm of a branch takes, is not counted;
  - a write of a register the load writes (a `.64` or `.128` load writes a run of registers) ends that path; a read in the
    same instruction as the write is a read.
The load counts once, naming the lowest-addressed reader it reaches."""
from __future__ import annotations

from collections.abc import Callable

from ..model import Instruction, Kernel, Loop
from ...manifest import Finding
from ..rule import Fixture, Rule


def trips(kernel: Kernel):
    """Each innermost natural loop with its instructions in address order, the dead ones left out."""
    for lp in kernel.innermost():
        yield lp, sorted((i for b in lp.blocks for i in kernel.block(b).instructions if not i.dead), key=lambda i: i.address)


def later_in_trip(kernel: Kernel, lp: Loop, ins: Instruction, hit: Callable[[Instruction], bool],
                  stop: Callable[[Instruction], bool]) -> list[Instruction]:
    """The instructions after `ins` that `hit` accepts, along the paths of the same trip: forward over the loop's blocks a
    converged warp executes, never over the back edge (an edge into the header) or out of the loop, each path ending at
    the first instruction `stop` accepts (an instruction is tested for `hit` first). A cold block is not entered. `ins`
    must sit in a block of the loop that is not cold."""
    found: list[Instruction] = []
    block = kernel.block(ins.block)
    seen = {block.id}
    work = [(block, [i for i in block.instructions if i.address > ins.address])]
    while work:
        block, body = work.pop()
        ended = False
        for later in body:
            if later.dead:
                continue
            if hit(later):
                found.append(later)
                ended = True
                break
            if stop(later):
                ended = True
                break
        if ended:
            continue
        for nxt in block.successors:
            if nxt in seen or nxt == lp.header or nxt not in lp.blocks or kernel.block(nxt).cold:
                continue
            seen.add(nxt)
            work.append((kernel.block(nxt), list(kernel.block(nxt).instructions)))
    return found


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for lp, trip in trips(kernel):
        for ins in trip:
            if ins.base != "LDG" or not ins.writes or kernel.block(ins.block).cold:
                continue
            readers = later_in_trip(kernel, lp, ins, lambda later: bool(later.reads & ins.writes),
                                    lambda later: bool(later.writes & ins.writes))
            if readers:
                first = min(readers, key=lambda r: r.address)
                found.append(Finding(f"{ins.address:#06x}", 1, f"the load at {ins.address:#06x} is read by the instruction "
                                     f"at {first.address:#06x} in the same trip (loop headed at {lp.address:#06x})"))
    return found


RULE = Rule(
    name="load_in_trip",
    statement="a global load whose value an instruction reads that the same trip of an innermost loop reaches from it, without the back edge, over blocks a converged warp executes",
    check=check,
    refusal="a loop trip reads the value of a global load it issued itself: it waits out the load's latency each trip",
    fix="issue the load a trip (a stage) ahead: load the next trip's operand at the top of this trip into a second set of "
        "registers and consume the previous trip's, or stage it with `cp.async` (`LDGSTS`) or the bulk copy",
    fixtures=(Fixture("read_in_trip", 1), Fixture("wide_pair", 1), Fixture("a_stage_ahead", 0), Fixture("outside_loop", 0),
              Fixture("cold_reader", 0), Fixture("cold_and_hot_reader", 1)),
)
