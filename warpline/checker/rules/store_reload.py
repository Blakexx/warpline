"""store_reload: a shared load of an address the same trip of an innermost loop stored to.

A value written to shared memory and read back by the same thread in the same trip goes through the shared-memory
port twice for nothing: it belongs in a register, unless another warp reads it (which this does not know: a
declaration says so). The address is the same operand text (`[R8+0x10]`); a write of a register of the address between
the store and the load, or the end of the trip, ends the scan. Read in address order over the blocks of each innermost
natural loop."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule
from .load_in_trip import trips


def check(kernel: Kernel) -> list[Finding]:
    found = []
    for lp, trip in trips(kernel):
        for k, ins in enumerate(trip):
            if ins.base not in ("STS", "STSM") or ins.address_operand is None:
                continue
            for later in trip[k + 1:]:
                if later.base in ("LDS", "LDSM") and later.address_operand == ins.address_operand:
                    found.append(Finding(f"{ins.address:#06x}", 1, f"the store at {ins.address:#06x} to "
                                         f"{ins.address_operand} is loaded back at {later.address:#06x} in the same trip"))
                    break
                if later.writes & ins.address_registers:
                    break
    return found


RULE = Rule(
    name="store_reload",
    statement="a shared load of an address the same trip of an innermost loop stored to",
    check=check,
    refusal="a loop trip stores a value to shared memory and loads it back: two passes through the shared-memory port "
            "for a value that could stay in a register",
    fix="keep the value in a register for the rest of the trip; keep the shared store only for a value another warp "
        "reads, and declare the count with that reason",
    fixtures=(Fixture("reload", 1), Fixture("other_address", 0), Fixture("base_rewritten", 0), Fixture("outside_loop", 0)),
)
