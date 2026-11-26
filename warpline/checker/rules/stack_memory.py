"""stack_memory: any use of stack memory.

The resource report's `STACK` is the kernel's stack frame in bytes: a register spill or a dynamically indexed local
array. The finding's count is those bytes; its detail names the stack instructions (`LDL`/`STL`) the listing holds."""
from __future__ import annotations

from ..model import Kernel
from ...manifest import Finding
from ..rule import Fixture, Rule


def check(kernel: Kernel) -> list[Finding]:
    if not kernel.stack_bytes:
        return []
    stack = [a for a in kernel.accesses if a.space == "local"]
    loads = sum(a.kind == "load" for a in stack)
    where = f"{stack[0].instruction.address:#06x}" if stack else "kernel"
    return [Finding(where, kernel.stack_bytes,
                    f"a {kernel.stack_bytes}-byte stack frame; {loads} LDL and {len(stack) - loads} STL "
                    f"instruction(s) in the listing")]


RULE = Rule(
    name="stack_memory",
    statement="any use of stack memory (a nonzero stack frame in the resource report)",
    check=check,
    refusal="the kernel uses stack memory (local memory): its accesses are far slower than registers",
    fix="remove the spill (fewer live values, a smaller tile, `__launch_bounds__`) or index the array with a "
        "compile-time constant so it lives in registers",
    fixtures=(Fixture("spill", 128), Fixture("no_stack", 0)),
)
