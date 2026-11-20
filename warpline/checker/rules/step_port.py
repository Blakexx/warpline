"""step_port: the bytes a tensor step puts through the shared-memory port, over its region.

A step reads its operands from shared memory through descriptors (A is m x k and B is k x n elements, at the input
type's size; a step whose A is in registers reads B alone), and the shared loads and stores of its region (an innermost
loop that holds steps, or an unrolled step's own run) go through the same port, every warp of the group executing them.
The rule reports the largest figure over the kernel's regions, in bytes a step (a wavefront is 128 bytes, so the figure over 128
is the wavefronts a step holds the port for); a project's declaration is its bound. A step whose form the model does
not read is named in the finding and counts only the region's own shared accesses."""
from __future__ import annotations

from ..model import Kernel
from ..rule import Finding, Fixture, Rule
from ._steps import WARPS, WAVEFRONT, figures, where


def check(kernel: Kernel) -> list[Finding]:
    rows = sorted((figures(r) for r in kernel.regions), key=lambda f: -f.port)
    if not rows:
        return []
    top = rows[0]
    shown = "; ".join(f"{where(f.region)}: {f.port:,} B a step" + (f" ({f.unread} step(s) read as 0 B: form not read)" if f.unread else "")
                      for f in rows[:6])
    detail = (f"{top.port:,} B a step ({-(-top.port // WAVEFRONT)} wavefronts) at {where(top.region)}: operands {top.operands:,} + "
              f"shared {top.shared:,}" + (f" (x {WARPS} warps)" if top.shared else "") + f"; regions: {shown}")
    return [Finding(f"{top.region.address:#06x}", top.port, detail)]


RULE = Rule(
    name="step_port",
    statement="the bytes a tensor step puts through the shared-memory port: its operands and its region's shared accesses",
    check=check,
    refusal="a tensor step's region moves more bytes through the shared-memory port than the declared bound",
    fix="stage fewer bytes a step: hold A in registers, widen the step's N so B is read once for more work, move shared "
        "loads and stores out of the step loop (stage them ahead), or declare a larger bound with its measurement",
    fixtures=(Fixture("operands_only", 2560), Fixture("loop_with_loads", 5120), Fixture("a_in_registers", 512),
              Fixture("no_steps", 0)),
    bound=True,
)
