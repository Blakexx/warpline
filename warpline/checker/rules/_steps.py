"""What the rules over a step's cost share: the figures of a step region (`model.StepRegion`), each the region's total over
its steps, rounded up. Not a rule (it defines no `RULE`)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..model import StepRegion

#: a warpgroup's warps: a shared load or store between two steps is executed by each of them
WARPS = 4
#: the bytes one wavefront of the shared-memory port moves
WAVEFRONT = 128


@dataclass(frozen=True)
class Figures:
    region: StepRegion
    operands: int          # the steps' own operand bytes a step (the steps this reads)
    shared: int            # the region's shared loads and stores, every warp of the group, a step
    unread: int            # steps whose operands the model does not read
    work: Counter          # instructions a step by class

    @property
    def port(self) -> int:
        return self.operands + self.shared

    @property
    def instructions(self) -> int:
        return sum(self.work.values())


def figures(region: StepRegion) -> Figures:
    per = lambda n: -(-n // region.steps)  # noqa: E731
    own = [t.operand_bytes for t in region.tensor if not t.null]
    shared = WARPS * sum(a.warp_bytes for a in region.accesses if a.space == "shared" and a.kind in ("load", "store"))
    work = Counter({c: per(n) for c, n in Counter(i.klass for i in region.instructions).items()})
    return Figures(region, per(sum(b for b in own if b is not None)), per(shared), sum(b is None for b in own), work)


def where(region: StepRegion) -> str:
    return f"{region.kind} at {region.address:#06x} ({region.steps} step{'s' if region.steps != 1 else ''})"
