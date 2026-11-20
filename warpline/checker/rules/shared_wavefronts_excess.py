"""shared_wavefronts_excess: shared-memory wavefronts in excess of the conflict-free count, by instruction.

A warp's shared-memory access takes one wavefront when no two lanes touch different words of one bank; each extra
wavefront is a bank conflict (or a misaligned wide access) that the memory pipe serves in another pass. The profiler's
source page states, for each instruction, the wavefronts it took and the count it would take without conflicts. The
finding of an instruction is its excess, EXECUTED: a count over the profiled window (what a declaration allows is a
count of that window, or zero). Where is the instruction's offset in the kernel (its address less the first row's)."""
from __future__ import annotations

from ..ncu import Profile
from ..rule import Finding, Fixture, ProfileRule


def check(profile: Profile) -> list[Finding]:
    if not profile.source:
        return []
    base = profile.source[0].address
    return [Finding(f"{row.address - base:#06x}", row.excess,
                    f"{row.source}: {row.wavefronts} wavefronts, {row.ideal} without conflicts, {row.excess} in excess")
            for row in profile.source if row.excess > 0]


PROFILE_RULE = ProfileRule(
    name="shared_wavefronts_excess",
    statement="shared-memory wavefronts in excess of the conflict-free count, by instruction (executed, from a profile)",
    check=check,
    refusal="shared-memory accesses take more wavefronts than their conflict-free count: bank conflicts or misaligned "
            "wide accesses, each extra wavefront a further pass of the memory pipe",
    fix="change the shared layout so the lanes of a warp touch distinct banks (padding, a swizzle, a wider access "
        "per lane), then profile again",
    fixtures=(Fixture("conflicted", 98304), Fixture("conflict_free", 0)),
    needs="source",
)
