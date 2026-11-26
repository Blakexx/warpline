"""profiler_verdicts: the profiler's own rule verdicts, passed through as findings.

Nsight Compute's built-in rules turn a section's metrics into a verdict: `OPT` (an optimization it sees), `WRN` (a
warning) or `INF` (information). Each `OPT` and `WRN` result is one finding, `where` the profiler's rule name, the text
its description with the estimated speedup it states. `INF` results are information, not findings. A declaration of
this rule allows a number of verdicts; it does not say which."""
from __future__ import annotations

from ..ncu import Profile
from ...manifest import Finding
from ..rule import Fixture, ProfileRule


def check(profile: Profile) -> list[Finding]:
    return [Finding(r.name, 1,
                    f"{r.type}" + (f", estimated speedup {r.speedup:g}% ({r.speedup_type})" if r.speedup is not None else "")
                    + f": {r.description}")
            for r in profile.rules if r.type in ("OPT", "WRN")]


PROFILE_RULE = ProfileRule(
    name="profiler_verdicts",
    statement="each optimization or warning the profiler's built-in rules report, named by the profiler's rule",
    check=check,
    refusal="the profiler's built-in rules report optimizations or warnings on the kernel",
    fix="read the named rule's description, change the kernel, and profile again; or declare the verdicts accepted",
    fixtures=(Fixture("fired", 3), Fixture("informational", 0)),
    needs="details",
)
