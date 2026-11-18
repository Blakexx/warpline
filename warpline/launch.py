"""The launch in time, from a record's ties.

The kernel stamps two forced, coarse marks, one at its ENTRY (`launch.begin`) and one at its EXIT (`launch.end`), each
with a tie, a pair a lane a launch. From the ties of a scope this reads:

  BEGIN   the earliest lane's `launch.begin`, in nanoseconds
  END     the latest lane's `launch.end`
  LENGTH  T = END - BEGIN, in nanoseconds; and T_CYCLES, the same two moments on the scope's cycle counter (the
          earliest lane's `launch.begin` stamp, the latest lane's `launch.end` stamp: the lanes of a scope share one
          counter). The samples are joined to the stamps on CYCLES (`placement`): the profiler's time sampling stretches
          the kernel's time and not its cycles.
  ANCHORS the steady window starts, in cycles, on the anchor lane (after the skipped windows; the last one opens a
          partial window), as fractions of the launch (`fractions`): where the record's windows fall in the launch
  SETUP   BEGIN to the first steady window anchor's tie, in nanoseconds
  PERIOD  P, the window's period in nanoseconds, from the anchor lane's ties (`rates.window_in_time`)
  WINDOWS the steady windows counted

A record without both edges carrying ties is REFUSED (`LaunchError`), naming what the kernel must stamp. A record that
has the two edges and no window anchors (the plain kernel with its two edges) is a launch WITHOUT windows
(`launch_of(..., need_windows=False)`): its windows are another record's, as fractions of that launch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import check, rates, window
from .record import LAUNCH_BEGIN, LAUNCH_END, Record, scope_of

NEEDS = ("the kernel must stamp, at its ENTRY and at its EXIT, on both clocks (a stamp with a tie, one of each a lane, "
         "outside every loop): `launch.begin` and `launch.end`; the importer makes them from the sites whose legend "
         "names are `entry` and `exit` (`--launch-begin`, `--launch-end`)")


class LaunchError(Exception):
    pass


@dataclass
class Launch:
    scope: str
    begin_ns: int
    end_ns: int
    begin_lane: str
    end_lane: str
    lanes: int                                   # lanes of the scope carrying both edges, tied
    begin_cycles: int = 0                        # the earliest `launch.begin` stamp, on the scope's cycle counter
    end_cycles: int = 0                          # the latest `launch.end` stamp
    setup_ns: float | None = None                # None: the record has no window anchors
    period_ns: float | None = None
    windows: int = 0
    anchor: str = ""
    anchor_lane: str = ""
    anchors: list[int] = field(default_factory=list)    # steady window starts, cycles, on the anchor lane
    findings: list[str] = field(default_factory=list)   # `check.launch_edges` findings of the scope

    @property
    def cycles(self) -> int:
        """T in cycles: the launch's length on the scope's cycle counter."""
        return self.end_cycles - self.begin_cycles

    @property
    def has_windows(self) -> bool:
        return bool(self.anchors)

    @property
    def fractions(self) -> list[float]:
        """The steady window starts as fractions of the launch (of T in cycles)."""
        return [(a - self.begin_cycles) / self.cycles for a in self.anchors]

    @property
    def period_cycles(self) -> float:
        """The mean steady window in cycles (anchor to anchor)."""
        return (self.anchors[-1] - self.anchors[0]) / (len(self.anchors) - 1)

    @property
    def length_ns(self) -> int:
        return self.end_ns - self.begin_ns

    @property
    def length_us(self) -> float:
        return self.length_ns / rates.US

    @property
    def setup_us(self) -> float | None:
        return None if self.setup_ns is None else self.setup_ns / rates.US

    @property
    def period_us(self) -> float | None:
        return None if self.period_ns is None else self.period_ns / rates.US

    def describe(self) -> str:
        s = (f"launch of scope {self.scope}: begin {self.begin_ns} ns (lane {self.begin_lane}), end {self.end_ns} ns "
             f"(lane {self.end_lane}), length T = {self.length_us:.3f} us = {self.cycles:,} cycles "
             f"(cycle {self.begin_cycles:,} to {self.end_cycles:,}, {self.cycles / self.length_us:,.0f} cycles an us)")
        if self.has_windows:
            s += (f"; {self.windows} steady windows (anchor `{self.anchor}` on {self.anchor_lane}), mean "
                  f"{self.period_cycles:,.1f} cycles = {self.period_us:.4f} us, the first at {self.fractions[0]:.4f} of "
                  f"the launch ({self.setup_us:.3f} us after its begin) and the last start at {self.fractions[-1]:.4f}")
        else:
            s += "; no window anchors in this record (its windows are another record's)"
        s += f"; {self.lanes} lane(s) carry both edges"
        return s + "".join(f"\n  FINDING {f}" for f in self.findings)

    def to_dict(self) -> dict:
        return {"scope": self.scope, "begin_ns": self.begin_ns, "end_ns": self.end_ns, "length_ns": self.length_ns,
                "begin_cycles": self.begin_cycles, "end_cycles": self.end_cycles, "length_cycles": self.cycles,
                "setup_ns": self.setup_ns, "period_ns": self.period_ns, "windows": self.windows,
                "anchor": self.anchor, "anchor_lane": self.anchor_lane, "lanes": self.lanes,
                "window_fractions": self.fractions if self.has_windows else None, "findings": self.findings}


def _edge_ns(rec: Record, lane: str, name: str) -> int | None:
    """The nanosecond reading of the lane's stamp `name` (its tie), None when it is not stamped or not tied."""
    at = dict(rec.ties.get(lane, []))
    for _i, nm, t, _p in rec.lane_stamps(lane):
        if nm == name:
            return at.get(t)
    return None


def _edge_cycles(rec: Record, lane: str, name: str) -> int | None:
    for _i, nm, t, _p in rec.lane_stamps(lane):
        if nm == name:
            return t
    return None


def launch_of(rec: Record, scope: str | None = None, anchor: str | None = None, skip: int = window.SKIP,
              need_windows: bool = True) -> Launch:
    sc = scope or rec.scopes()[0]
    if sc not in rec.scopes():
        raise LaunchError(f"no scope {sc!r}; the record has {rec.scopes()}")
    lanes = [ln for ln in rec.lanes if scope_of(ln) == sc]
    if not any(LAUNCH_BEGIN in {nm for _i, nm, _t, _p in rec.lane_stamps(ln)} for ln in lanes):
        raise LaunchError(f"REFUSED: scope {sc}: the record carries no `{LAUNCH_BEGIN}` and `{LAUNCH_END}`; {NEEDS}")
    findings = [str(f) for f in check.launch_edges(rec) if f.lane is not None and scope_of(f.lane) == sc]
    begins = {ln: _edge_ns(rec, ln, LAUNCH_BEGIN) for ln in lanes}
    ends = {ln: _edge_ns(rec, ln, LAUNCH_END) for ln in lanes}
    bs = {ln: v for ln, v in begins.items() if v is not None}
    es = {ln: v for ln, v in ends.items() if v is not None}
    if not bs or not es:
        raise LaunchError(f"REFUSED: scope {sc}: the record has no tied "
                          f"{'`' + LAUNCH_BEGIN + '`' if not bs else '`' + LAUNCH_END + '`'}"
                          f"{'' if bs or es else ' and no tied `' + LAUNCH_END + '`'}; {NEEDS}"
                          + "".join(f"; {f}" for f in findings[:3]))
    bl = min(bs, key=lambda ln: bs[ln])
    el = max(es, key=lambda ln: es[ln])
    if es[el] <= bs[bl]:
        raise LaunchError(f"REFUSED: scope {sc}: the launch ends ({es[el]} ns) at or before it begins ({bs[bl]} ns)")
    cb = {ln: _edge_cycles(rec, ln, LAUNCH_BEGIN) for ln in lanes}
    ce = {ln: _edge_cycles(rec, ln, LAUNCH_END) for ln in lanes}
    cb = [v for v in cb.values() if v is not None]
    ce = [v for v in ce.values() if v is not None]
    if not cb or not ce or max(ce) <= min(cb):
        raise LaunchError(f"REFUSED: scope {sc}: the launch's two edges are not on the cycle counter in order: {NEEDS}")
    both = [ln for ln in bs if ln in es]
    lc = Launch(sc, bs[bl], es[el], bl, el, len(both), min(cb), max(ce), findings=findings)
    try:
        name, _how, lane, starts = window.anchor_starts(rec, sc, anchor, skip)
    except window.WindowError as e:
        if need_windows:
            raise LaunchError(f"scope {sc}: the setup and the window period need the window: {e}")
        return lc
    tie = dict(rec.ties.get(lane, []))
    w = rates.window_in_time(rec, sc, name, skip)
    if starts[0] not in tie or w is None:
        if need_windows:
            raise LaunchError(f"scope {sc}: the `{name}` stamps that start the windows carry no ties: the setup and "
                              f"the window's period cannot be read in time")
        return lc
    lc.setup_ns, lc.period_ns, lc.windows = tie[starts[0]] - bs[bl], w["period_ns"], w["windows"]
    lc.anchor, lc.anchor_lane, lc.anchors = name, lane, list(starts)
    return lc
