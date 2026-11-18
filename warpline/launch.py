"""The launch in time, from a record's ties.

The kernel stamps two forced, coarse marks, one at its ENTRY (`launch.begin`) and one at its EXIT (`launch.end`), each
with a tie, a pair a lane a launch. From the ties of a scope this reads:

  BEGIN   the earliest lane's `launch.begin`, in nanoseconds
  END     the latest lane's `launch.end`
  LENGTH  T = END - BEGIN
  SETUP   BEGIN to the first steady window anchor's tie (the first window start after the skipped windows)
  PERIOD  P, the window's period in nanoseconds, from the anchor lane's ties (`rates.window_in_time`)
  WINDOWS the steady windows counted

A record without both edges carrying ties is REFUSED (`LaunchError`), naming what the kernel must stamp.
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
    setup_ns: float
    period_ns: float
    windows: int
    anchor: str
    anchor_lane: str
    begin_lane: str
    end_lane: str
    lanes: int                                   # lanes of the scope carrying both edges, tied
    findings: list[str] = field(default_factory=list)   # `check.launch_edges` findings of the scope

    @property
    def length_ns(self) -> int:
        return self.end_ns - self.begin_ns

    @property
    def length_us(self) -> float:
        return self.length_ns / rates.US

    @property
    def setup_us(self) -> float:
        return self.setup_ns / rates.US

    @property
    def period_us(self) -> float:
        return self.period_ns / rates.US

    def describe(self) -> str:
        s = (f"launch of scope {self.scope}: begin {self.begin_ns} ns (lane {self.begin_lane}), end {self.end_ns} ns "
             f"(lane {self.end_lane}), length T = {self.length_us:.3f} us; setup {self.setup_us:.3f} us to the first "
             f"steady window (anchor `{self.anchor}`); window period {self.period_us:.4f} us over {self.windows} "
             f"steady windows; {self.lanes} lane(s) carry both edges")
        return s + "".join(f"\n  FINDING {f}" for f in self.findings)

    def to_dict(self) -> dict:
        return {"scope": self.scope, "begin_ns": self.begin_ns, "end_ns": self.end_ns, "length_ns": self.length_ns,
                "setup_ns": self.setup_ns, "period_ns": self.period_ns, "windows": self.windows,
                "anchor": self.anchor, "anchor_lane": self.anchor_lane, "lanes": self.lanes,
                "findings": self.findings}


def _edge_ns(rec: Record, lane: str, name: str) -> int | None:
    """The nanosecond reading of the lane's stamp `name` (its tie), None when it is not stamped or not tied."""
    at = dict(rec.ties.get(lane, []))
    for _i, nm, t, _p in rec.lane_stamps(lane):
        if nm == name:
            return at.get(t)
    return None


def launch_of(rec: Record, scope: str | None = None, anchor: str | None = None, skip: int = window.SKIP) -> Launch:
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
    try:
        name, _how, lane, starts = window.anchor_starts(rec, sc, anchor, skip)
    except window.WindowError as e:
        raise LaunchError(f"scope {sc}: the setup and the window period need the window: {e}")
    tie = dict(rec.ties.get(lane, []))
    if starts[0] not in tie:
        raise LaunchError(f"scope {sc}: the first steady `{name}` on {lane} carries no tie: the setup (launch begin to "
                          f"the window's start) cannot be read in time")
    w = rates.window_in_time(rec, sc, name, skip)
    if w is None:
        raise LaunchError(f"scope {sc}: the `{name}` stamps that start the windows carry no ties: the window's period "
                          f"cannot be read in time")
    both = [ln for ln in bs if ln in es]
    return Launch(sc, bs[bl], es[el], tie[starts[0]] - bs[bl], w["period_ns"], w["windows"], name, lane, bl, el,
                  len(both), findings)
