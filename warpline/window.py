"""The average window of a record.

A window is the span between consecutive `begin` stamps of the ANCHOR region on a scope's first lane carrying it.
Over the steady windows (past the first `skip`, the last partial one dropped), for every lane and every stamp name,
the stamp's OFFSET from its window's start: its mean, spread (min, max, standard deviation) and how many windows
held it.

`average_window` reports every lane as it falls in the SCOPE's anchor lane's window `[start, next start)` (a stamp that
falls in no window of the span is not counted). A role whose own window is shifted against that lane's would be smeared
by it, so the per-role view (`role_windows`) takes each ROLE's window from the role's own lanes' anchor event where the
role has one (else the scope's, said): a table of the role's PHASES -- where each starts in the role's own window, how
long it lasts (begin to end), the spread across lanes and across windows, in cycles -- and the role's own period.

A name a lane stamps more than once in a window is told apart by its OCCURRENCE (0, 1, ...): the order it is
written in within the window.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from . import record as rec_mod
from .record import Record, classify, scope_of

SKIP = 2          # windows dropped at the start
MIN_BEGINS = 5    # fewer begins than this on the carrying lane: not a recurring region


class WindowError(Exception):
    """The record has no usable window (no recurring anchor, too few begins)."""


@dataclass(frozen=True)
class Row:
    """One stamp of one lane in the average window."""
    lane: str
    name: str
    occ: int
    mean: float
    std: float
    min: int
    max: int
    n: int  # windows that held it


@dataclass
class ScopeWindow:
    scope: str
    anchor: str            # the anchor stamp name, e.g. `phase.window.begin`
    anchor_lane: str
    chosen: str            # how: "requested" | "most regular begin (cv=...)"
    period: float
    period_std: float
    period_min: int
    period_max: int
    windows: int           # steady windows used
    skip: int
    rows: list[Row] = field(default_factory=list)


@dataclass
class Average:
    source: Record
    scopes: dict[str, ScopeWindow]


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _lanes_of(rec: Record, scope: str) -> list[str]:
    return [ln for ln in rec.lanes if scope_of(ln) == scope]


def _begins(rec: Record, scope: str, anchor: str) -> tuple[str, list[int]] | None:
    """The first lane of the scope carrying the stamp `anchor`, and its times."""
    for ln in _lanes_of(rec, scope):
        ts = [t for _i, name, t, _p in rec.lane_stamps(ln) if name == anchor]
        if ts:
            return ln, ts
    return None


def _cv(ts: list[int]) -> float:
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    m = _mean(gaps)
    return _std(gaps) / m if m > 0 else math.inf


def candidates(rec: Record, scope: str) -> dict[str, tuple[str, list[int]]]:
    """Region begins that recur on the first lane carrying them: {stamp name: (lane, times)}."""
    out = {}
    for nid, name in rec.names.items():
        if classify(name)[0] != "region_begin":
            continue
        found = _begins(rec, scope, name)
        if found and len(found[1]) >= MIN_BEGINS:
            out[name] = found
    return out


def resolve_anchor(rec: Record, wanted: str) -> str:
    """`phase.window` and `phase.window.begin` both name the stamp `phase.window.begin`; any other stamp by its name."""
    names = set(rec.names.values())
    if wanted in names:
        return wanted
    if wanted + ".begin" in names:
        return wanted + ".begin"
    raise WindowError(f"anchor {wanted!r} is not a stamp name of the record")


def choose_anchor(rec: Record, scope: str, allowed: set[str] | None = None) -> tuple[str, str]:
    """The region whose begin recurs most regularly (lowest coefficient of variation of its gaps) on the first lane
    carrying it; ties go to the earlier name id. Returns (stamp name, how chosen)."""
    cands = candidates(rec, scope)
    if allowed is not None:
        cands = {k: v for k, v in cands.items() if k in allowed}
    if not cands:
        raise WindowError(f"scope {scope!r}: no region begin recurs {MIN_BEGINS}+ times on its first carrying lane")
    ids = {name: nid for nid, name in rec.names.items()}
    best = min(cands, key=lambda n: (_cv(cands[n][1]), ids[n]))
    others = ", ".join(f"{n} {_cv(cands[n][1]):.4f}" for n in sorted(cands, key=lambda n: (_cv(cands[n][1]), ids[n])))
    return best, f"most regular begin, cv={_cv(cands[best][1]):.4f} of {len(cands)} candidates ({others})"


def anchor_starts(rec: Record, scope: str, anchor: str | None, skip: int) -> tuple[str, str, str, list[int]]:
    """(anchor stamp name, how chosen, the anchor lane, the steady window starts) of a scope: the begins of the anchor
    after the first `skip`; the last start opens a partial window and ends none."""
    if anchor is None:
        name, how = choose_anchor(rec, scope)
    else:
        name, how = resolve_anchor(rec, anchor), "requested"
    found = _begins(rec, scope, name)
    if found is None:
        raise WindowError(f"scope {scope!r}: no lane carries {name!r}")
    lane, begins = found
    if len(begins) - 1 - skip < 1:
        raise WindowError(f"scope {scope!r}: {len(begins)} `{name}` stamps leave no steady window after "
                          f"skipping {skip} and the last partial one")
    return name, how, lane, begins[skip:]


def average_window(rec: Record, anchor: str | dict[str, str] | None = None, skip: int = SKIP,
                   scopes: list[str] | None = None, clock_of=None) -> Average:
    """The average window per scope. `anchor`: a name for every scope, a {scope: name}, or None (chosen).

    `clock_of(lane, t)` (optional) maps a stamp's cycle count to a TIME on its own lane, or None where it cannot be
    placed (see `rates.placer`): offsets and periods are then in that time unit (the stamps' membership of a window is
    still by cycles, the counter of the scope), and a stamp that cannot be placed is not counted. Without it: cycles."""
    out: dict[str, ScopeWindow] = {}
    for scope in scopes if scopes is not None else rec.scopes():
        want = anchor.get(scope) if isinstance(anchor, dict) else anchor
        name, how, lane, starts = anchor_starts(rec, scope, want, skip)
        at = (lambda ln, t: t) if clock_of is None else clock_of
        origin = [at(lane, b) for b in starts]   # window k = [starts[k], starts[k+1]); the last opens a partial one
        if any(o is None for o in origin):
            raise WindowError(f"scope {scope!r}: a `{name}` stamp that starts a window carries no tie on {lane}")
        spans = [b - a for a, b in zip(origin, origin[1:])]
        sw = ScopeWindow(scope, name, lane, how, _mean(spans), _std(spans), min(spans), max(spans), len(spans), skip)
        edges = starts
        acc: dict[tuple[str, str, int], list[float]] = {}
        order: list[tuple[str, str, int]] = []
        for ln in _lanes_of(rec, scope):
            stamps = sorted(((t, i, nm) for i, nm, t, _p in rec.lane_stamps(ln)), key=lambda x: (x[0], x[1]))
            times = [s[0] for s in stamps]
            for k in range(len(spans)):
                lo = bisect.bisect_left(times, edges[k])
                hi = bisect.bisect_left(times, edges[k + 1])
                seen: dict[str, int] = {}
                for t, _i, nm in stamps[lo:hi]:
                    o = seen.get(nm, 0)
                    seen[nm] = o + 1
                    x = at(ln, t)
                    if x is None:
                        continue
                    key = (ln, nm, o)
                    if key not in acc:
                        acc[key] = []
                        order.append(key)
                    acc[key].append(x - origin[k])
        for key in order:
            xs = acc[key]
            sw.rows.append(Row(key[0], key[1], key[2], _mean(xs), _std(xs), min(xs), max(xs), len(xs)))
        sw.rows.sort(key=lambda r: (list(rec.lanes).index(r.lane), r.mean, r.occ))
        out[scope] = sw
    return Average(rec, out)


# ---------------------------------------------------------------------------------------------------- by role

@dataclass(frozen=True)
class RoleRow:
    role: str
    name: str
    occ: int
    multi: bool        # the name recurs within a window on some lane of the role: show its occurrence
    mean: float        # mean over the role's lanes of each lane's mean offset
    lane_min: float    # across lanes: the earliest and latest lane mean
    lane_max: float
    jitter: float      # mean over lanes of each lane's standard deviation across windows
    lanes: int         # lanes of the role carrying it
    n: int             # fewest windows that held it on a carrying lane
    straddles: bool = False  # on some lane its offsets span over half the period: it sits on the window's edge


def by_role(rec: Record, sw: ScopeWindow) -> list[tuple[str, int, list[RoleRow]]]:
    """[(role, lanes in the role, rows by mean offset)] in the order roles first appear."""
    roles: dict[str, list[str]] = {}
    for ln in _lanes_of(rec, sw.scope):
        roles.setdefault(rec.lanes[ln].role, []).append(ln)
    out = []
    for role, lanes in roles.items():
        per: dict[tuple[str, int], list[Row]] = {}
        for r in sw.rows:
            if r.lane in lanes:
                per.setdefault((r.name, r.occ), []).append(r)
        multi = {nm for (nm, o) in per if o > 0}
        rows = []
        for (nm, o), rs in per.items():
            ms = [r.mean for r in rs]
            rows.append(RoleRow(role, nm, o, nm in multi, _mean(ms), min(ms), max(ms), _mean([r.std for r in rs]),
                                len(rs), min(r.n for r in rs),
                                any(r.max - r.min > sw.period / 2 for r in rs)))
        rows.sort(key=lambda r: (r.mean, r.name, r.occ))
        out.append((role, len(lanes), rows))
    return out


def label(r: RoleRow) -> str:
    return f"{r.name}#{r.occ}" if r.multi else r.name


def format_table(av: Average) -> str:
    """Per scope: the anchor and the window's period, then per ROLE the phases in the role's own window."""
    rec = av.source
    lines = []
    for sw in av.scopes.values():
        lines.append(f"scope {sw.scope}: anchor `{sw.anchor}` on lane {sw.anchor_lane} ({sw.chosen})")
        lines.append(f"  period {sw.period:,.1f} cycles (std {sw.period_std:,.1f}, min {sw.period_min:,}, "
                     f"max {sw.period_max:,}); {sw.windows} steady windows (first {sw.skip} skipped, last partial "
                     f"one dropped)")
        lines.append("  a role's window is its OWN: taken from the anchor event of the role's own lanes where it has "
                     "one, else the scope's; offsets and durations below are in cycles within that window")
        lines.append("")
        lines.append(format_roles(rec, role_windows(rec, sw.scope, sw.anchor, sw.skip)).rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------------- the record

def window_facts(av: Average, rows: list[tuple[Row, str]] | None = None) -> dict:
    """`facts.window`: per scope the period, anchor and windows used; `stamps[i]` the spread of record stamp i."""
    scopes = {}
    for sc, sw in av.scopes.items():
        scopes[sc] = {"period": sw.period, "period_std": sw.period_std, "period_min": sw.period_min,
                      "period_max": sw.period_max, "anchor": sw.anchor, "anchor_lane": sw.anchor_lane,
                      "chosen": sw.chosen, "windows": sw.windows, "skip": sw.skip}
    return {"scopes": scopes, "stamps": []}


def spread_entry(r: Row) -> dict:
    return {"lane": r.lane, "name": r.name, "occ": r.occ, "mean": r.mean, "std": r.std, "min": r.min, "max": r.max,
            "n": r.n}


def to_record(av: Average) -> Record:
    """An "average window" record: the same format; `stamps` at mean offsets (rounded to cycles) from the window's
    start, per lane in time order; `facts.window` holds the period, anchor, windows used and each stamp's spread
    (`facts.window.stamps[i]` belongs to `stamps[i]`). The source's declared facts are kept."""
    src = av.source
    names: dict[str, int] = {}
    stamps = []
    spread = []
    for sw in av.scopes.values():
        for r in sw.rows:
            nid = names.setdefault(r.name, len(names))
            stamps.append((list(src.lanes).index(r.lane), [r.lane, nid, int(round(r.mean))], r))
    stamps.sort(key=lambda x: (x[0], x[1][2]))  # per lane in lane order, then by offset
    lanes_used = []
    for _o, row, r in stamps:
        spread.append(spread_entry(r))
        if row[0] not in lanes_used:
            lanes_used.append(row[0])
    facts = {k: v for k, v in src.facts.items() if k not in ("window", "runs", "merge")}
    facts.setdefault("barriers", {})
    facts.setdefault("operations", {})
    wf = window_facts(av)
    wf["stamps"] = spread
    facts["window"] = wf
    doc = {
        "format": rec_mod.FORMAT,
        "clock": src.clock,
        "names": {str(i): n for n, i in names.items()},
        "lanes": [{k: v for k, v in (("id", ln), ("role", src.lanes[ln].role), ("group", src.lanes[ln].group))
                   if v is not None} for ln in lanes_used],
        "stamps": [row for _o, row, _r in stamps],
        "dropped": {},
        "facts": facts,
        "twin": src.twin,
        "source": {"average_window_of": (src.source or {}).get("input", "a record"), "windows_used": {
            sc: sw.windows for sc, sw in av.scopes.items()}},
    }
    return rec_mod.from_dict(doc)


# ---------------------------------------------------------------------------------------------------- a role's own window

SUM_TOL = 0.005   # a role's phases sum to its period within this fraction, else it is said


@dataclass(frozen=True)
class PhaseRow:
    """One phase (a region) of a role, in the role's own window."""
    role: str
    name: str             # the region's key, e.g. `phase.issuing_steps`
    occ: int              # the order it begins in within the window
    multi: bool
    start: float          # cycles from the role's window start, mean over the role's lanes of each lane's mean
    start_min: float      # across lanes: the earliest and the latest lane mean
    start_max: float
    start_jit: float      # across windows: mean over lanes of each lane's standard deviation
    dur: float            # begin to end, same statistics
    dur_min: float
    dur_max: float
    dur_jit: float
    lanes: int
    n: int                # fewest windows that held it on a carrying lane
    share: float          # mean over its lanes of (windows that held it / steady windows)


@dataclass
class RoleWindow:
    role: str
    lanes: list[str]
    anchor: str
    anchor_lane: str
    own: bool                      # the role's own lanes carry the anchor event (else the scope's anchor lane is used)
    period: float                  # cycles
    period_std: float
    period_min: int
    period_max: int
    windows: int
    offset: float                  # the role's window start against the scope's, mean (cycles; 0 when not own)
    rows: list[PhaseRow] = field(default_factory=list)
    lane_phases: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)   # lane -> (name, start, dur)
    total: float = 0.0             # the phases' durations summed over a window (a lane's, averaged over the lanes)
    total_min: float = 0.0
    total_max: float = 0.0
    own_lanes: int = 0                                  # lanes of the role that carry the anchor event themselves
    lane_offset: dict[str, float] = field(default_factory=dict)   # lane -> its window start against the scope's, cycles

    @property
    def gap(self) -> float:
        return self.total - self.period

    @property
    def sums(self) -> bool:
        return abs(self.gap) <= SUM_TOL * self.period

    def sum_note(self) -> str:
        what = f"phase durations sum to {self.total:,.1f} cycles (lanes {self.total_min:,.1f}..{self.total_max:,.1f})"
        if self.sums:
            return f"{what}, the role's period {self.period:,.1f}: they sum to it (within {100 * SUM_TOL:g}%)"
        return (f"{what}, the role's period {self.period:,.1f}: they do NOT sum to it ({self.gap:+,.1f} cycles = "
                f"{100 * self.gap / self.period:+.2f}%; bound {100 * SUM_TOL:g}%): the lanes' phases do not tile "
                f"the window (an idle stretch between phases, or a phase that is not stamped)")


def _regions_of(rec: Record, lane: str) -> list[tuple[str, int, int]]:
    """[(key, begin time, duration)] of a lane's regions: the n-th begin of a key with the n-th end of that key."""
    begins: dict[str, list[int]] = {}
    ends: dict[str, list[int]] = {}
    for _i, nm, t, _p in rec.lane_stamps(lane):
        kind, key = classify(nm)
        if kind == "region_begin":
            begins.setdefault(key, []).append(t)
        elif kind == "region_end":
            ends.setdefault(key, []).append(t)
    out = []
    for key, bs in begins.items():
        es = ends.get(key, [])
        for k, b in enumerate(bs):
            if k < len(es) and es[k] >= b:
                out.append((key, b, es[k] - b))
    out.sort(key=lambda x: (x[1], x[0]))
    return out


def _circular_offset(xs: list[float], period: float) -> float:
    """The mean of offsets taken modulo the period, in (-period/2, period/2]: a window's start against another's."""
    c = sum(math.cos(2 * math.pi * x / period) for x in xs)
    s = sum(math.sin(2 * math.pi * x / period) for x in xs)
    return math.atan2(s, c) / (2 * math.pi) * period


def _own_begins(rec: Record, lane: str, name: str, skip: int) -> list[int] | None:
    ts = [t for _i, nm, t, _p in rec.lane_stamps(lane) if nm == name]
    return ts[skip:] if len(ts) - 1 - skip >= 1 else None


def role_windows(rec: Record, scope: str | None = None, anchor: str | None = None, skip: int = SKIP) -> list[RoleWindow]:
    """Per role of the scope, the role's phases in the role's OWN window. A lane that carries the anchor event takes
    its window from its own begins of it (the span between consecutive begins, past the first `skip`); a lane of a role
    that has such a lane but not itself uses the role's first carrying lane's; a role none of whose lanes carries it
    uses the scope's anchor lane (said). A phase belongs to the window its begin falls in; its duration is begin to
    end; its start is from its lane's window start."""
    sc = scope or rec.scopes()[0]
    name, how, sc_lane, sc_starts = anchor_starts(rec, sc, anchor, skip)
    sc_period = _mean([b - a for a, b in zip(sc_starts, sc_starts[1:])])
    roles: dict[str, list[str]] = {}
    for ln in _lanes_of(rec, sc):
        roles.setdefault(rec.lanes[ln].role, []).append(ln)
    out = []
    for role, lanes in roles.items():
        own = {ln: _own_begins(rec, ln, name, skip) for ln in lanes}
        first_own = next((ln for ln in lanes if own[ln] is not None), None)
        begins: dict[str, list[int]] = {}
        offset: dict[str, float] = {}
        for ln in lanes:
            src = own[ln] if own[ln] is not None else (own[first_own] if first_own else sc_starts)
            begins[ln] = src
            if first_own is not None:
                offs = []
                for t in src[:-1]:
                    k = min(range(len(sc_starts)), key=lambda i: abs(sc_starts[i] - t))
                    offs.append(((t - sc_starts[k] + sc_period / 2) % sc_period) - sc_period / 2)
                offset[ln] = _circular_offset(offs, sc_period)
            else:
                offset[ln] = 0.0
        n_own = sum(1 for ln in lanes if own[ln] is not None)
        spans = {ln: [b - a for a, b in zip(begins[ln], begins[ln][1:])] for ln in lanes}
        W = min(len(v) for v in spans.values())
        allspans = [x for v in spans.values() for x in v]
        rw = RoleWindow(role, lanes, name, sc_lane if first_own is None else first_own, first_own is not None,
                        _mean([_mean(v) for v in spans.values()]), _mean([_std(v) for v in spans.values()]),
                        min(allspans), max(allspans), W,
                        _circular_offset(list(offset.values()), sc_period) if first_own is not None else 0.0)
        rw.own_lanes = n_own
        rw.lane_offset = offset
        per: dict[str, dict[tuple[str, int], tuple[list[float], list[float]]]] = {}
        totals = []
        for ln in lanes:
            regs = _regions_of(rec, ln)
            times = [r[1] for r in regs]
            acc: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
            tot = 0.0
            bg = begins[ln]
            for k in range(len(spans[ln])):
                lo = bisect.bisect_left(times, bg[k])
                hi = bisect.bisect_left(times, bg[k + 1])
                seen: dict[str, int] = {}
                for key, b, d in regs[lo:hi]:
                    o = seen.get(key, 0)
                    seen[key] = o + 1
                    st, du = acc.setdefault((key, o), ([], []))
                    st.append(b - bg[k])
                    du.append(d)
                    tot += d
            per[ln] = acc
            totals.append(tot / len(spans[ln]))
        rw.total, rw.total_min, rw.total_max = _mean(totals), min(totals), max(totals)
        keys: dict[tuple[str, int], None] = {}
        for ln in lanes:
            for k in per[ln]:
                keys[k] = None
        multi = {k for k, o in keys if o > 0}
        rows = []
        for (key, o) in keys:
            carry = [ln for ln in lanes if (key, o) in per[ln]]
            sm = [_mean(per[ln][(key, o)][0]) for ln in carry]
            dm = [_mean(per[ln][(key, o)][1]) for ln in carry]
            rows.append(PhaseRow(
                role, key, o, key in multi, _mean(sm), min(sm), max(sm),
                _mean([_std(per[ln][(key, o)][0]) for ln in carry]), _mean(dm), min(dm), max(dm),
                _mean([_std(per[ln][(key, o)][1]) for ln in carry]), len(carry),
                min(len(per[ln][(key, o)][0]) for ln in carry),
                _mean([len(per[ln][(key, o)][0]) / len(spans[ln]) for ln in carry])))
            for ln in carry:
                rw.lane_phases.setdefault(ln, []).append((key, _mean(per[ln][(key, o)][0]), _mean(per[ln][(key, o)][1])))
        rows.sort(key=lambda r: (r.start, r.name, r.occ))
        rw.rows = rows
        out.append(rw)
    return out


def format_roles(rec: Record, rws: list[RoleWindow]) -> str:
    lines = []
    for rw in rws:
        where = (f"each lane's own `{rw.anchor}` ({rw.own_lanes} of {len(rw.lanes)} lanes carry it; the others use lane "
                 f"{rw.anchor_lane}'s)" if rw.own else
                 f"the scope's anchor `{rw.anchor}` on lane {rw.anchor_lane} (the role's lanes carry none)")
        lines.append(f"role {rw.role} ({len(rw.lanes)} lanes): window from {where}")
        lines.append(f"  period {rw.period:,.1f} cycles (std {rw.period_std:,.1f}, min {rw.period_min:,}, max "
                     f"{rw.period_max:,}); {rw.windows} steady windows"
                     + (f"; its window starts {rw.offset:+,.1f} cycles from the scope's (positive: later)" if rw.own else ""))
        lines.append(f"  {'start':>10}  {'lanes min..max':>19}  {'windows':>7}  {'duration':>10}  {'lanes min..max':>19}  "
                     f"{'windows':>7}  {'lanes':>5}  {'held':>5}  phase")
        lines.append(f"  {'(cycles)':>10}  {'':>19}  {'jitter':>7}  {'(cycles)':>10}  {'':>19}  {'jitter':>7}")
        for r in rw.rows:
            lab = f"{r.name}#{r.occ}" if r.multi else r.name
            lines.append(f"  {r.start:10.1f}  {r.start_min:9.1f}..{r.start_max:<9.1f}  {r.start_jit:7.1f}  {r.dur:10.1f}  "
                         f"{r.dur_min:9.1f}..{r.dur_max:<9.1f}  {r.dur_jit:7.1f}  {r.lanes:5d}  {100 * r.share:4.0f}%  {lab}")
        lines.append("  " + rw.sum_note())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
