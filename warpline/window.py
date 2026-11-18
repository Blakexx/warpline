"""The average window of a record.

A window is the span between consecutive `begin` stamps of the ANCHOR region on a scope's first lane carrying it.
Over the steady windows (past the first `skip`, the last partial one dropped), for every lane and every stamp name,
the stamp's OFFSET from its window's start: its mean, spread (min, max, standard deviation) and how many windows
held it.

A lane's own window may be shifted against the anchor lane's. Offsets are taken modulo nothing: a lane's stamps are
reported as they fall in the anchor lane's window `[start, next start)`, and a stamp that falls in no window of the
span is not counted.

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


def average_window(rec: Record, anchor: str | dict[str, str] | None = None, skip: int = SKIP,
                   scopes: list[str] | None = None) -> Average:
    """The average window per scope. `anchor`: a name for every scope, a {scope: name}, or None (chosen)."""
    out: dict[str, ScopeWindow] = {}
    for scope in scopes if scopes is not None else rec.scopes():
        want = anchor.get(scope) if isinstance(anchor, dict) else anchor
        if want is None:
            name, how = choose_anchor(rec, scope)
        else:
            name, how = resolve_anchor(rec, want), "requested"
        found = _begins(rec, scope, name)
        if found is None:
            raise WindowError(f"scope {scope!r}: no lane carries {name!r}")
        lane, begins = found
        if len(begins) - 1 - skip < 1:
            raise WindowError(f"scope {scope!r}: {len(begins)} `{name}` stamps leave no steady window after "
                              f"skipping {skip} and the last partial one")
        starts = begins[skip:]           # window k = [starts[k], starts[k+1]); the last start opens a partial window
        spans = [b - a for a, b in zip(starts, starts[1:])]
        sw = ScopeWindow(scope, name, lane, how, _mean(spans), _std(spans), min(spans), max(spans), len(spans), skip)
        edges = starts
        acc: dict[tuple[str, str, int], list[int]] = {}
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
                    key = (ln, nm, o)
                    if key not in acc:
                        acc[key] = []
                        order.append(key)
                    acc[key].append(t - edges[k])
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
    rec = av.source
    lines = []
    for sw in av.scopes.values():
        lines.append(f"scope {sw.scope}: anchor `{sw.anchor}` on lane {sw.anchor_lane} ({sw.chosen})")
        lines.append(f"  period {sw.period:,.1f} cycles (std {sw.period_std:,.1f}, min {sw.period_min:,}, "
                     f"max {sw.period_max:,}); {sw.windows} steady windows (first {sw.skip} skipped, last partial "
                     f"one dropped)")
        lines.append("  a lane's stamps are reported as they fall in the ANCHOR lane's window: a lane whose own "
                     "window is shifted shows its early stamps late (or its late ones missing)")
        for role, nlanes, rows in by_role(rec, sw):
            lines.append(f"  role {role} ({nlanes} lanes)")
            lines.append(f"    {'offset':>9}  {'lanes min..max':>17}  {'jitter':>7}  {'lanes':>5}  {'windows':>7}  stamp")
            for r in rows:
                lines.append(f"    {r.mean:9.1f}  {r.lane_min:8.1f}..{r.lane_max:<8.1f}  {r.jitter:7.1f}  "
                             f"{r.lanes:5d}  {r.n:7d}  {label(r)}{'  [straddles the window start]' if r.straddles else ''}")
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
