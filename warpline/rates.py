"""The two clocks: the rate between a record's cycle counter and time, read from its TIES.

A stamp's clock is a processor's cycle counter; a profiler's samples, and anything outside that processor, are on
time. A TIE is one moment's cycle count and its nanosecond reading, on one lane (`record.ties`): an exact point
between the two axes. Two ties give the rate between the axes; a stamp between two ties is placed in time by
INTERPOLATION (`placer`), exact only to the ties' spacing, and a stamp outside a lane's tied span is not placed. Ties
are used on their own lane: no lane's clock is assumed another's.

The rate is in cycles a microsecond. Its UNCERTAINTY comes from the GRAIN: the smallest nonzero difference between
consecutive nanosecond readings on any lane, the time counter's step (an upper bound on it: it is what the readings
show). Two readings that each floor to the grain differ from the true interval by less than a grain, so a rate over an
interval of `ns` nanoseconds is uncertain by the fraction grain / ns.

`analyze` reads, per scope: the rate between consecutive ties of each lane; per REGION (a region whose begin stamp
carries a tie, and whose end -- or, where the end carries none, the next tied stamp -- carries one) its cycles, its
nanoseconds and its rate, averaged over the steady windows of `window.py`; by ROLE (lanes of a role together, the
spread across lanes shown) the same, and the window's own period; and the CHECK that the lanes of one scope, which
share one processor's counter, show the same rate over the same window (default tolerance 0.5%): a lane that does not
is a named finding.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field

from . import check, reader, window
from .record import Finding, Record, scope_of

TOLERANCE = 0.005     # the lanes of a scope agree on the rate over one window within this fraction
US = 1000.0           # nanoseconds in a microsecond


def _rate(cycles: float, ns: float) -> float:
    """Cycles a microsecond."""
    return cycles / (ns / US)


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ------------------------------------------------------------------------------------------------ placement

def placer(rec: Record):
    """`clock_of(lane, cycles) -> nanoseconds | None` for `window.average_window`: a tie's own reading exactly,
    linear interpolation between the two ties of the same lane that bracket the cycle count (exact for a rate that is
    constant between them), None outside the lane's tied span or on a lane without ties. Raises `ValueError` where a
    lane's ties go backwards (`check.ties_ordered` names where): they place nothing."""
    bad = check.ties_ordered(rec)
    if bad:
        raise ValueError(f"the ties go backwards, so no stamp can be placed by them: {bad[0]}")
    cs = {lane: [c for c, _ in lst] for lane, lst in rec.ties.items()}

    def clock_of(lane: str, t: int):
        lst = rec.ties.get(lane)
        if not lst:
            return None
        c = cs[lane]
        i = bisect.bisect_left(c, t)
        if i < len(c) and c[i] == t:
            return float(lst[i][1])
        if i == 0 or i == len(c):
            return None
        (c0, n0), (c1, n1) = lst[i - 1], lst[i]
        return n0 + (n1 - n0) * (t - c0) / (c1 - c0)

    return clock_of


# ------------------------------------------------------------------------------------------------ ties, pairs

@dataclass(frozen=True)
class Pair:
    lane: str
    c0: int
    c1: int
    ns: int          # nanoseconds between the two ties
    cycles: int
    rate: float      # cycles a microsecond
    unc: float       # the rate's uncertainty, a fraction: the grain over the interval's nanoseconds


def smallest_difference(rec: Record) -> int | None:
    """The smallest nonzero difference between consecutive nanosecond readings on any lane."""
    best = None
    for lst in rec.ties.values():
        for (_c0, a), (_c1, b) in zip(lst, lst[1:]):
            if b - a > 0 and (best is None or b - a < best):
                best = b - a
    return best


def grain(rec: Record) -> int | None:
    """THE GRAIN, the time counter's step. Ties are sparse (a phase's start), so the smallest difference between
    consecutive readings is the shortest interval between two ties, not the counter's step. A counter that steps by
    `g` gives readings whose every difference is a multiple of `g`: the greatest common divisor of the nonzero
    differences is then `g` exactly, and it never exceeds the smallest difference (which divides to it). Readings
    on no lattice give 1, the unit of the integers: the least a reading can resolve."""
    g = 0
    for lst in rec.ties.values():
        for (_c0, a), (_c1, b) in zip(lst, lst[1:]):
            if b - a > 0:
                g = math.gcd(g, b - a)
    return g or None


def pairs(rec: Record, g: int | None) -> dict[str, list[Pair]]:
    """Per lane, the rate between consecutive ties that advance in both coordinates (a pair that does not -- a tie that
    goes backwards, or no time between two ties -- gives no rate)."""
    out: dict[str, list[Pair]] = {}
    for lane, lst in rec.ties.items():
        ps = []
        for (c0, n0), (c1, n1) in zip(lst, lst[1:]):
            if n1 > n0 and c1 > c0:
                ps.append(Pair(lane, c0, c1, n1 - n0, c1 - c0, _rate(c1 - c0, n1 - n0), (g or 0) / (n1 - n0)))
        out[lane] = ps
    return out


# ------------------------------------------------------------------------------------------------ regions

@dataclass(frozen=True)
class Instance:
    lane: str
    name: str
    begin: int       # the begin stamp's cycles
    cycles: int
    ns: int
    via: str         # "end" | "next tie"


def instances(rec: Record) -> dict[str, list[Instance]]:
    """Each lane's regions whose begin carries a tie and that are bounded by a second: the end's tie, or where the end
    has none (or the region never closes) the lane's next tie after the begin's."""
    regs, _findings = reader.regions(rec)
    out: dict[str, list[Instance]] = {}
    for lane, rs in regs.items():
        ties = rec.ties.get(lane, [])
        if not ties:
            continue
        at = {}
        for c, ns in ties:
            at.setdefault(c, ns)
        cs = [c for c, _ in ties]
        lst = []
        for r in sorted(rs, key=lambda r: r.begin):
            if r.begin not in at:
                continue
            if r.end is not None and r.end in at:
                c1, n1, via = r.end, at[r.end], "end"
            else:
                i = bisect.bisect_right(cs, r.begin)
                if i == len(cs):
                    continue
                c1, n1, via = cs[i], ties[i][1], "next tie"
            n0 = at[r.begin]
            if n1 > n0 and c1 > r.begin:
                lst.append(Instance(lane, r.name, r.begin, c1 - r.begin, n1 - n0, via))
        out[lane] = lst
    return out


@dataclass
class Agg:
    """One region (a name, its occurrence in the window) on one lane, over the steady windows."""
    lane: str
    role: str
    name: str
    occ: int
    n: int
    cycles: float            # mean over the windows
    ns: float
    rate: float              # total cycles over total nanoseconds
    spread: float            # standard deviation of the instances' rates
    unc: float               # grain over the mean nanoseconds
    via: str


@dataclass
class RoleRegion:
    role: str
    name: str
    occ: int
    lanes: int
    cycles: float            # mean over the role's lanes
    ns: float
    rate: float              # mean of the lanes' rates
    lane_min: float
    lane_max: float
    unc: float


@dataclass
class RolePeriod:
    role: str
    lanes: int
    cycles: float
    ns: float
    rate: float
    lane_min: float
    lane_max: float
    unc: float


@dataclass
class ScopeRates:
    scope: str
    anchor: str
    chosen: str
    anchor_lane: str
    windows: int
    skip: int
    regions: list[Agg] = field(default_factory=list)
    by_role: list[RoleRegion] = field(default_factory=list)
    periods: list[RolePeriod] = field(default_factory=list)
    lane_rates: dict[str, dict] = field(default_factory=dict)   # lane -> {rate, windows, worst, over}
    notes: list[str] = field(default_factory=list)


@dataclass
class Rates:
    grain: int | None
    smallest: int | None
    tolerance: float
    lanes: dict[str, dict]          # lane -> {ties, role, pairs: [Pair]}
    scopes: dict[str, ScopeRates]
    findings: list[Finding]
    time_unit: str = "ns"
    time_origin: str | None = None


def analyze(rec: Record, anchor: str | None = None, skip: int = window.SKIP, scopes: list[str] | None = None,
            tolerance: float = TOLERANCE) -> Rates:
    g = grain(rec)
    findings = list(check.ties_ordered(rec))
    ps = pairs(rec, g)
    lanes = {ln: {"ties": len(rec.ties.get(ln, [])), "role": rec.lanes[ln].role, "pairs": ps.get(ln, [])}
             for ln in rec.lanes if rec.ties.get(ln)}
    res = Rates(g, smallest_difference(rec), tolerance, lanes, {}, findings, rec.clock.get("time_unit", "ns"), rec.clock.get("time_origin"))
    if not rec.ties:
        return res
    inst = instances(rec)
    for scope in scopes if scopes is not None else rec.scopes():
        if not any(scope_of(ln) == scope for ln in lanes):
            continue
        try:
            name, how, alane, starts = window.anchor_starts(rec, scope, anchor, skip)
        except window.WindowError as e:
            res.findings.append(Finding("no_window", str(e)))
            continue
        sr = ScopeRates(scope, name, how, alane, len(starts) - 1, skip)
        _regions(rec, sr, inst, starts, g)
        _periods(rec, sr, name, skip, g)
        _lane_check(rec, sr, starts, tolerance, g, res.findings)
        res.scopes[scope] = sr
    return res


def _regions(rec: Record, sr: ScopeRates, inst: dict[str, list[Instance]], starts: list[int], g) -> None:
    acc: dict[tuple[str, str, int], list[Instance]] = {}
    order = []
    for lane, lst in inst.items():
        if scope_of(lane) != sr.scope:
            continue
        for k in range(len(starts) - 1):
            seen: dict[str, int] = {}
            for it in lst:
                if starts[k] <= it.begin < starts[k + 1]:
                    o = seen.get(it.name, 0)
                    seen[it.name] = o + 1
                    key = (lane, it.name, o)
                    if key not in acc:
                        acc[key] = []
                        order.append(key)
                    acc[key].append(it)
    for key in order:
        its = acc[key]
        lane, nm, occ = key
        cyc, ns = _mean([i.cycles for i in its]), _mean([i.ns for i in its])
        rate = _rate(sum(i.cycles for i in its), sum(i.ns for i in its))
        spread = _std([_rate(i.cycles, i.ns) for i in its])
        via = "end" if all(i.via == "end" for i in its) else "next tie"
        sr.regions.append(Agg(lane, rec.lanes[lane].role, nm, occ, len(its), cyc, ns, rate, spread,
                              (g or 0) / ns, via))
    roles: dict[tuple[str, str, int], list[Agg]] = {}
    for a in sr.regions:
        roles.setdefault((a.role, a.name, a.occ), []).append(a)
    for (role, nm, occ), as_ in roles.items():
        rs = [a.rate for a in as_]
        sr.by_role.append(RoleRegion(role, nm, occ, len(as_), _mean([a.cycles for a in as_]),
                                     _mean([a.ns for a in as_]), _mean(rs), min(rs), max(rs),
                                     _mean([a.unc for a in as_])))
    if not sr.regions:
        sr.notes.append("no region has a tied begin and a second tie to close it in the steady windows")


def _periods(rec: Record, sr: ScopeRates, anchor: str, skip: int, g) -> None:
    """The window's own period on each lane that stamps the anchor with ties: cycles and nanoseconds between its
    consecutive tied anchors (after the first `skip`)."""
    per_role: dict[str, list[tuple[float, float]]] = {}
    for ln in rec.lanes:
        if scope_of(ln) != sr.scope or not rec.ties.get(ln):
            continue
        at = dict(reversed(rec.ties[ln]))
        begins = [t for _i, nm, t, _p in rec.lane_stamps(ln) if nm == anchor][skip:]
        gaps = [(b - a, at[b] - at[a]) for a, b in zip(begins, begins[1:])
                if a in at and b in at and at[b] > at[a] and b > a]
        if gaps:
            per_role.setdefault(rec.lanes[ln].role, []).append((_mean([c for c, _ in gaps]),
                                                                 _mean([n for _, n in gaps])))
    for role, lst in per_role.items():
        rs = [_rate(c, n) for c, n in lst]
        ns = _mean([n for _, n in lst])
        sr.periods.append(RolePeriod(role, len(lst), _mean([c for c, _ in lst]), ns, _mean(rs), min(rs), max(rs),
                                     (g or 0) / ns))


def _lane_check(rec: Record, sr: ScopeRates, starts: list[int], tol: float, g, findings: list[Finding]) -> None:
    """Every lane of the scope over each steady window: the rate from its first tie in the window to its last. The
    lanes share the scope's counter, so they must agree within `tol` of the window's median."""
    lanes = [ln for ln in rec.lanes if scope_of(ln) == sr.scope and rec.ties.get(ln)]
    per: dict[int, dict[str, tuple[float, float]]] = {}
    for k in range(len(starts) - 1):
        for ln in lanes:
            inside = [(c, n) for c, n in rec.ties[ln] if starts[k] <= c < starts[k + 1]]
            if len(inside) >= 2 and inside[-1][1] > inside[0][1] and inside[-1][0] > inside[0][0]:
                dn = inside[-1][1] - inside[0][1]
                per.setdefault(k, {})[ln] = (_rate(inside[-1][0] - inside[0][0], dn), (g or 0) / dn)
    stat = {ln: {"windows": 0, "compared": 0, "worst": 0.0, "over": 0, "rate": None, "unc": 0.0} for ln in lanes}
    cyc: dict[str, list[float]] = {ln: [] for ln in lanes}
    for k, d in per.items():
        for ln, (r, u) in d.items():
            stat[ln]["windows"] += 1
            cyc[ln].append(r)
            stat[ln]["unc"] = max(stat[ln]["unc"], u)
        if len(d) < 2:
            continue
        med = _median([r for r, _u in d.values()])
        for ln, (r, _u) in d.items():
            dev = abs(r / med - 1)
            stat[ln]["compared"] += 1
            stat[ln]["worst"] = max(stat[ln]["worst"], dev)
            if dev > tol:
                stat[ln]["over"] += 1
    for ln in lanes:
        stat[ln]["rate"] = _mean(cyc[ln]) if cyc[ln] else None
        s = stat[ln]
        if s["over"]:
            findings.append(Finding(
                "lane_rate_disagrees",
                f"over {s['over']} of {s['compared']} steady windows its rate differs from the scope's median by more "
                f"than {100 * tol:g}% (worst {100 * s['worst']:.2f}%; its rate's uncertainty from the grain is up to "
                f"{100 * s['unc']:.2f}%): the lanes of {sr.scope} share one counter, so a lane's clock or its ties "
                f"are not what the record says", ln))
    sr.lane_rates = stat


# ------------------------------------------------------------------------------------------------ for variants

def window_in_time(rec: Record, scope: str | None = None, anchor: str | None = None,
                   skip: int = window.SKIP) -> dict | None:
    """The window's own period read from the record's ties on the anchor lane: nanoseconds and cycles between
    consecutive window starts over the steady windows, None where the anchor stamps carry no ties."""
    if not rec.ties:
        return None
    sc = scope or rec.scopes()[0]
    name, _how, lane, starts = window.anchor_starts(rec, sc, anchor, skip)
    at = dict(reversed(rec.ties.get(lane, [])))
    if any(t not in at for t in starts):
        return None
    gaps = [(b - a, at[b] - at[a]) for a, b in zip(starts, starts[1:])]
    return {"scope": sc, "anchor": name, "lane": lane, "windows": len(gaps),
            "period_cycles": _mean([c for c, _ in gaps]), "period_ns": _mean([n for _, n in gaps])}


# ------------------------------------------------------------------------------------------------ output

def _f(x: float, w: int = 0, d: int = 1) -> str:
    return f"{x:{w},.{d}f}"


def format_rates(r: Rates, rows: int = 6) -> str:
    L = []
    if not r.lanes:
        return ("the record carries no ties: a cycle count becomes time only by its own record's ties, so there is no "
                "rate to read (time comes from samples alone)\n")
    g = r.grain
    L.append(f"ties on {len(r.lanes)} lane(s); time axis: {r.time_unit}, origin: {r.time_origin or 'not stated'}")
    L.append(f"THE GRAIN, the time counter's step: {g if g is not None else 'not observed'} ns -- the greatest common "
             f"divisor of the nonzero differences between consecutive nanosecond readings on any lane (every difference "
             f"is a multiple of the step). The smallest nonzero difference is {r.smallest} ns: the shortest interval "
             f"between two ties, which is the step only when the ties sit that close. A rate's uncertainty is the "
             f"grain over its interval's nanoseconds")
    L.append("rates are cycles a microsecond (cycles over microseconds)")
    L.append("")
    L.append("RATE BETWEEN CONSECUTIVE TIES, per lane")
    for ln, d in r.lanes.items():
        ps = d["pairs"]
        if not ps:
            L.append(f"  {ln} ({d['role']}): {d['ties']} tie(s), no consecutive pair advances in both coordinates")
            continue
        rs = [p.rate for p in ps]
        L.append(f"  {ln} ({d['role']}): {d['ties']} ties, {len(ps)} pairs; rate min {_f(min(rs))}, median "
                 f"{_f(_median(rs))}, max {_f(max(rs))}; whole span "
                 f"{_f(_rate(sum(p.cycles for p in ps), sum(p.ns for p in ps)))} (+-{100 * max(p.unc for p in ps):.2f}% "
                 f"worst pair)")
        for p in ps[:rows]:
            L.append(f"      {p.c0:>10,} -> {p.c1:>10,} cycles  {p.ns:>9,} ns  {_f(p.rate, 9)}  +-{100 * p.unc:.2f}%")
        if len(ps) > rows:
            L.append(f"      ... {len(ps) - rows} more (--json has all)")
    for sc, s in r.scopes.items():
        L.append("")
        L.append(f"scope {sc}: window anchor `{s.anchor}` on {s.anchor_lane} ({s.chosen}); {s.windows} steady windows "
                 f"(first {s.skip} skipped, last partial one dropped)")
        L.append("  REGIONS by role (lanes of a role together; the rate across lanes min..max): cycles and time a window")
        L.append(f"    {'role':<12} {'region':<28} {'lanes':>5} {'cycles':>10} {'time us':>9} {'rate c/us':>10} "
                 f"{'across lanes':>17} {'+-%':>5}")
        for a in s.by_role:
            nm = f"{a.name}#{a.occ}" if any(x.name == a.name and x.occ > 0 for x in s.by_role) else a.name
            L.append(f"    {a.role:<12} {nm:<28} {a.lanes:>5} {_f(a.cycles, 10)} {_f(a.ns / US, 9, 3)} "
                     f"{_f(a.rate, 10)} {_f(a.lane_min, 8)}..{_f(a.lane_max, 8)} {100 * a.unc:5.2f}")
        L.append("  per lane, with the spread of the rate across windows:")
        for a in s.regions:
            nm = f"{a.name}#{a.occ}" if any(x.name == a.name and x.occ > 0 for x in s.regions) else a.name
            L.append(f"    {a.lane:<14} {nm:<28} n {a.n:>3}  {_f(a.cycles, 10)} cycles  {_f(a.ns / US, 9, 3)} us  "
                     f"{_f(a.rate, 9)} +-{a.spread:.2f} (spread)  +-{100 * a.unc:.2f}% (grain)"
                     + ("" if a.via == "end" else f"  [closed by the next tie]"))
        L.append("  THE WINDOW's own period, by role (between its tied window starts):")
        for p in s.periods:
            L.append(f"    {p.role:<12} {p.lanes:>2} lane(s)  {_f(p.cycles, 10)} cycles  {_f(p.ns / US, 9, 3)} us  rate "
                     f"{_f(p.rate, 9)}  across lanes {_f(p.lane_min, 8)}..{_f(p.lane_max, 8)}  +-{100 * p.unc:.2f}%")
        for n in s.notes:
            L.append(f"  note: {n}")
        L.append(f"  CHECK: the lanes of {sc} share one counter: their rate over the same window agrees within "
                 f"{100 * r.tolerance:g}%")
        for ln, st in s.lane_rates.items():
            if st["compared"]:
                L.append(f"    {ln}: rate {_f(st['rate'])} c/us over {st['windows']} windows; worst deviation from the "
                         f"median {100 * st['worst']:.3f}% over {st['compared']} compared windows"
                         + ("  -> DISAGREES" if st["over"] else ""))
            else:
                L.append(f"    {ln}: no window to compare (needs two lanes with two ties each in a window)")
    L.append("")
    if r.findings:
        L.append("FINDINGS")
        L += [f"  {f}" for f in r.findings]
    else:
        L.append("findings: none")
    return "\n".join(L) + "\n"


def to_dict(r: Rates) -> dict:
    return {
        "time_unit": r.time_unit, "time_origin": r.time_origin, "grain_ns": r.grain, "smallest_difference_ns": r.smallest, "tolerance": r.tolerance,
        "lanes": {ln: {"role": d["role"], "ties": d["ties"],
                       "pairs": [{"c0": p.c0, "c1": p.c1, "cycles": p.cycles, "ns": p.ns,
                                  "cycles_per_us": p.rate, "uncertainty": p.unc} for p in d["pairs"]]}
                  for ln, d in r.lanes.items()},
        "scopes": {sc: {
            "anchor": s.anchor, "anchor_lane": s.anchor_lane, "chosen": s.chosen, "windows": s.windows, "skip": s.skip,
            "regions": [{"lane": a.lane, "role": a.role, "name": a.name, "occ": a.occ, "n": a.n, "cycles": a.cycles,
                         "ns": a.ns, "cycles_per_us": a.rate, "rate_spread": a.spread, "uncertainty": a.unc,
                         "closed_by": a.via} for a in s.regions],
            "by_role": [{"role": a.role, "name": a.name, "occ": a.occ, "lanes": a.lanes, "cycles": a.cycles,
                         "ns": a.ns, "cycles_per_us": a.rate, "lane_min": a.lane_min, "lane_max": a.lane_max,
                         "uncertainty": a.unc} for a in s.by_role],
            "window": [{"role": p.role, "lanes": p.lanes, "cycles": p.cycles, "ns": p.ns, "cycles_per_us": p.rate,
                        "lane_min": p.lane_min, "lane_max": p.lane_max, "uncertainty": p.unc} for p in s.periods],
            "lane_check": s.lane_rates, "notes": s.notes} for sc, s in r.scopes.items()},
        "findings": [{"code": f.code, "lane": f.lane, "time": f.time, "message": f.message} for f in r.findings],
    }


def dumps(r: Rates) -> str:
    return json.dumps(to_dict(r), indent=1)
