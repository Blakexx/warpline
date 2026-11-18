"""The sampled layer laid under the stamps, PLACED BY THE LAUNCH'S TWO EDGES.

The kernel stamps `launch.begin` at its entry and `launch.end` at its exit, each with a tie (see `launch.py`); the
samples see the same two moments as the series going live and going quiet (see `sampling.edge_placement`). Each
sample's window position is ((j + 0.5)*dt - i0*dt - u - SETUP) / P mod 1, u the entry's position inside the first
live sample i0 (the midpoint of an interval, narrowed by every run), SETUP the launch's begin to the first steady
window and P the window's period, both from the ties. Each series is folded at ITS OWN placement (each is its own
pass) and carries its placement's half-width; one wider than half a bin is LOOSE.

A record without both edges (with ties) is REFUSED, naming what the kernel must stamp: the correlation of the samples'
`barrier` fold with the stamps' barrier waits (`align`, below) is never a substitute; where the record has such
waits it is reported beside the edges' placement as a CROSS-CHECK.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import record as rec_mod
from . import launch as launch_mod
from . import rates, sampling, window
from .record import Record, classify, scope_of

FINE = 16   # cells a bin when the profile is smoothed


class OverlayError(Exception):
    pass


@dataclass(frozen=True)
class VRow:
    lane: str
    role: str
    name: str
    occ: int
    mean: float
    n: int


@dataclass
class View:
    scope: str
    period: float          # cycles
    windows: int
    anchor: str
    rows: list[VRow]
    lanes: dict[str, str]  # lane -> role, the lanes of the scope
    rec: Record            # the average-window record (for Perfetto)
    source: str = ""
    runs: list[int] = field(default_factory=list)
    axis: str = "cycles"               # the unit of `rows`' offsets: "cycles", or "ns" where the ties placed them
    period_ns: float | None = None     # the window's period in nanoseconds, from the anchor lane's ties
    axis_note: str = "window position = cycles from the window's start over the period in cycles"

    @property
    def span(self) -> float:
        """The window's length in the unit of `rows`: the divisor of a window position."""
        return self.period_ns if self.axis == "ns" and self.period_ns else self.period


def view_of(rec: Record, scope: str | None = None, source: str = "") -> View:
    """The average window of a stamp record (computed), or of a record that already is one (a `window` or `merge`
    output: `facts.window`)."""
    wf = rec.facts.get("window")
    if wf and wf.get("stamps") and wf.get("scopes"):
        scopes = wf["scopes"]
        sc = scope or next(iter(scopes))
        if sc not in scopes:
            raise OverlayError(f"no scope {sc!r}; the record has {list(scopes)}")
        rows = [VRow(e["lane"], rec.lanes[e["lane"]].role, e["name"], e["occ"], e["mean"], e["n"])
                for e in wf["stamps"] if scope_of(e["lane"]) == sc]
        info = scopes[sc]
        lanes = {ln: x.role for ln, x in rec.lanes.items() if scope_of(ln) == sc}
        return View(sc, info["period"], info["windows"], info["anchor"], rows, lanes, rec, source)
    scopes = rec.scopes()
    sc = scope or scopes[0]
    if sc not in scopes:
        raise OverlayError(f"no scope {sc!r}; the record has {scopes}")
    try:
        av = window.average_window(rec, None, window.SKIP, [sc])
        tv = None
        if rec.ties:
            try:
                tv = window.average_window(rec, av.scopes[sc].anchor, window.SKIP, [sc], clock_of=rates.placer(rec))
            except (ValueError, window.WindowError) as e:
                raise OverlayError(f"the record has ties, so cycles are not used for the window position, and the "
                                   f"ties cannot place it: {e}")
    except window.WindowError as e:
        raise OverlayError(str(e))
    sw = av.scopes[sc]
    lanes = {ln: x.role for ln, x in rec.lanes.items() if scope_of(ln) == sc}
    if tv is None:
        rows = [VRow(r.lane, rec.lanes[r.lane].role, r.name, r.occ, r.mean, r.n) for r in sw.rows]
        return View(sc, sw.period, sw.windows, sw.anchor, rows, lanes, window.to_record(av), source)
    tw = tv.scopes[sc]
    rows = [VRow(r.lane, rec.lanes[r.lane].role, r.name, r.occ, r.mean, r.n) for r in tw.rows]
    note = (f"window position = nanoseconds from the window's start over the period in nanoseconds, by the record's "
            f"ties (an untied stamp is interpolated between its lane's ties; one outside them is not placed); cycles "
            f"were NOT used")
    return View(sc, sw.period, sw.windows, sw.anchor, rows, lanes, window.to_record(av), source, axis="ns",
                period_ns=tw.period, axis_note=note)


# ------------------------------------------------------------------------------------------------ intervals

def _circular(view: View, kind_begin: str, kind_end: str):
    """[(lane, key, begin offset, duration, weight)]: each begin paired with the next end of the same key on the
    lane, going round the window (a wait that straddles the window's end has its end early in the window)."""
    per: dict[tuple[str, str], dict[str, list[VRow]]] = {}
    for r in view.rows:
        kind, key = classify(r.name)
        if kind == kind_begin:
            per.setdefault((r.lane, key), {}).setdefault("b", []).append(r)
        elif kind == kind_end:
            per.setdefault((r.lane, key), {}).setdefault("e", []).append(r)
    out = []
    P = view.span
    for (lane, key), d in per.items():
        ends = d.get("e", [])
        for b in d.get("b", []):
            if not ends:
                continue
            e = min(ends, key=lambda x: (x.mean - b.mean) % P)
            out.append((lane, key, b.mean, (e.mean - b.mean) % P, min(b.n, e.n) / max(1, view.windows)))
    return out


def waits(view: View):
    return _circular(view, "wait_begin", "wait_end")


def regions(view: View):
    return _circular(view, "region_begin", "region_end")


def _smooth(cells: list[float], width: float) -> list[float]:
    """Circular box of `width` cells (a real number) by exact integration of the piecewise-constant cells: cell i
    becomes the mean of the profile over [i, i + width], the span a sample STARTING there totals."""
    m = len(cells)
    pre = [0.0]
    for c in cells:
        pre.append(pre[-1] + c)
    tot = pre[-1]

    def F(x: float) -> float:
        q, r = divmod(x, m)
        i = min(int(r), m - 1)
        return q * tot + pre[i] + cells[i] * (r - i)

    if width <= 1e-9:
        return list(cells)
    return [(F(i + 0.5 + width) - F(i + 0.5)) / width for i in range(m)]


def wait_profile(view: View, n_bins: int, blur: float) -> list[float]:
    """The mean number of lanes inside a `barrier.*.wait` region at each of `n_bins` window positions, smoothed by
    a box `blur` (a fraction of the window) wide that starts at the position: a sample is a total over its interval, and
    a sample at window position x (its interval's start) sees the waits in [x, x + blur]."""
    w = waits(view)
    if not w:
        raise OverlayError(
            f"the stamp record has no barrier waits in its average window: it cannot be aligned to the samples this "
            f"way. A fine record is needed: `barrier.<B>.wait.begin` and `.wait.end` stamps on the lanes, in every "
            f"window (the sampled `barrier` stall counts warps inside such waits)")
    m = n_bins * FINE
    cells = [0.0] * m
    P = view.span
    for _lane, _key, b, dur, wt in w:
        if wt <= 0 or dur <= 0:
            continue
        x0 = b / P * m
        x1 = x0 + dur / P * m
        k = int(math.floor(x0))
        while k < x1:
            lo, hi = max(x0, k), min(x1, k + 1)
            if hi > lo:
                cells[k % m] += wt * (hi - lo)
            k += 1
    fine = _smooth(cells, blur * m)
    return [sum(fine[b * FINE:(b + 1) * FINE]) / FINE for b in range(n_bins)]


def phases(view: View, n_bins: int) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """Per bin, per role: the phase most of the role's lanes are in at the bin's centre, and the share of the role's
    lanes in it (`-` when none are in any region)."""
    P = view.span
    reg = regions(view)
    roles: list[str] = []
    for ln, role in view.lanes.items():
        if role not in roles:
            roles.append(role)
    by_lane: dict[str, list] = {}
    for lane, key, b, dur, wt in reg:
        if wt >= 0.5:
            by_lane.setdefault(lane, []).append((key, b, dur))
    out = []
    for b in range(n_bins):
        centre = (b + 0.5) / n_bins * P
        row = []
        for role in roles:
            lanes = [ln for ln, r in view.lanes.items() if r == role]
            count: dict[str, int] = {}
            for ln in lanes:
                inside = [(dur, key) for key, st, dur in by_lane.get(ln, []) if (centre - st) % P < dur]
                name = min(inside)[1] if inside else "-"
                count[name] = count.get(name, 0) + 1
            top = max(sorted(count), key=lambda k: count[k])
            row.append((top[6:] if top.startswith("phase.") else top, count[top] / len(lanes)))
        out.append(row)
    return roles, out


# ------------------------------------------------------------------------------------------------ alignment

@dataclass
class Alignment:
    view: View
    folded: sampling.Folded
    series: str                # the folded series the shift is measured on
    bins: int
    blur: float
    profile: list[float]
    fold: list[float]          # the series' fold, unrotated
    shift: sampling.Shift
    n_waits: int

    @property
    def accepted(self) -> bool:
        return self.shift.accepted

    def rotated(self, name: str) -> list[sampling.Bin]:
        n = self.bins
        return [self.folded.bins[name][(b + self.shift.shift) % n] for b in range(n)]

    @property
    def origin(self) -> float:
        """Where the samples' phase 0 falls in the window (a fraction)."""
        return ((self.bins - self.shift.shift) % self.bins) / self.bins


def choose_series(f: sampling.Folded) -> str:
    var = f.varying()
    if not var:
        raise OverlayError("no series of the samples has enough variation to align")
    for n in var:
        if n.endswith("stalled_barrier.avg"):
            return n
    return max(var, key=lambda n: f.score.get(n, 0.0))


def align(view: View, f: sampling.Folded, series: str | None = None) -> Alignment:
    name = series or choose_series(f)
    if name not in f.bins:
        raise OverlayError(f"the series {name!r} is not in the fold")
    n = f.cfg.bins
    profile = wait_profile(view, n, f.blur)
    fold = [b.mean for b in f.bins[name]]
    sh = sampling.best_shift(profile, fold, f.cfg.min_corr, f.cfg.min_gap, sampling.guard_for(f.blur, n))
    return Alignment(view, f, name, n, f.blur, profile, fold, sh, len(waits(view)))


# ------------------------------------------------------------------------------------------------ placement

SHAPE_CORR, SHAPE_GAP = sampling.MIN_CORR, sampling.MIN_GAP
PERIOD_TOL = 0.02       # the stamps' window (ties) and the samples' agree within this fraction, else a finding


def launch_for(rec: Record, scope: str | None = None) -> launch_mod.Launch:
    """The record's launch, or an OverlayError naming what the kernel must stamp."""
    try:
        return launch_mod.launch_of(rec, scope)
    except launch_mod.LaunchError as e:
        raise OverlayError(str(e))


@dataclass
class SeriesPlace:
    name: str
    short: str
    key: str                       # its pass
    bins: list[sampling.Bin]       # folded at its placement: bin b is window position b/N .. (b+1)/N
    half_us: float
    half_windows: float
    half_bins: float
    loose: bool
    fed: int                       # runs that fed its placement
    more: int | None               # more runs needed at the observed narrowing (None: not estimable)


@dataclass
class Cross:
    series: str | None
    note: str
    old_origin: float | None = None
    edge_origin: float | None = None
    tol: float | None = None
    diff: float | None = None
    agree: bool | None = None


@dataclass
class Placement:
    view: View
    launch: launch_mod.Launch
    folded: sampling.Folded
    bins: int
    edges: dict[str, sampling.PassPlacement]
    series: dict[str, SeriesPlace]
    findings: list[str]
    cross: Cross
    profile: list[float] | None = None

    def values(self) -> dict[str, list[sampling.Bin]]:
        return {n: s.bins for n, s in self.series.items()}

    def varying(self) -> list[str]:
        f = self.folded
        return sorted((n for n in f.varying() if n in self.series), key=lambda x: -f.swing(x))


def _circ(a: float, b: float) -> float:
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def _more_runs(pp: sampling.PassPlacement, bin_us: float) -> int | None:
    """Runs still needed, at the narrowing observed, for the interval's width to reach a bin (half-width half a bin):
    the width falls as 1/runs."""
    if len(pp.steps) < 2 or not pp.steps[0][1] > pp.steps[-1][1] * (1 + 1e-9):
        return None
    n, w = pp.steps[-1]
    if w <= 0:
        return 0
    return max(1, math.ceil(n * w / bin_us) - n)


def place(view: View, lc: launch_mod.Launch, exports: list[sampling.Export], f: sampling.Folded) -> Placement:
    cfg, n = f.cfg, f.cfg.bins
    P = lc.period_us
    bin_us = P / n
    edges = sampling.edge_placement(exports, f, lc.length_us, P, cfg)
    findings = list(f"launch edge: {x}" for x in lc.findings)
    d = f.period_us / P - 1
    if abs(d) > PERIOD_TOL:
        findings.append(f"time_bases_disagree: the stamps' window from their ties is {P:.4f} us, the samples' "
                        f"{f.period_us:.4f} us ({100 * d:+.2f}% by the samples; bound {100 * PERIOD_TOL:g}%): the "
                        f"placement uses the ties' window, so a sample's position drifts by that over the launch")
    series: dict[str, SeriesPlace] = {}
    for key, pp in edges.items():
        findings += [f"{x}" for x in pp.findings]
        if not pp.placed:
            continue
        ck = {c.run: c for c in pp.checks}
        for name in pp.names:
            if name not in f.bins:
                continue
            parts = []
            for i in pp.fed:
                sr = next((x for x in f.runs[i].series if x.name == name), None)
                if sr is None or not sr.folded:
                    continue
                vals = next(x for x in exports[i].series if x.name == name).values
                c = ck[i]
                b = sampling.placed_bins(vals, c.i0, c.i1, c.dt, pp.u[i], lc.setup_us, P, cfg)
                if b is not None:
                    parts.append(b)
            if not parts:
                continue
            hw = pp.half / P
            series[name] = SeriesPlace(name, _short(name), key, sampling.pool_bins(parts), pp.half, hw, hw * n,
                                       hw * n > 0.5 + 1e-9, len(pp.fed), _more_runs(pp, bin_us))
    cross = _cross(view, f, edges, series, lc, n)
    if cross.agree is False:
        findings.append(f"cross_check_disagrees: {cross.note}")
    try:
        profile = wait_profile(view, n, f.blur)
    except OverlayError:
        profile = None
    return Placement(view, lc, f, n, edges, series, findings, cross, profile)


def _cross(view: View, f: sampling.Folded, edges: dict[str, sampling.PassPlacement], series: dict[str, SeriesPlace],
           lc: launch_mod.Launch, n: int) -> Cross:
    try:
        al = align(view, f)
    except OverlayError as e:
        return Cross(None, f"no cross-check: the old correlation cannot be made ({e})")
    if not al.accepted:
        return Cross(al.series, f"no cross-check: the correlation of the stamps' barrier waits with the samples' "
                                f"`{_short(al.series)}` fold was itself refused ({al.shift.why}); the edges' placement "
                                f"stands alone")
    sp = series.get(al.series)
    pp = edges.get(sp.key) if sp else None
    ref = next((s.index for s in f.run_shifts if s.status == "reference"), 0)
    if pp is None or ref not in pp.u:
        return Cross(al.series, f"no cross-check: `{_short(al.series)}` is not placed through the fold's reference "
                                f"run ({ref}) by the edges")
    edge = (-pp.u[ref] / 1.0 - lc.setup_us) / lc.period_us % 1.0
    tol = sp.half_windows + 1.0 / n
    diff = _circ(al.origin, edge)
    agree = diff <= tol + 1e-9
    note = (f"on `{_short(al.series)}`: the correlation puts the start of the first live sample at window position "
            f"{al.origin:.3f} (shift {al.shift.shift} bins, correlation {al.shift.corr:.3f}), the edges at {edge:.3f}: "
            f"{diff:.3f} of a window apart; bound the placement's half-width {sp.half_windows:.3f} + one bin "
            f"{1.0 / n:.3f} = {tol:.3f}")
    return Cross(al.series, note + (": agree" if agree else ": DISAGREE"), al.origin, edge, tol, diff, agree)


# ------------------------------------------------------------------------------------------------ output

def _short(s: str) -> str:
    return sampling.Series(s, "", "", [], 0).short


def table_lines(view: View, n: int, cols: list[tuple[str, list[sampling.Bin]]], profile: list[float] | None) -> list[str]:
    """Per window position: each role's phase (and the share of its lanes in it), the stamps' mean lanes in a wait
    (when the record has waits), and each column's folded value."""
    roles, ph = phases(view, n)
    head = f"{'position':<13} " + (f"{'lanes in wait':>13} " if profile else "") + " ".join(
        f"{r[:17]:>17}" for r in roles) + "  " + " ".join(f"{label[:19]:>19}" for label, _b in cols)
    out = [head]
    for b in range(n):
        cells = [f"{p[:12]} {100 * s:3.0f}%" for p, s in ph[b]]
        vals = [f"{(bins[b].mean if bins is not None else float('nan')):19.2f}" for _l, bins in cols]
        out.append(f"{f'{b / n:.3f}-{(b + 1) / n:.3f}':<13} " + (f"{profile[b]:13.2f} " if profile else "")
                   + " ".join(f"{c:>17}" for c in cells) + "  " + " ".join(vals))
    return out


def placement_lines(pl: Placement) -> list[str]:
    """The placement's own report: the launch, each pass's length checks and entry interval, each series' half-width,
    the cross-check and the findings."""
    f, n = pl.folded, pl.bins
    P = pl.launch.period_us
    L = [pl.launch.describe()]
    L.append(f"ASSUMED AND STATED: the kernel's entry falls in the first live sample of a pass and its exit in the last "
             f"(live: any series of the pass over {f.cfg.live:g}); within {f.cfg.length_tol:g} of a sample, else the "
             f"length check refuses. Each pass has its own start, so each is placed by its own entry position u "
             f"(0 <= u < one sample), an INTERVAL; the runs' intervals are intersected through the run alignment "
             f"(known to {f.cfg.shift_slack:g} bin = {f.cfg.shift_slack * P / n:.3f} us)")
    L.append(f"sample j sits at window position ((j + 0.5)*dt - i0*dt - u - SETUP) / P mod 1; SETUP "
             f"{pl.launch.setup_us:.3f} us, P {P:.4f} us, u the interval's midpoint; bins of {1.0 / n:.3f} of a window")
    for key, pp in pl.edges.items():
        L.append("")
        L.append(f"pass `{_short(key)}` ({len(pp.names)} series: {', '.join(_short(x) for x in pp.names)})")
        for c in pp.checks:
            L.append(f"  length check, run {c.run}: {c.line()}"
                     + ("" if c.entry.ok else "  [FAILS]" if c.entry.gap > 0 else "")
                     + (f"  [within the slack: empty by {c.entry.gap:.3f} us]" if 0 < c.entry.gap and c.entry.ok else ""))
        if pp.placed:
            hw = pp.half / P
            L.append(f"  entry u in [{pp.lo:.3f}, {pp.hi:.3f}) us of run {pp.ref}'s first live sample: midpoint "
                     f"{pp.mid:.3f} us, half-width {pp.half:.3f} us = {hw:.4f} of a window = {hw * n:.2f} bins; "
                     f"{len(pp.fed)} run(s) fed it {pp.fed}"
                     + (f" (aligned on `{_short(pp.align_series)}`)" if pp.align_series else ""))
            if len(pp.steps) > 1:
                L.append("  narrowing: " + ", ".join(f"{k} run(s) width {w:.3f} us" for k, w in pp.steps))
            for i, sh in pp.shifts.items():
                if sh.accepted:
                    L.append(f"  run {i}: shift {sh.shift} bins, correlation {sh.corr:.3f}; u_run - u_{pp.ref} = "
                             f"{pp.d[i]:+.3f} us")
        for x in pp.findings:
            L.append(f"  FINDING {x}")
    L.append("")
    L.append("PLACEMENT of each series (half-width of its entry position; LOOSE when over half a bin):")
    for name in pl.varying() + [x for x in pl.series if x not in pl.varying()]:
        s = pl.series[name]
        tag = ""
        if s.loose:
            tag = "  LOOSE" + (f" (about {s.more} more run(s) at the observed narrowing)" if s.more is not None
                              else " (one run, or no narrowing observed: not estimable)")
        L.append(f"  {s.short:<22} pass `{_short(s.key)}`: half-width {s.half_us:.3f} us = {s.half_windows:.4f} of a "
                 f"window = {s.half_bins:.2f} bins; {s.fed} run(s){tag}")
    L.append("")
    L.append(f"CROSS-CHECK: {pl.cross.note}")
    extra = [x for x in pl.findings if not any(x in pp.findings for pp in pl.edges.values())]
    for x in extra:
        L.append(f"FINDING {x}")
    return L


def format_overlay(pl: Placement, stamps_path: str, sample_paths: list[str]) -> str:
    v, f, n = pl.view, pl.folded, pl.bins
    L = []
    L.append(f"stamps: {stamps_path}: scope {v.scope}, anchor `{v.anchor}`, window {v.period:,.1f} cycles over "
             f"{v.windows} steady windows")
    ru = sorted({i for x in f.runs_of.values() for i in x})
    L.append(f"samples: {len(sample_paths)} run(s), runs {ru} in the average; window {f.period:.4f} samples = "
             f"{f.period_us:.3f} us (interval {f.interval:g} us)")
    L.append(f"the two windows: {v.period:,.1f} cycles in the stamps, {f.period_us:.3f} us in the samples: "
             f"{v.period / f.period_us:,.0f} cycles an us ({v.period / f.period_us / 1000:.3f} GHz), a measurement of "
             f"THIS PAIR of launches (stamps and samples come from different launches)")
    if v.axis == "ns":
        L.append(f"the stamps' own window in time, from their ties: {v.period_ns / 1000:.3f} us against "
                 f"{f.period_us:.3f} us in the samples ({100 * (v.period_ns / 1000 / f.period_us - 1):+.2f}%)")
    L.append(f"clock used: {v.axis_note}")
    L.append(f"blur: a sample is a total over its interval: the fold is the profile smoothed by a box {f.blur:.3f} of a "
             f"window ({f.blur * n:.2f} of {n} bins)")
    L += placement_lines(pl)
    L.append("")
    var = pl.varying()
    flat = [x for x in f.names if x not in var]
    cols = [(pl.series[x].short + ("~" if pl.series[x].loose else ""), pl.series[x].bins) for x in var]
    L.append("OVERLAY: per window position (the launch's edges place the samples), each role's phase (and the share of "
             "its lanes in it), the stamps' mean lanes in a wait (smoothed, where the record has waits), and each "
             "varying stall reason's value (warps); ~ marks a LOOSE placement")
    L += table_lines(v, n, cols, pl.profile)
    if flat:
        L.append("flat reasons (not shown): " + ", ".join(_short(x) for x in flat))
    return "\n".join(L) + "\n"


def place_to_dict(pl: Placement) -> dict:
    f, n = pl.folded, pl.bins
    return {
        "launch": pl.launch.to_dict(), "bins": n, "blur": f.blur,
        "passes": {k: {"series": pp.names, "placed": pp.placed, "u_lo_us": pp.lo, "u_hi_us": pp.hi,
                       "u_mid_us": pp.mid if pp.placed else None, "half_us": pp.half if pp.placed else None,
                       "runs_fed": pp.fed, "narrowing": [[a, b] for a, b in pp.steps], "findings": pp.findings,
                       "checks": [{"run": c.run, "i0": c.i0, "i1": c.i1, "dt_us": c.dt, "length_us": c.length_us,
                                   "diff_us": c.entry.diff, "ok": c.entry.ok} for c in pp.checks]}
                   for k, pp in pl.edges.items()},
        "series": {s.short: {"pass": _short(s.key), "half_us": s.half_us, "half_windows": s.half_windows,
                             "half_bins": s.half_bins, "loose": s.loose, "runs": s.fed, "more_runs": s.more,
                             "bins": [{"mean": b.mean, "se": b.se, "n": b.n} for b in s.bins]}
                   for s in pl.series.values()},
        "cross_check": {"note": pl.cross.note, "series": pl.cross.series, "agree": pl.cross.agree,
                        "old_origin": pl.cross.old_origin, "edge_origin": pl.cross.edge_origin},
        "findings": pl.findings,
        "profile": pl.profile,
        "overlay": {"bins": [{"position": [b / n, (b + 1) / n],
                              "phase": {r: {"phase": phases(pl.view, n)[1][b][i][0], "share": phases(pl.view, n)[1][b][i][1]}
                                        for i, r in enumerate(phases(pl.view, n)[0])},
                              "reasons": {pl.series[x].short: {"mean": pl.series[x].bins[b].mean,
                                                               "se": pl.series[x].bins[b].se,
                                                               "n": pl.series[x].bins[b].n}
                                          for x in pl.varying()}} for b in range(n)]},
    }


def perfetto_doc(view: View, placed: dict[str, list[sampling.Bin]], n: int, note: dict) -> dict:
    """The stamp lanes of the average window as `perfetto.build` writes them, plus one counter track a series over one
    average window (a value a bin, at the bin's start)."""
    from . import perfetto
    rec = view.rec
    doc = perfetto.build(rec)
    hz = rec.clock.get("hz")
    scale = 1e6 / hz if hz else 1.0
    pid = max((e.get("pid", 0) for e in doc["traceEvents"]), default=0) + 1
    ev = [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
           "args": {"name": f"sampled counters placed by the launch's edges, one average window ({view.scope})"}}]
    P = view.period
    for x, bins in placed.items():
        name = _short(x)
        for b in range(n):
            ev.append({"ph": "C", "name": name, "pid": pid, "ts": b / n * P * scale, "args": {name: bins[b].mean}})
        ev.append({"ph": "C", "name": name, "pid": pid, "ts": P * scale, "args": {name: bins[-1].mean}})
    doc["traceEvents"] += ev
    doc["otherData"]["sampled_layer"] = {
        "bins": n, **note,
        "note": "a fold of samples from other launches, an average window, smoothed by the sampling interval"}
    return doc
