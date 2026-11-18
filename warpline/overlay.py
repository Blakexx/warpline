"""The sampled layer laid under the stamps, JOINED ON CYCLES, each pass scaled by its own two edges (`placement.py`).

The kernel stamps `launch.begin` at its entry and `launch.end` at its exit (see `launch.py`); the samples see the same
two moments as the series going live and going quiet. A profiler's time sampling stretches the kernel in real time and
leaves its cycles alone, so a sample is placed at the cycle f * T past the entry (f its fraction of the pass's own two
edges, T the record's launch in cycles) and its window position is read from a frame of the record's window anchors (a
local fit by default: the anchors' window-to-window jitter is not the samples' rhythm). A pass's runs are fitted JOINTLY
to one pooled profile; a series is folded at the placement of its own pass and carries the placement's half-width: the
edges' box of common shifts (a hard bound) and the spread of its profile's phase measured by resampling the runs.

A record without both edges (with ties) is REFUSED, naming what the kernel must stamp. Where the record has barrier
waits, the placed `barrier` fold is compared with the waits (a CROSS-CHECK).

The table: a row a window bin; a column a ROLE (the phase most of its lanes are in, each lane in its own window: see
`window.role_windows`) and a column a varying stall reason (its folded value with its standard error); a footer with
each series' placement half-width.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import record as rec_mod
from . import launch as launch_mod
from . import placement, rates, sampling, window
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
    src: Record | None = None          # the stamp record itself (each ROLE's own window is read from it)

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
        return View(sc, sw.period, sw.windows, sw.anchor, rows, lanes, window.to_record(av), source, src=rec)
    tw = tv.scopes[sc]
    rows = [VRow(r.lane, rec.lanes[r.lane].role, r.name, r.occ, r.mean, r.n) for r in tw.rows]
    note = (f"window position = nanoseconds from the window's start over the period in nanoseconds, by the record's "
            f"ties (an untied stamp is interpolated between its lane's ties; one outside them is not placed); cycles "
            f"were NOT used")
    return View(sc, sw.period, sw.windows, sw.anchor, rows, lanes, window.to_record(av), source, axis="ns",
                period_ns=tw.period, axis_note=note, src=rec)


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


def _smooth(cells: list[float], width: float, start: float = 0.0) -> list[float]:
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
    return [(F(i + 0.5 + start + width) - F(i + 0.5 + start)) / width for i in range(m)]


def wait_profile(view: View, n_bins: int, blur: float, centered: bool = False) -> list[float]:
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
    fine = _smooth(cells, blur * m, -0.5 * blur * m if centered else 0.0)
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

def launch_for(rec: Record, scope: str | None = None, need_windows: bool = True) -> launch_mod.Launch:
    """The record's launch, or an OverlayError naming what the kernel must stamp. `need_windows=False` takes a record
    that has the launch's two edges and no window anchors (an EDGES variant: its windows are another record's)."""
    try:
        return launch_mod.launch_of(rec, scope, need_windows=need_windows)
    except launch_mod.LaunchError as e:
        raise OverlayError(str(e))


@dataclass
class Cross:
    series: str | None
    note: str
    rotation: float | None = None       # bins: the placed fold lies this much later than the waits' profile
    corr: float | None = None
    tol: float | None = None            # bins
    agree: bool | None = None


@dataclass
class Placement:
    view: View                          # whose stamps supply the roles' phases (the stamped variant's)
    launch: Launch                      # this variant's own launch: its T in cycles
    vp: placement.VariantPlacement
    bins: int
    findings: list[str]
    cross: Cross
    profile: list[float] | None = None
    cps_tol: float = 0.03
    windows_from: str = ""              # what the windows (as fractions of the launch) were read from
    roles: tuple | None = None          # (roles, per-bin phases) in each role's own window

    @property
    def series(self) -> dict[str, placement.SeriesPlace]:
        return self.vp.series()

    @property
    def edges(self) -> dict[str, placement.PassPlacement]:
        return self.vp.passes

    def values(self) -> dict[str, list[sampling.Bin]]:
        return {n: s.bins for n, s in self.series.items()}

    def varying(self) -> list[str]:
        return self.vp.varying()


def _circ(a: float, b: float) -> float:
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def role_phases(view: View, n_bins: int) -> tuple[list[str], list[list[tuple[str, float]]]]:
    """Per window bin (of the SCOPE's window), per role: the phase most of the role's lanes are in at the bin's centre,
    each lane in its OWN window (`window.role_windows`: the role's anchor event where it has one), and the share of the
    lanes in it (`-`: none). A view with no stamp record behind it falls back to the anchor lane's window."""
    if view.src is None:
        return phases(view, n_bins)
    try:
        rws = window.role_windows(view.src, view.scope, None, window.SKIP)
    except window.WindowError:
        return phases(view, n_bins)
    roles = [rw.role for rw in rws]
    out = []
    for b in range(n_bins):
        x = (b + 0.5) / n_bins * view.period
        row = []
        for rw in rws:
            count: dict[str, int] = {}
            for ln in rw.lanes:
                u = (x - rw.lane_offset.get(ln, 0.0)) % rw.period
                for name, st, du in rw.lane_phases.get(ln, []):
                    if (u - st) % rw.period < du:
                        count[name] = count.get(name, 0) + 1
                        break
            if count:
                name = max(sorted(count), key=lambda k: count[k])
                row.append((name.removeprefix("phase."), count[name] / len(rw.lanes)))
            else:
                row.append(("-", 1.0))
        out.append(row)
    return roles, out


def place(view: View, lc: launch_mod.Launch, exports: list[sampling.Export], cfg: sampling.Config,
          frame_launch: launch_mod.Launch | None = None, frame_source: str = "",
          frame_fracs: list[float] | None = None) -> Placement:
    """The samples of one variant placed on CYCLES: `lc` is the variant's own launch (T in cycles), `frame_launch` the
    launch whose window anchors give the windows as fractions of the launch (its own, or a named stamped variant's);
    `frame_fracs` those fractions where they are not the launch's own (an edges variant's: see `Launch.fractions_on`)."""
    fl = frame_launch or lc
    if not fl.has_windows:
        raise OverlayError("the record carries the launch's two edges and no window anchors: its windows are another "
                           "record's (name a stamped variant of the same kernel: `windows_from` in a manifest)")
    frame = placement.Frame(frame_fracs or fl.fractions, frame_source or "this record's own window anchors")
    vp = placement.place_variant(exports, lc.cycles, frame, cfg)
    findings = [f"launch edge: {x}" for x in lc.findings] + list(vp.findings)
    pl = Placement(view, lc, vp, cfg.bins, findings, Cross(None, ""), None, cfg.cps_tol, frame_source or frame.source)
    pl.cross = _cross(view, pl, cfg.bins)
    if pl.cross.agree is False:
        findings.append(f"cross_check_disagrees: {pl.cross.note}")
    if pl.cross.series is not None and pl.cross.agree is not None:
        pl.profile = _profile_for(view, pl, cfg.bins)
    pl.roles = role_phases(view, cfg.bins)
    return pl


def _barrier_series(pl: Placement) -> str | None:
    s = pl.series
    for n in pl.varying():
        if n.endswith("stalled_barrier.avg"):
            return n
    return None


def _profile_for(view: View, pl: Placement, n: int) -> list[float] | None:
    name = _barrier_series(pl)
    if name is None:
        return None
    try:
        return wait_profile(view, n, pl.vp.passes[pl.series[name].key].kappa, centered=True)
    except OverlayError:
        return None


def _cross(view: View, pl: Placement, n: int) -> Cross:
    """The old correlation as a CROSS-CHECK: where the record has barrier waits, the placed `barrier` fold against the
    waits' profile (smoothed by the samples' box)."""
    if not waits(view):
        return Cross(None, "no cross-check: the stamp record has no barrier waits in its average window (a fine record "
                           "has them)")
    name = _barrier_series(pl)
    if name is None:
        return Cross(None, "no cross-check: no `barrier` series varies in the placed fold")
    sp = pl.series[name]
    prof = _profile_for(view, pl, n)
    fold = [b.mean for b in sp.bins]
    rot = placement.fold_rotation(prof, fold)
    corr = sampling.pearson(prof, fold)
    if rot is None or corr is None:
        return Cross(name, "no cross-check: a profile is flat")
    tol = sp.half * n + 1.0
    agree = abs(rot) <= tol + 1e-9
    note = (f"on `{_short(name)}`: the placed fold lies {rot:+.2f} bins from the stamps' barrier waits (correlation "
            f"{corr:.3f}); bound the placement's half-width {sp.half * n:.2f} bins + one bin = {tol:.2f}: "
            + ("agree" if agree else "DISAGREE"))
    return Cross(name, note, rot, corr, tol, agree)


# ------------------------------------------------------------------------------------------------ output

def _short(s: str) -> str:
    return sampling.Series(s, "", "", [], 0).short


def table_lines(view: View, n: int, cols: list[tuple[str, list[sampling.Bin]]], profile: list[float] | None,
                rp: tuple | None = None) -> list[str]:
    """Per window position: each role's phase (and the share of its lanes in it; each lane in its own window), the
    stamps' mean lanes in a wait (when the record has waits), and each column's folded value with its standard error."""
    roles, ph = rp or role_phases(view, n)
    head = f"{'position':<13} " + (f"{'lanes in wait':>13} " if profile else "") + " ".join(
        f"{r[:17]:>17}" for r in roles) + "  " + " ".join(f"{label[:17]:>17}" for label, _b in cols)
    out = [head]
    for b in range(n):
        cells = [f"{p[:12]} {100 * s:3.0f}%" for p, s in ph[b]]
        vals = [(f"{bins[b].mean:7.2f} +-{bins[b].se:5.2f}" if bins is not None else f"{'-':>17}") for _l, bins in cols]
        out.append(f"{f'{b / n:.3f}-{(b + 1) / n:.3f}':<13} " + (f"{profile[b]:13.2f} " if profile else "")
                   + " ".join(f"{c:>17}" for c in cells) + "  " + " ".join(f"{v:>17}" for v in vals))
    return out


def _tag(s: placement.SeriesPlace, n: int) -> str:
    t = ""
    if s.half * n > 0.5 + 1e-9:
        t += "~"
    if s.ambiguous is not None:
        t += "?"
    if s.binds:
        t += "!"
    return t


def footer_lines(pl: Placement) -> list[str]:
    """Each series' placement half-width: the edges' box (a hard bound) and the measured spread of the profile's phase
    in quadrature, in windows and bins, and the runs that fed it."""
    n = pl.bins
    out = ["placement half-width of each column (windows; bins of 1/%d; ~ = over half a bin; ? = AMBIGUOUS BY A SAMPLE; ! = EDGES BIND):" % n]
    for name in pl.varying():
        s = pl.series[name]
        pp = pl.edges[s.key]
        sp = "not measurable" if s.spread is None else f"spread {s.spread:.4f} windows = {s.spread * n:.2f} bins"
        out.append(f"  {s.short + _tag(s, n):<24} {s.half:.4f} windows = {s.half * n:.2f} bins "
                   f"(box +-{s.half_a:.4f}, {sp}); {s.fed} / {len(pp.runs)} runs"
                   + (f"; the other candidate {s.ambiguous:+.2f} bins" if s.ambiguous is not None else ""))
    return out


def frame_lines(vp: placement.VariantPlacement) -> list[str]:
    """The sharpness of every pass on every frame, and the frame chosen."""
    keys = [k for k, pp in vp.passes.items() if pp.fit_names]
    L = [f"windows: {vp.frame.label}: {vp.frame_why}"]
    L.append("sharpness of the pooled joint fit (share of the pooled samples' variance in the first 3 harmonics, summed over a "
             "pass's series) by frame:")
    L.append(f"  {'frame':<32}" + "".join(f"{_short(k)[:14]:>16}" for k in keys) + f"{'sum':>8}")
    for lab, q in vp.frame_q.items():
        mark = " <- the frame" if lab == vp.frame.label else ""
        L.append(f"  {lab:<32}" + "".join(f"{q.get(k, 0.0):16.3f}" for k in keys) + f"{sum(q.get(k, 0.0) for k in keys):8.3f}" + mark)
    return L


def placement_lines(pl: Placement) -> list[str]:
    """The placement's own report: the launch, the frame, each pass's cycles a sample, the joint fit's runs, the box,
    the measured spread and the alias check, each series' half-width, the cross-check and the findings."""
    n = pl.bins
    lc, vp = pl.launch, pl.vp
    L = [lc.describe()]
    L.append(f"windows from: {pl.windows_from}; the frame's {len(vp.frame.fracs) - 1} steady windows, mean "
             f"{vp.frame.mean:.5f} of the launch")
    L += frame_lines(vp)
    L.append("THE JOIN IS ON CYCLES. A profiler's time sampling stretches the kernel in real time and leaves its cycles "
             "unchanged, so a pass is scaled by its OWN two edges: its entry at sample-time i0 + a, its exit at i1 + b "
             "(a, b in [0, 1) unknown); sample j sits at the fraction f = ((j + 0.5) - (i0 + a)) / ((i1 + b) - (i0 + a)) "
             "of the launch, at cycle f * T past the entry, and its window position is read from the frame (above)")
    L.append("STATED ASSUMPTION: over a pass the profiler's time is linear in the kernel's cycles (the stretch is even "
             "over a window and over the launch, not within a phase: a sample's position inside a phase is as good as "
             "that, and no better)")
    L.append(f"THE FIT of a pass is JOINT: every run's entry offset a and scale d (inside what its two edges allow) are "
             f"fitted at once to one pooled profile, maximizing the pooled fold's sharpness (first {placement.HARMONICS} "
             f"harmonics); no run is left out for a weak fold. The objective is blind to a common shift of every run: "
             f"the edges' BOX (the interval of common shifts that keep every entry inside its first live sample and every "
             f"exit inside its last) bounds it and the placement is its middle. THE SPREAD is measured by refitting "
             f"{vp_boot(vp)} bootstrap resamples of the runs and every leave-one-run-out, and reporting the spread of "
             f"each series' profile phase; bins of {1.0 / n:.3f} of a window")
    L.append(f"CYCLES A SAMPLE (T = {vp.t_cycles:,.0f} cycles over the pass's live span), median over the variant's "
             f"passes {vp.cps_median:,.1f}; a pass further than {100 * pl.cps_tol:g}% from it is flagged and left out")
    for key, pp in vp.passes.items():
        L.append("")
        L.append(f"pass `{_short(key)}` ({len(pp.names)} series: {', '.join(_short(x) for x in pp.names)}; fitted: "
                 f"{', '.join(_short(x) for x in pp.fit_names) or 'none'})")
        for r, rp in pp.runs.items():
            tag = "  [FLAGGED, left out]" if rp.flagged else ""
            fit = (f"; placed at a = {rp.a:.3f}, d = {rp.d:+.2f} samples (scale within one noise sigma: "
                   f"[{rp.d_plateau[0]:+.1f}, {rp.d_plateau[1]:+.1f}]), its own sharpness {rp.share:.2f}"
                   if rp.a is not None else "")
            L.append(f"  run {r}: live {rp.p.live} samples ({rp.p.i0}..{rp.p.i1}); {rp.cps:,.1f} cycles a sample "
                     f"({100 * (rp.ratio - 1):+.2f}% from the median){tag}{fit}")
            if rp.note:
                L.append(f"      {rp.note}")
        if pp.placed:
            L.append(f"  pooled sharpness {pp.q:.3f}; box of the common shift [{pp.box[0]:+.4f}, {pp.box[1]:+.4f}] windows "
                     f"= half-width {pp.half_a:.4f} windows = {pp.half_a * n:.2f} bins; {len(pp.fed)} run(s) fed it "
                     f"{pp.fed}" + ("; THE BOX IS EMPTY: the runs' scales leave no common shift that keeps every entry "
                                    "and exit inside its live sample (the fit and the edges disagree)" if pp.tight else ""))
            for nm in pp.fit_names:
                sp = pp.spread.get(nm)
                if sp is None:
                    L.append(f"  spread of `{_short(nm)}`'s profile phase: not measurable ({len(pp.fed)} run(s))")
                else:
                    L.append(f"  spread of `{_short(nm)}`'s profile phase: {sp:.4f} windows = {sp * n:.2f} bins (bootstrap sd "
                             f"{pp.spread_boot[nm] * n:.2f} bins, leave-one-run-out {pp.spread_loo[nm] * n:.2f} bins; "
                             f"{pp.boot} resamples)")
            if pp.alias:
                L.append("  ALIAS CHECK, the pooled objective at the placement against its value with a run (or every run) "
                         "shifted (best minus candidate, margin = twice the resampled sd of the difference; a candidate is "
                         "WITHIN the margin when it is under it):")
                for a in pp.alias:
                    mv = ", ".join(f"{_short(k)} {v:+.2f}" for k, v in a.moves.items())
                    mg = "" if a.margin is None else f", margin {a.margin:.4f}"
                    L.append(f"      {a.label:<22} objective {a.q:.4f}, difference {a.dq:+.4f}{mg}"
                             f"{'  WITHIN' if a.within else ''}; profile moves (bins): {mv}"
                             + (f"  [{a.note}]" if a.note else ""))
        for x in pp.findings:
            L.append(f"  FINDING {x}")
    L.append("")
    L.append("PLACEMENT of each series (half-width = the box's and SPREAD_Z=%g spreads, in quadrature; ~ LOOSE when over half "
             "a bin; AMBIGUOUS BY A SAMPLE when named):" % placement.SPREAD_Z)
    for name in pl.varying() + [x for x in pl.series if x not in pl.varying()]:
        s = pl.series[name]
        sp = "spread not measurable" if s.spread is None else f"spread {s.spread * n:.2f} bins"
        L.append(f"  {s.short:<22} pass `{_short(s.key)}`: half-width {s.half:.4f} windows = {s.half * n:.2f} bins "
                 f"(box +-{s.half_a * n:.2f} bins, {sp}); {s.fed} run(s) fed it; "
                 f"{'varies' if s.varying else 'flat'}" + ("  LOOSE" if s.half * n > 0.5 + 1e-9 else "")
                 + (f"  AMBIGUOUS BY A SAMPLE (other candidate {s.ambiguous:+.2f} bins)" if s.ambiguous is not None else "")
                 + ("  EDGES BIND" if s.binds else ""))
    L.append("")
    L.append(f"CROSS-CHECK: {pl.cross.note}")
    extra = [x for x in pl.findings if not any(x in pp.findings for pp in pl.edges.values())]
    for x in extra:
        L.append(f"FINDING {x}")
    return L


def vp_boot(vp: placement.VariantPlacement) -> int:
    return max((pp.boot for pp in vp.passes.values()), default=0)


def overlay_table(pl: Placement) -> list[str]:
    n = pl.bins
    var = pl.varying()
    cols = [(pl.series[x].short + _tag(pl.series[x], n), pl.series[x].bins) for x in var]
    L = ["OVERLAY: per window position (the samples placed on cycles by the pass's own two edges), each role's phase "
         "(in the role's own window; the share of its lanes in it), the stamps' mean lanes in a wait (where the record "
         "has waits), and each varying stall reason's folded value +- its standard error (warps); ~ marks a placement "
         "over half a bin, ? a placement AMBIGUOUS BY A SAMPLE, ! a placement where the EDGES BIND"]
    L += table_lines(pl.view, n, cols, pl.profile, pl.roles)
    L += footer_lines(pl)
    flat = [s.short for x, s in pl.series.items() if x not in var]
    if flat:
        L.append("flat reasons (not shown): " + ", ".join(flat))
    return L


def format_overlay(pl: Placement, stamps_path: str, sample_paths: list[str]) -> str:
    v = pl.view
    L = [f"stamps: {stamps_path}: scope {v.scope}, anchor `{v.anchor}`, window {v.period:,.1f} cycles over "
         f"{v.windows} steady windows",
         f"samples: {len(sample_paths)} run(s)"]
    L += placement_lines(pl)
    L.append("")
    L += overlay_table(pl)
    return "\n".join(L) + "\n"


def place_to_dict(pl: Placement) -> dict:
    n = pl.bins
    roles, ph = pl.roles
    return {
        "launch": pl.launch.to_dict(), "bins": n, "windows_from": pl.windows_from,
        "cycles_a_sample": {"median": pl.vp.cps_median, "tolerance": pl.cps_tol,
                            "checks": [{"run": c.run, "pass": c.key, "live": c.live, "cycles_a_sample": c.cps,
                                        "ratio": c.ratio, "flagged": c.flagged} for c in pl.vp.checks]},
        "assumption": "over a pass the profiler's time is linear in the kernel's cycles",
        "frame": {"chosen": vp_frame(pl), "why": pl.vp.frame_why, "sharpness": pl.vp.frame_q},
        "passes": {k: {"series": pp.names, "fitted": pp.fit_names, "placed": pp.placed,
                       "box_windows": list(pp.box) if pp.box else None, "pooled_sharpness": pp.q,
                       "runs_fed": pp.fed, "findings": pp.findings, "resamples": pp.boot,
                       "alias": [{"label": a.label, "objective": a.q, "difference": a.dq, "margin": a.margin,
                                  "within": a.within, "moves_bins": {_short(x): v for x, v in a.moves.items()}}
                                 for a in pp.alias],
                       "runs": {r: {"live": rp.p.live, "i0": rp.p.i0, "i1": rp.p.i1, "cycles_a_sample": rp.cps,
                                    "flagged": rp.flagged, "d": rp.d, "a": rp.a}
                                for r, rp in pp.runs.items()}}
                   for k, pp in pl.edges.items()},
        "series": {s.short: {"pass": _short(s.key), "half_windows": s.half, "half_bins": s.half * n,
                             "box_half_windows": s.half_a, "spread_windows": s.spread,
                             "spread_bins": None if s.spread is None else s.spread * n, "varying": s.varying,
                             "ambiguous_by_a_sample_bins": s.ambiguous, "edges_bind": s.binds,
                             "runs_fed": s.fed, "loose": s.half * n > 0.5 + 1e-9,
                             "bins": [{"mean": b.mean, "se": b.se, "n": b.n} for b in s.bins]}
                   for s in pl.series.values()},
        "cross_check": {"note": pl.cross.note, "series": pl.cross.series, "agree": pl.cross.agree,
                        "rotation_bins": pl.cross.rotation, "corr": pl.cross.corr},
        "findings": pl.findings,
        "profile": pl.profile,
        "overlay": {"bins": [{"position": [b / n, (b + 1) / n],
                              "phase": {r: {"phase": ph[b][i][0], "share": ph[b][i][1]} for i, r in enumerate(roles)},
                              "reasons": {pl.series[x].short: {"mean": pl.series[x].bins[b].mean,
                                                               "se": pl.series[x].bins[b].se,
                                                               "n": pl.series[x].bins[b].n}
                                          for x in pl.varying()}} for b in range(n)]},
    }


def vp_frame(pl: Placement) -> str:
    return pl.vp.frame.label


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
