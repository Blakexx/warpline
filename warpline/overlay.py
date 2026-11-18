"""The sampled layer laid under the stamps.

From a stamp record's average window (`window.py`) this builds what the samples should see: for each window position,
the mean number of lanes inside a `barrier.*.wait` region, smoothed by the same box as the samples' blur. THE SHIFT is
the rotation of the samples' `barrier` fold that maximizes its correlation with that profile. It is REFUSED (a named
finding, no overlay) below a correlation of 0.7 or when the runner-up shift is within 0.05: an alignment that could be
another is none. With an accepted shift, each window position shows each role's phase (the phase most of the role's
lanes are in) beside each varying stall reason's folded value.

A record without waits cannot be aligned this way: the command says so and stops, naming what is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import record as rec_mod
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


# ------------------------------------------------------------------------------------------------ output

def _short(s: str) -> str:
    return sampling.Series(s, "", "", [], 0).short


def format_overlay(a: Alignment, stamps_path: str, sample_paths: list[str]) -> str:
    v, f, sh, n = a.view, a.folded, a.shift, a.bins
    L = []
    L.append(f"stamps: {stamps_path}: scope {v.scope}, anchor `{v.anchor}`, window {v.period:,.1f} cycles over "
             f"{v.windows} steady windows; {a.n_waits} lane-waits in the average window")
    ru = sorted({i for x in f.runs_of.values() for i in x})
    L.append(f"samples: {len(sample_paths)} run(s), runs {ru} in the average; window {f.period:.4f} samples = "
             f"{f.period_us:.3f} us (interval {f.interval:g} us)")
    L.append(f"the two windows: {v.period:,.1f} cycles in the stamps, {f.period_us:.3f} us in the samples: "
             f"{v.period / f.period_us:,.0f} cycles an us ({v.period / f.period_us / 1000:.3f} GHz), a measurement of "
             f"THIS PAIR of launches (stamps and samples come from different launches); the overlay uses window "
             f"position alone")
    if v.axis == "ns":
        L.append(f"the stamps' own window in time, from their ties: {v.period_ns / 1000:.3f} us against "
                 f"{f.period_us:.3f} us in the samples ({100 * (v.period_ns / 1000 / f.period_us - 1):+.2f}%)")
    L.append(f"clock used: {v.axis_note}")
    L.append(f"blur: the profile is smoothed by a box {a.blur:.3f} of a window ({a.blur * n:.2f} of {n} bins), the "
             f"samples' own")
    L.append(f"series aligned: `{a.series}` ({_short(a.series)}), score {f.score.get(a.series, 0):.3f}")
    L.append("")
    L.append(f"{'shift':>5} {'corr':>7}   (rotated[b] = fold[(b + shift) mod {n}])")
    for s, c in enumerate(sh.corrs):
        mark = "  <- best" if s == sh.shift else ("  <- runner-up" if s == sh.runner_shift else "")
        L.append(f"{s:5d} {('n/a' if c is None else f'{c:7.3f}'):>7}{mark}")
    corr = "n/a" if sh.corr is None else f"{sh.corr:.3f}"
    ru_ = ("none" if sh.runner_corr is None else f"shift {sh.runner_shift}, correlation {sh.runner_corr:.3f} (the best "
           f"at {sampling.guard_for(a.blur, n)} or more bins from the best: nearer ones are the same alignment smeared by the blur)")
    L.append(f"SHIFT {sh.shift} bins ({sh.shift / n:.3f} of a window); correlation {corr}; runner-up {ru_}")
    if not sh.accepted:
        L.append(f"ALIGNMENT REFUSED: {sh.why}. No overlay: an alignment that could be another is none "
                 f"(bounds: correlation >= {f.cfg.min_corr}, runner-up more than {f.cfg.min_gap} below).")
        return "\n".join(L) + "\n"
    L.append(f"ACCEPTED: the start of the first live sample's interval falls at window position {a.origin:.3f}")
    L.append("")
    roles, ph = phases(v, n)
    var = sorted(f.varying(), key=lambda x: -f.swing(x))
    flat = [x for x in f.names if x not in var]
    head = f"{'position':<13} {'lanes in wait':>13} " + " ".join(f"{r[:17]:>17}" for r in roles) + "  " + " ".join(
        f"{_short(x)[:14]:>14}" for x in var)
    L.append("OVERLAY: per window position, each role's phase (and the share of its lanes in it), the stamps' mean lanes "
             "in a wait (smoothed), and each varying stall reason's folded value (warps)")
    L.append(head)
    for b in range(n):
        cells = [f"{p[:12]} {100 * s:3.0f}%" for p, s in ph[b]]
        vals = [f"{a.rotated(x)[b].mean:14.2f}" for x in var]
        L.append(f"{f'{b / n:.3f}-{(b + 1) / n:.3f}':<13} {a.profile[b]:13.2f} " + " ".join(f"{c:>17}" for c in cells)
                 + "  " + " ".join(vals))
    if flat:
        L.append("flat reasons (not shown): " + ", ".join(_short(x) for x in flat))
    return "\n".join(L) + "\n"


def to_dict(a: Alignment, stamps_path: str) -> dict:
    v, f, sh, n = a.view, a.folded, a.shift, a.bins
    out = {
        "stamps": {"path": stamps_path, "scope": v.scope, "anchor": v.anchor, "period_cycles": v.period,
                   "period_ns": v.period_ns, "axis": v.axis, "axis_note": v.axis_note, "windows": v.windows, "lane_waits": a.n_waits},
        "samples": {"period_samples": f.period, "period_us": f.period_us, "interval_us": f.interval,
                    "flags": f.flags},
        "blur": a.blur, "bins": n, "series": a.series,
        "profile": a.profile, "fold": a.fold,
        "alignment": {"shift": sh.shift, "corr": sh.corr, "runner_up_shift": sh.runner_shift,
                      "runner_up_corr": sh.runner_corr, "runner_up_guard_bins": sampling.guard_for(a.blur, n), "accepted": sh.accepted, "finding": sh.why,
                      "corrs": sh.corrs, "bounds": {"min_corr": f.cfg.min_corr, "min_gap": f.cfg.min_gap}},
        "overlay": None,
    }
    if sh.accepted:
        roles, ph = phases(v, n)
        out["alignment"]["samples_phase0_at"] = a.origin
        out["overlay"] = {
            "roles": roles,
            "bins": [{"position": [b / n, (b + 1) / n], "lanes_in_wait": a.profile[b],
                      "phase": {r: {"phase": ph[b][i][0], "share": ph[b][i][1]} for i, r in enumerate(roles)},
                      "reasons": {_short(x): {"mean": a.rotated(x)[b].mean, "se": a.rotated(x)[b].se,
                                              "n": a.rotated(x)[b].n}
                                  for x in f.varying()}} for b in range(n)]}
    return out


def perfetto_doc(a: Alignment) -> dict:
    """The stamp lanes of the average window as `perfetto.build` writes them, plus one counter track a varying
    reason over one average window (a value a bin, at the bin's start)."""
    from . import perfetto
    rec = a.view.rec
    doc = perfetto.build(rec)
    hz = rec.clock.get("hz")
    scale = 1e6 / hz if hz else 1.0
    pid = max((e.get("pid", 0) for e in doc["traceEvents"]), default=0) + 1
    ev = [{"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
           "args": {"name": f"sampled counters, one average window ({a.view.scope}); samples' phase 0 at "
                            f"{a.origin:.3f}"}}]
    P, n = a.view.period, a.bins
    for x in a.folded.varying():
        bins = a.rotated(x)
        name = _short(x)
        for b in range(n):
            ev.append({"ph": "C", "name": name, "pid": pid, "ts": b / n * P * scale,
                       "args": {name: bins[b].mean}})
        ev.append({"ph": "C", "name": name, "pid": pid, "ts": P * scale, "args": {name: bins[-1].mean}})
    doc["traceEvents"] += ev
    doc["otherData"]["sampled_layer"] = {
        "blur_fraction_of_window": a.blur, "bins": n, "shift": a.shift.shift, "correlation": a.shift.corr,
        "note": "a fold of samples from another launch, an average window, smoothed by the sampling interval"}
    return doc
