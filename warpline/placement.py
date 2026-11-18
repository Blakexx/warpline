"""The samples joined to the stamps on CYCLES: a pass is scaled by its own two edges.

A profiler's time sampling stretches the kernel in real time and leaves its cycles unchanged, and the stretch is even
over a window and over the launch but not within a phase. So a sample is placed by the kernel's own cycle counter, not
by a mean period of the samples:

  A PASS   the series that were collected together (one execution of the kernel): its first live sample i0 and its last
           i1 (live: any series of the pass over `cfg.live`). The kernel's ENTRY is at sample-time i0 + a and its EXIT at
           i1 + b, a and b in [0, 1) unknown, so its length in samples is S = (i1 - i0) + d with d = b - a in (-1, 1).
  THE JOIN sample j's centre is then the fraction f = ((j + 0.5) - (i0 + a)) / S of the launch, and the record gives
           the launch in cycles (T, entry to exit): the sample sits at cycle c = f * T past the entry. Its window
           position is read from the record's own window anchors (c falls between two consecutive anchors: the fraction
           of that window), never from one mean period (the period jitters). STATED ASSUMPTION: over a pass the
           profiler's time is linear in the kernel's cycles (even over a window and a launch, not within a phase).
  THE CHECK that replaces the length check: a pass's cycles a sample, T over its live span, against the median of the
           variant's passes (default 3%).
  THE SCALE d is fixed by the fold's sharpness (the share of the pass's variance the first harmonics of the fold
           explain, summed over its series), scanned over the range the edges allow; its plateau (within one standard
           deviation of the score's own sampling noise of the best) is carried as the scale's uncertainty.
  THE OFFSET a is not seen by one pass. Runs of the same series tie it: each run's fold is the same profile rotated by
           its own a (times the windows a sample spans); the rotation of each run against the first is estimated to a
           fraction of a bin (the phase of the first harmonics of the samples at their placed positions, with its
           bootstrap error), and the runs' intervals for a are intersected through those rotations.
"""
from __future__ import annotations

import bisect
import cmath
import math
import random
import zlib
from dataclasses import dataclass, field

from . import sampling
from .sampling import Bin, Config, Export, fold_bins, pool_bins

HARMONICS = 3          # the blur (a sample is a total over an interval) leaves little above the third
SCAN_STEP = 0.025      # samples
ROT_GRID = 720         # the rotation's grid over one window (then refined by a parabola)
BOOT = 40              # bootstrap resamples of the rotation
ROT_Z = 2.0            # the rotation's slack in the intersection: this many bootstrap standard deviations
MIN_SERIES = 0.25      # a series is informative to a pass's scale when its best harmonic share is at least this


MEAN_NOTE = "a mean period over the same span (the anchors' jitter is not followed)"


class PlacementError(Exception):
    pass


# ------------------------------------------------------------------------------------------------------- passes

@dataclass
class Pass:
    key: str                 # the pass's name across runs: its smallest series name
    names: list[str]
    i0: int                  # first live sample
    i1: int                  # last live sample
    dt: float                # the sampling interval of this pass, microseconds

    @property
    def n(self) -> int:
        return self.i1 - self.i0

    @property
    def live(self) -> int:
        """The live span in samples."""
        return self.i1 - self.i0 + 1


def passes_of(exports: list[Export], live: float = 0.0) -> list[dict[str, Pass]]:
    """Per run, {pass key: Pass}. The series that share a sample count were collected together; two passes can have
    the same count by chance, which would merge them (and move the first and last live sample), so the partition of
    the series into passes is the COMMON REFINEMENT of the runs' own: two series are one pass only when they have one
    sample count in every run of the variant (the profiler assigns the same counters to a pass every time)."""
    sig: dict[str, tuple] = {}
    for e in exports:
        for s in e.series:
            sig.setdefault(s.name, ())
    for name in list(sig):
        sig[name] = tuple(next((s.group for s in e.series if s.name == name), None) for e in exports)
    classes: dict[tuple, list[str]] = {}
    for name, g in sig.items():
        classes.setdefault(g, []).append(name)
    out: list[dict[str, Pass]] = []
    for r, e in enumerate(exports):
        by = {s.name: s for s in e.series}
        groups = sorted({s.group for s in e.series})
        res: dict[str, Pass] = {}
        for names in classes.values():
            names = sorted(names)
            first = last = None
            for nm in names:
                s = by.get(nm)
                if s is None:
                    continue
                idx = [i for i, v in enumerate(s.values) if abs(v) > live]
                if idx:
                    first = idx[0] if first is None else min(first, idx[0])
                    last = idx[-1] if last is None else max(last, idx[-1])
            if first is None:
                continue
            g = by[names[0]].group
            dt = e.intervals[groups.index(g)] if len(e.intervals) == len(groups) else e.interval
            res[names[0]] = Pass(names[0], names, first, last, dt)
        out.append(res)
    return out


@dataclass
class CpsCheck:
    """A pass's cycles a sample: T over its live span, against the median of the variant's passes."""
    run: int
    key: str
    live: int
    cps: float
    ratio: float = 1.0         # to the median
    flagged: bool = False


def cps_checks(passes: list[dict[str, Pass]], t_cycles: float, tol: float) -> tuple[list[CpsCheck], float]:
    rows = [CpsCheck(r, k, p.live, t_cycles / p.live) for r, d in enumerate(passes) for k, p in d.items()]
    if not rows:
        return rows, 0.0
    v = sorted(x.cps for x in rows)
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    for x in rows:
        x.ratio = x.cps / med
        x.flagged = abs(x.ratio - 1.0) > tol
    return rows, med


# ------------------------------------------------------------------------------------------------------- the frame

@dataclass
class Frame:
    """Where a launch's windows fall, as fractions of the launch."""
    fracs: list[float]       # steady window starts; window k is [fracs[k], fracs[k + 1])
    source: str              # what the windows were read from

    @property
    def mean(self) -> float:
        return (self.fracs[-1] - self.fracs[0]) / (len(self.fracs) - 1)

    def uniform(self) -> "Frame":
        """The same steady span on a MEAN period: every window the same fraction of the launch."""
        n = len(self.fracs) - 1
        return Frame([self.fracs[0] + (self.fracs[-1] - self.fracs[0]) * k / n for k in range(n + 1)],
                     "a mean period over the same span (the anchors' jitter is not followed)")

    def position(self, f: float) -> float | None:
        """The fraction of its window the launch fraction `f` falls at, None outside the steady windows."""
        if not (self.fracs[0] <= f < self.fracs[-1]):
            return None
        k = bisect.bisect_right(self.fracs, f) - 1
        return (f - self.fracs[k]) / (self.fracs[k + 1] - self.fracs[k])


def sample_positions(p: Pass, a: float, d: float, frame: Frame) -> tuple[list[int], list[float]]:
    """The samples of the pass that fall in the steady windows and their window positions, for the entry at i0 + a
    and the length S = (i1 - i0) + d samples."""
    s = p.n + d
    js, xs = [], []
    for j in range(p.i0, p.i1 + 1):
        x = frame.position((j + 0.5 - p.i0 - a) / s)
        if x is not None:
            js.append(j)
            xs.append(x)
    return js, xs


def allowed_a(d: float) -> tuple[float, float]:
    """The entry offsets a in [0, 1) with b = a + d in [0, 1)."""
    return max(0.0, -d), min(1.0, 1.0 - d)


# ------------------------------------------------------------------------------------------------------- the scale

def _norm(values: list[float]) -> tuple[float, float] | None:
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return (m, math.sqrt(var)) if var > 1e-12 else None


def _zs(xs: list[float]) -> list[tuple[complex, complex, complex]]:
    out = []
    for x in xs:
        z = cmath.exp(-2j * math.pi * x)
        z2 = z * z
        out.append((z, z2, z2 * z))
    return out


def harmonic_shares(zs: list[tuple[complex, complex, complex]], vals: list[float]) -> float | None:
    """The share of the variance of `vals` (at the window positions whose e^(-2 pi i x), squared and cubed are `zs`)
    that the first three harmonics explain; None for a constant series."""
    nm = _norm(vals)
    if nm is None:
        return None
    m, sd = nm
    n = len(zs)
    f1 = f2 = f3 = 0j
    for (z1, z2, z3), v in zip(zs, vals):
        w = v - m
        f1 += w * z1
        f2 += w * z2
        f3 += w * z3
    return 2 * (abs(f1) ** 2 + abs(f2) ** 2 + abs(f3) ** 2) / (n * n * sd * sd)


@dataclass
class Scan:
    ds: list[float]
    totals: list[float]               # the pass's sharpness at each d (over the informative series)
    best: float                       # d of the best score within the range the edges allow
    best_score: float
    lo: float                         # the plateau: the connected d around the best within one sigma of the best
    hi: float
    sigma: float                      # the score's own sampling noise at the best
    informative: list[str]
    at_edge: bool                     # the best lies at the edge of the allowed range
    wide_best: float                  # the best of the scan a little wider than the edges allow
    wide_score: float

    @property
    def beyond(self) -> bool:
        """The sharpest scale is outside what the edges allow, by more than the score's noise."""
        return abs(self.wide_best) > 1.0 + 1e-9 and self.wide_score > self.best_score + self.sigma


def scan_scale(p: Pass, series: dict[str, list[float]], frame: Frame) -> Scan | None:
    """Sharpness of the pass's fold against d. `series` maps a name to its values (all samples). The pass's
    informative series (best harmonic share over the scan >= MIN_SERIES) are summed."""
    wide = 1.2
    k_max = int(round(wide / SCAN_STEP))
    ds = [k * SCAN_STEP for k in range(-k_max, k_max + 1)]
    per: dict[str, list[float]] = {nm: [] for nm in series}
    nsamp = 0
    for d in ds:
        a = sum(allowed_a(max(-0.999, min(0.999, d)))) / 2
        js, xs = sample_positions(p, a, d, frame)
        nsamp = max(nsamp, len(js))
        zs = _zs(xs)
        for nm, vals in series.items():
            sh = harmonic_shares(zs, [vals[j] for j in js]) if len(js) >= 20 else None
            per[nm].append(sh if sh is not None else 0.0)
    inform = [nm for nm, sc in per.items() if sc and max(sc) >= MIN_SERIES]
    if not inform:
        return None
    totals = [sum(per[nm][i] for nm in inform) for i in range(len(ds))]
    inside = [i for i, d in enumerate(ds) if abs(d) <= 1.0 + 1e-9]
    ib = max(inside, key=lambda i: totals[i])
    best, bs = ds[ib], totals[ib]
    sigma = math.sqrt(2 * bs / max(1, nsamp))
    lo = hi = ib
    while lo - 1 in inside and totals[lo - 1] >= bs - sigma:
        lo -= 1
    while hi + 1 in inside and totals[hi + 1] >= bs - sigma:
        hi += 1
    iw = max(range(len(ds)), key=lambda i: totals[i])
    edge = ib in (inside[0], inside[-1])
    return Scan(ds, totals, best, bs, ds[lo], ds[hi], sigma, inform, edge, ds[iw], totals[iw])


# ------------------------------------------------------------------------------------------------------- rotations

def _unit(xs: list[float], vals: list[float], H: int = HARMONICS):
    """Per harmonic, the terms (v - mean)/sd * exp(-2 pi i h x) of each sample, so a sum over a set of samples is that
    set's normalized harmonic (times its size)."""
    nm = _norm(vals)
    if nm is None:
        return None
    m, sd = nm
    return [[(v - m) / sd * cmath.exp(-2j * math.pi * h * x) for x, v in zip(xs, vals)] for h in range(1, H + 1)]


def _best_rotation(c: list[complex]) -> float:
    """The rotation s (a fraction of a window, in (-0.5, 0.5]) that maximizes sum_h Re(c_h exp(2 pi i h s)): the
    cross-spectrum c_h of run r with the reference is |c| exp(-2 pi i h s) when r's profile lies s later."""
    def r(s):
        return sum((ch * cmath.exp(2j * math.pi * (h + 1) * s)).real for h, ch in enumerate(c))
    grid = [(g / ROT_GRID) for g in range(ROT_GRID)]
    vals = [r(s) for s in grid]
    i = max(range(ROT_GRID), key=lambda k: vals[k])
    y0, y1, y2 = vals[i - 1], vals[i], vals[(i + 1) % ROT_GRID]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    s = (grid[i] + off / ROT_GRID) % 1.0
    return s - 1.0 if s > 0.5 else s


@dataclass
class Rotation:
    s: float                  # windows: the run's profile lies this much later than the reference's
    sd: float                 # bootstrap standard deviation, windows
    strength: float           # |cross-spectrum| over the product of the two folds' harmonic amplitudes, 0..1


def rotation_between(ref: dict[str, tuple[list[float], list[float]]], run: dict[str, tuple[list[float], list[float]]],
                     seed: int, boot: int = BOOT) -> Rotation | None:
    """Rotation of `run` against `ref`, each {series: (positions, values)}: the phase of the cross-spectrum summed over
    the series both carry, to a fraction of a bin; its bootstrap standard deviation resamples the samples of each."""
    names = [n for n in ref if n in run]
    ur = {n: _unit(*ref[n]) for n in names}
    us = {n: _unit(*run[n]) for n in names}
    names = [n for n in names if ur[n] is not None and us[n] is not None]
    if not names:
        return None
    nr = {n: len(ref[n][0]) for n in names}
    ns = {n: len(run[n][0]) for n in names}

    def spectra(ir, is_):
        out = []
        for h in range(HARMONICS):
            c = 0j
            for n in names:
                fr = sum(ur[n][h][i] for i in ir[n]) / len(ir[n])
                fs = sum(us[n][h][i] for i in is_[n]) / len(is_[n])
                c += fs * fr.conjugate()
            out.append(c)
        return out

    full_r = {n: range(nr[n]) for n in names}
    full_s = {n: range(ns[n]) for n in names}
    c = spectra(full_r, full_s)
    s0 = _best_rotation(c)
    amp = sum(abs(sum(ur[n][h][i] for i in full_r[n]) / nr[n]) * abs(sum(us[n][h][i] for i in full_s[n]) / ns[n])
              for n in names for h in range(HARMONICS))
    strength = sum(abs(x) for x in c) / amp if amp > 0 else 0.0
    rng = random.Random(seed)
    ests = []
    for _ in range(boot):
        ir = {n: [rng.randrange(nr[n]) for _ in range(nr[n])] for n in names}
        is_ = {n: [rng.randrange(ns[n]) for _ in range(ns[n])] for n in names}
        d = _best_rotation(spectra(ir, is_)) - s0
        ests.append(d - round(d))
    m = sum(ests) / len(ests)
    sd = math.sqrt(sum((x - m) ** 2 for x in ests) / max(1, len(ests) - 1))
    return Rotation(s0, sd, strength)


# ------------------------------------------------------------------------------------------------------- one pass

@dataclass
class RunPass:
    """A pass in one run: its length check, its scale, its place."""
    run: int
    p: Pass
    cps: float
    ratio: float
    flagged: bool
    scan: Scan | None = None
    a_lo: float = 0.0               # the entry offsets the edges allow over the scale's plateau
    a_hi: float = 1.0
    a_nom: float = 0.5
    d_nom: float = 0.0
    kappa: float = 0.0              # windows a sample spans, at this run's scale
    rot: Rotation | None = None
    alt_score: float | None = None  # the best sharpness of the same run on the OTHER frame (a mean period)
    k_alias: int = 0
    a: float | None = None          # the placement (None: not placed)
    d: float | None = None
    fed: bool = False
    note: str = ""


@dataclass
class PassPlacement:
    key: str
    names: list[str]
    runs: dict[int, RunPass]
    ref: int | None = None
    a_lo: float | None = None       # the reference run's entry offset a, the intersection (samples)
    a_hi: float | None = None
    kappa: float = 0.0
    half_a: float = 0.0             # windows: half the interval, in windows
    half_d: float = 0.0             # windows: the scales' plateaus' effect, pooled over the runs
    steps: list[tuple[int, float]] = field(default_factory=list)   # (runs, width of a's interval in samples)
    findings: list[str] = field(default_factory=list)
    series: dict[str, "SeriesPlace"] = field(default_factory=dict)
    frame_diag: str = ""            # the sharpness of this pass on the record's anchors against a mean period

    @property
    def placed(self) -> bool:
        return self.a_lo is not None

    @property
    def fed(self) -> list[int]:
        return [r for r, x in self.runs.items() if x.fed]

    @property
    def half(self) -> float:
        return self.half_a + self.half_d

    @property
    def mid(self) -> float:
        return 0.5 * (self.a_lo + self.a_hi)


@dataclass
class SeriesPlace:
    name: str
    short: str
    key: str
    bins: list[Bin]
    half_a: float                   # windows
    half_d: float
    fed: int
    score: float                    # the share of the series' variance the pooled placed fold's bins explain
    varying: bool                   # the score is over VARY_Z standard deviations above what noise gives (see `varies`)
    null: float = 0.0               # the share bin means of pure noise explain, on average

    @property
    def half(self) -> float:
        return self.half_a + self.half_d


def _bin_share(points: list[tuple[float, float]], n_bins: int) -> float:
    vals = [v for _x, v in points]
    nm = _norm(vals)
    if nm is None:
        return 0.0
    m, sd = nm
    cnt = [0] * n_bins
    sm = [0.0] * n_bins
    for x, v in points:
        b = min(n_bins - 1, int(x * n_bins))
        cnt[b] += 1
        sm[b] += v
    between = sum(s * s / c for s, c in zip(sm, cnt) if c) / len(points) - m * m
    return between / (sd * sd)


VARY_Z = 5.0   # a placed series VARIES when its fold's between-bin share is this many standard deviations over noise's


def varies(score: float, n_points: int, n_bins: int) -> tuple[bool, float]:
    """A fold of noise has a between-bin share of about (k - 1)/(N - 1) (a Beta with variance 2(k - 1)/N^2, for k bins and
    N samples); a series varies when its share exceeds that by `VARY_Z` standard deviations. (The period fit's 25% bar
    is for a series whose period is being searched for; a placed fold is not searched, and a clear profile under the
    noise of single samples explains well under 25% of their variance.)"""
    null = (n_bins - 1) / max(1, n_points - 1)
    sd = math.sqrt(2 * (n_bins - 1)) / max(1, n_points)
    return score > null + VARY_Z * sd, null


def place_pass(key: str, passes: dict[int, Pass], values: dict[int, dict[str, list[float]]],
               checks: dict[int, CpsCheck], frame: Frame, cfg: Config) -> PassPlacement:
    """Place one pass across the runs that carry it."""
    names = next(iter(passes.values())).names
    runs = {r: RunPass(r, p, checks[r].cps, checks[r].ratio, checks[r].flagged) for r, p in passes.items()}
    pp = PassPlacement(key, names, runs)
    w = frame.mean
    pts: dict[int, dict[str, tuple[list[float], list[float]]]] = {}
    for r, rp in runs.items():
        if rp.flagged:
            rp.note = (f"cycles a sample {rp.cps:,.0f} is {100 * (rp.ratio - 1):+.1f}% from the variant's median "
                       f"(bound {100 * cfg.cps_tol:g}%): left out")
            pp.findings.append(f"CYCLES A SAMPLE, run {r}, pass `{key}`: {rp.note}")
            continue
        sc = scan_scale(rp.p, {nm: values[r][nm] for nm in rp.p.names if nm in values[r]}, frame)   # THIS pass's series
        if sc is None:
            rp.note = "no series of the pass varies in its steady span (no harmonic share over the scan): not placed"
            pp.findings.append(f"pass `{key}`, run {r}: {rp.note}")
            continue
        rp.scan = sc
        alt = scan_scale(rp.p, {nm: values[r][nm] for nm in rp.p.names if nm in values[r]}, frame.uniform()
                         if frame.source != MEAN_NOTE else frame)
        rp.alt_score = alt.best_score if alt is not None else 0.0
        los, his = zip(*[allowed_a(max(-0.999, min(0.999, d))) for d in sc.ds if sc.lo - 1e-9 <= d <= sc.hi + 1e-9])
        rp.a_lo, rp.a_hi = min(los), max(his)
        rp.d_nom = sc.best
        rp.a_nom = sum(allowed_a(sc.best)) / 2
        rp.kappa = 1.0 / ((rp.p.n + rp.d_nom) * w)
        js, xs = sample_positions(rp.p, rp.a_nom, rp.d_nom, frame)
        pts[r] = {nm: (xs, [values[r][nm][j] for j in js]) for nm in sc.informative}
        if len(js) < cfg.bins * sampling.MIN_PER_BIN:
            rp.note = f"{len(js)} steady samples leave under {sampling.MIN_PER_BIN} a bin of {cfg.bins}: not placed"
            pp.findings.append(f"pass `{key}`, run {r}: {rp.note}")
            del pts[r]
            continue
        rp.fed = True                                      # candidate; the intersection may drop it
    both = [rp for rp in runs.values() if rp.scan is not None and rp.alt_score is not None]
    if both and frame.source != MEAN_NOTE:
        a_sum, m_sum = sum(rp.scan.best_score for rp in both), sum(rp.alt_score for rp in both)
        noise = math.sqrt(sum(rp.scan.sigma ** 2 for rp in both))
        pp.frame_diag = (f"sharpness summed over {len(both)} run(s): {a_sum:.2f} on the record's window anchors, {m_sum:.2f} "
                         f"on a mean period (noise {noise:.2f})")
        if m_sum > a_sum + noise:
            pp.frame_diag += (": the samples do NOT follow the anchors' jitter (a mean period is sharper): the anchors' "
                              "window-to-window jitter is not the kernel's rhythm for this pass; `--frame mean` places "
                              "on a mean period")
            pp.findings.append(f"ANCHORS NOT FOLLOWED, pass `{key}`: {pp.frame_diag}")
    cand = [r for r in runs if runs[r].fed]
    if not cand:
        pp.findings.append(f"pass `{key}`: no run places it: left out")
        return pp
    ref = cand[0]
    pp.ref = ref
    rr = runs[ref]
    # e_ref: the reference's entry offset less its nominal one; the interval it is in
    e_lo, e_hi = rr.a_lo - rr.a_nom, rr.a_hi - rr.a_nom
    pp.steps = [(1, e_hi - e_lo)]
    s_of: dict[int, float] = {ref: 0.0}
    n_in = 1
    for r in cand[1:]:
        rp = runs[r]
        rot = rotation_between(pts[ref], pts[r], zlib.crc32(f"{key}/{r}".encode()))
        if rot is None:
            rp.fed = False
            rp.note = "no series both runs carry varies: not tied to the reference"
            pp.findings.append(f"pass `{key}`: run {r} is not tied to run {ref}: left out of the intersection")
            continue
        rp.rot = rot
        sig = ROT_Z * rot.sd / rp.kappa                    # the rotation's slack, in samples of this run
        best = None
        for k in (-1, 0, 1):
            sk = rot.s + k
            lo = (rp.kappa * (rp.a_lo - rp.a_nom - sig) - sk) / rr.kappa
            hi = (rp.kappa * (rp.a_hi - rp.a_nom + sig) - sk) / rr.kappa
            ov = min(e_hi, hi) - max(e_lo, lo)
            if best is None or ov > best[0] + 1e-9:
                best = (ov, k, lo, hi)
        ov, k, lo, hi = best
        if ov < -cfg.length_tol:
            rp.fed = False
            rp.note = (f"its interval for the entry, carried to run {ref} through the rotation {rot.s:+.4f} windows, lies "
                       f"{-ov:.3f} samples from the others': left out")
            pp.findings.append(f"pass `{key}`: INTERSECTION EMPTY, run {r}: {rp.note}")
            continue
        rp.k_alias = k
        s_of[r] = rot.s + k
        e_lo, e_hi = (max(e_lo, lo), min(e_hi, hi)) if ov >= 0 else (0.5 * (max(e_lo, lo) + min(e_hi, hi)),) * 2
        n_in += 1
        pp.steps.append((n_in, e_hi - e_lo))
    fed = [r for r in cand if runs[r].fed]
    pp.a_lo, pp.a_hi = rr.a_nom + e_lo, rr.a_nom + e_hi
    e_mid = 0.5 * (e_lo + e_hi)
    pp.kappa = rr.kappa
    for r in fed:
        rp = runs[r]
        e_r = (s_of[r] + e_mid * rr.kappa) / rp.kappa
        a = min(max(rp.a_nom + e_r, rp.a_lo), rp.a_hi)
        # the scale: the best score within the plateau where this offset is allowed
        sc = rp.scan
        ok = [i for i, d in enumerate(sc.ds) if sc.lo - 1e-9 <= d <= sc.hi + 1e-9
              and allowed_a(d)[0] - 1e-9 <= a <= allowed_a(d)[1] + 1e-9]
        rp.a = a
        rp.d = sc.ds[max(ok, key=lambda i: sc.totals[i])] if ok else sc.best
    pp.half_a = 0.5 * (e_hi - e_lo) * rr.kappa
    d2 = sum((0.5 * (runs[r].scan.hi - runs[r].scan.lo) * runs[r].kappa / 2) ** 2 for r in fed)
    pp.half_d = math.sqrt(d2) / len(fed)
    return pp


def fold_pass(pp: PassPlacement, values: dict[int, dict[str, list[float]]], frame: Frame, cfg: Config) -> None:
    """Fold every series of the placed pass at its runs' placements and pool the runs: `pp.series`."""
    from .sampling import Series
    n = cfg.bins
    for nm in pp.names:
        parts, pts = [], []
        for r in pp.fed:
            rp = pp.runs[r]
            if nm not in values[r]:
                continue
            js, xs = sample_positions(rp.p, rp.a, rp.d, frame)
            vs = [values[r][nm][j] for j in js]
            if len(js) < n * sampling.MIN_PER_BIN or _norm(vs) is None:
                continue
            parts.append(fold_bins(xs, vs, 1.0, n))
            pts += list(zip(xs, vs))
        if not parts:
            continue
        score = _bin_share(pts, n)
        vary, null = varies(score, len(pts), n)
        pp.series[nm] = SeriesPlace(nm, Series(nm, "", "", [], 0).short, pp.key, pool_bins(parts), pp.half_a, pp.half_d,
                                    len(parts), score, vary, null)


# ------------------------------------------------------------------------------------------------------- a variant

@dataclass
class VariantPlacement:
    frame: Frame
    t_cycles: float
    checks: list[CpsCheck]
    cps_median: float
    passes: dict[str, PassPlacement]
    findings: list[str]
    bins: int

    def series(self) -> dict[str, SeriesPlace]:
        return {nm: s for pp in self.passes.values() for nm, s in pp.series.items()}

    def varying(self) -> list[str]:
        s = self.series()
        return sorted((n for n, x in s.items() if x.varying),
                      key=lambda n: -(max(b.mean for b in s[n].bins) - min(b.mean for b in s[n].bins)))


def place_variant(exports: list[Export], t_cycles: float, frame: Frame, cfg: Config) -> VariantPlacement:
    """Every pass of a variant placed on cycles: `t_cycles` is the variant's own launch in cycles, `frame` its windows
    as fractions of that launch (its own record's, or another variant's)."""
    if getattr(cfg, "frame", "anchors") == "mean":
        frame = frame.uniform()
    per_run = passes_of(exports, cfg.live)
    checks, med = cps_checks(per_run, t_cycles, cfg.cps_tol)
    ck = {(c.run, c.key): c for c in checks}
    keys: list[str] = []
    for d in per_run:
        for k in d:
            if k not in keys:
                keys.append(k)
    values = [{s.name: s.values for s in e.series} for e in exports]
    passes: dict[str, PassPlacement] = {}
    findings: list[str] = []
    for key in sorted(keys):
        ps = {r: d[key] for r, d in enumerate(per_run) if key in d}
        pp = place_pass(key, ps, {r: values[r] for r in ps}, {r: ck[(r, key)] for r in ps}, frame, cfg)
        passes[key] = pp
        if pp.placed:
            fold_pass(pp, {r: values[r] for r in ps}, frame, cfg)
        findings += pp.findings
    return VariantPlacement(frame, t_cycles, checks, med, passes, findings, cfg.bins)


# ------------------------------------------------------------------------------------------------------- comparing folds

def fold_rotation(a: list[float], b: list[float], H: int = HARMONICS) -> float | None:
    """The rotation, in bins and to a fraction of one, by which fold `b` lies later than fold `a`: the phase of the
    cross-spectrum of their first harmonics. None when either is flat."""
    n = len(a)
    H = min(H, n // 2 - 1 if n % 2 == 0 else n // 2)
    if _norm(a) is None or _norm(b) is None:
        return None
    ma, mb = sum(a) / n, sum(b) / n

    def f(v, m, h):
        return sum((x - m) * cmath.exp(-2j * math.pi * h * (i + 0.5) / n) for i, x in enumerate(v)) / n
    c = [f(b, mb, h) * f(a, ma, h).conjugate() for h in range(1, H + 1)]
    return _best_rotation(c + [0j] * (HARMONICS - len(c))) * n


@dataclass
class FoldCompare:
    series: str
    short: str
    rotation: float | None       # bins: `b` lies this much later than `a`
    corr: float | None           # correlation of the two folds as they are
    corr_rot: float | None       # correlation after the whole-bin rotation nearest the one found
    finding: str = ""


def compare_folds(name: str, short: str, a: list[float], b: list[float], max_rot: float = 0.5,
                  min_corr: float = sampling.MIN_CORR, strict: bool = True) -> FoldCompare:
    """Two folds of one series (the same kernel): the rotation between them and the correlation. With `strict` a
    rotation over `max_rot` bins or a correlation under `min_corr` (as they are) is a named finding."""
    n = len(a)
    rot = fold_rotation(a, b)
    corr = sampling.pearson(a, b)
    corr_rot = sampling.pearson(a, sampling.rotate(b, int(round(rot)) % n)) if rot is not None else None
    why = []
    if rot is None:
        why.append("a fold is flat: nothing to compare")
    else:
        if abs(rot) > max_rot:
            why.append(f"rotated by {rot:+.2f} bins (over {max_rot:g})")
        if corr is not None and corr < min_corr:
            why.append(f"correlation {corr:.3f} (under {min_corr:g})")
    return FoldCompare(name, short, rot, corr, corr_rot, "; ".join(why) if strict else "")
