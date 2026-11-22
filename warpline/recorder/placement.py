"""The samples joined to the stamps on CYCLES: a pass is scaled by its own two edges, its runs fitted JOINTLY.

A profiler's time sampling stretches the kernel in real time and leaves its cycles unchanged, and the stretch is even
over a window and over the launch but not within a phase. So a sample is placed by the kernel's own cycle counter:

  A PASS   the series that were collected together (one execution of the kernel): its first live sample i0 and its last
           i1 (live: any series of the pass over `cfg.live`). The kernel's ENTRY is at sample-time i0 + a and its EXIT at
           i1 + b, a and b in [0, 1) unknown, so its length in samples is S = (i1 - i0) + d with d = b - a in (-1, 1).
  THE JOIN sample j's centre is then the fraction f = ((j + 0.5) - (i0 + a)) / S of the launch, and the record gives
           the launch in cycles (T, entry to exit): the sample sits at cycle c = f * T past the entry. Its window
           position is read from a FRAME: where the launch's windows fall, as fractions of the launch.
  THE FRAME follows the SAMPLES, not one lane's anchors: the anchors' window-to-window jitter is not the samples' rhythm
           (the intervals' lag-1 autocorrelation is -0.5: an early window is followed by a late one). The default frame
           is a LOCAL fit of the anchors (a Gaussian-weighted local line, sd `h` windows), `h` chosen by the pooled
           sharpness of the variant's passes among a stated grid, the mean period being the limit `h` -> infinity; the
           raw anchors (`--frame anchors`) and the mean period (`--frame mean`) are stated alternatives, and the sharpness
           of every pass under each is reported.
  THE CHECK that replaces the length check: a pass's cycles a sample, T over its live span, against the median of the
           variant's passes (default 3%).
  THE FIT  of a pass is JOINT. Its runs are all fitted at once to one pooled profile: each run has an entry offset a and
           a scale d inside what its two edges allow, and the pooled fold's SHARPNESS is maximized (the share of the
           pooled samples' variance the first three harmonics of the window explain, summed over the pass's series: a
           sample is a total over an interval, which leaves little above the third; a harmonic fit has no bin edges, so
           the objective is smooth in the offsets). No run is left out for a weak fold.
  THE ABSOLUTE OFFSET is not in the objective: a common shift of every run's profile rotates the pooled fold and leaves
           its sharpness unchanged. The runs' edges alone bound it: the profile's common shift lies in an interval (the
           BOX) where every run's entry stays inside its first live sample and its exit inside its last. The placement is
           the box's midpoint, and the box's half-width is the placement's hard bound.
  THE UNCERTAINTY is MEASURED by resampling the RUNS (a bootstrap over runs and a leave-one-run-out, each refitted from
           scratch): the spread of each series' profile phase. The ALIAS CHECK evaluates the pooled objective at the
           placement against its value with a run (or every run) shifted by one whole sample, or one window.
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
D_STEP = 0.1           # the scale's grid, samples
A_STEP = 0.025         # the entry offset's grid, samples
SWEEPS = 12            # coordinate sweeps of the joint fit
ROT_GRID = 720         # the rotation's grid over one window (then refined by a parabola)
MIN_FIT = 0.15         # a series enters the fit when one run's best harmonic share is at least this
H_GRID = (2.0, 4.0, 8.0, 16.0, 32.0)    # the local fit's bandwidths (windows) the frame chooses among, and the mean
SPREAD_Z = 2.0         # the placement's half-width: the box's half-width and this many spreads, in quadrature
ALIAS_BINS = 0.5       # an alias moves a series' profile by more than this many bins to matter
BIND = 0.01            # a box narrower than this (windows) is one the runs fill: EDGES BIND
FLIP = 0.4             # a resampled fit that lands over this many samples (in windows: x kappa) from the placement FLIPPED
FLIP_SHARE = 0.10      # and a series is AMBIGUOUS BY A SAMPLE when this share of the resampled fits flipped


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

def local_track(fracs: list[float], h: float) -> list[float]:
    """The anchors' local fit: at each anchor k a line fitted to the anchors i with Gaussian weights exp(-(i - k)^2 / 2h^2),
    and its value at k. A line, not a mean: the track keeps a slow drift of the window's length (the line follows it)
    and loses a wobble shorter than about 2 pi h windows. At the two ends the fit is one-sided."""
    n = len(fracs)
    reach = int(math.ceil(4 * h))
    out = []
    for k in range(n):
        sw = swx = swxx = swy = swxy = 0.0
        for i in range(max(0, k - reach), min(n, k + reach + 1)):
            w = math.exp(-0.5 * ((i - k) / h) ** 2)
            x = i - k
            sw += w
            swx += w * x
            swxx += w * x * x
            swy += w * fracs[i]
            swxy += w * x * fracs[i]
        det = sw * swxx - swx * swx
        out.append((swxx * swy - swx * swxy) / det if det > 1e-12 else swy / sw)
    for k in range(1, n):                       # a track must advance
        out[k] = max(out[k], out[k - 1] + 1e-9)
    return out


@dataclass
class Frame:
    """Where a launch's windows fall, as fractions of the launch."""
    fracs: list[float]       # steady window starts; window k is [fracs[k], fracs[k + 1])
    source: str              # what the windows were read from
    kind: str = "anchors"    # anchors | local | mean
    h: float | None = None   # a local fit's bandwidth, windows

    @property
    def mean(self) -> float:
        return (self.fracs[-1] - self.fracs[0]) / (len(self.fracs) - 1)

    def uniform(self) -> "Frame":
        """The same steady span on a MEAN period: every window the same fraction of the launch."""
        n = len(self.fracs) - 1
        return Frame([self.fracs[0] + (self.fracs[-1] - self.fracs[0]) * k / n for k in range(n + 1)],
                     "a mean period over the same span (the anchors' jitter is not followed)", "mean")

    def local(self, h: float) -> "Frame":
        """The anchors' local fit with bandwidth `h` windows."""
        return Frame(local_track(self.fracs, h), f"a local fit of the anchors (Gaussian weights, sd {h:g} windows)",
                     "local", h)

    def position(self, f: float) -> float | None:
        """The fraction of its window the launch fraction `f` falls at, None outside the steady windows."""
        if not (self.fracs[0] <= f < self.fracs[-1]):
            return None
        k = bisect.bisect_right(self.fracs, f) - 1
        return (f - self.fracs[k]) / (self.fracs[k + 1] - self.fracs[k])

    @property
    def label(self) -> str:
        if self.kind == "local":
            return f"a local fit (h = {self.h:g} windows)"
        return {"anchors": "the raw anchors", "mean": "a mean period"}[self.kind]


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


# ------------------------------------------------------------------------------------------------------- the objective

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


def _best_rotation(c: list[complex]) -> float:
    """The rotation s (a fraction of a window, in (-0.5, 0.5]) that maximizes sum_h Re(c_h exp(2 pi i h s)): the
    cross-spectrum c_h of a profile with a reference is |c| exp(-2 pi i h s) when the profile lies s later."""
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


def _wrap(x: float) -> float:
    x = x % 1.0
    return x - 1.0 if x > 0.5 else x


def best_share(p: Pass, vals: list[float], frame: Frame) -> float:
    """A series' best harmonic share over the scales the edges allow (entry in the middle of what they allow): whether
    it varies at all in the pass, whatever the frame's fine structure."""
    best = 0.0
    for k in range(-10, 11):
        d = k * D_STEP
        a = sum(allowed_a(max(-0.999, min(0.999, d)))) / 2
        js, xs = sample_positions(p, a, d, frame)
        if len(js) < 20:
            continue
        sh = harmonic_shares(_zs(xs), [vals[j] for j in js])
        best = max(best, sh or 0.0)
    return best


# ------------------------------------------------------------------------------------------------------- one pass

@dataclass
class RunPass:
    """A pass in one run: its length check and its place."""
    run: int
    p: Pass
    cps: float
    ratio: float
    flagged: bool
    a: float | None = None          # the placement (None: not placed)
    d: float | None = None
    fed: bool = False
    share: float = 0.0              # the run's own sharpness (summed over the fitted series) at its placement
    d_plateau: tuple[float, float] | None = None   # the scales within one noise sigma of the best
    note: str = ""


@dataclass
class AliasRow:
    """The pooled objective with a run, or every run, shifted by whole samples or a window."""
    label: str
    q: float                        # the objective there
    dq: float                       # best minus this
    margin: float | None            # twice the resampled standard deviation of dq (None: not resampled)
    within: bool
    moves: dict[str, float]         # series -> bins the pooled profile moves against the placement
    note: str = ""


@dataclass
class PassPlacement:
    key: str
    names: list[str]
    runs: dict[int, RunPass]
    fit_names: list[str] = field(default_factory=list)
    box: tuple[float, float] | None = None     # the common shift's interval about the placement, windows
    kappa: float = 0.0                         # windows a sample
    q: float = 0.0                             # the pooled sharpness at the placement
    q_frames: dict[str, float] = field(default_factory=dict)       # the same on each frame, by label
    spread: dict[str, float | None] = field(default_factory=dict)  # series -> measured spread of its phase, windows
    spread_boot: dict[str, float] = field(default_factory=dict)
    spread_loo: dict[str, float] = field(default_factory=dict)
    boot: int = 0
    alias: list[AliasRow] = field(default_factory=list)
    ambiguous: dict[str, float] = field(default_factory=dict)      # series -> bins the alias candidate lies from the placement
    outward: list[str] = field(default_factory=list)
    tight: bool = False                        # the runs' scale plateaus leave no common shift (the box is empty)
    binds: bool = False                        # ... or a hair of one
    flips: dict[str, tuple[float, float]] = field(default_factory=dict)   # series -> (share of resamples, bins) that land a sample away
    findings: list[str] = field(default_factory=list)
    series: dict[str, "SeriesPlace"] = field(default_factory=dict)
    frame_diag: str = ""

    @property
    def placed(self) -> bool:
        return self.box is not None

    @property
    def fed(self) -> list[int]:
        return [r for r, x in self.runs.items() if x.fed]

    @property
    def half_a(self) -> float:
        """Half the box, windows."""
        return 0.0 if self.box is None else 0.5 * (self.box[1] - self.box[0])


@dataclass
class SeriesPlace:
    name: str
    short: str
    key: str
    bins: list[Bin]
    half_a: float                   # windows: half the edges' box
    spread: float | None            # windows: the measured spread of the profile's phase (None: not measurable)
    fed: int
    score: float                    # the share of the series' variance the pooled placed fold's bins explain
    varying: bool                   # the score is over VARY_Z standard deviations above what noise gives (see `varies`)
    null: float = 0.0               # the share bin means of pure noise explain, on average
    ambiguous: float | None = None  # bins the alias candidate's profile lies from the placement, when the check fired
    binds: bool = False             # the edges' box is empty or a hair wide: the runs fill all the edges allow

    @property
    def half(self) -> float:
        """The placement's half-width, windows: the box's half-width and SPREAD_Z spreads, in quadrature."""
        s = 0.0 if self.spread is None else SPREAD_Z * self.spread
        return math.sqrt(self.half_a ** 2 + s ** 2)


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


# ------------------------------------------------------------------------------------------------------- the joint fit

@dataclass
class Table:
    """One run's samples as harmonic sums at each scale d, the entry offset a = 0: the sums at another offset are these
    rotated (a shift of the entry by a samples moves every window position by kappa * a windows: exact on a uniform
    frame, to the frame's unevenness on another). Series are in the order of `names`, three harmonics each."""
    run: int
    p: Pass
    ds: list[float]
    F: list[list[complex]]         # [d index] -> flat (series, harmonic)
    cnt: list[int]
    kap: list[float]               # windows a sample at each d

    def contribution(self, di: int, a: float) -> list[complex]:
        k = self.kap[di]
        e = cmath.exp(2j * math.pi * k * a)
        es = (e, e * e, e * e * e)
        return [x * es[i % 3] for i, x in enumerate(self.F[di])]


def _grid(lo: float, hi: float) -> list[float]:
    if hi <= lo + 1e-12:
        return [lo]
    n = max(1, int(math.ceil((hi - lo) / A_STEP - 1e-9)))
    return [lo + (hi - lo) * k / n for k in range(n + 1)]


def make_table(run: int, p: Pass, nv: dict[str, list[float]], names: list[str], frame: Frame) -> Table:
    ds = [round(k * D_STEP, 10) for k in range(-10, 11)]
    F, cnt, kap = [], [], []
    w = frame.mean
    for d in ds:
        a0 = 0.0
        js, xs = sample_positions(p, a0, d, frame)
        zs = _zs(xs)
        vec = []
        for nm in names:
            v = nv[nm]
            f1 = f2 = f3 = 0j
            for (z1, z2, z3), j in zip(zs, js):
                u = v[j]
                f1 += u * z1
                f2 += u * z2
                f3 += u * z3
            vec += [f1, f2, f3]
        F.append(vec)
        cnt.append(len(js))
        kap.append(1.0 / ((p.n + d) * w))
    return Table(run, p, ds, F, cnt, kap)


def _q(T: list[complex], n: int) -> float:
    if n <= 0:
        return 0.0
    return 2.0 * sum(x.real * x.real + x.imag * x.imag for x in T) / (n * n)


def _cands(d: float) -> list[float]:
    lo, hi = allowed_a(max(-1.0, min(1.0, d)))
    return _grid(lo, hi)


def _per_d(tab: Table, S: list[complex], n0: int) -> list[tuple[float, float] | None]:
    """For each scale d of the table, the best (pooled objective, entry offset a) given the others' sum `S` over `n0`
    samples (None where no sample falls in the steady windows)."""
    L = len(S)
    out: list[tuple[float, float] | None] = []
    for di, d in enumerate(tab.ds):
        cnt = tab.cnt[di]
        if cnt == 0:
            out.append(None)
            continue
        F = tab.F[di]
        k = tab.kap[di]
        den = (n0 + cnt) ** 2
        lo, hi = allowed_a(d)
        mid = 0.5 * (lo + hi)
        best = (-1.0, 0.0)
        for a in _cands(d):
            e = cmath.exp(2j * math.pi * k * a)
            e2 = e * e
            e3 = e2 * e
            s = 0.0
            for i in range(0, L, 3):
                v = S[i] + F[i] * e
                s += v.real * v.real + v.imag * v.imag
                v = S[i + 1] + F[i + 1] * e2
                s += v.real * v.real + v.imag * v.imag
                v = S[i + 2] + F[i + 2] * e3
                s += v.real * v.real + v.imag * v.imag
            q = 2.0 * s / den - 1e-9 * abs(a - mid)        # ties along the ridge: the middle of the allowed range
            if q > best[0]:
                best = (q, a)
        out.append(best)
    return out


def _best_move(tab: Table, S: list[complex], n0: int, cur: tuple[int, float] | None):
    """The (pooled objective, d index, a) of the table with the best pooled objective given the others' sum `S`."""
    best = (-1.0, 0, 0.0)
    for di, r in enumerate(_per_d(tab, S, n0)):
        if r is not None and r[0] > best[0]:
            best = (r[0], di, r[1])
    return best


def fit_pool(tabs: list[Table], order: list[int] | None = None) -> tuple[list[tuple[int, float]], float]:
    """The joint fit: every table's (d index, a) maximizing the pooled objective. Runs are inserted one at a time
    (strongest first) and then swept by coordinate ascent until nothing moves."""
    R = len(tabs)
    if R == 0:
        return [], 0.0
    L = len(tabs[0].F[0])
    order = order if order is not None else list(range(R))
    cur: list[tuple[int, float] | None] = [None] * R
    T = [0j] * L
    N = 0
    for i in order:
        q, di, a = _best_move(tabs[i], T, N, None)
        c = tabs[i].contribution(di, a)
        T = [x + y for x, y in zip(T, c)]
        N += tabs[i].cnt[di]
        cur[i] = (di, a)
    for _ in range(SWEEPS):
        moved = False
        for i in order:
            di0, a0 = cur[i]
            c0 = tabs[i].contribution(di0, a0)
            S = [x - y for x, y in zip(T, c0)]
            n0 = N - tabs[i].cnt[di0]
            q, di, a = _best_move(tabs[i], S, n0, cur[i])
            q_old = _q(T, N)
            if q > q_old + 1e-9:
                c = tabs[i].contribution(di, a)
                T = [x + y for x, y in zip(S, c)]
                N = n0 + tabs[i].cnt[di]
                cur[i] = (di, a)
                moved = True
        if not moved:
            break
    return cur, _q(T, N)


def pooled(tabs: list[Table], cfg: list[tuple[int, float]]) -> tuple[list[complex], int]:
    T = [0j] * len(tabs[0].F[0])
    N = 0
    for t, (di, a) in zip(tabs, cfg):
        T = [x + y for x, y in zip(T, t.contribution(di, a))]
        N += t.cnt[di]
    return T, N


@dataclass
class Joint:
    """The joint fit of a pass's runs, its box and its placement."""
    cur: list[tuple[int, float]]                       # per table: (d index, a) at the placement
    q: float                                           # the pooled objective at the fit's best
    q_place: float                                     # and at the placement
    T: list[complex]                                   # the pooled sums at the placement
    N: int
    box: tuple[float, float]                           # the common shift's interval about the placement, windows
    plateau: list[tuple[float, float]]                 # per table: the scales within one noise sigma of the best
    sigma: float
    tight: bool                                        # the runs' plateaus leave no common shift: the box is empty


def joint_fit(tabs: list[Table]) -> Joint:
    """THE JOINT FIT of a pass's runs. (1) Every run's (a, d) maximizes the pooled objective (insertion, then sweeps; two
    insertion orders, the better kept). (2) The objective is blind to a common shift of every run, so the edges alone
    bound it: a run's scale d is known only to within the plateau where the objective is within one noise sigma
    (sqrt(2 Q / N)) of the best, and at each such d the run's entry (and its exit, a + d) must stay in its first (last)
    live sample, which gives the shifts the run allows; the BOX is the intersection over runs of those shifts. (3) The
    placement is the middle of the box: every run's offset moves by that common shift (and its scale to the plateau's
    best where the box needs it)."""
    R = len(tabs)
    order = sorted(range(R), key=lambda i: -_q(*pooled([tabs[i]], [(10, 0.5)])))
    cfg0, q0 = fit_pool(tabs, order)
    cfg1, q1 = fit_pool(tabs, order[::-1])
    cur, q = (cfg0, q0) if q0 >= q1 else (cfg1, q1)
    T, N = pooled(tabs, cur)
    sigma = math.sqrt(2.0 * q / max(1, N))
    allow: list[list[tuple[int, float, float, float, float]]] = []   # per run: (di, a, q, lo, hi) over the plateau
    plateau = []
    for i, t in enumerate(tabs):
        di, a = cur[i]
        c = t.contribution(di, a)
        S = [x - y for x, y in zip(T, c)]
        n0 = N - t.cnt[di]
        per = _per_d(t, S, n0)
        rows = []
        for dj, r in enumerate(per):
            if r is None or r[0] < q - sigma:
                continue
            al, ah = allowed_a(t.ds[dj])
            if ah < al:
                continue
            k = t.kap[dj]
            rows.append((dj, r[1], r[0], k * (al - r[1]), k * (ah - r[1])))
        if not rows:
            al, ah = allowed_a(t.ds[di])
            k = t.kap[di]
            rows = [(di, a, q, k * (al - a), k * (ah - a))]
        allow.append(rows)
        ds = [t.ds[r[0]] for r in rows]
        plateau.append((min(ds), max(ds)))
    lo = max(min(r[3] for r in rows) for rows in allow)
    hi = min(max(r[4] for r in rows) for rows in allow)
    tight = lo > hi
    if tight:
        lo = hi = 0.5 * (lo + hi)
    m = 0.5 * (lo + hi)
    out = []
    for t, rows in zip(tabs, allow):
        ok = [r for r in rows if r[3] - 1e-9 <= m <= r[4] + 1e-9]
        r = max(ok, key=lambda x: x[2]) if ok else min(rows, key=lambda x: min(abs(m - x[3]), abs(m - x[4])))
        al, ah = allowed_a(t.ds[r[0]])
        out.append((r[0], min(max(r[1] + m / t.kap[r[0]], al), ah)))
    T2, N2 = pooled(tabs, out)
    return Joint(out, q, _q(T2, N2), T2, N2, (lo - m, hi - m), plateau, sigma, tight)


def rot_of(T: list[complex], ref: list[complex], s: int) -> float:
    """The rotation, windows, of the profile whose pooled harmonic sums are `T` against `ref`, for series number `s`."""
    return _best_rotation([T[3 * s + h] * ref[3 * s + h].conjugate() for h in range(3)])


# ------------------------------------------------------------------------------------------------------- exact evaluation

def contribs(p: Pass, nv: dict[str, list[float]], names: list[str], frame: Frame, a: float, d: float) -> dict[int, list[complex]]:
    """Each live sample's harmonic terms at the entry a and scale d, exact on the frame: {sample: flat (series, harmonic)}."""
    s = p.n + d
    out = {}
    for j in range(p.i0, p.i1 + 1):
        x = frame.position((j + 0.5 - p.i0 - a) / s)
        if x is None:
            continue
        z = cmath.exp(-2j * math.pi * x)
        zs = (z, z * z, z * z * z)
        out[j] = [nv[nm][j] * zs[h] for nm in names for h in range(3)]
    return out


def _sum(c: dict[int, list[complex]], w: dict[int, int] | None, L: int) -> tuple[list[complex], int]:
    T = [0j] * L
    n = 0
    for j, v in c.items():
        k = 1 if w is None else w.get(j, 0)
        if k == 0:
            continue
        n += k
        for i in range(L):
            T[i] += k * v[i]
    return T, n


# ------------------------------------------------------------------------------------------------------- a pass's context

@dataclass
class Ctx:
    key: str
    names: list[str]
    runs: dict[int, RunPass]
    use: list[int]                        # the runs that enter the fit
    fit_names: list[str]
    nv: dict[int, dict[str, list[float]]]  # each run's values, normalized by the pass's pooled mean and sd
    raw: dict[int, dict[str, list[float]]]
    findings: list[str]


def make_ctx(key: str, passes: dict[int, Pass], values: dict[int, dict[str, list[float]]], checks: dict[int, CpsCheck],
             mean_frame: Frame, cfg: Config) -> Ctx:
    names = next(iter(passes.values())).names
    runs = {r: RunPass(r, p, checks[r].cps, checks[r].ratio, checks[r].flagged) for r, p in passes.items()}
    findings: list[str] = []
    use: list[int] = []
    for r, rp in runs.items():
        if rp.flagged:
            rp.note = (f"cycles a sample {rp.cps:,.0f} is {100 * (rp.ratio - 1):+.1f}% from the variant's median "
                       f"(bound {100 * cfg.cps_tol:g}%): left out")
            findings.append(f"CYCLES A SAMPLE, run {r}, pass `{key}`: {rp.note}")
            continue
        js, _ = sample_positions(rp.p, 0.5, 0.0, mean_frame)
        if len(js) < cfg.bins * sampling.MIN_PER_BIN:
            rp.note = f"{len(js)} steady samples leave under {sampling.MIN_PER_BIN} a bin of {cfg.bins}: not placed"
            findings.append(f"pass `{key}`, run {r}: {rp.note}")
            continue
        use.append(r)
    fit_names: list[str] = []
    for nm in names:
        if any(nm in values[r] and best_share(runs[r].p, values[r][nm], mean_frame) >= MIN_FIT for r in use):
            fit_names.append(nm)
    nv: dict[int, dict[str, list[float]]] = {r: {} for r in use}
    for nm in fit_names:
        pts = []
        for r in use:
            js, _ = sample_positions(runs[r].p, 0.5, 0.0, mean_frame)
            pts += [values[r][nm][j] for j in js]
        nm_ = _norm(pts)
        if nm_ is None:
            continue
        m, sd = nm_
        for r in use:
            nv[r][nm] = [(v - m) / sd for v in values[r][nm]]
    fit_names = [nm for nm in fit_names if all(nm in nv[r] for r in use)]
    return Ctx(key, names, runs, use, fit_names, nv, {r: values[r] for r in use}, findings)


def tables_for(ctx: Ctx, frame: Frame) -> list[Table]:
    return [make_table(r, ctx.runs[r].p, ctx.nv[r], ctx.fit_names, frame) for r in ctx.use]


def pass_q(ctx: Ctx, frame: Frame) -> float:
    """The pooled sharpness a pass reaches on a frame, the joint fit's maximum."""
    tabs = tables_for(ctx, frame)
    if not tabs or not ctx.fit_names:
        return 0.0
    return fit_pool(tabs)[1]


def resolve_frame(base: Frame, ctxs: list[Ctx], cfg: Config) -> tuple[Frame, dict[str, dict[str, float]], str]:
    """The frame of the variant. `anchors`: the raw anchors; `mean`: a mean period; `local` with `cfg.smooth` set: the
    local fit of that bandwidth; `local` otherwise: the bandwidth chosen from the raw anchors (h = 0), H_GRID and the
    mean period (h -> infinity) by the pooled sharpness summed over the variant's passes, the SMOOTHEST frame whose sum is
    within one noise sigma of the best (the anchors are followed only where the samples follow them). Returns the frame,
    the sharpness of every pass on every candidate (label -> pass key -> Q) and a line saying what was chosen and why."""
    kind = getattr(cfg, "frame", "local")
    smooth = getattr(cfg, "smooth", None)
    cands: list[Frame] = [base] + [base.local(h) for h in H_GRID] + [base.uniform()]
    if kind == "anchors":
        chosen, why = base, "the raw anchors, by request"
    elif kind == "mean":
        chosen, why = cands[-1], "a mean period, by request"
    elif smooth:
        chosen, why = base.local(smooth), f"the local fit of bandwidth {smooth:g} windows, by request"
        cands.append(chosen)
    else:
        chosen, why = None, ""
    if not getattr(cfg, "report_frames", True) and chosen is not None:
        cands = [chosen]
    qs: dict[str, dict[str, float]] = {}
    for f in cands:
        qs[f.label] = {c.key: pass_q(c, f) for c in ctxs if c.fit_names}
    if chosen is not None:
        return chosen, qs, why
    tot = {f.label: sum(qs[f.label].values()) for f in cands}
    n_tot = sum(c.runs[r].p.live for c in ctxs if c.fit_names for r in c.use) or 1
    top = max(tot.values())
    sigma = math.sqrt(2.0 * top / n_tot)
    ok = [f for f in cands if tot[f.label] >= top - sigma]
    best = ok[-1]                              # the candidates run from the least to the most smoothed
    why = ("the frame follows the samples: the local fit's bandwidth is chosen by the pooled sharpness summed over the "
           "variant's passes, the smoothest within one noise sigma (" + f"{sigma:.2f}" + ") of the best: "
           + ", ".join(f"{f.label.replace('a local fit ', '').replace('the raw anchors', 'anchors').strip('()')} "
                       f"{tot[f.label]:.2f}" for f in cands) + f"; chosen: {best.label}"
           + (" (the anchors' wobble is not followed at all: the limit of the local fit is the mean period)"
              if best.kind == "mean" else ""))
    return best, qs, why


# ------------------------------------------------------------------------------------------------------- placing a pass

def _boot_dq(base: list[dict[int, list[complex]]], alts: dict[str, dict[int, dict[int, list[complex]]]], runs_p: list[Pass],
             L: int, B: int, rng: random.Random) -> dict[str, float]:
    """For each alternative (a dict run index -> that run's contributions in the alternative), twice the standard
    deviation over B resamplings of the samples (within each run) of best objective minus the alternative's."""
    out: dict[str, list[float]] = {k: [] for k in alts}
    for _ in range(B):
        ws = []
        for p in runs_p:
            w: dict[int, int] = {}
            for _k in range(p.live):
                j = p.i0 + rng.randrange(p.live)
                w[j] = w.get(j, 0) + 1
            ws.append(w)
        sums = [_sum(c, w, L) for c, w in zip(base, ws)]
        Tb = [0j] * L
        Nb = 0
        for t, n in sums:
            Tb = [x + y for x, y in zip(Tb, t)]
            Nb += n
        qb = _q(Tb, Nb)
        for k, alt in alts.items():
            T, N = list(Tb), Nb
            for i, c in alt.items():
                t0, n0 = sums[i]
                t1, n1 = _sum(c, ws[i], L)
                T = [x - y + z for x, y, z in zip(T, t0, t1)]
                N += n1 - n0
            out[k].append(qb - _q(T, N))
    res = {}
    for k, v in out.items():
        m = sum(v) / len(v)
        res[k] = 2.0 * math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
    return res


def place_ctx(ctx: Ctx, frame: Frame, cfg: Config) -> PassPlacement:
    """Place one pass: the joint fit of its runs, the box, the spread by resampling the runs, the alias check."""
    pp = PassPlacement(ctx.key, ctx.names, ctx.runs, ctx.fit_names, findings=list(ctx.findings))
    if not ctx.use or not ctx.fit_names:
        pp.findings.append(f"pass `{ctx.key}`: no series of the pass varies in a run's steady span: not placed"
                           if ctx.use else f"pass `{ctx.key}`: no run places it: left out")
        return pp
    tabs = tables_for(ctx, frame)
    jf = joint_fit(tabs)
    cur, T, N = jf.cur, jf.T, jf.N
    pp.box = jf.box
    pp.q = jf.q_place
    pp.tight = jf.tight
    pp.binds = jf.tight or (jf.box[1] - jf.box[0]) < BIND
    if pp.binds:
        pp.findings.append(
            f"EDGES BIND, pass `{ctx.key}`: the fitted runs fill all that the first and last live sample allow (the box of the "
            f"common shift is {jf.box[1] - jf.box[0]:.4f} windows wide): the placement rests on the edges' limits, and the "
            f"half-width below (the box's, which is nothing, and the measured spread) does not count how far the edges are "
            f"from the kernel's own entry and exit")
    ns = len(ctx.fit_names)
    pp.kappa = sum(t.kap[di] for t, (di, _a) in zip(tabs, cur)) / len(tabs)
    for i, (t, (di, a)) in enumerate(zip(tabs, cur)):
        rp = ctx.runs[t.run]
        rp.a, rp.d, rp.fed = a, t.ds[di], True
        rp.d_plateau = jf.plateau[i]
        rp.share = _q(t.contribution(di, a), t.cnt[di])
    if len(tabs) < 2:
        pp.findings.append(f"pass `{ctx.key}`: one run: the joint fit has nothing to tie, and the spread cannot be measured "
                           f"by resampling runs")
    # the spread: resample the runs, refit each from scratch, the rotation of each series' pooled profile
    R = len(tabs)
    rng = random.Random(zlib.crc32(ctx.key.encode()))
    B = int(getattr(cfg, "boot", 24))
    boots: list[list[float]] = []
    if R >= 3:
        for _ in range(B):
            idx = [rng.randrange(R) for _ in range(R)]
            sub = [tabs[i] for i in idx]
            Tb = joint_fit(sub).T
            boots.append([rot_of(Tb, T, s) for s in range(ns)])
        loo: list[list[float]] = []
        for k in range(R):
            sub = [t for i, t in enumerate(tabs) if i != k]
            Tb = joint_fit(sub).T
            loo.append([rot_of(Tb, T, s) for s in range(ns)])
        pp.boot = len(boots)
        for s, nm in enumerate(ctx.fit_names):
            allr = [r[s] for r in boots] + [r[s] for r in loo]
            far = [x for x in allr if abs(x) > FLIP * pp.kappa]
            if len(far) >= FLIP_SHARE * len(allr):
                pp.flips[nm] = (len(far) / len(allr), sorted(far)[len(far) // 2] * cfg.bins)
            xb = [r[s] for r in boots]
            mb = sum(xb) / len(xb)
            sdb = math.sqrt(sum((x - mb) ** 2 for x in xb) / max(1, len(xb) - 1))
            xl = [r[s] for r in loo]
            ml = sum(xl) / len(xl)
            sdl = math.sqrt((R - 1) / R * sum((x - ml) ** 2 for x in xl))
            pp.spread_boot[nm], pp.spread_loo[nm] = sdb, sdl
            pp.spread[nm] = max(sdb, sdl)
    else:
        for nm in ctx.fit_names:
            pp.spread[nm] = None
        pp.findings.append(f"pass `{ctx.key}`: {R} run(s): the spread of the placement cannot be measured by resampling "
                           f"runs (it needs three): the half-width is the edges' box alone")
    alias_check(ctx, pp, tabs, cur, frame, T, cfg, rng)
    for nm, (share, bins_) in pp.flips.items():
        if abs(pp.ambiguous.get(nm, 0.0)) < abs(bins_):
            pp.ambiguous[nm] = bins_
    if pp.flips:
        pp.findings.append(
            f"PLACEMENT AMBIGUOUS BY A SAMPLE, pass `{pp.key}`: refitting from resampled runs, "
            + ", ".join(f"`{sampling.Series(n, '', '', [], 0).short}` landed {100 * sh:.0f}% of the time {b:+.2f} bins from the "
                        f"placement" for n, (sh, b) in sorted(pp.flips.items()))
            + f" (a sample is {pp.kappa * cfg.bins:.2f} bins): two candidates, the placement and the placement shifted by "
              f"those bins; the objective cannot tell them apart (every run shifted by a sample changes it by "
              + ", ".join(f"{r.dq:+.3f}" for r in pp.alias if r.label.startswith("every run") and "sample" in r.label)
              + f" against the best's {pp.q:.3f}) and the edges' box ({pp.box[1] - pp.box[0]:.3f} windows wide) alone decides")
    return pp


def alias_check(ctx: Ctx, pp: PassPlacement, tabs: list[Table], cur: list[tuple[int, float]], frame: Frame, T0: list[complex],
                cfg: Config, rng: random.Random) -> None:
    """THE ALIAS CHECK, on the objective evaluated exactly on the frame: (a) each run alone shifted by +-1 whole sample
    (its scale as fitted), (b) every run together by +-1 sample, (c) every run together by +-1 window. A candidate is
    WITHIN the margin when its objective is under twice the resampled standard deviation of the difference from the
    best's; a run's candidate within the margin that moves a series' pooled profile by more than half a bin names that
    series AMBIGUOUS BY A SAMPLE, and both profiles are given. A shift of every run together is an exact rotation of the
    pooled profile (its objective is the same up to the frame's unevenness and the span's ends): it is listed, and the
    edges alone decide it."""
    names = ctx.fit_names
    L = 3 * len(names)
    R = len(tabs)
    cfgx = [(t.ds[di], a) for t, (di, a) in zip(tabs, cur)]
    base = [contribs(t.p, ctx.nv[t.run], names, frame, a, d) for t, (d, a) in zip(tabs, cfgx)]
    sums = [_sum(c, None, L) for c in base]
    Tb = [0j] * L
    Nb = 0
    for t, n in sums:
        Tb = [x + y for x, y in zip(Tb, t)]
        Nb += n
    qb = _q(Tb, Nb)
    n_bins = cfg.bins
    cand: dict[str, dict[int, dict[int, list[complex]]]] = {}
    labels: dict[str, str] = {}
    for i, t in enumerate(tabs):
        d, a = cfgx[i]
        for k in (-1, 1):
            lab = f"run {t.run} {'+' if k > 0 else '-'}1 sample"
            cand[lab] = {i: contribs(t.p, ctx.nv[t.run], names, frame, a + k, d)}
    whole = {}
    for k in (-1, 1):
        whole[f"every run {'+' if k > 0 else '-'}1 sample"] = {
            i: contribs(t.p, ctx.nv[t.run], names, frame, cfgx[i][1] + k, cfgx[i][0]) for i, t in enumerate(tabs)}
        whole[f"every run {'+' if k > 0 else '-'}1 window"] = {
            i: contribs(t.p, ctx.nv[t.run], names, frame, cfgx[i][1] + k / t.kap[cur[i][0]], cfgx[i][0])
            for i, t in enumerate(tabs)}
    B = max(8, int(getattr(cfg, "boot", 24)) // 2)
    margins = _boot_dq(base, cand, [t.p for t in tabs], L, B, rng)
    rows: list[AliasRow] = []

    def total(alt):
        T, N = list(Tb), Nb
        for i, c in alt.items():
            t0, n0 = sums[i]
            t1, n1 = _sum(c, None, L)
            T = [x - y + z for x, y, z in zip(T, t0, t1)]
            N += n1 - n0
        return T, N

    def moves(T):
        out = {}
        for s, nm in enumerate(names):
            out[nm] = rot_of(T, Tb, s) * n_bins
        return out

    for lab, alt in list(cand.items()) + list(whole.items()):
        T, N = total(alt)
        q = _q(T, N)
        m = margins.get(lab)
        mv = moves(T)
        every = lab.startswith("every")
        within = (abs(qb - q) <= (m if m is not None else 0.0)) if not every else True
        rows.append(AliasRow(lab, q, qb - q, m, within, mv,
                             "an exact rotation of the pooled profile: the objective cannot tell it; the edges decide"
                             if every else ""))
    pp.alias = rows
    flagged: dict[str, float] = {}
    for row in rows:
        if row.label.startswith("every") or not row.within:
            continue
        for nm, mvb in row.moves.items():
            if abs(mvb) > ALIAS_BINS and abs(mvb) > abs(flagged.get(nm, 0.0)):
                flagged[nm] = mvb
    pp.ambiguous = flagged
    if flagged:
        worst = max(flagged, key=lambda n: abs(flagged[n]))
        pp.findings.append(
            f"PLACEMENT AMBIGUOUS BY A SAMPLE, pass `{pp.key}`: with a run's entry shifted by one whole sample the pooled "
            f"objective is within the resampled margin of the best, and the profile of "
            + ", ".join(f"`{sampling.Series(n, '', '', [], 0).short}` {v:+.2f} bins" for n, v in sorted(flagged.items()))
            + f" lies that far from the placement: two candidates, the placement and the placement shifted by those bins "
              f"(the edges' box decides between them; worst `{sampling.Series(worst, '', '', [], 0).short}`)")
    # the fit against the edges: a run's entry moved outside its first live sample, the objective better by more than
    # the resampled margin
    out_cand: dict[str, dict[int, dict[int, list[complex]]]] = {}
    for i, t in enumerate(tabs):
        d, a = cfgx[i]
        di = cur[i][0]
        best = None
        for step in range(-12, 13):
            a2 = a + step * 0.1
            al, ah = allowed_a(d)
            if al - 1e-9 <= a2 <= ah + 1e-9:
                continue
            c = t.contribution(di, a2)
            T = [x - y + z for x, y, z in zip(Tb, sums[i][0], c)]
            q = _q(T, Nb - sums[i][1] + t.cnt[di])
            if best is None or q > best[0]:
                best = (q, a2)
        if best is not None and best[0] > qb:
            out_cand[f"run {t.run} to a = {best[1]:+.2f}"] = {i: contribs(t.p, ctx.nv[t.run], names, frame, best[1], d)}
    if out_cand:
        mg = _boot_dq(base, out_cand, [t.p for t in tabs], L, B, rng)
        for lab, alt in out_cand.items():
            T, N = total(alt)
            gain = _q(T, N) - qb
            if gain > mg[lab] and gain > 0:
                pp.outward.append(f"{lab}: the objective is higher by {gain:.4f} (margin {mg[lab]:.4f}), outside the first live "
                                  f"sample")
    for x in pp.outward:
        pp.findings.append(f"FIT AND EDGES DISAGREE, pass `{pp.key}`, {x}: the placement keeps the edges' bound; the fit "
                           f"would move the run")


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
        sp = pp.spread.get(nm)
        has = nm in pp.spread
        pp.series[nm] = SeriesPlace(nm, Series(nm, "", "", [], 0).short, pp.key, pool_bins(parts), pp.half_a,
                                    sp if has else None, len(parts), score, vary, null, pp.ambiguous.get(nm), pp.binds)


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
    frame_q: dict[str, dict[str, float]] = field(default_factory=dict)   # frame label -> pass key -> pooled sharpness
    frame_why: str = ""

    def series(self) -> dict[str, SeriesPlace]:
        return {nm: s for pp in self.passes.values() for nm, s in pp.series.items()}

    def varying(self) -> list[str]:
        s = self.series()
        return sorted((n for n, x in s.items() if x.varying),
                      key=lambda n: -(max(b.mean for b in s[n].bins) - min(b.mean for b in s[n].bins)))


def place_variant(exports: list[Export], t_cycles: float, frame: Frame, cfg: Config) -> VariantPlacement:
    """Every pass of a variant placed on cycles: `t_cycles` is the variant's own launch in cycles, `frame` its windows
    as fractions of that launch (its own record's, or another variant's): the raw anchors, which this reduces to the
    frame `cfg.frame` names."""
    per_run = passes_of(exports, cfg.live)
    checks, med = cps_checks(per_run, t_cycles, cfg.cps_tol)
    ck = {(c.run, c.key): c for c in checks}
    keys: list[str] = []
    for d in per_run:
        for k in d:
            if k not in keys:
                keys.append(k)
    values = [{s.name: s.values for s in e.series} for e in exports]
    mean_frame = frame.uniform()
    ctxs = []
    for key in sorted(keys):
        ps = {r: d[key] for r, d in enumerate(per_run) if key in d}
        ctxs.append(make_ctx(key, ps, {r: values[r] for r in ps}, {r: ck[(r, key)] for r in ps}, mean_frame, cfg))
    chosen, qs, why = resolve_frame(frame, ctxs, cfg)
    passes: dict[str, PassPlacement] = {}
    findings: list[str] = []
    for ctx in ctxs:
        pp = place_ctx(ctx, chosen, cfg)
        pp.q_frames = {lab: q.get(ctx.key, 0.0) for lab, q in qs.items()}
        passes[ctx.key] = pp
        if pp.placed:
            fold_pass(pp, {r: values[r] for r in ctx.use}, chosen, cfg)
        findings += pp.findings
    vp = VariantPlacement(chosen, t_cycles, checks, med, passes, findings, cfg.bins, qs, why)
    return vp


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
    bound: float | None = None   # bins: the two placements' measured spreads (quadrature, SPREAD_Z each) and one bin
    within: bool | None = None   # the rotation is within `bound`


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
