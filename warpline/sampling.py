"""A profiler's time-sampled counters: read, find the window's period, check the assumptions, fold.

A file is one RUN: the export of the sampled counters of one kernel run at one CTA, one value a sample (see
docs/sampling.md). Each sampled series (name `pmsampling:...`) was collected in its own PASS of the kernel, so each
has its own start. Everything here is computed from the series alone:

  START      the first nonzero sample of the series' PASS (the series that share a sample count were collected
             together; a sparse series' own first nonzero sample is not its kernel's start). To within a sample.
  STEADY     the span past `skip` windows from the start to `tail` samples before the live end.
  PERIOD     in samples: the scan that maximizes the folded profile's between-bin variance over `lo..hi` at
             `step`, the best refined to `REFINE` (a period is taken as the shortest whose score is within
             `HARMONIC` of the best, so a multiple of the period is not mistaken for it); its SCORE is the share of
             the series' variance the fold explains.
  CHECKS     passes agree (each series' period within the tolerance of the median; one that does not is flagged
             and left out of the fold; a series with too little variation is FLAT and is folded on the agreed
             period), and steady (a period from the first half of the steady span and one from the second agree).
  FOLD       every series on the agreed period into N bins: mean, standard error, count; the BLUR is stated: a
             sample is a total over its interval, so the fold is the true profile smoothed by a box of
             interval/period of a window.
  RUNS       several runs of one kernel: their periods (in microseconds) must agree; each is aligned to the first
             by the rotation that maximizes the correlation of one series' fold, then the folds are averaged.
  PLACEMENT  (`placement.py`) the samples joined to the stamps on cycles, a pass scaled by its own two edges: no
             period is fitted there, so none of the checks above can drop a series from it.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field

PREFIX = "pmsampling:"
INTERVAL = "profiler__pmsampler_interval_time"

SCAN_LO, SCAN_HI, SCAN_STEP, REFINE = 2.0, 6.0, 0.002, 1e-4
HALF_RANGE, HALF_STEP = 0.10, 0.0005   # the half-spans' periods are searched within 10% of the whole span's
HARMONIC = 0.9
HARM_BAND = 0.007       # a sub-multiple of the period is sought within this fraction of period/k
MIN_PER_BIN = 10        # the fold refuses an N that leaves fewer samples than this a bin
MIN_CORR, MIN_GAP = 0.7, 0.05


class SamplingError(Exception):
    pass


@dataclass
class Config:
    bins: int = 8
    tolerance: float = 0.01      # periods agree within this fraction
    skip: float = 4.0            # windows past the start before the steady span
    tail: int = 3                # samples dropped before the live end
    lo: float = SCAN_LO
    hi: float = SCAN_HI
    step: float = SCAN_STEP
    min_score: float = 0.25      # a series whose best fold explains less of its variance than this is FLAT
    min_corr: float = MIN_CORR
    min_gap: float = MIN_GAP
    live: float = 0.0            # a sample is LIVE when any series of its pass exceeds this in absolute value
    length_tol: float = 0.25     # the entry intervals of two runs may miss each other by this many samples
    cps_tol: float = 0.03        # a pass's cycles a sample may differ from its variant's median by this fraction
    frame: str = "local"         # window positions from a local fit of the window anchors (default), the raw "anchors", or a "mean" period
    smooth: float | None = None  # the local fit's bandwidth in windows; None: chosen by the pooled sharpness of the passes
    boot: int = 24               # resamplings of the runs (bootstrap) that measure a placement's spread
    report_frames: bool = True   # the sharpness of every candidate frame is computed and reported (False: only the chosen)


# ------------------------------------------------------------------------------------------------------- reading

@dataclass
class Series:
    name: str            # without the `pmsampling:` prefix
    unit: str
    aggregate: str
    values: list[float]
    group: int           # the pass: series with one sample count were collected together
    start: int | None = None   # the pass's first nonzero sample; None: the series is all zero
    end: int | None = None     # one past the pass's last nonzero sample

    @property
    def short(self) -> str:
        s = self.name
        for p in ("smsp__warps_issue_stalled_", "smsp__"):
            if s.startswith(p):
                s = s[len(p):]
                break
        return s[:-4] if s.endswith(".avg") else s


@dataclass
class Export:
    path: str
    intervals: list[float]      # microseconds, one a pass as the export lists them
    series: list[Series]

    @property
    def interval(self) -> float:
        return sum(self.intervals) / len(self.intervals)


_VALUE = re.compile(r"^\s*([^()]*?)\s*\((.*)\)\s*$", re.S)


def _parse(cell: str) -> tuple[str, list[float]] | None:
    m = _VALUE.match(cell)
    if not m:
        return None
    try:
        return m.group(1), [float(x) for x in m.group(2).split(";")]
    except ValueError:
        return None


def read_export(path: str) -> Export:
    csv.field_size_limit(1 << 30)
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except OSError as e:
        raise SamplingError(f"{path}: {e}")
    if len(rows) < 3 or not (len(rows[0]) == len(rows[1]) == len(rows[2])):
        raise SamplingError(f"{path}: not an export of three rows (names, units, values) of equal width")
    intervals, raw = None, []
    for name, unit, cell in zip(*rows[:3]):
        if name == INTERVAL:
            p = _parse(cell)
            if p is None:
                raise SamplingError(f"{path}: `{INTERVAL}` is not `<aggregate> (<v>; ...)`")
            intervals = p[1]
        elif name.startswith(PREFIX):
            p = _parse(cell)
            if p is None or not p[1]:
                raise SamplingError(f"{path}: `{name}` is not `<aggregate> (<v0>; <v1>; ...)`")
            raw.append((name[len(PREFIX):], unit, p[0], p[1]))
    if not raw:
        raise SamplingError(f"{path}: no sampled metric (a name starting `{PREFIX}`)")
    if not intervals:
        raise SamplingError(f"{path}: no `{INTERVAL}`: the sampling interval is not known")
    groups: dict[int, int] = {}
    series = [Series(n, u, a, v, groups.setdefault(len(v), len(groups))) for n, u, a, v in raw]
    for g in set(s.group for s in series):
        members = [s for s in series if s.group == g]
        firsts = [next(i for i, x in enumerate(s.values) if x != 0) for s in members if any(s.values)]
        lasts = [max(i for i, x in enumerate(s.values) if x != 0) for s in members if any(s.values)]
        for s in members:
            if firsts and any(s.values):
                s.start, s.end = min(firsts), max(lasts) + 1
    return Export(path, intervals, series)


# ------------------------------------------------------------------------------------------------------- the period

def _between(xs, vs, period, n_bins, min_bin):
    """Between-bin variance of the values folded on `period` (None when a bin holds fewer than `min_bin`)."""
    cnt = [0] * n_bins
    sm = [0.0] * n_bins
    inv = 1.0 / period
    for x, v in zip(xs, vs):
        b = int((x * inv % 1.0) * n_bins)
        cnt[b] += 1
        sm[b] += v
    if min(cnt) < min_bin:
        return None
    n = len(xs)
    mean = sum(vs) / n
    return sum(s * s / c for s, c in zip(sm, cnt)) / n - mean * mean


def _total(vs):
    n = len(vs)
    m = sum(vs) / n
    return sum(v * v for v in vs) / n - m * m


def _scan(xs, vs, lo, hi, step, n_bins, min_bin):
    best = (None, -1.0)
    k = 0
    while True:
        p = lo + k * step
        if p > hi + 1e-12:
            break
        k += 1
        b = _between(xs, vs, p, n_bins, min_bin)
        if b is not None and b > best[1]:
            best = (p, b)
    return best


def _refine(xs, vs, p, step, n_bins, min_bin):
    best = (p, _between(xs, vs, p, n_bins, min_bin) or -1.0)
    k = -int(round(step / REFINE))
    while k <= int(round(step / REFINE)):
        q = p + k * REFINE
        b = _between(xs, vs, q, n_bins, min_bin)
        if b is not None and b > best[1]:
            best = (q, b)
        k += 1
    return best


def find_period(xs, vs, cfg: Config, lo=None, hi=None, step=None):
    """(period, between-bin variance, share of the total variance) of the values `vs` at sample offsets `xs`;
    None when the series is constant or no period leaves every bin occupied."""
    lo = cfg.lo if lo is None else lo
    hi = cfg.hi if hi is None else hi
    step = cfg.step if step is None else step
    tot = _total(vs)
    if tot <= 1e-12:
        return None
    min_bin = max(2, int(0.2 * len(xs) / cfg.bins))
    p, b = _scan(xs, vs, lo, hi, step, cfg.bins, min_bin)
    if p is None:
        return None
    p, b = _refine(xs, vs, p, step, cfg.bins, min_bin)
    k = 2                                   # the shortest period that explains nearly as much
    while p / k >= lo - 1e-9:
        q, bq = _scan(xs, vs, p / k * (1 - HARM_BAND), p / k * (1 + HARM_BAND), HALF_STEP, cfg.bins, min_bin)
        if q is not None:
            q, bq = _refine(xs, vs, q, HALF_STEP, cfg.bins, min_bin)
            if bq >= HARMONIC * b:
                p, b = q, bq
                k = 1
        k += 1
    return p, b, b / tot


# ------------------------------------------------------------------------------------------------------- the fold

@dataclass
class Bin:
    mean: float
    se: float | None
    n: int


def fold_bins(xs, vs, period, n_bins) -> list[Bin]:
    sums = [[] for _ in range(n_bins)]
    for x, v in zip(xs, vs):
        sums[int((x / period % 1.0) * n_bins)].append(v)
    out = []
    for vals in sums:
        n = len(vals)
        if n == 0:
            out.append(Bin(0.0, None, 0))
            continue
        m = sum(vals) / n
        se = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1) / n) if n > 1 else None
        out.append(Bin(m, se, n))
    return out


@dataclass
class SeriesResult:
    name: str
    short: str
    unit: str
    group: int
    start: int | None
    end: int | None
    n_steady: int = 0
    status: str = "ok"         # ok | flat | dead | disagrees | short
    period: float | None = None
    score: float | None = None          # the share of the series' variance the best fold explains
    between: float | None = None
    halves: tuple[float, float] | None = None
    drift: bool | None = None
    folded: bool = False
    bins: list[Bin] | None = None
    note: str = ""

    @property
    def swing(self) -> float:
        if not self.bins:
            return 0.0
        ms = [b.mean for b in self.bins]
        return max(ms) - min(ms)


@dataclass
class RunResult:
    path: str
    interval: float            # microseconds
    intervals: list[float]
    cfg: Config
    series: list[SeriesResult]
    agreed: float              # samples
    spread: float              # (max - min) / agreed over the series that agree
    flags: list[str] = field(default_factory=list)
    skipped: int = 0
    half_noise: float | None = None   # the median |period of the second half - the first| / period over the series

    @property
    def period_us(self) -> float:
        return self.agreed * self.interval

    @property
    def blur(self) -> float:
        """A sample is a total over its interval: a box this share of a window wide."""
        return 1.0 / self.agreed

    def get(self, short: str) -> SeriesResult | None:
        for r in self.series:
            if r.short == short or r.name == short:
                return r
        return None

    @property
    def varying(self) -> list[SeriesResult]:
        return [r for r in self.series if r.folded and r.status == "ok"]


def _steady(s: Series, period: float, cfg: Config):
    first = s.start + int(math.ceil(cfg.skip * period))
    last = s.end - cfg.tail
    idx = list(range(first, max(first, last)))
    return [i - s.start for i in idx], [s.values[i] for i in idx]


def _analyze_series(s: Series, cfg: Config) -> SeriesResult:
    r = SeriesResult(s.name, s.short, s.unit, s.group, s.start, s.end)
    if s.start is None:
        r.status, r.note = "dead", "all samples zero"
        return r
    # a first period from the whole live span (past a few samples), to place the steady span
    first = s.start + 10
    xs0 = list(range(first - s.start, s.end - cfg.tail - s.start))
    vs0 = [s.values[s.start + x] for x in xs0]
    if len(xs0) < 20 or _total(vs0) <= 1e-12:
        r.n_steady = len(xs0)
        if len(xs0) < 20:
            r.status, r.note = "short", f"{len(xs0)} live samples"
        else:
            r.status, r.note = "flat", "no variation over the live span"
        return r
    pre = find_period(xs0, vs0, cfg, step=0.01)
    p0 = pre[0] if pre else (cfg.lo + cfg.hi) / 2
    xs, vs = _steady(s, p0, cfg)
    r.n_steady = len(xs)
    if len(xs) < cfg.bins * MIN_PER_BIN:
        r.status, r.note = "short", f"{len(xs)} steady samples leave under {MIN_PER_BIN} a bin of {cfg.bins}"
        return r
    found = find_period(xs, vs, cfg)
    if found is None:
        r.status, r.note = "flat", "no variation in the steady span"
        return r
    r.period, r.between, r.score = found
    if r.score < cfg.min_score:
        r.status = "flat"
        r.note = f"the best fold explains {100 * r.score:.0f}% of the variance (under {100 * cfg.min_score:.0f}%)"
        return r
    h = len(xs) // 2
    ph = []
    for a, b in ((0, h), (h, len(xs))):
        f = find_period(xs[a:b], vs[a:b], cfg, lo=r.period * (1 - HALF_RANGE), hi=r.period * (1 + HALF_RANGE),
                        step=HALF_STEP)
        ph.append(f[0] if f else math.nan)
    r.halves = (ph[0], ph[1])
    r.drift = not (abs(ph[0] - ph[1]) <= cfg.tolerance * r.period)
    return r


def analyze_run(exp: Export, cfg: Config | None = None) -> RunResult:
    cfg = cfg or Config()
    srcs = {s.name: s for s in exp.series}
    res = [_analyze_series(s, cfg) for s in exp.series]
    cands = [r for r in res if r.period is not None and r.status == "ok"]
    if not cands:
        longest = max((r.n_steady for r in res), default=0)
        if longest and longest < cfg.bins * MIN_PER_BIN:
            raise SamplingError(f"{exp.path}: {cfg.bins} bins leave under {MIN_PER_BIN} samples a bin on every series "
                                f"(the longest has {longest} steady samples): use fewer bins")
        raise SamplingError(f"{exp.path}: no series has enough variation to give a period")
    # the reference: the series whose period most other series' periods lie within the tolerance of (ties: the one
    # nearest the median); the series that do not agree with it are flagged
    ps = sorted(r.period for r in cands)
    med = ps[len(ps) // 2] if len(ps) % 2 else (ps[len(ps) // 2 - 1] + ps[len(ps) // 2]) / 2
    centre = max(cands, key=lambda c: (sum(1 for r in cands if abs(r.period - c.period) <= cfg.tolerance * c.period),
                                       -abs(c.period - med)))
    ref = centre.period
    flags = []
    for r in cands:
        if abs(r.period - ref) > cfg.tolerance * ref:
            r.status = "disagrees"
            r.note = f"period {r.period:.4f} against {ref:.4f} samples ({100 * (r.period - ref) / ref:+.2f}%)"
            flags.append(f"PASSES DISAGREE: {r.short} (pass {r.group}): {r.note}; left out of the fold")
    ok = [r for r in cands if r.status == "ok"]
    agreed = sum(r.period for r in ok) / len(ok)
    spread = (max(r.period for r in ok) - min(r.period for r in ok)) / agreed
    drifting = [r for r in cands if r.drift]
    if drifting:
        flags.append(f"DRIFT in {len(drifting)} of {len(cands)} series (the period of the steady span's first half and of "
                     f"its second differ by more than {100 * cfg.tolerance:.1f}%): " + ", ".join(
                         f"{r.short} {100 * (r.halves[1] - r.halves[0]) / r.period:+.1f}%" for r in drifting))
    by_pass: dict[int, list[float]] = {}
    for r in cands:
        by_pass.setdefault(r.group, []).append(r.period)
    if len(by_pass) > 1:
        means = {g: sum(v) / len(v) for g, v in by_pass.items()}
        if (max(means.values()) - min(means.values())) > cfg.tolerance * agreed:
            flags.append("the passes' own periods differ: " + ", ".join(f"pass {g} {m:.4f}" for g, m in sorted(means.items()))
                         + " samples")
    if agreed < 1.15 * cfg.lo:
        flags.append(f"NEAR THE SAMPLING LIMIT: the window is {agreed:.3f} samples, {cfg.lo:g} being the shortest that "
                     f"can be told from its mirror image (a period under {cfg.lo:g} samples aliases to one over it): "
                     f"the period and the fold's direction are uncertain")
    skipped = 0
    for r in res:
        if r.status == "dead" or r.status == "disagrees":
            continue
        s = srcs[r.name]
        xs, vs = _steady(s, agreed, cfg)
        r.n_steady = len(xs)
        if len(xs) < cfg.bins * MIN_PER_BIN:
            r.status, r.note = "short", f"{len(xs)} steady samples leave under {MIN_PER_BIN} a bin of {cfg.bins}"
            skipped += 1
            continue
        r.bins = fold_bins(xs, vs, agreed, cfg.bins)
        r.folded = True
    dp = sorted(abs(r.halves[1] - r.halves[0]) / r.period for r in ok if r.halves)
    floor = dp[len(dp) // 2] if dp else None
    return RunResult(exp.path, exp.interval, exp.intervals, cfg, res, agreed, spread, flags, skipped, floor)


# ------------------------------------------------------------------------------------------------------- shifts

def pearson(a, b) -> float | None:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-12 * max(1.0, abs(ma)) or vb <= 1e-12 * max(1.0, abs(mb)):
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def rotate(xs, s):
    """`out[b] = xs[(b + s) % n]`."""
    n = len(xs)
    return [xs[(b + s) % n] for b in range(n)]


@dataclass
class Shift:
    shift: int                      # the fold is rotated by this many bins: rotated[b] = fold[(b + shift) % N]
    corr: float | None
    runner_shift: int | None
    runner_corr: float | None
    accepted: bool
    why: str                        # the named finding when not accepted
    corrs: list[float | None] = field(default_factory=list)


def guard_for(blur: float, n: int) -> int:
    """Shifts closer than this many bins to the best are the same alignment smeared by the blur (a box `blur` of a
    window wide, at least 2 bins): the runner-up is sought beyond them."""
    return max(2, int(round(blur * n)))


def best_shift(target, fold, min_corr=MIN_CORR, min_gap=MIN_GAP, guard=2) -> Shift:
    """The rotation of `fold` that maximizes its correlation with `target`; the runner-up: the best correlation at
    a shift `guard` or more bins from the best (a different alignment, not the best's own neighbour); and the
    verdict: REFUSED below `min_corr` or when the runner-up is within `min_gap` -- an alignment that could be
    another is none."""
    n = len(target)
    c = [pearson(target, rotate(fold, s)) for s in range(n)]
    if all(x is None for x in c):
        what = "the profile" if pearson(target, list(target)) is None else "the folded series"
        return Shift(0, None, None, None, False, f"FLAT: {what} has no variation: nothing to align", c)
    cc = [-2.0 if x is None else x for x in c]
    best = max(range(n), key=lambda s: (cc[s], -s))
    rest = [s for s in range(n) if min((s - best) % n, (best - s) % n) >= guard]
    runner = max(rest, key=lambda s: (cc[s], -s)) if rest else None
    rc = cc[runner] if runner is not None else None
    why = ""
    if cc[best] < min_corr:
        why = f"WEAK: the best correlation {cc[best]:.3f} is under {min_corr}"
    elif rc is not None and cc[best] - rc < min_gap:
        why = (f"AMBIGUOUS: the runner-up (shift {runner}, correlation {rc:.3f}) is within {min_gap} of the best "
               f"(shift {best}, {cc[best]:.3f})")
    return Shift(best, cc[best], runner, rc, not why, why, c)


# ------------------------------------------------------------------------------------------------------- runs

@dataclass
class RunShift:
    index: int
    path: str
    period_us: float
    status: str                  # reference | accepted | period | alignment
    shift: Shift | None
    note: str = ""


@dataclass
class Folded:
    cfg: Config
    runs: list[RunResult]
    run_shifts: list[RunShift]
    align_series: str | None
    flags: list[str]
    names: list[str]                       # series in the average, in the order of the first run
    shorts: dict[str, str]
    bins: dict[str, list[Bin]]             # name -> the average fold
    runs_of: dict[str, list[int]]
    score: dict[str, float]
    status: dict[str, str]                 # flat | ok (in the average: folded in some accepted run)
    period: float                          # samples, the accepted runs' mean
    interval: float
    unit: dict[str, str]

    @property
    def period_us(self) -> float:
        return self.period * self.interval

    @property
    def blur(self) -> float:
        return 1.0 / self.period

    def varying(self) -> list[str]:
        return [n for n in self.names if self.status.get(n) == "ok"]

    def swing(self, name: str) -> float:
        ms = [b.mean for b in self.bins[name]]
        return max(ms) - min(ms)


def _pick_align(runs: list[RunResult]) -> str | None:
    common = None
    for r in runs:
        have = {x.name for x in r.series if x.folded and x.status == "ok"}
        common = have if common is None else common & have
    if not common:
        return None
    for name in sorted(common):
        if name.endswith("stalled_barrier.avg"):
            return name
    return max(sorted(common), key=lambda n: min(next(x for x in r.series if x.name == n).score for r in runs))


def fold(exports: list[Export], cfg: Config | None = None) -> Folded:
    cfg = cfg or Config()
    runs = [analyze_run(e, cfg) for e in exports]
    flags = [f"{r.path}: {f}" for r in runs for f in r.flags]
    us = sorted(r.period_us for r in runs)
    med = us[len(us) // 2] if len(us) % 2 else (us[len(us) // 2 - 1] + us[len(us) // 2]) / 2
    shifts: list[RunShift] = []
    keep = []
    n_bins = cfg.bins
    for i, r in enumerate(runs):
        if len(runs) > 2 and abs(r.period_us - med) > cfg.tolerance * med:
            note = (f"period {r.period_us:.4f} us against the median {med:.4f} "
                    f"({100 * (r.period_us - med) / med:+.2f}%; bound {100 * cfg.tolerance:.1f}%)")
            flags.append(f"RUNS DISAGREE: {r.path}: {note}; left out of the average")
            shifts.append(RunShift(i, r.path, r.period_us, "period", None, note))
        else:
            keep.append(i)
            shifts.append(RunShift(i, r.path, r.period_us, "accepted", None))
    if len(runs) == 2 and abs(runs[0].period_us - runs[1].period_us) > cfg.tolerance * min(us):
        d = runs[1].period_us - runs[0].period_us
        flags.append(f"RUNS DISAGREE: the two runs' periods are {runs[0].period_us:.4f} and {runs[1].period_us:.4f} us "
                     f"({100 * d / runs[0].period_us:+.2f}%; bound {100 * cfg.tolerance:.1f}%): the second is left out "
                     f"(two runs cannot say which is wrong)")
        shifts[1].status, shifts[1].note = "period", "disagrees with the first"
        keep = [0]
    if not keep:
        raise SamplingError("no run is left")
    ref = keep[0]
    if ref != 0:
        flags.append(f"the first run is flagged: run {ref} is the reference the others are aligned to")
    shifts[ref].status = "reference"
    align = _pick_align([runs[i] for i in keep])
    rot = {ref: 0}
    if len(keep) > 1:
        if align is None:
            raise SamplingError("the runs have no varying series in common to align them on")
        refbins = [b.mean for b in next(x for x in runs[ref].series if x.name == align).bins]
        for i in keep[1:]:
            mine = [b.mean for b in next(x for x in runs[i].series if x.name == align).bins]
            sh = best_shift(refbins, mine, cfg.min_corr, cfg.min_gap, guard_for(1.0 / runs[ref].agreed, n_bins))
            shifts[i].shift = sh
            if sh.accepted:
                rot[i] = sh.shift
            else:
                shifts[i].status, shifts[i].note = "alignment", sh.why
                flags.append(f"RUN NOT ALIGNED: {runs[i].path}: {sh.why}; left out of the average")
    elif align is None:
        align = None
    used = [i for i in keep if i in rot]
    names, shorts, unit = [], {}, {}
    for r in runs[ref].series:
        names.append(r.name)
        shorts[r.name], unit[r.name] = r.short, r.unit
    bins: dict[str, list[Bin]] = {}
    runs_of: dict[str, list[int]] = {}
    score: dict[str, float] = {}
    status: dict[str, str] = {}
    n_bins = cfg.bins
    for name in names:
        parts = []
        for i in used:
            x = next((x for x in runs[i].series if x.name == name), None)
            if x is not None and x.folded:
                parts.append((i, x))
        if not parts:
            continue
        out = []
        for b in range(n_bins):
            n = tot = 0
            m = 0.0
            var = 0.0
            for i, x in parts:
                bb = x.bins[(b + rot[i]) % n_bins]
                n += bb.n
                m += bb.n * bb.mean
                var += (bb.n * (bb.se or 0.0)) ** 2
            out.append(Bin(m / n, math.sqrt(var) / n, n))
        bins[name] = out
        runs_of[name] = [i for i, _x in parts]
        sc = [x.score for _i, x in parts if x.score is not None]
        score[name] = sum(sc) / len(sc) if sc else 0.0
        status[name] = "ok" if any(x.status == "ok" for _i, x in parts) else "flat"
    period = sum(runs[i].agreed for i in used) / len(used)
    interval = sum(runs[i].interval for i in used) / len(used)
    return Folded(cfg, runs, shifts, align, flags, [n for n in names if n in bins], shorts, bins, runs_of, score,
                  status, period, interval, unit)


# ------------------------------------------------------------------------------------------------------- output

def _pos(b: int, n: int) -> str:
    return f"{b / n:.3f}-{(b + 1) / n:.3f}"


def format_fold(f: Folded) -> str:
    cfg = f.cfg
    L = []
    for i, r in enumerate(f.runs):
        L.append(f"run {i}: {r.path}")
        iv = ", ".join(f"{x:g}" for x in r.intervals)
        L.append(f"  interval {r.interval:g} us ({iv} by pass); period {r.agreed:.4f} samples = {r.period_us:.3f} us "
                 f"(series spread {100 * r.spread:.2f}%)"
                 + (f"; the half-span periods differ by a median {100 * r.half_noise:.1f}% (the steady check resolves no "
                    f"better than that)" if r.half_noise is not None else ""))
        for x in r.series:
            if x.status == "dead":
                L.append(f"  {x.short:<22} pass {x.group}: DEAD ({x.note})")
            elif x.period is None:
                L.append(f"  {x.short:<22} pass {x.group}: {x.status.upper()} ({x.note}); start {x.start}")
            else:
                h = (f"; halves {x.halves[0]:.4f} / {x.halves[1]:.4f}" + (" DRIFT" if x.drift else "")
                     if x.halves else "")
                L.append(f"  {x.short:<22} pass {x.group}: period {x.period:.4f}, score {x.score:.3f}, start {x.start}, "
                         f"{x.n_steady} steady samples{h}" + (f"  [{x.status.upper()}: {x.note}]"
                                                              if x.status in ("disagrees", "flat", "short") else ""))
    L.append("")
    L.append(f"scan: periods {cfg.lo:g}..{cfg.hi:g} samples at step {cfg.step:g}, the best refined to {REFINE:g}; a "
             f"shorter period within {int(100 * HARMONIC)}% of the best score is taken (a multiple is not the period); "
             f"score = the share of the series' variance the fold explains; flat below {cfg.min_score:g}")
    L.append(f"checks: periods agree within {100 * cfg.tolerance:g}% (of the series most others agree with); halves agree within {100 * cfg.tolerance:g}%; "
             f"steady span = past {cfg.skip:g} windows from the start to {cfg.tail} samples before the end")
    if f.flags:
        L.append("FLAGS")
        for x in f.flags:
            L.append(f"  {x}")
    else:
        L.append("flags: none")
    if len(f.runs) > 1:
        L.append(f"runs: periods in us " + ", ".join(f"{s.period_us:.3f}" for s in f.run_shifts)
                 + f"; aligned on `{f.align_series}` (rotation of its fold)")
        for s in f.run_shifts:
            if s.shift is not None:
                u = f"; runner-up shift {s.shift.runner_shift} corr {s.shift.runner_corr:.3f}" \
                    if s.shift.runner_corr is not None else ""
                L.append(f"  run {s.index}: shift {s.shift.shift} bins, correlation "
                         f"{'n/a' if s.shift.corr is None else f'{s.shift.corr:.3f}'}{u}  [{s.status}]")
            else:
                L.append(f"  run {s.index}: " + (f"left out ({s.status}): {s.note}" if s.status == "period" else s.status))
    L.append("")
    n = cfg.bins
    L.append(f"THE FOLD: {n} bins of the window, mean (standard error) in {next(iter(f.unit.values()), '')} of "
             f"the runs {sorted({i for v in f.runs_of.values() for i in v})}; the window = {f.period:.4f} samples = "
             f"{f.period_us:.3f} us")
    var = sorted(f.varying(), key=lambda x: -f.swing(x))
    flat = [x for x in f.names if x not in var]
    if var:
        L.append(f"  {'window':<13} " + " ".join(f"{f.shorts[x][:16]:>16}" for x in var))
        for b in range(n):
            cells = []
            for x in var:
                bb = f.bins[x][b]
                cells.append(f"{bb.mean:7.2f} ({bb.se if bb.se is not None else float('nan'):4.2f})")
            L.append(f"  {_pos(b, n):<13} " + " ".join(f"{c:>16}" for c in cells))
        L.append("  count a bin: " + ", ".join(str(b.n) for b in f.bins[var[0]]))
    if flat:
        L.append("  flat (folded on the agreed period, not shown): " + ", ".join(f.shorts[x] for x in flat))
    dead = [r.short for r in f.runs[0].series if r.name not in f.bins]
    if dead:
        L.append("  left out of every fold: " + ", ".join(dead))
    L.append(f"THE BLUR: a sample is a total over its interval ({f.interval:g} us), so the fold is the true profile "
             f"smoothed by a box {f.blur:.3f} of a window wide ({f.interval:g} us over {f.period_us:.3f} us; "
             f"{f.blur * n:.2f} of {n} bins)")
    return "\n".join(L) + "\n"


def to_dict(f: Folded) -> dict:
    cfg = f.cfg

    def ser(x: SeriesResult):
        return {"name": x.name, "pass": x.group, "start": x.start, "end": x.end, "status": x.status,
                "period": x.period, "score": x.score, "halves": list(x.halves) if x.halves else None,
                "drift": x.drift, "n_steady": x.n_steady, "note": x.note,
                "bins": [{"mean": b.mean, "se": b.se, "n": b.n} for b in x.bins] if x.bins else None}

    return {
        "bins": cfg.bins, "tolerance": cfg.tolerance,
        "scan": {"lo": cfg.lo, "hi": cfg.hi, "step": cfg.step, "refine": REFINE, "harmonic": HARMONIC,
                 "min_score": cfg.min_score, "skip_windows": cfg.skip, "tail": cfg.tail},
        "period_samples": f.period, "period_us": f.period_us, "interval_us": f.interval,
        "blur": {"fraction_of_window": f.blur, "bins": f.blur * cfg.bins,
                 "note": "a sample is a total over its interval: the fold is the true profile smoothed by this box"},
        "flags": f.flags, "align_series": f.align_series,
        "runs": [{"index": i, "path": r.path, "interval_us": r.interval, "period_samples": r.agreed,
                  "period_us": r.period_us, "series_spread": r.spread, "half_span_noise": r.half_noise, "flags": r.flags,
                  "series": [ser(x) for x in r.series]} for i, r in enumerate(f.runs)],
        "run_shifts": [{"index": s.index, "status": s.status, "period_us": s.period_us, "note": s.note,
                        "shift": None if s.shift is None else
                        {"bins": s.shift.shift, "corr": s.shift.corr, "runner_up_shift": s.shift.runner_shift,
                         "runner_up_corr": s.shift.runner_corr, "accepted": s.shift.accepted}}
                       for s in f.run_shifts],
        "series": {n: {"short": f.shorts[n], "unit": f.unit[n], "status": f.status[n], "score": f.score[n],
                       "runs": f.runs_of[n],
                       "bins": [{"mean": b.mean, "se": b.se, "n": b.n} for b in f.bins[n]]} for n in f.names},
    }


# ------------------------------------------------------------------------------------------------------- pooling

def pool_bins(parts: list[list[Bin]]) -> list[Bin]:
    """The runs' folds averaged bin by bin, weighted by count (as the fold's average)."""
    out = []
    for b in range(len(parts[0])):
        n = sum(p[b].n for p in parts)
        if n == 0:
            out.append(Bin(0.0, None, 0))
            continue
        m = sum(p[b].n * p[b].mean for p in parts) / n
        var = sum((p[b].n * (p[b].se or 0.0)) ** 2 for p in parts)
        out.append(Bin(m, math.sqrt(var) / n, n))
    return out
