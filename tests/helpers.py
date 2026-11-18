"""Build small records from a compact description."""
from __future__ import annotations

from warpline import record


def make(lanes, stamps, facts=None, dropped=None, issued=None):
    """lanes: {id: role}; stamps: (lane, name, time[, payload]); facts: the declared facts."""
    names: dict[str, int] = {}
    rows = []
    for s in stamps:
        nid = names.setdefault(s[1], len(names))
        rows.append([s[0], nid, s[2]] + ([s[3]] if len(s) > 3 else []))
    doc = {
        "format": "warpline/0",
        "clock": {"unit": "cycles", "hz": None},
        "names": {str(i): n for n, i in names.items()},
        "lanes": [{"id": k, "role": v, **({"issued": issued[k]} if issued and k in issued else {})}
                  for k, v in lanes.items()],
        "stamps": rows,
        "dropped": dropped or {},
        "facts": facts or {},
    }
    return record.from_dict(doc)


def three_lane():
    """Three lanes in one scope `k`; the answer of its critical path is stated in test_path.

    c: region load 0..50; arrives at Y and X at 50; waits on X 50..90.
    b: region work 0..80; waits on Y 20..70 (released by c's arrival at 50); arrives at X at 80.
    a: region w 0..100; waits on X 10..90 (X has count 2: c's arrival at 50 and b's at 80 -- b's, the last, releases).
    """
    lanes = {"k/a": "consumer", "k/b": "middle", "k/c": "producer"}
    stamps = [
        ("k/a", "w.begin", 0), ("k/a", "barrier.X.wait.begin", 10), ("k/a", "barrier.X.wait.end", 90),
        ("k/a", "w.end", 100),
        ("k/b", "work.begin", 0), ("k/b", "barrier.Y.wait.begin", 20), ("k/b", "barrier.Y.wait.end", 70),
        ("k/b", "barrier.X.arrive", 80), ("k/b", "work.end", 80),
        ("k/c", "load.begin", 0), ("k/c", "load.end", 50), ("k/c", "barrier.Y.arrive", 50),
        ("k/c", "barrier.X.arrive", 50), ("k/c", "barrier.X.wait.begin", 50), ("k/c", "barrier.X.wait.end", 90),
    ]
    facts = {"barriers": {"X": {"count": 2}, "Y": {"count": 1}}}
    return make(lanes, stamps, facts)


def slack_example():
    """Four lanes in one scope `k`; the slack of q's wait on S is stated in test_reader.Slack.

    r: region load 0..85; arrives at S at 30 (S has one arrival) and at Z at 85.
    q: region side 0..70; waits on S 10..40 (released by r's arrival at 30); works; arrives at Z at 60.
    m: region run 0..100; waits on Z 50..90 (Z has count 2: q at 60, r at 85 -- r's, the last, releases).
    The critical path from m's end is r's work 0..85, Z's release latency 85..90 on m, m's work 90..100.
    q's wait on S is off the path. Its next dependency on the path is its arrival at Z (60), which the path needs at
    85: the releaser of S could have arrived 25 cycles later without q holding Z back.
    """
    lanes = {"k/m": "main", "k/q": "side", "k/r": "load"}
    stamps = [
        ("k/m", "run.begin", 0), ("k/m", "barrier.Z.wait.begin", 50), ("k/m", "barrier.Z.wait.end", 90),
        ("k/m", "run.end", 100),
        ("k/q", "side.begin", 0), ("k/q", "barrier.S.wait.begin", 10), ("k/q", "barrier.S.wait.end", 40),
        ("k/q", "barrier.Z.arrive", 60), ("k/q", "side.end", 70),
        ("k/r", "load.begin", 0), ("k/r", "barrier.S.arrive", 30), ("k/r", "barrier.Z.arrive", 85),
        ("k/r", "load.end", 85),
    ]
    facts = {"barriers": {"Z": {"count": 2}, "S": {"count": 1}}}
    return make(lanes, stamps, facts)


def regular(period=1000, windows=12, jitter=(-4, -2, 0, 2, 4), stretch=1.0, extra=(), warm=300):
    """A regular kernel with jitter, as a record. Scope `k`, lanes a (role front) and b (role back).

    Window w begins at period*w on a (`window.begin`, the anchor). Offsets from the begin, before jitter:
      a: window.begin 0, phase.x.begin 400, phase.x.end 700      (fixed)
      b: phase.y.begin 300 + jitter[w % 5], phase.y.end 600      (the first two windows are `warm` late)
      b also stamps `phase.y.begin` once more after the last window begin (a partial window: must not count).
    `stretch` scales every offset and the period. `extra`: (lane, name, offset) stamps added to every window.
    """
    stamps = []
    for w in range(windows):
        t0 = int(period * stretch * w)
        late = warm if w < 2 else 0
        stamps += [("k/a", "window.begin", t0), ("k/a", "phase.x.begin", t0 + int(400 * stretch)),
                   ("k/a", "phase.x.end", t0 + int(700 * stretch)),
                   ("k/b", "phase.y.begin", t0 + int((300 + jitter[w % len(jitter)]) * stretch) + late),
                   ("k/b", "phase.y.end", t0 + int(600 * stretch) + late)]
        for lane, name, off in extra:
            stamps.append((lane, name, t0 + int(off * stretch)))
    t_last = int(period * stretch * windows)
    stamps += [("k/a", "window.begin", t_last), ("k/b", "phase.y.begin", t_last + 300)]
    stamps.sort(key=lambda s: s[2])
    return make({"k/a": "front", "k/b": "back"}, stamps)


# ---------------------------------------------------------------------------------------------- sampled series
import math
import random

BARRIER = "pmsampling:smsp__warps_issue_stalled_barrier.avg"
INTERVAL = "profiler__pmsampler_interval_time"


def hump(centre=0.3, width=0.1, height=10.0, base=2.0):
    """A periodic profile of a window position in [0, 1): a base and one raised-cosine hump of `width` either side."""
    def f(pos):
        d = abs((pos - centre + 0.5) % 1.0 - 0.5)
        return base + (height * 0.5 * (1 + math.cos(math.pi * d / width)) if d < width else 0.0)
    return f


def synth(profile, period, n, start=40, noise=0.2, seed=1, tail=30, origin=0.0, phase=None, sub=16):
    """The samples of a kernel whose window repeats every `period` samples with the true profile `profile` (a function
    of a window position): zeros, then `n` live samples, then zeros. A sample is a total over its interval, so each
    value is the profile averaged over the sample's span (a box of 1/period of a window), plus noise. `origin`: the
    window position of the first live sample; `phase`: a function of the live time (in samples) to a window position,
    replacing the steady `origin + t / period` (a drifting period)."""
    rng = random.Random(seed)
    ph = phase or (lambda t: origin + t / period)
    vals = [0.0] * start
    for j in range(n):
        v = sum(profile(ph(j + (k + 0.5) / sub) % 1.0) for k in range(sub)) / sub
        vals.append(max(0.01, v + rng.gauss(0, noise)))
    return vals + [0.0] * tail


def blurred_truth(profile, period, lo, hi, sub=48, inner=24):
    """The mean over window positions [lo, hi) of the profile boxed by one sample: what a bin's mean estimates."""
    tot = 0.0
    for a in range(sub):
        pos = lo + (a + 0.5) / sub * (hi - lo)
        tot += sum(profile((pos + (k + 0.5) / inner / period) % 1.0) for k in range(inner)) / inner
    return tot / sub


def write_csv(path, series, interval=1.5, passes=6):
    """An export of three rows (names, units, values): an ID column, the sampled series, the sampling interval."""
    names, units, cells = ["ID"], [""], ["0"]
    for name, vals in series.items():
        full = name if name.startswith("pmsampling:") else "pmsampling:smsp__warps_issue_stalled_" + name + ".avg"
        names.append(full)
        units.append("warp")
        cells.append(f"{sum(vals):.2f} (" + "; ".join(f"{v:g}" for v in vals) + ")")
    names.append(INTERVAL)
    units.append("us")
    cells.append(f"{interval:.2f} (" + "; ".join(f"{interval:.2f}" for _ in range(passes)) + ")")
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in (names, units, cells):
            w.writerow(row)
    return path
