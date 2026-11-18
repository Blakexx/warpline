"""A synthetic launch: a known entry inside a sample, a setup, a window period, a window count and a periodic profile.

The tests use it, and it writes a SET under a directory (`bazel run //:launch_set -- build/set_synth`): a plain variant
and a stamped variant, each with several runs of sampled series, and the stamped variant's record carrying the
launch's two edges (`launch.begin`, `launch.end`, each with a tie) and the window's anchors.

THE TRUTH it keeps: a run's entry is at i0*dt + u (microseconds) in each PASS (each series group has its own start and
its own entry position u in 0..dt); the kernel then runs T = SETUP + windows*P + tail; the profile of the window,
`prof(position)`, is integrated over each sampling interval (a sample is a total over its interval) and the samples
are zero outside the launch. The stamped record is the same launch on the cycle counter, 1.8 cycles a nanosecond.

The plain variant is the same kernel without stamps: its window is 2% shorter (no stamp cost) and its profile is the
stamped variant's rotated by `plain_rot` of a window (its setup is not the stamped variant's), or, with
`plain_shape="other"`, a different profile altogether.

THE REAL PROFILER'S TRUTH, optional (the defaults keep the even launch above), each a parameter of `make_set`:
  `jitter`    each window's length varies (a standard deviation, a fraction of the period): the window anchors are
              the jittered ones, so a mean period places no sample right;
  `stretch`   the profiler's time sampling STRETCHES the kernel in real time and leaves its cycles alone: the samples'
              clock runs `stretch` times the record's;
  `ripple`    and the stretch is not even inside a window (time = cycles plus a ripple of this amplitude that vanishes
              at the window's two ends: even over a window and over the launch, not within a phase);
  `follow`    False: the samples do NOT follow the window anchors' jitter (the kernel's rhythm is regular; the record's
              anchors wobble around it), as the real kernel's do;
  `len_vary`  per run, a fraction by which that run's kernel is longer in cycles (a pass's length varies);
  `edges`     a third variant: the plain kernel with its two launch edges and nothing else (`edges_record.json`, its own
              samples), its windows the stamped variant's as fractions of its launch (manifest `windows_from`); its
              profile rotated by `edges_rot` of a window (a shifted one the check must name).
  `alternate` the window anchors alternate early and late (window k is `1 +- alternate` of the period long): the lag-1
              autocorrelation of the real record's intervals is -0.5; with `follow=False` the samples keep the regular
              rhythm;
  `entry_seed` the runs' entry positions inside their first live sample drawn at random from this seed (default: fixed ones);
  `profiles`  {series: (pass index, profile)} replacing SERIES (a profile with two humps a sample apart, say);
  `edges_windows` the edges variant's windows are this fraction of the stamped variant's, its setup the same cycles (the
              stamps add cycles to the windows and none to the setup): 1/1.04 makes the stamped windows 4% longer.
The truth of each sample's window position is returned (`pos`), so a placement is checked against it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from bisect import bisect_right

from tests.helpers import hump, write_csv

E0 = 12_000          # the scope's nanosecond origin of the entry
RATE = 1.8           # cycles a nanosecond
CYC0 = 4000          # the cycle count at the entry


def step(lo, hi, high, low):
    return lambda pos: high if lo <= pos < hi else low


def sparse(centre, width, height):
    def f(pos):
        d = abs((pos - centre + 0.5) % 1.0 - 0.5)
        return height * 0.5 * (1 + math.cos(math.pi * d / width)) if d < width else 0.0
    return f


# series -> (pass index, profile of the window position). `mio` and `drain` share a pass; `drain` is zero off its hump.
SERIES = {
    "barrier": (0, step(0.2, 0.5, 5.0, 1.0)),
    "wait": (1, hump(0.6, 0.15, 5.0, 1.0)),
    "mio": (2, hump(0.82, 0.12, 4.0, 1.0)),
    "drain": (2, sparse(0.3, 0.1, 3.0)),
}
def bumps(*centres, width=0.1, height=4.0, base=1.0):
    return lambda pos: base + sum(hump(c, width, height, 0.0)(pos) for c in centres)


OTHER = {                     # a plain variant whose sampled profiles differ in SHAPE from the stamped variant's
    "barrier": (0, bumps(0.1, 0.6)),
    "wait": (1, bumps(0.05, 0.3, 0.55, 0.8)),
    "mio": (2, bumps(0.2, 0.7)),
    "drain": (2, bumps(0.0, 0.5)),
}


def sample_values(prof, dt, i0, u, length, setup, period, total, noise, rng, sub=24):
    """Each sample a mean of `prof` over its interval within [entry, entry + length], zero outside; `total` samples."""
    te = i0 * dt + u
    out = []
    for j in range(total):
        a, b = max(j * dt, te), min((j + 1) * dt, te + length)
        if b <= a:
            out.append(0.0)
            continue
        s = sum(prof(((a + (k + 0.5) * (b - a) / sub) - te - setup) / period % 1.0) for k in range(sub)) * (b - a) / sub
        out.append(max(0.01, s / dt + rng.gauss(0, noise)))
    return out


ENTRY_SEED = None     # None: the fixed entries below; an integer: entries drawn at random from it (see `make_set(entry_seed=)`)


def entry_u(run: int, pass_index: int, dt: float, variant: int = 0) -> float:
    """The truth: where the entry falls inside the pass's first live sample, microseconds, in 0.05..0.95 of an interval."""
    if ENTRY_SEED is not None:
        return random.Random(f"{ENTRY_SEED}/{run}/{pass_index}/{variant}").uniform(0.02, 0.98) * dt
    f = (0.13 + 0.37 * run + 0.21 * pass_index + 0.11 * variant) % 0.9 + 0.05
    return f * dt


def pass_start(pass_index: int) -> int:
    return 40 + 2 * pass_index


def series_files(path, profiles, variant, runs, dt, length, setup, period, noise, seed, label):
    """One export a run: every series of every pass, each pass with its own start, entry position and sample count.
    Returns (paths, truth) with truth {(run, pass_index): u}."""
    paths, truth = [], {}
    for r in range(runs):
        cols = {}
        for pi in sorted({pi for pi, _ in profiles.values()}):
            u = entry_u(r, pi, dt, variant)
            truth[(r, pi)] = u
            i0 = pass_start(pi)
            i1 = int((i0 * dt + u + length) // dt)
            total = i1 + 1 + 25 + 7 * pi
            for k, (name, (p, prof)) in enumerate(profiles.items()):
                if p == pi:
                    rng = random.Random(seed + 1000 * r + 10 * pi + k)
                    cols[name] = sample_values(prof, dt, i0, u, length, setup, period, total, noise, rng)
        paths.append(write_csv(os.path.join(path, f"{label}_r{r}.csv"), cols, interval=dt))
    return paths, truth


class Cycles:
    """One run's kernel on the cycle counter and in the profiler's time. Windows (their anchors, cycles from the entry)
    are `anchors`; the profiler's time at cycle c is `te + stretch * (c + ripple(c)) / 1800` microseconds, the ripple
    `ripple * D / (2 pi) * sin(2 pi pos)` inside a window of length D (it vanishes at the window's ends)."""

    def __init__(self, scale, setup, period, mult, tail, stretch, ripple, win_scale=1.0):
        per = 1000 * RATE
        self.anchors = [scale * per * (setup + win_scale * sum(mult[:w]) * period) for w in range(len(mult) + 1)]
        self.t_c = scale * per * (setup + win_scale * (sum(mult) * period + tail))
        self.stretch, self.ripple, self.per = stretch, ripple, per

    def pos(self, c):
        a = self.anchors
        k = min(max(bisect_right(a, c) - 1, 0), len(a) - 2)
        return (c - a[k]) / (a[k + 1] - a[k]) % 1.0

    def warp(self, c):
        a = self.anchors
        if not self.ripple or c < a[0] or c >= a[-1]:
            return c
        k = bisect_right(a, c) - 1
        d = a[k + 1] - a[k]
        return c + self.ripple * d / (2 * math.pi) * math.sin(2 * math.pi * (c - a[k]) / d)

    def cycle_at(self, x):
        """The cycle whose warped value is x (cycles of profiler time at the nominal rate)."""
        lo, hi = 0.0, self.t_c
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if self.warp(mid) < x:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def length_us(self):
        return self.stretch * self.warp(self.t_c) / self.per


def cyc_files(path, profiles, variant, runs, dt, setup, period, mult, tail, noise, seed, label, stretch, ripple,
              len_vary, scale=1.0, rot=0.0, sub=24, win_scale=1.0):
    """One export a run, as `series_files`, from the kernel on the cycle counter: run r is longer by `len_vary[r]`, the
    profiler's clock runs `stretch` times the cycle counter's with a ripple inside each window. Returns (paths, truth,
    pos): pos[(run, pass_index)] = {sample index: the TRUE window position of the sample's centre}."""
    paths, truth, pos = [], {}, {}
    for r in range(runs):
        k = Cycles(scale * (1 + (len_vary[r] if r < len(len_vary) else 0.0)), setup, period, mult, tail, stretch, ripple,
                   win_scale)
        cols = {}
        for pi in sorted({pi for pi, _ in profiles.values()}):
            u = entry_u(r, pi, dt, variant)
            truth[(r, pi)] = u
            i0 = pass_start(pi)
            te = i0 * dt + u
            x_end = te + k.length_us()
            i1 = int(x_end // dt)
            total = i1 + 1 + 25 + 7 * pi
            pts, pp = [], {}
            for j in range(total):
                a, b = max(j * dt, te), min((j + 1) * dt, x_end)
                if b <= a:
                    pts.append(None)
                    continue
                pts.append(((b - a) / dt, [k.pos(k.cycle_at((a + (q + 0.5) * (b - a) / sub - te) * k.per / stretch))
                                           for q in range(sub)]))
                c = k.cycle_at((0.5 * (a + b) - te) * k.per / stretch)
                if k.anchors[2] <= c < k.anchors[-1]:
                    pp[j] = k.pos(c)
            pos[(r, pi)] = pp
            for q, (name, (p, prof)) in enumerate(profiles.items()):
                if p != pi:
                    continue
                rng = random.Random(seed + 1000 * r + 10 * pi + q)
                cols[name] = [0.0 if pt is None else
                              max(0.01, pt[0] * sum(prof((x - rot) % 1.0) for x in pt[1]) / sub + rng.gauss(0, noise))
                              for pt in pts]
        paths.append(write_csv(os.path.join(path, f"{label}_r{r}.csv"), cols, interval=dt))
    return paths, truth, pos


def stamped_record(windows, setup, period, length, mult=None):
    """The stamped variant's record, times in nanoseconds on the scope's axis, the cycle counter 1.8 a nanosecond:
    four lanes of one scope, each `launch.begin` .. `launch.end` (a tie each) with phases tiling between them; lane 0
    carries the window's anchor `phase.a.begin` (tied) once a window; lanes 2 and 3 wait on barrier B (declared
    without arrivals) from 0.2 to 0.5 of every window, which the sampled `barrier` series sees."""
    cyc = lambda ns: CYC0 + round((ns - E0) * RATE)
    names: dict[str, int] = {}
    nid = lambda n: names.setdefault(n, len(names))
    lanes, stamps, ties = [], [], []
    roles = ["front", "back", "x", "x"]
    for k, role in enumerate(roles):
        lane = f"cta0/warp{k}"
        lanes.append({"id": lane, "role": role})
        t_begin = E0 + 40 * k
        t_end = E0 + round(length * 1000) - 40 * (3 - k)
        ev = [(t_begin, 0, "launch.begin"), (t_begin, 1, "phase.setup.begin")]
        t_first = E0 + round(setup * 1000)
        mult = mult or [1.0] * windows
        anchors = [t_first + round(sum(mult[:w]) * period * 1000) for w in range(windows + 1)]
        wob = lambda w, salt: (round(((w * 7919 + salt * 104729) % 13 - 6) / 6 * 0.15 * period * 1000)
                               if any(m != 1.0 for m in mult) else 0)    # the other begins are less regular than a's
        wl = lambda w: (anchors[min(w + 1, windows)] - anchors[w]) / 1000.0 if w < windows else period
        if k == 0:
            for w, t in enumerate(anchors):
                ev += [(t, 2, "phase.setup.end" if w == 0 else "phase.b.end"), (t, 3, "phase.a.begin")]
                tb = t + round(0.5 * wl(w) * 1000) + wob(w, 1)
                ev += [(tb, 4, "phase.a.end"), (tb, 5, "phase.b.begin")]
                ties.append([lane, cyc(t), t])
        elif k == 1:
            for w, t in enumerate(anchors):
                t1, t2 = t + round(0.1 * wl(w) * 1000) + wob(w, 2), t + round(0.6 * wl(w) * 1000) + wob(w, 3)
                ev += [(t1, 2, "phase.setup.end" if w == 0 else "phase.d.end"), (t1, 3, "phase.c.begin")]
                ev += [(t2, 4, "phase.c.end"), (t2, 5, "phase.d.begin")]
        else:
            ev += [(t_first, 2, "phase.setup.end"), (t_first, 3, "phase.work.begin")]
            for w, t in enumerate(anchors[:-1]):
                ev += [(t + round(0.2 * wl(w) * 1000), 4, "barrier.B.wait.begin"),
                       (t + round(0.5 * wl(w) * 1000), 5, "barrier.B.wait.end")]
        ev = [e for e in ev if e[0] < t_end]
        ev.sort(key=lambda e: (e[0], e[1]))
        # the launch's end closes the lane's open phase
        open_ = None
        for t, _o, n in ev:
            if n.endswith(".begin") and n.startswith("phase."):
                open_ = n[:-len(".begin")]
            elif n.endswith(".end") and n.startswith("phase."):
                open_ = None
        if open_:
            ev.append((t_end, 6, open_ + ".end"))
        ev.append((t_end, 7, "launch.end"))
        for t, _o, n in ev:
            stamps.append([lane, nid(n), cyc(t)])
        ties += [[lane, cyc(t_begin), t_begin], [lane, cyc(t_end), t_end]]
    ties.sort(key=lambda x: (x[0], x[1]))
    # a lane's ties in order; dedupe equal cycle counts
    seen, uniq = set(), []
    for t in ties:
        if (t[0], t[1]) not in seen:
            seen.add((t[0], t[1]))
            uniq.append(t)
    return {"format": "warpline/0",
            "clock": {"unit": "cycles", "hz": None, "time_unit": "ns", "time_origin": "the synthetic launch's own"},
            "names": {str(i): n for n, i in names.items()}, "lanes": lanes, "stamps": stamps,
            "dropped": {}, "facts": {"barriers": {"B": {"count": 0}}, "operations": {}}, "ties": uniq}


def edges_record(length):
    """The plain kernel with its two edges and nothing else: four lanes, each `launch.begin` and `launch.end` with a
    tie (times in nanoseconds on the scope's axis, the cycle counter 1.8 a nanosecond)."""
    cyc = lambda ns: CYC0 + round((ns - E0) * RATE)
    lanes, stamps, ties, names = [], [], [], {"launch.begin": 0, "launch.end": 1}
    for k in range(4):
        lane = f"cta0/warp{k}"
        lanes.append({"id": lane, "role": "x"})
        t0, t1 = E0 + 40 * k, E0 + round(length * 1000) - 40 * (3 - k)
        stamps += [[lane, 0, cyc(t0)], [lane, 1, cyc(t1)]]
        ties += [[lane, cyc(t0), t0], [lane, cyc(t1), t1]]
    return {"format": "warpline/0",
            "clock": {"unit": "cycles", "hz": None, "time_unit": "ns", "time_origin": "the synthetic launch's own"},
            "names": {str(i): n for n, i in names.items()}, "lanes": lanes, "stamps": stamps, "dropped": {},
            "facts": {}, "ties": ties}


def make_set(out_dir, runs=3, dt=1.5, period=3.62, windows=64, setup=8.7, tail=2.2, plain_rot=0.25,
             plain_shape="same", noise=0.1, seed=7, plain_cost=0.98, jitter=0.0, stretch=1.0, ripple=0.0,
             len_vary=(), edges=False, edges_rot=0.0, edges_cost=0.98, follow=True, alternate=0.0, profiles=None,
             edges_windows=None, entry_seed=None):
    """Write the set under `out_dir` and return {"manifest", "truth", "length", "setup", "period", "plain_rot", ...}.
    `jitter`, `stretch`, `ripple`, `len_vary`, `edges`, `edges_rot`: see the module's docstring."""
    global ENTRY_SEED
    ENTRY_SEED = entry_seed
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed + 99)
    profiles = profiles or SERIES
    mult = [max(0.5, 1 + rng.gauss(0, jitter)) for _ in range(windows)] if jitter else [1.0] * windows
    if alternate:
        mult = [1 + alternate * (-1) ** k for k in range(windows)]
    length = setup + sum(mult) * period + tail
    rec = stamped_record(windows, setup, period, length, mult)
    rec_path = os.path.join(out_dir, "record.json")
    with open(rec_path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
    general = bool(jitter or alternate or stretch != 1.0 or ripple or any(len_vary))
    smult = mult if follow else [1.0] * windows              # what the samples' kernel does
    pos = {}
    if general:
        st_paths, st_truth, pos = cyc_files(out_dir, profiles, 0, runs, dt, setup, period, smult, tail, noise, seed,
                                            "stamped", stretch, ripple, list(len_vary))
    else:
        st_paths, st_truth = series_files(out_dir, profiles, 0, runs, dt, length, setup, period, noise, seed, "stamped")
    # the plain variant: a window plain_cost of the stamped one's, its setup the stamped one's plus plain_rot windows
    pp = period * plain_cost
    plain_setup = setup + plain_rot * pp
    plain_len = plain_setup + windows * pp + tail
    pl_paths, pl_truth = series_files(out_dir, OTHER if plain_shape == "other" else profiles, 1, runs, dt, plain_len,
                                      plain_setup, pp, noise, seed + 500, "plain")
    manifest = {"kernel": "a synthetic launch", "variants": [
        {"name": "plain", "plain": True, "stamps": None, "samples": [os.path.basename(p) for p in pl_paths]},
        {"name": "stamped", "plain": False, "stamps": "record.json", "samples": [os.path.basename(p) for p in st_paths]}]}
    out = {"truth": st_truth, "plain_truth": pl_truth, "length": length, "setup": setup, "period": period, "dt": dt,
           "windows": windows, "plain_rot": plain_rot, "mult": mult, "pos": pos}
    if edges:
        # the same kernel without the stamps: every duration edges_cost of the stamped one's, its two edges recorded
        e_len = length * edges_cost if edges_windows is None else setup + edges_windows * (length - setup)
        e_rec = os.path.join(out_dir, "edges_record.json")
        with open(e_rec, "w") as f:
            json.dump(edges_record(e_len), f, separators=(",", ":"))
        e_paths, e_truth, e_pos = cyc_files(out_dir, profiles, 2, runs, dt, setup, period, smult, tail, noise, seed + 900,
                                            "edges", stretch, ripple, list(len_vary),
                                            scale=edges_cost if edges_windows is None else 1.0, rot=edges_rot,
                                            win_scale=1.0 if edges_windows is None else edges_windows)
        manifest["variants"].append({"name": "edges", "plain": False, "stamps": "edges_record.json",
                                     "samples": [os.path.basename(p) for p in e_paths], "windows_from": "stamped"})
        out.update({"edges_record": e_rec, "edges": e_paths, "edges_truth": e_truth, "edges_pos": e_pos,
                    "edges_length": e_len})
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    out.update({"manifest": mpath, "record": rec_path, "stamped": st_paths, "plain": pl_paths})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="launch_set", description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--plain-rot", type=float, default=0.25, help="the plain variant's rotation, windows")
    ap.add_argument("--plain-shape", choices=("same", "other"), default="same")
    ap.add_argument("--noise", type=float, default=0.1)
    a = ap.parse_args(argv)
    os.chdir(os.environ.get("BUILD_WORKING_DIRECTORY", "."))     # as `bazel run //:cli` does
    r = make_set(a.out_dir, runs=a.runs, plain_rot=a.plain_rot, plain_shape=a.plain_shape, noise=a.noise)
    print(f"wrote {r['manifest']}: T {r['length']:.3f} us = {r['length'] / r['dt']:.1f} samples, setup {r['setup']} us, "
          f"window {r['period']} us = {r['period'] / r['dt']} samples x {r['windows']}, plain rotation {r['plain_rot']}")
    print("truth, entry position u in the first live sample (us), stamped variant: "
          + ", ".join(f"run {k[0]} pass {k[1]}: {v:.3f}" for k, v in sorted(r["truth"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
