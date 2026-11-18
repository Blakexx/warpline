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
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

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


def entry_u(run: int, pass_index: int, dt: float, variant: int = 0) -> float:
    """The truth: where the entry falls inside the pass's first live sample, microseconds, in 0.05..0.95 of an interval."""
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


def stamped_record(windows, setup, period, length):
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
        anchors = [t_first + round(w * period * 1000) for w in range(windows + 1)]
        if k == 0:
            for w, t in enumerate(anchors):
                ev += [(t, 2, "phase.setup.end" if w == 0 else "phase.b.end"), (t, 3, "phase.a.begin")]
                ev += [(t + round(0.5 * period * 1000), 4, "phase.a.end"), (t + round(0.5 * period * 1000), 5, "phase.b.begin")]
                ties.append([lane, cyc(t), t])
        elif k == 1:
            for w, t in enumerate(anchors):
                t1, t2 = t + round(0.1 * period * 1000), t + round(0.6 * period * 1000)
                ev += [(t1, 2, "phase.setup.end" if w == 0 else "phase.d.end"), (t1, 3, "phase.c.begin")]
                ev += [(t2, 4, "phase.c.end"), (t2, 5, "phase.d.begin")]
        else:
            ev += [(t_first, 2, "phase.setup.end"), (t_first, 3, "phase.work.begin")]
            for w, t in enumerate(anchors[:-1]):
                ev += [(t + round(0.2 * period * 1000), 4, "barrier.B.wait.begin"),
                       (t + round(0.5 * period * 1000), 5, "barrier.B.wait.end")]
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


def make_set(out_dir, runs=3, dt=1.5, period=3.62, windows=64, setup=8.7, tail=2.2, plain_rot=0.25,
             plain_shape="same", noise=0.1, seed=7, plain_cost=0.98):
    """Write the set under `out_dir` and return {"manifest", "truth", "length", "setup", "period", "plain_rot", ...}."""
    os.makedirs(out_dir, exist_ok=True)
    length = setup + windows * period + tail
    rec = stamped_record(windows, setup, period, length)
    rec_path = os.path.join(out_dir, "record.json")
    with open(rec_path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
    st_paths, st_truth = series_files(out_dir, SERIES, 0, runs, dt, length, setup, period, noise, seed, "stamped")
    # the plain variant: a window plain_cost of the stamped one's, its setup the stamped one's plus plain_rot windows
    pp = period * plain_cost
    plain_setup = setup + plain_rot * pp
    plain_len = plain_setup + windows * pp + tail
    pl_paths, pl_truth = series_files(out_dir, OTHER if plain_shape == "other" else SERIES, 1, runs, dt, plain_len,
                                      plain_setup, pp, noise, seed + 500, "plain")
    manifest = {"kernel": "a synthetic launch", "variants": [
        {"name": "plain", "plain": True, "stamps": None, "samples": [os.path.basename(p) for p in pl_paths]},
        {"name": "stamped", "plain": False, "stamps": "record.json", "samples": [os.path.basename(p) for p in st_paths]}]}
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    return {"manifest": mpath, "record": rec_path, "stamped": st_paths, "plain": pl_paths, "truth": st_truth,
            "plain_truth": pl_truth, "length": length, "setup": setup, "period": period, "dt": dt,
            "windows": windows, "plain_rot": plain_rot}


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
