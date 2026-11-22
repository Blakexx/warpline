"""The samples joined to the stamps on CYCLES, each pass scaled by its own two edges and its runs fitted JOINTLY.

The synthetic launch (`launch_gen`) here has the real profiler's behaviour put in with known truth: the window's length
jitters (or alternates early and late), the profiler's clock runs a known factor faster than the cycle counter with a
ripple inside the window, each run's kernel is a little longer or shorter than the record's, and each run's entry falls
at a known place inside its first live sample. What the placement recovers, and what its stated uncertainty claims, is
checked against that truth."""
import cmath
import contextlib
import io
import json
import math
import os
import tempfile
import unittest

from warpline.recorder import launch, overlay, placement, record, sampling, variants, window
from warpline.recorder.cli import main
from tests.recorder import launch_gen
from tests.recorder.helpers import hump

PASS_INDEX = {"barrier": 0, "wait": 1, "drain": 2}
RIPPLE = 0.3
LEN_VARY = [0.0, 0.012, -0.01, 0.015]
FAST = dict(boot=8, report_frames=False)           # fewer resamples, only the chosen frame's sharpness


def pass_index(key: str) -> int:
    return PASS_INDEX[key.split("stalled_")[-1].split(".")[0]]


def circ(a, b):
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def phase(vals):
    lo, n = min(vals), len(vals)
    c = sum((v - lo) * math.cos(2 * math.pi * (b + 0.5) / n) for b, v in enumerate(vals))
    s = sum((v - lo) * math.sin(2 * math.pi * (b + 0.5) / n) for b, v in enumerate(vals))
    return (math.atan2(s, c) / (2 * math.pi)) % 1.0


def place(set_, cfg):
    lc = launch.launch_of(record.load(set_["record"]))
    ex = [sampling.read_export(p) for p in set_["stamped"]]
    return lc, ex, placement.place_variant(ex, lc.cycles, placement.Frame(lc.fractions, "r"), cfg)


def true_fold_error(set_, vp, pp, series="barrier"):
    """The placed fold of a series of an EVEN synthetic launch (no ripple, one entry a run) against the fold of the same
    samples at their TRUE window positions: the rotation, in bins, by which the placement lies later than the truth."""
    pi = pass_index(pp.key)
    nm = next(n for n in pp.names if n.endswith(series + ".avg"))
    n, dt, per, setup = 8, set_["dt"], set_["period"], set_["setup"]
    placed, true = [], []
    for r in pp.fed:
        rp = pp.runs[r]
        js, xs = placement.sample_positions(rp.p, rp.a, rp.d, vp.frame)
        vals = next(s for s in sampling.read_export(set_["stamped"][r]).series if s.name == nm).values
        v = [vals[j] for j in js]
        u = set_["truth"][(r, pi)]
        xt = [(((j + 0.5 - rp.p.i0) * dt - u - setup) / per) % 1.0 for j in js]
        placed.append(sampling.fold_bins(xs, v, 1.0, n))
        true.append(sampling.fold_bins(xt, v, 1.0, n))
    return placement.fold_rotation([b.mean for b in sampling.pool_bins(true)], [b.mean for b in sampling.pool_bins(placed)])


class Real(unittest.TestCase):
    """Jitter 6% a window, time stretched 1.15 against cycles with a ripple, four runs of lengths 0, +1.2%, -1%, +1.5%."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.set = launch_gen.make_set(cls.dir.name, runs=4, jitter=0.06, stretch=1.15, ripple=RIPPLE, len_vary=LEN_VARY)
        cls.rec = record.load(cls.set["record"])
        cls.lc = launch.launch_of(cls.rec)
        cls.exports = [sampling.read_export(p) for p in cls.set["stamped"]]
        cls.frame = placement.Frame(cls.lc.fractions, "the record")
        cls.vp = placement.place_variant(cls.exports, cls.lc.cycles, cls.frame, sampling.Config(boot=12))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_passes_lengths_vary_and_each_is_reported_in_cycles_a_sample(self):
        live = {c.run: c.live for c in self.vp.checks if pass_index(c.key) == 0}
        self.assertGreater(len(set(live.values())), 1, live)          # the live spans differ by several samples
        self.assertGreaterEqual(max(live.values()) - min(live.values()), 3)
        for c in self.vp.checks:
            self.assertAlmostEqual(c.cps, self.lc.cycles / c.live)
            self.assertFalse(c.flagged, (c.run, c.key, c.ratio))      # all within 3% of the median: the old one-length
        self.assertTrue(2000 < self.vp.cps_median < 2600, self.vp.cps_median)   # check would have refused most of them

    def test_every_run_is_fed_and_the_placement_recovers_the_true_window_position(self):
        # the true window position of every sample's centre (the generator keeps it) against the placed one: the stated
        # half-width plus the ripple inside the window, which the placement cannot see
        allow = RIPPLE / (2 * math.pi)
        for key, pp in self.vp.passes.items():
            self.assertTrue(pp.placed, (key, pp.findings))
            self.assertEqual(len(pp.fed), 4, (key, pp.findings))              # no run left out
            half = max(s.half for s in pp.series.values() if s.varying)
            for r in pp.fed:
                rp = pp.runs[r]
                js, xs = placement.sample_positions(rp.p, rp.a, rp.d, self.vp.frame)
                truth = self.set["pos"][(r, pass_index(key))]
                errs = sorted(circ(x, truth[j]) for j, x in zip(js, xs) if j in truth)
                self.assertGreater(len(errs), 100)
                med = errs[len(errs) // 2]
                self.assertLessEqual(med, half + allow, (key, r, med, half))
                self.assertLessEqual(errs[int(0.9 * len(errs))], 2 * half + 2 * allow)

    def test_the_scale_is_the_one_that_was_put_in(self):
        dt = self.set["dt"]
        for key, pp in self.vp.passes.items():
            for r in pp.fed:
                rp = pp.runs[r]
                k = launch_gen.Cycles(1 + LEN_VARY[r], 8.7, 3.62, self.set["mult"], 2.2, 1.15, RIPPLE)
                d_true = k.length_us() / dt - rp.p.n
                self.assertLess(abs(rp.d - d_true), 0.45, (key, r, rp.d, d_true))
                self.assertGreaterEqual(rp.d_plateau[1], rp.d_plateau[0])

    def test_the_folded_profile_has_the_true_shape(self):
        n = self.vp.bins
        s = self.vp.series()
        # `barrier`: high over window positions 0.2..0.5 (blurred by a sample's box and by the placement)
        # (to within the placement's own stated half-width and the blur)
        sb, sw = s["smsp__warps_issue_stalled_barrier.avg"], s["smsp__warps_issue_stalled_wait.avg"]
        vals = [b.mean for b in sb.bins]
        self.assertLess(circ(phase(vals), 0.35), sb.half + 0.03, vals)
        # `wait`: a hump at 0.6
        vals = [b.mean for b in sw.bins]
        self.assertLess(circ(phase(vals), 0.6), sw.half + 0.03, vals)
        for x in s.values():
            self.assertLessEqual(x.half * n, 2.0, (x.short, x.half))      # tight enough to be useful: about a bin

    def test_the_overlay_reports_the_frame_the_fit_the_spread_the_alias_check_and_the_assumption(self):
        view = overlay.view_of(self.rec)
        pl = overlay.place(view, self.lc, self.exports, sampling.Config(boot=8))
        text = overlay.format_overlay(pl, "record.json", ["x"])
        for needle in ("THE JOIN IS ON CYCLES", "STATED ASSUMPTION", "linear in the kernel's cycles", "cycles a sample",
                       "THE FIT of a pass is JOINT", "sharpness of the pooled joint fit", "ALIAS CHECK", "spread of",
                       "box of the common shift", "half-width", "windows =", "bins", "OVERLAY:", "front", "back", "x ",
                       "run(s) fed it", "+-", "the raw anchors", "a mean period", "a local fit"):
            self.assertIn(needle, text)
        self.assertEqual(len(pl.series), 4)
        json.dumps(overlay.place_to_dict(pl))


class TheFrameFollowsTheSamples(unittest.TestCase):
    """The anchors' window-to-window jitter is not the samples' rhythm on the real data (an early window is followed by
    a late one, lag-1 autocorrelation -0.5). A local fit of the anchors folds sharper than the raw anchors there, and the
    raw anchors are kept where the samples do follow them."""

    def run_set(self, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name, runs=4, stretch=1.1, **kw)
        lc, ex, vp = place(s, sampling.Config(boot=6))
        return s, lc, vp

    def test_local_track_removes_an_alternation_and_keeps_a_drift(self):
        n = 62
        regular = [0.0368 + 0.0155 * k for k in range(n)]
        alt = [r + 0.12 * 0.0155 * (k % 2) for k, r in enumerate(regular)]
        for h in (2.0, 4.0):
            tr = placement.local_track(alt, h)
            mid = [r + 0.06 * 0.0155 for r in regular]                  # the alternation's mean: the track's own centre
            before = max(abs(a - m) for a, m in zip(alt, mid))
            after = max(abs(a - m) for a, m in zip(tr[3:-3], mid[3:-3]))
            self.assertLess(after, 0.1 * before, (h, before, after))
        drift = [r + 0.003 * math.sin(math.pi * k / n) for k, r in enumerate(regular)]    # a slow bow of 0.2 window
        tr = placement.local_track(drift, 3.0)
        self.assertLess(max(abs(a - b) for a, b in zip(tr, drift)), 0.0005)
        self.assertTrue(all(b > a for a, b in zip(tr, tr[1:])))

    def test_alternating_anchors_the_local_frame_folds_sharper_than_the_raw_anchors_in_every_pass(self):
        s, lc, vp = self.run_set(alternate=0.08, follow=False)
        raw = vp.frame_q["the raw anchors"]
        self.assertEqual(len(raw), 3)
        local = {lab: q for lab, q in vp.frame_q.items() if lab.startswith("a local fit")}
        self.assertEqual(len(local), len(placement.H_GRID))
        for lab, q in local.items():
            for key in raw:
                self.assertGreater(q[key], raw[key], (lab, key))      # the smoothed track beats the raw anchors
        self.assertNotEqual(vp.frame.kind, "anchors")
        for key, q in raw.items():
            self.assertGreater(vp.passes[key].q_frames[vp.frame.label], q)
        self.assertIn("chosen", vp.frame_why)

    def test_where_the_samples_follow_the_anchors_the_raw_anchors_are_the_frame(self):
        s, lc, vp = self.run_set(jitter=0.06, follow=True)
        self.assertEqual(vp.frame.kind, "anchors", vp.frame_why)
        mean = vp.frame_q["a mean period"]
        for key, q in vp.frame_q["the raw anchors"].items():
            self.assertGreater(q, 1.4 * mean[key], key)

    def test_where_they_do_not_the_smooth_frame_is_chosen_and_each_frame_can_be_asked_for(self):
        s, lc, vp = self.run_set(jitter=0.06, follow=False)
        self.assertNotEqual(vp.frame.kind, "anchors")
        for kind, want in (("anchors", "anchors"), ("mean", "mean")):
            _, _, v = place(s, sampling.Config(frame=kind, **FAST))
            self.assertEqual(v.frame.kind, want)
            self.assertIn("by request", v.frame_why)
        _, _, v = place(s, sampling.Config(smooth=4.0, **FAST))
        self.assertEqual((v.frame.kind, v.frame.h), ("local", 4.0))


class WeakNoisySeries(unittest.TestCase):
    """A weak, noisy series (the step's height under half the noise's spread): the runs' relative rotations, each read
    from a pair of folds, are off by a fifth of a window or more for some run; the joint fit, which asks every run at once
    for one pooled profile, recovers the truth, and no run is left out."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.set = launch_gen.make_set(cls.dir.name, runs=6, noise=2.0, seed=5)
        cls.lc, cls.ex, cls.vp = place(cls.set, sampling.Config(frame="anchors", **FAST))
        cls.pp = next(p for k, p in cls.vp.passes.items() if "barrier" in k)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def pairwise_rotation(self, r):
        """Run r's profile against run 0's from the two folds alone (first three harmonics' cross-spectrum, entries and
        scales at their nominal values): what a rotation tied pair by pair would read."""
        def spec(run):
            p = self.pp.runs[run].p
            js, xs = placement.sample_positions(p, 0.5, 0.0, self.vp.frame)
            v = next(x for x in self.ex[run].series if x.name.endswith("barrier.avg")).values
            vals = [v[j] for j in js]
            m = sum(vals) / len(vals)
            return [sum((y - m) * cmath.exp(-2j * math.pi * h * x) for x, y in zip(xs, vals)) / len(vals) for h in (1, 2, 3)]
        a, b = spec(0), spec(r)
        return placement._best_rotation([y * x.conjugate() for x, y in zip(a, b)])

    def test_a_pair_by_pair_rotation_is_off_for_some_run(self):
        kappa = self.set["dt"] / self.set["period"]
        off = []
        for r in range(1, 6):
            truth = kappa * (self.set["truth"][(r, 0)] - self.set["truth"][(0, 0)]) / self.set["dt"]
            off.append(circ(self.pairwise_rotation(r), truth))
        self.assertGreater(max(off), 0.2, off)            # a fifth of a window: 1.6 bins

    def test_the_joint_fit_recovers_the_truth_with_every_run_fed(self):
        self.assertTrue(self.pp.placed, self.pp.findings)
        self.assertEqual(len(self.pp.fed), 6)
        err = true_fold_error(self.set, self.vp, self.pp)
        self.assertLess(abs(err), 1.0, err)
        s = self.vp.series()["smsp__warps_issue_stalled_barrier.avg"]
        self.assertTrue(s.varying)
        self.assertLessEqual(abs(err), s.half * 8 + 1e-9)


class TheAliasCheck(unittest.TestCase):
    """The pooled objective at the placement against its value with a run, or every run, shifted by one sample or one
    window. A shift of every run together is an exact rotation of the pooled profile: the objective cannot tell it, and
    the edges alone decide. A run alone shifted is told apart by the other runs; with one run there is nobody to tell."""

    def one_series(self, profile, runs, noise, seed):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name, runs=runs, noise=noise, seed=seed, entry_seed=seed,
                                profiles={"barrier": (0, profile)})
        lc, ex, vp = place(s, sampling.Config(frame="anchors", **FAST))
        return s, vp, next(p for k, p in vp.passes.items() if "barrier" in k)

    def test_two_equal_humps_a_sample_apart_one_run_is_ambiguous_by_a_sample_and_both_candidates_are_given(self):
        kap = 1.5 / 3.62                                     # a sample, in windows
        two = lambda pos: 1.0 + 4.0 * (hump(0.3, 0.06, 1, 0)(pos) + hump(0.3 + kap, 0.06, 1, 0)(pos))
        s, vp, pp = self.one_series(two, 1, 0.3, 3)
        self.assertTrue(pp.placed, pp.findings)
        within = [a for a in pp.alias if a.label.startswith("run 0") and a.within]
        self.assertEqual(len(within), 2, [(a.label, a.dq, a.margin) for a in pp.alias])    # +1 and -1 sample
        self.assertTrue(any("AMBIGUOUS BY A SAMPLE" in f for f in pp.findings), pp.findings)
        name = "smsp__warps_issue_stalled_barrier.avg"
        self.assertAlmostEqual(abs(pp.ambiguous[name]), kap * 8, delta=0.4)       # the other candidate: a sample away
        sp = vp.series()[name]
        self.assertAlmostEqual(abs(sp.ambiguous), kap * 8, delta=0.4)

    def test_six_runs_tell_one_runs_sample_alias_apart_and_the_check_is_silent(self):
        one = lambda pos: 1.0 + 3.0 * hump(0.3, 0.1, 1, 0)(pos)
        s, vp, pp = self.one_series(one, 6, 0.5, 4)
        self.assertTrue(pp.placed, pp.findings)
        alone = [a for a in pp.alias if not a.label.startswith("every")]
        self.assertEqual(len(alone), 12)
        for a in alone:
            self.assertFalse(a.within, (a.label, a.dq, a.margin))        # the objective falls far under the margin
            self.assertGreater(a.dq, 1.5 * a.margin)
        self.assertEqual(pp.ambiguous, {})
        self.assertFalse(any("AMBIGUOUS" in f for f in pp.findings), pp.findings)

    def test_every_run_shifted_together_is_listed_as_a_rotation_the_edges_decide(self):
        one = lambda pos: 1.0 + 3.0 * hump(0.3, 0.1, 1, 0)(pos)
        s, vp, pp = self.one_series(one, 4, 0.4, 6)
        every = [a for a in pp.alias if a.label.startswith("every")]
        self.assertEqual(len(every), 4)                       # +-1 sample and +-1 window
        for a in every:
            self.assertTrue(a.within)
            self.assertIn("edges decide", a.note)
            self.assertLess(abs(a.dq), 0.05 * pp.q)           # the objective is the same to a few percent
        name = "smsp__warps_issue_stalled_barrier.avg"
        kap = 1.5 / 3.62
        for a in (x for x in every if "sample" in x.label):
            self.assertAlmostEqual(abs(a.moves[name]), kap * 8, delta=0.4)     # the profile moves a sample's worth
        for a in (x for x in every if "window" in x.label):
            self.assertLess(abs(a.moves[name]), 0.4)           # and a window moves nothing: it is the period


class SpreadIsMeasured(unittest.TestCase):
    """The resampled spread of the profile's phase covers the truth at its stated rate over many seeds, with the runs'
    entries drawn at random: a spread that claimed 0.1 bin and covered the truth a third of the time would fail."""

    SEEDS = 20

    def test_twice_the_spread_covers_the_truth_and_the_half_width_does_not_claim_the_impossible(self):
        rows = []
        for seed in range(1, self.SEEDS + 1):
            with tempfile.TemporaryDirectory() as d:
                s = launch_gen.make_set(d, runs=6, noise=1.0, seed=seed * 17, entry_seed=seed)
                lc, ex, vp = place(s, sampling.Config(frame="anchors", boot=12, report_frames=False))
                for tag in ("barrier", "wait"):
                    pp = next(p for k, p in vp.passes.items() if tag in k)
                    sp = vp.series()[f"smsp__warps_issue_stalled_{tag}.avg"]
                    rows.append((true_fold_error(s, vp, pp, tag), sp))
        n = len(rows)
        cover_spread = sum(abs(e) <= placement.SPREAD_Z * sp.spread * 8 for e, sp in rows) / n
        cover_half = sum(abs(e) <= sp.half * 8 for e, sp in rows) / n
        mean_half = sum(sp.half * 8 for _e, sp in rows) / n
        mean_err = sum(abs(e) for e, _sp in rows) / n
        mean_spread = sum(sp.spread * 8 for _e, sp in rows) / n
        # twice a standard deviation is a 95% claim: not a third, not half. (the box's hard bound would cover them all)
        self.assertGreaterEqual(cover_spread, 0.8, (cover_spread, mean_spread, mean_err))
        self.assertGreaterEqual(cover_half, 0.95, (cover_half, mean_half))
        # and the claim is not empty: the spread is of the size of the error (not a tenth of it), the half-width useful
        self.assertGreater(mean_spread, 0.4 * mean_err, (mean_spread, mean_err))
        self.assertLess(mean_half, 2.0, mean_half)

    def test_a_spread_claiming_a_tenth_of_a_bin_would_fail_this_coverage(self):
        # the same kind of errors against a spread of 0.1 bin: the coverage the test above demands is not met by such a claim
        rows = []
        for seed in range(1, 9):
            with tempfile.TemporaryDirectory() as d:
                s = launch_gen.make_set(d, runs=4, noise=2.5, seed=seed * 31, entry_seed=seed)
                lc, ex, vp = place(s, sampling.Config(frame="anchors", boot=8, report_frames=False))
                pp = next(p for k, p in vp.passes.items() if "barrier" in k)
                rows.append(true_fold_error(s, vp, pp))
        claim = 0.1 * placement.SPREAD_Z
        covered = sum(abs(e) <= claim for e in rows) / len(rows)
        self.assertLess(covered, 0.8, (covered, rows))

    def test_runs_fewer_than_three_cannot_be_resampled_and_say_so(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=2, noise=0.3, seed=3)
            lc, ex, vp = place(s, sampling.Config(frame="anchors", **FAST))
            pp = next(p for k, p in vp.passes.items() if "barrier" in k)
            self.assertTrue(all(v is None for v in pp.spread.values()))
            self.assertTrue(any("cannot be measured" in f for f in pp.findings), pp.findings)
            sp = vp.series()["smsp__warps_issue_stalled_barrier.avg"]
            self.assertIsNone(sp.spread)
            self.assertAlmostEqual(sp.half, sp.half_a)      # the box alone


class EdgesBind(unittest.TestCase):
    """Where the fitted runs fill all that the first and last live sample allow, the box of the common shift is nothing
    wide and the placement rests on the edges' limits: named, and marked in the table."""

    def test_a_box_of_no_width_is_named_and_marked(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=4, noise=0.3, seed=3)
            lc = launch.launch_of(record.load(s["record"]))
            ex = [sampling.read_export(p) for p in s["stamped"]]
            old = placement.BIND
            try:
                placement.BIND = 10.0                      # every box is narrower than this
                pl = overlay.place(overlay.view_of(record.load(s["record"])), lc, ex, sampling.Config(frame="anchors", **FAST))
            finally:
                placement.BIND = old
            for pp in pl.edges.values():
                if pp.placed:
                    self.assertTrue(pp.binds)
                    self.assertTrue(any(f.startswith("EDGES BIND") for f in pp.findings), pp.findings)
            self.assertTrue(all(x.binds for x in pl.series.values()))
            text = overlay.format_overlay(pl, "r.json", ["x"])
            self.assertIn("EDGES BIND", text)
            self.assertIn("!", "".join(overlay.footer_lines(pl)))


class CyclesASampleFlag(unittest.TestCase):
    def test_a_pass_whose_cycles_a_sample_is_off_the_median_is_flagged_and_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=4, len_vary=[0.0, 0.0, 0.0, 0.07], stretch=1.1)
            lc = launch.launch_of(record.load(s["record"]))
            ex = [sampling.read_export(p) for p in s["stamped"]]
            vp = placement.place_variant(ex, lc.cycles, placement.Frame(lc.fractions, "r"), sampling.Config(**FAST))
            by_run = {}
            for c in vp.checks:
                by_run.setdefault(c.run, []).append(c)
            self.assertTrue(all(c.flagged for c in by_run[3]), [c.ratio for c in by_run[3]])
            self.assertFalse(any(c.flagged for r in (0, 1, 2) for c in by_run[r]))
            self.assertLess(by_run[3][0].ratio, 0.95)           # 7% longer: that many more samples a launch
            for pp in vp.passes.values():
                self.assertNotIn(3, pp.fed)
                self.assertTrue(any("CYCLES A SAMPLE, run 3" in x for x in pp.findings), pp.findings)

    def test_the_median_and_the_tolerance_are_the_variants_own(self):
        rows, med = placement.cps_checks([{"a": placement.Pass("a", ["a"], 0, 99, 1.5)},
                                          {"a": placement.Pass("a", ["a"], 0, 109, 1.5)}], 22000.0, 0.03)
        self.assertEqual([r.live for r in rows], [100, 110])
        self.assertAlmostEqual(med, 0.5 * (220.0 + 200.0))
        self.assertTrue(all(r.flagged for r in rows))           # the two differ by 10%: each is 5% off the median


class SubBinRotation(unittest.TestCase):
    def test_fold_rotation_is_known_to_a_fraction_of_a_bin(self):
        n = 8
        prof = lambda x: 2.0 + math.cos(2 * math.pi * (x - 0.3)) + 0.4 * math.cos(4 * math.pi * (x - 0.1))
        for shift in (0.0, 0.17, 0.3, -0.42, 1.25):                    # bins
            a = [prof((b + 0.5) / n) for b in range(n)]
            b = [prof((k + 0.5 - shift) / n) for k in range(n)]        # b lies `shift` bins later than a
            est = placement.fold_rotation(a, b)
            self.assertLess(abs(((est - shift + n / 2) % n) - n / 2), 0.03, (shift, est))

    def test_compare_folds_names_a_rotation_over_half_a_bin_and_a_correlation_under_0_7(self):
        n = 8
        a = [3.0 + math.cos(2 * math.pi * (b + 0.5) / n) for b in range(n)]
        same = placement.compare_folds("x", "x", a, [v + 0.01 for v in a])
        self.assertEqual(same.finding, "")
        shifted = placement.compare_folds("x", "x", a, [a[(b - 1) % n] for b in range(n)])   # one bin later
        self.assertAlmostEqual(shifted.rotation, 1.0, places=1)
        self.assertIn("rotated by", shifted.finding)
        anti = placement.compare_folds("x", "x", a, [6.0 - v for v in a])
        self.assertIn("correlation", anti.finding)


class EdgesVariantFrame(unittest.TestCase):
    """A stamped variant's stamps add cycles to its windows and none to its setup, so its windows do not sit at the same
    fractions of the launch as the plain kernel's. Here the stamped windows are 4% longer."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.win = 1 / 1.04
        cls.set = launch_gen.make_set(cls.dir.name, runs=3, edges=True, jitter=0.04, ripple=0.2, len_vary=[0.0, 0.01, -0.01],
                                      edges_windows=cls.win)
        cls.rec = record.load(cls.set["record"])
        cls.lc = launch.launch_of(cls.rec)
        cls.elc = launch.launch_of(record.load(cls.set["edges_record"]), need_windows=False)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_corrected_fractions_are_the_edges_kernels_own_and_the_stamped_launchs_are_not(self):
        k = launch_gen.Cycles(1.0, 8.7, 3.62, self.set["mult"], 2.2, 1.0, 0.0, self.win)
        true = [a / k.t_c for a in k.anchors[window.SKIP:]]
        got = self.lc.fractions_on(self.elc.cycles)
        self.assertEqual(len(got), len(true))
        self.assertLess(max(abs(a - b) for a, b in zip(got, true)), 3e-4, max(abs(a - b) for a, b in zip(got, true)))
        raw = self.lc.fractions
        self.assertGreater(abs(raw[0] - true[0]), 1e-3)                   # a bin of a 64-window launch is 0.0019
        self.assertGreater(max(abs(a - b) for a, b in zip(raw, true)) * len(true), 0.05)      # a twentieth of a window
        self.assertLess(self.elc.cycles, self.lc.cycles)
        self.assertEqual(self.lc.setup_cycles, round(8.7 * 1800))         # the code before the first window

    def test_the_placed_edges_fold_agrees_with_the_stamped_variants_and_the_check_passes(self):
        res = variants.run(self.set["manifest"], sampling.Config(boot=8, report_frames=False))
        e = next(v for v in res.variants if v.name == "edges")
        self.assertTrue(e.compare)
        for c in e.compare:
            self.assertTrue(c.within, (c.short, c.rotation, c.bound))
            self.assertLessEqual(abs(c.rotation), 1.0, (c.short, c.rotation))
            self.assertGreater(c.corr, 0.7, (c.short, c.corr))
        text = variants.format_set(res)
        self.assertIn("the code before the first window", text)
        self.assertIn("within the two placements' measured spreads", text)


class EdgesVariant(unittest.TestCase):
    """The plain kernel with its two launch edges: placed by its own edges and the stamped variant's windows."""

    def run_set(self, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name, runs=3, edges=True, jitter=0.04, ripple=0.2, len_vary=[0.0, 0.01, -0.01], **kw)
        return s, variants.run(s["manifest"], sampling.Config(boot=8, report_frames=False))

    def test_an_edges_record_is_a_launch_without_windows(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, edges=True)
            rec = record.load(s["edges_record"])
            with self.assertRaises(launch.LaunchError) as e:
                launch.launch_of(rec)
            self.assertIn("window", str(e.exception))
            lc = launch.launch_of(rec, need_windows=False)
            self.assertFalse(lc.has_windows)
            self.assertAlmostEqual(lc.cycles, s["edges_length"] * 1800, delta=2)
            self.assertIn("another record's", lc.describe())

    def test_it_is_placed_through_the_stamped_variant_and_agrees_with_it(self):
        s, res = self.run_set()
        by = {v.name: v for v in res.variants}
        e, st = by["edges"], by["stamped"]
        self.assertTrue(e.edges_only)
        self.assertIsNotNone(e.placement)
        self.assertEqual(e.placement.cps_tol, 0.03)
        self.assertIn("variant `stamped`'s window anchors", e.placement.windows_from)
        self.assertTrue(e.compare)
        n = e.placement.bins
        for c in e.compare:
            # the two placed folds differ by no more than the two placements' stated half-widths allow
            hw = (e.placement.series[c.series].half + st.placement.series[c.series].half) * n
            self.assertLessEqual(abs(c.rotation), hw + 0.25, (c.short, c.rotation, hw))
            self.assertGreater(c.corr, 0.7, (c.short, c.corr))
        # the edges variant's own T (its record's launch in cycles), not the stamped one's, scales its passes
        self.assertAlmostEqual(e.placement.launch.cycles, s["edges_length"] * 1800, delta=2)
        self.assertLess(e.placement.launch.cycles, st.placement.launch.cycles)
        text = variants.format_set(res)
        for needle in ("an EDGES variant", "THE CHECK: `edges`'s fold", "THE PLAIN VARIANT `plain` PLACED THROUGH `edges`",
                       "THE OVERLAY OF `edges`'S OWN SAMPLES", "windows and roles from `stamped`"):
            self.assertIn(needle, text)
        json.dumps(variants.to_dict(res))

    def test_a_shifted_edges_variant_is_named_by_the_check(self):
        s, res = self.run_set(edges_rot=0.25)                          # its profile two bins later than the stamped one's
        e = next(v for v in res.variants if v.name == "edges")
        named = [c for c in e.compare if c.finding]
        self.assertGreaterEqual(len(named), 3, [(c.short, c.rotation, c.corr) for c in e.compare])
        for c in named:
            self.assertTrue(c.rotation is None or abs(c.rotation) > 0.5 or c.corr < 0.7)
        shifts = [c.rotation for c in e.compare if c.rotation is not None]
        self.assertTrue(any(abs(abs(x) - 2.0) < 0.6 for x in shifts), shifts)    # 0.25 of a window is two bins
        self.assertIn("FINDING: rotated by", variants.format_set(res))

    def test_the_overlay_command_places_an_edges_record_through_a_named_stamped_one(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, edges=True)
            out = io.StringIO()
            args = ["overlay", s["edges_record"]] + s["edges"] + ["--windows-from", s["record"], "--boot", "6"]
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(args), 0)
            self.assertIn("OVERLAY:", out.getvalue())
            self.assertIn("window anchors on this launch's cycles", out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):                      # without `--windows-from`: no windows
                self.assertEqual(main(["overlay", s["edges_record"]] + s["edges"]), 2)
            self.assertIn("window", err.getvalue())


class Refused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.set = launch_gen.make_set(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_a_record_without_the_launch_events_is_refused_naming_what_to_stamp(self):
        with open(self.set["record"]) as f:
            doc = json.load(f)
        keep = {i for i, n in doc["names"].items() if not n.startswith("launch.")}
        doc["stamps"] = [s for s in doc["stamps"] if str(s[1]) in keep]
        rec = record.from_dict(doc)
        with self.assertRaises(overlay.OverlayError) as e:
            overlay.launch_for(rec)
        for needle in ("REFUSED", "launch.begin", "launch.end", "tie", "entry", "exit"):
            self.assertIn(needle, str(e.exception))
        p = os.path.join(self.dir.name, "nolaunch.json")
        record.dump(rec, p)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["overlay", p] + self.set["stamped"])
        self.assertEqual(code, 2)
        self.assertIn("launch.begin", err.getvalue())

    def test_the_overlay_command_prints_the_placement_json_and_perfetto(self):
        out = io.StringIO()
        pf = os.path.join(self.dir.name, "p.json")
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--perfetto", pf, "--boot", "6"]), 0)
        text = out.getvalue()
        for needle in ("PLACEMENT", "half-width", "CROSS-CHECK", "OVERLAY:", "front", "cycles a sample",
                       "THE JOIN IS ON CYCLES", "STATED ASSUMPTION", "ALIAS CHECK"):
            self.assertIn(needle, text)
        with open(pf) as f:
            doc = json.load(f)
        self.assertTrue(any(e["ph"] == "C" for e in doc["traceEvents"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--json", "--boot", "6"]), 0)
        d = json.loads(out.getvalue())
        self.assertEqual(len(d["overlay"]["bins"]), 8)
        self.assertIn("wait", d["series"])
        self.assertIn("cycles_a_sample", d)
        self.assertIn("spread_bins", d["series"]["wait"])
        self.assertIn("frame", d)

    def test_the_old_correlation_is_only_a_cross_check_and_agrees(self):
        rec = record.load(self.set["record"])
        lc = launch.launch_of(rec)
        ex = [sampling.read_export(p) for p in self.set["stamped"]]
        pl = overlay.place(overlay.view_of(rec), lc, ex, sampling.Config(boot=6, report_frames=False))
        self.assertTrue(pl.cross.agree, pl.cross.note)
        self.assertIn("one bin", pl.cross.note)


if __name__ == "__main__":
    unittest.main()
