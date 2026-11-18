"""The samples joined to the stamps on CYCLES, each pass scaled by its own two edges.

The synthetic launch (`launch_gen`) here has the real profiler's behaviour put in with known truth: the window's length
jitters, the profiler's clock runs a known factor faster than the cycle counter with a ripple inside the window, and each
run's kernel is a little longer or shorter than the record's. What the placement recovers is checked against it."""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest

from warpline import launch, overlay, placement, record, sampling, variants
from warpline.__main__ import main
from tests import launch_gen

PASS_INDEX = {"barrier": 0, "wait": 1, "drain": 2}
RIPPLE = 0.3
LEN_VARY = [0.0, 0.012, -0.01, 0.015]


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
        cls.vp = placement.place_variant(cls.exports, cls.lc.cycles, cls.frame, sampling.Config())

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

    def test_the_placement_recovers_the_true_window_position_within_its_stated_half_width(self):
        # the true window position of every sample's centre (the generator keeps it) against the placed one: the stated
        # half-width (the entry's and the scale's) plus the ripple inside the window, which the placement cannot see
        allow = RIPPLE / (2 * math.pi)
        for key, pp in self.vp.passes.items():
            self.assertTrue(pp.placed, (key, pp.findings))
            self.assertGreaterEqual(len(pp.fed), 3, (key, pp.findings))
            for r in pp.fed:
                rp = pp.runs[r]
                js, xs = placement.sample_positions(rp.p, rp.a, rp.d, self.frame)
                truth = self.set["pos"][(r, pass_index(key))]
                errs = sorted(circ(x, truth[j]) for j, x in zip(js, xs) if j in truth)
                self.assertGreater(len(errs), 100)
                med = errs[len(errs) // 2]
                self.assertLessEqual(med, pp.half_a + pp.half_d + allow, (key, r, med, pp.half_a, pp.half_d))
                self.assertLessEqual(errs[int(0.9 * len(errs))], 2 * (pp.half_a + pp.half_d) + 2 * allow)

    def test_the_scale_and_the_offset_are_those_that_were_put_in(self):
        dt = self.set["dt"]
        for key, pp in self.vp.passes.items():
            pi = pass_index(key)
            for r in pp.fed:
                rp = pp.runs[r]
                k = launch_gen.Cycles(1 + LEN_VARY[r], 8.7, 3.62, self.set["mult"], 2.2, 1.15, RIPPLE)
                d_true = k.length_us() / dt - rp.p.n
                a_true = self.set["truth"][(r, pi)] / dt
                self.assertLess(abs(rp.d - d_true), 0.45, (key, r, rp.d, d_true))
                self.assertLess(abs(rp.a - a_true), 0.45, (key, r, rp.a, a_true))

    def test_the_folded_profile_has_the_true_shape(self):
        n = self.vp.bins
        s = self.vp.series()
        # `barrier`: high over window positions 0.2..0.5 (blurred by a sample's box and by the placement)
        vals = [b.mean for b in s["smsp__warps_issue_stalled_barrier.avg"].bins]
        self.assertLess(circ(phase(vals), 0.35), 0.06, vals)
        # `wait`: a hump at 0.6
        vals = [b.mean for b in s["smsp__warps_issue_stalled_wait.avg"].bins]
        self.assertLess(circ(phase(vals), 0.6), 0.06, vals)
        for x in s.values():
            self.assertLessEqual(x.half * n, 1.5, (x.short, x.half))      # tight enough to be useful: about a bin

    def test_the_overlay_reports_cycles_a_sample_the_half_widths_and_the_assumption(self):
        view = overlay.view_of(self.rec)
        pl = overlay.place(view, self.lc, self.exports, sampling.Config())
        text = overlay.format_overlay(pl, "record.json", ["x"])
        for needle in ("THE JOIN IS ON CYCLES", "STATED ASSUMPTION", "linear in the kernel's cycles", "cycles a sample",
                       "half-width", "windows =", "bins", "OVERLAY:", "front", "back", "x ", "runs fed", "+-"):
            self.assertIn(needle, text)
        self.assertEqual(len(pl.series), 4)


class AnchorsAgainstAMeanPeriod(unittest.TestCase):
    """Where the samples follow the window anchors' jitter the anchors place them best; where they do not (the real
    kernel's: a regular rhythm the anchors wobble around) the pass says so, and `frame="mean"` places them."""

    def place(self, follow, frame):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=3, jitter=0.06, stretch=1.1, follow=follow)
            lc = launch.launch_of(record.load(s["record"]))
            ex = [sampling.read_export(p) for p in s["stamped"]]
            cfg = sampling.Config(frame=frame)
            fr = placement.Frame(lc.fractions, "r")
            vp = placement.place_variant(ex, lc.cycles, fr, cfg)
            return s, vp, (fr.uniform() if frame == "mean" else fr)

    def test_the_anchors_are_followed_when_the_kernel_follows_them(self):
        s, vp, fr = self.place(True, "anchors")
        for pp in vp.passes.values():
            self.assertTrue(pp.placed)
            self.assertFalse(any("ANCHORS NOT FOLLOWED" in x for x in pp.findings), pp.frame_diag)
            self.assertIn("on the record's window anchors", pp.frame_diag)

    def test_a_pass_whose_samples_do_not_follow_them_says_so_and_the_mean_period_places_it(self):
        s, vp, _ = self.place(False, "anchors")
        for pp in vp.passes.values():
            self.assertTrue(any("ANCHORS NOT FOLLOWED" in x for x in pp.findings), pp.frame_diag)
        s, vp, fr = self.place(False, "mean")
        for key, pp in vp.passes.items():
            self.assertTrue(pp.placed, pp.findings)
            self.assertFalse(any("ANCHORS NOT FOLLOWED" in x for x in pp.findings))
            self.assertGreaterEqual(len(pp.fed), 2)
            self.assertLessEqual((pp.half_a + pp.half_d) * 8, 1.5)      # a useful placement, in bins


class CyclesASampleFlag(unittest.TestCase):
    def test_a_pass_whose_cycles_a_sample_is_off_the_median_is_flagged_and_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=4, len_vary=[0.0, 0.0, 0.0, 0.07], stretch=1.1)
            lc = launch.launch_of(record.load(s["record"]))
            ex = [sampling.read_export(p) for p in s["stamped"]]
            vp = placement.place_variant(ex, lc.cycles, placement.Frame(lc.fractions, "r"), sampling.Config())
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

    def test_the_rotation_of_two_runs_samples_is_known_to_a_fraction_of_a_bin(self):
        import random
        rng = random.Random(3)
        prof = lambda x: 2.0 + math.cos(2 * math.pi * (x - 0.3)) + 0.5 * math.cos(4 * math.pi * (x - 0.45))
        for shift in (0.013, -0.031, 0.1):                            # windows; a bin is 0.125 of one
            xs = [rng.random() for _ in range(900)]
            ys = [rng.random() for _ in range(900)]
            ref = {"s": (xs, [prof(x) + rng.gauss(0, 0.15) for x in xs])}
            run = {"s": (ys, [prof((y - shift) % 1.0) + rng.gauss(0, 0.15) for y in ys])}
            rot = placement.rotation_between(ref, run, 7)
            self.assertLess(abs(rot.s - shift), 0.012, (shift, rot))
            self.assertLess(rot.sd, 0.03)
            self.assertGreater(rot.strength, 0.9)

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


class EdgesVariant(unittest.TestCase):
    """The plain kernel with its two launch edges: placed by its own edges and the stamped variant's windows."""

    def run_set(self, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name, runs=3, edges=True, jitter=0.04, ripple=0.2, len_vary=[0.0, 0.01, -0.01], **kw)
        return s, variants.run(s["manifest"])

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
        self.assertTrue(e.placement.windows_from.startswith("variant `stamped`"))
        self.assertTrue(e.compare)
        n = e.placement.bins
        for c in e.compare:
            # the two placed folds differ by no more than the two placements' stated half-widths allow (3 runs a series:
            # a bin or so each); a rotation over half a bin is a finding of the check, and it can be one by chance here
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
            args = ["overlay", s["edges_record"]] + s["edges"] + ["--windows-from", s["record"]]
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(args), 0)
            self.assertIn("OVERLAY:", out.getvalue())
            self.assertIn("window anchors as fractions of its launch", out.getvalue())
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
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--perfetto", pf]), 0)
        text = out.getvalue()
        for needle in ("PLACEMENT", "half-width", "CROSS-CHECK", "OVERLAY:", "front", "cycles a sample",
                       "THE JOIN IS ON CYCLES", "STATED ASSUMPTION"):
            self.assertIn(needle, text)
        with open(pf) as f:
            doc = json.load(f)
        self.assertTrue(any(e["ph"] == "C" for e in doc["traceEvents"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--json"]), 0)
        d = json.loads(out.getvalue())
        self.assertEqual(len(d["overlay"]["bins"]), 8)
        self.assertIn("wait", d["series"])
        self.assertIn("cycles_a_sample", d)

    def test_the_old_correlation_is_only_a_cross_check_and_agrees(self):
        rec = record.load(self.set["record"])
        lc = launch.launch_of(rec)
        ex = [sampling.read_export(p) for p in self.set["stamped"]]
        pl = overlay.place(overlay.view_of(rec), lc, ex, sampling.Config())
        self.assertTrue(pl.cross.agree, pl.cross.note)
        self.assertIn("one bin", pl.cross.note)


if __name__ == "__main__":
    unittest.main()
