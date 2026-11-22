import os
import tempfile
import unittest

from warpline.recorder import sampling
from tests.recorder.helpers import BARRIER, blurred_truth, hump, synth, write_csv

P = 2.62


class Setup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def csv(self, name, series, **kw):
        return write_csv(os.path.join(self.dir.name, name), series, **kw)

    def run_of(self, series, cfg=None, **kw):
        return sampling.analyze_run(sampling.read_export(self.csv("r.csv", series, **kw)), cfg)


class Reading(Setup):
    def test_series_interval_passes_and_starts(self):
        a = synth(hump(), P, 100, start=40, tail=20)
        b = synth(hump(0.6), P, 100, start=40, tail=20, seed=2)
        c = [0.0] * 30 + [1.0] * 5 + [0.0] * 45          # a different sample count: its own pass
        exp = sampling.read_export(self.csv("a.csv", {"barrier": a, "wait": b, "drain": c}, interval=2.0))
        self.assertEqual(exp.interval, 2.0)
        self.assertEqual([s.short for s in exp.series], ["barrier", "wait", "drain"])
        self.assertEqual([s.group for s in exp.series], [0, 0, 1])
        self.assertEqual((exp.series[0].start, exp.series[0].end), (40, 140))
        self.assertEqual((exp.series[2].start, exp.series[2].end), (30, 35))

    def test_a_sparse_series_starts_with_its_pass(self):
        full = synth(hump(), P, 100, start=40, tail=20)
        sparse = [0.0] * len(full)
        sparse[120] = 3.0                                   # its own first nonzero sample is far from the start
        exp = sampling.read_export(self.csv("a.csv", {"barrier": full, "drain": sparse}))
        self.assertEqual(exp.series[1].start, 40)

    def test_an_all_zero_series_is_dead(self):
        full = synth(hump(), P, 150)
        r = self.run_of({"barrier": full, "sleeping": [0.0] * len(full)})
        self.assertEqual(r.get("sleeping").status, "dead")

    def test_a_file_without_samples_or_interval_is_refused(self):
        p = os.path.join(self.dir.name, "x.csv")
        with open(p, "w") as f:
            f.write("a,b\nu,u\n1,2\n")
        with self.assertRaises(sampling.SamplingError):
            sampling.read_export(p)


class Period(Setup):
    def test_the_period_is_recovered_with_its_score(self):
        for true in (2.62, 3.4, 4.9):
            r = self.run_of({"barrier": synth(hump(), true, 170, noise=0.3, seed=3)})
            s = r.get("barrier")
            self.assertAlmostEqual(s.period, true, delta=0.01 * true, msg=true)
            self.assertGreater(s.score, 0.6)
            self.assertEqual(s.status, "ok")

    def test_a_multiple_of_the_period_is_not_taken_for_it(self):
        r = self.run_of({"barrier": synth(hump(), P, 300, noise=0.4, seed=5)})
        self.assertAlmostEqual(r.get("barrier").period, P, delta=0.02)

    def test_a_start_offset_does_not_change_the_period(self):
        for start in (10, 57, 120):
            r = self.run_of({"barrier": synth(hump(), P, 170, start=start)})
            self.assertAlmostEqual(r.get("barrier").period, P, delta=0.02)

    def test_a_flat_series_is_reported_flat_and_folded_on_the_agreed_period(self):
        import random
        rng = random.Random(9)
        flat = [0.0] * 40 + [3.0 + rng.gauss(0, 0.05) for _ in range(170)] + [0.0] * 30
        r = self.run_of({"barrier": synth(hump(), P, 170), "wait": flat})
        w = r.get("wait")
        self.assertEqual(w.status, "flat")
        self.assertTrue(w.folded)
        self.assertEqual(len(w.bins), 8)
        self.assertAlmostEqual(r.agreed, P, delta=0.02)


class Checks(Setup):
    def three_and(self, odd_period, **kw):
        s = {"barrier": synth(hump(), P, 200, seed=1), "wait": synth(hump(0.55), P, 200, seed=2),
             "mio": synth(hump(0.8), P, 200, seed=3), "odd": synth(hump(0.1), odd_period, 200, seed=4, **kw)}
        return self.run_of(s)

    def test_passes_that_disagree_are_flagged_and_left_out_of_the_fold(self):
        r = self.three_and(P * 1.04)
        self.assertEqual(r.get("odd").status, "disagrees")
        self.assertFalse(r.get("odd").folded)
        self.assertTrue(all(r.get(n).folded for n in ("barrier", "wait", "mio")))
        self.assertTrue(any("PASSES DISAGREE" in f and "odd" in f for f in r.flags))
        self.assertAlmostEqual(r.agreed, P, delta=0.02)

    def test_passes_within_the_tolerance_are_not_flagged(self):
        r = self.three_and(P * 1.003)
        self.assertEqual(r.get("odd").status, "ok")
        self.assertFalse(any("PASSES DISAGREE" in f for f in r.flags))

    def test_a_drifting_period_is_flagged(self):
        # the period is 2.55 samples in the first half and 2.70 in the second: phase is continuous
        def drift(t):
            return t / 2.55 if t < 100 else 100 / 2.55 + (t - 100) / 2.7
        s = {"barrier": synth(hump(), P, 200, seed=1), "wait": synth(hump(0.55), P, 200, seed=2),
             "mio": synth(hump(0.8), P, 200, seed=3), "odd": synth(hump(0.1), P, 200, seed=4, phase=drift)}
        r = self.run_of(s)
        self.assertTrue(r.get("odd").drift)
        self.assertTrue(any(f.startswith("DRIFT") and "odd" in f for f in r.flags))
        for n in ("barrier", "wait", "mio"):
            self.assertFalse(r.get(n).drift, n)

    def test_a_window_near_two_samples_is_flagged_as_near_the_limit(self):
        r = self.run_of({"barrier": synth(hump(), 2.08, 200, noise=0.1)})
        self.assertTrue(any("NEAR THE SAMPLING LIMIT" in f for f in r.flags))

    def test_too_many_bins_are_refused(self):
        with self.assertRaises(sampling.SamplingError) as e:
            self.run_of({"barrier": synth(hump(), P, 170)}, sampling.Config(bins=40))
        self.assertIn("fewer bins", str(e.exception))


class Fold(Setup):
    def test_the_bins_lie_within_their_standard_errors_of_the_blurred_truth(self):
        prof = hump(0.3, 0.15, 12.0, 2.0)
        r = self.run_of({"barrier": synth(prof, P, 420, noise=0.6, seed=11, tail=10)})
        bins = r.get("barrier").bins
        n = len(bins)
        worst = 0.0
        for b, bb in enumerate(bins):
            truth = blurred_truth(prof, P, b / n, (b + 1) / n)
            worst = max(worst, abs(bb.mean - truth) / bb.se)
            self.assertLess(abs(bb.mean - truth), 4 * bb.se + 0.1, f"bin {b}: {bb.mean} against {truth}")
        self.assertGreater(min(bb.n for bb in bins), 40)
        self.assertLess(worst, 4)

    def test_the_fold_is_the_boxed_profile_not_the_profile(self):
        # a narrow spike is smeared by the box: the folded peak is below the spike's height
        prof = hump(0.3, 0.03, 20.0, 1.0)
        r = self.run_of({"barrier": synth(prof, P, 420, noise=0.2, seed=12, tail=10)}, sampling.Config(bins=16))
        self.assertLess(max(b.mean for b in r.get("barrier").bins), 0.5 * 21.0)
        self.assertAlmostEqual(r.blur, 1 / P, places=2)


class Runs(Setup):
    def export(self, name, period, origin=0.0, seed=1, interval=1.5, n=260):
        s = {"barrier": synth(hump(), period, n, origin=origin, seed=seed, noise=0.25),
             "wait": synth(hump(0.6, 0.15, 6.0, 1.0), period, n, origin=origin, seed=seed + 50, noise=0.25),
             "mio": synth(hump(0.8, 0.1, 5.0, 1.0), period, n, origin=origin, seed=seed + 99, noise=0.25)}
        return sampling.read_export(self.csv(name, s, interval=interval))

    def test_runs_are_aligned_by_the_rotation_of_their_barrier_fold_then_averaged(self):
        exps = [self.export("a.csv", P, 0.0, 1), self.export("b.csv", P, 0.25, 2), self.export("c.csv", P, 0.5, 3)]
        f = sampling.fold(exps)
        self.assertEqual(f.flags, [])
        self.assertEqual(f.align_series, BARRIER[len("pmsampling:"):])
        by = {s.index: s for s in f.run_shifts}
        # a run whose first sample is 0.25 of a window later is rotated by -2 bins (6 of 8)
        self.assertEqual((by[1].shift.shift, by[2].shift.shift), (6, 4))
        self.assertGreater(by[1].shift.corr, 0.95)
        prof = hump()
        for b, bb in enumerate(f.bins[BARRIER[len("pmsampling:"):]]):
            truth = blurred_truth(prof, P, b / 8, (b + 1) / 8)
            self.assertLess(abs(bb.mean - truth), 4 * bb.se + 0.1, f"bin {b}")
        self.assertEqual(sum(bb.n for bb in f.bins[BARRIER[len("pmsampling:"):]]) // 3 > 200, True)

    def test_a_run_whose_period_disagrees_is_flagged_and_left_out(self):
        exps = [self.export("a.csv", P, 0.0, 1), self.export("b.csv", P, 0.0, 2), self.export("c.csv", P, 0.0, 3),
                self.export("d.csv", P * 1.05, 0.0, 4)]
        f = sampling.fold(exps)
        self.assertTrue(any("RUNS DISAGREE" in x and "d.csv" in x for x in f.flags))
        self.assertEqual([s.status for s in f.run_shifts], ["reference", "accepted", "accepted", "period"])
        self.assertNotIn(3, f.runs_of[BARRIER[len("pmsampling:"):]])

    def test_the_periods_are_compared_in_microseconds(self):
        # the same window in samples, but a different interval: the window in us differs
        exps = [self.export("a.csv", P, 0.0, 1), self.export("b.csv", P, 0.0, 2), self.export("c.csv", P, 0.0, 3, 2.0)]
        f = sampling.fold(exps)
        self.assertTrue(any("RUNS DISAGREE" in x and "c.csv" in x for x in f.flags))

    def test_a_run_that_cannot_be_aligned_is_flagged_and_left_out(self):
        n = 260
        a, b = hump(0.1, 0.1, 10.0, 2.0), hump(0.5, 0.1, 8.0, 2.0)
        two = lambda pos: a(pos) + b(pos) - 2.0          # two humps against the reference's one
        s = {"barrier": synth(two, P, n, seed=7, noise=0.25), "wait": synth(hump(0.6, 0.15, 6.0, 1.0), P, n, seed=8),
             "mio": synth(hump(0.8, 0.1, 5.0, 1.0), P, n, seed=9)}
        f = sampling.fold([self.export("a.csv", P, 0.0, 1), sampling.read_export(self.csv("b.csv", s))])
        self.assertEqual(f.run_shifts[1].status, "alignment")
        self.assertTrue(any("RUN NOT ALIGNED" in x and "b.csv" in x for x in f.flags))
        self.assertEqual(f.runs_of[BARRIER[len("pmsampling:"):]], [0])


class Shifts(unittest.TestCase):
    def test_best_shift_recovers_a_rotation(self):
        target = [0, 1, 4, 9, 4, 1, 0, 0]
        fold = sampling.rotate(target, 3)           # fold[b] = target[b + 3]  =>  rotated[b] = fold[b + s] needs s = 5
        sh = sampling.best_shift(target, fold)
        self.assertEqual(sh.shift, 5)
        self.assertAlmostEqual(sh.corr, 1.0)
        self.assertTrue(sh.accepted)

    def test_two_equal_humps_half_a_window_apart_are_refused_as_ambiguous(self):
        target = [0, 5, 9, 5, 0, 5, 9, 5]
        sh = sampling.best_shift(target, [0, 5, 9, 5, 0, 5, 9, 5])
        self.assertFalse(sh.accepted)
        self.assertTrue(sh.why.startswith("AMBIGUOUS"))
        self.assertEqual((sh.shift, sh.runner_shift) in {(0, 4), (4, 0)}, True)
        self.assertAlmostEqual(sh.corr, sh.runner_corr)

    def test_a_flat_profile_or_fold_is_refused(self):
        sh = sampling.best_shift([1.0] * 8, [0, 1, 4, 9, 4, 1, 0, 0])
        self.assertFalse(sh.accepted)
        self.assertTrue(sh.why.startswith("FLAT"))
        self.assertFalse(sampling.best_shift([0, 1, 4, 9, 4, 1, 0, 0], [2.0] * 8).accepted)

    def test_a_weak_correlation_is_refused(self):
        sh = sampling.best_shift([0, 1, 4, 9, 4, 1, 0, 0], [3, 0, 4, 1, 5, 0, 2, 6])
        self.assertFalse(sh.accepted)
        self.assertTrue(sh.why.startswith("WEAK") or sh.why.startswith("AMBIGUOUS"))


class Output(Setup):
    def test_the_table_and_json_state_the_scan_the_checks_and_the_blur(self):
        exps = [sampling.read_export(self.csv("a.csv", {"barrier": synth(hump(), P, 200),
                                                        "wait": synth(hump(0.5), P, 200, seed=2)}))]
        f = sampling.fold(exps)
        text = sampling.format_fold(f)
        for needle in ("scan: periods", "THE BLUR", "THE FOLD", "count a bin", "0.000-0.125"):
            self.assertIn(needle, text)
        d = sampling.to_dict(f)
        self.assertEqual(d["bins"], 8)
        self.assertAlmostEqual(d["blur"]["fraction_of_window"], 1 / P, places=2)
        self.assertEqual(len(d["series"]["smsp__warps_issue_stalled_barrier.avg"]["bins"]), 8)


if __name__ == "__main__":
    unittest.main()
