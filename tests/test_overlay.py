import json
import os
import tempfile
import unittest

from warpline import overlay, sampling
from tests.helpers import hump, make, regular, synth, write_csv

P = 2.62


def stamp_record(waits, windows=12, phases=True, period=1000):
    """Scope `k`: lane a (role front, the anchor `window.begin`, a phase `phase.main` 50..600) and `waits`: a list of
    (begin, end) offsets, one lane (role x) each, waiting on barrier B in every window."""
    stamps = []
    for w in range(windows + 1):
        t0 = period * w
        stamps.append(("k/a", "window.begin", t0))
        if w == windows:
            break
        if phases:
            stamps += [("k/a", "phase.main.begin", t0 + 50), ("k/a", "phase.main.end", t0 + 600)]
        for i, (b, e) in enumerate(waits):
            stamps += [(f"k/w{i}", "barrier.B.wait.begin", t0 + b), (f"k/w{i}", "barrier.B.wait.end", t0 + e)]
    stamps.sort(key=lambda s: s[2])
    lanes = {"k/a": "front"}
    lanes.update({f"k/w{i}": "x" for i in range(len(waits))})
    return make(lanes, stamps)


def samples_for(tmp, profile, origin, n=400, seed=1):
    """An export whose series see `profile(window position)`, the first live sample at window position `origin`."""
    s = {"barrier": synth(profile, P, n, origin=origin, seed=seed, noise=0.15),
         "wait": synth(hump(0.5, 0.1, 4.0, 1.0), P, n, origin=origin, seed=seed + 1, noise=0.15),
         "mio": synth(hump(0.1, 0.1, 3.0, 1.0), P, n, origin=origin, seed=seed + 2, noise=0.15)}
    return write_csv(os.path.join(tmp, f"s{seed}.csv"), s)


def step(lo, hi, high=5.0, low=1.0):
    return lambda pos: high if lo <= pos < hi else low


class Setup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def fold(self, profile, origin, **kw):
        return sampling.fold([sampling.read_export(samples_for(self.dir.name, profile, origin, **kw))])


class Profile(unittest.TestCase):
    def test_the_profile_counts_lanes_in_a_wait_and_smooths_by_the_blur(self):
        view = overlay.view_of(stamp_record([(200, 500)] * 4))
        sharp = overlay.wait_profile(view, 8, 0.0)
        # 4 lanes wait 0.2..0.5: bins of 0.125: [0.125,0.25) holds 0.05 of it (0.4 of the bin), [0.25,0.375) and
        # [0.375,0.5) are full, [0.5,0.625) none
        self.assertAlmostEqual(sharp[1], 4 * 0.05 / 0.125)
        self.assertAlmostEqual(sharp[2], 4.0)
        self.assertAlmostEqual(sharp[3], 4.0)
        self.assertAlmostEqual(sharp[4], 0.0)
        smooth = overlay.wait_profile(view, 8, 0.38)
        self.assertAlmostEqual(sum(smooth), sum(sharp), places=6)         # a box moves mass, never makes it
        self.assertLess(max(smooth), max(sharp))
        # a sample starting at a position totals the waits AHEAD of it: the hump is carried back by the box
        self.assertGreater(smooth[0], 1.0)
        self.assertLess(smooth[4], 1e-9)

    def test_a_wait_that_straddles_the_window_edge_is_counted_at_both_ends(self):
        view = overlay.view_of(stamp_record([(900, 1100)] * 2))
        prof = overlay.wait_profile(view, 10, 0.0)
        self.assertAlmostEqual(prof[9], 2.0 * 1.0, places=1)
        self.assertAlmostEqual(prof[0], 2.0, places=1)
        self.assertAlmostEqual(prof[5], 0.0)

    def test_a_record_without_waits_cannot_be_aligned_and_the_error_names_what_is_needed(self):
        view = overlay.view_of(regular())
        with self.assertRaises(overlay.OverlayError) as e:
            overlay.wait_profile(view, 8, 0.38)
        self.assertIn("barrier.<B>.wait.begin", str(e.exception))
        self.assertIn("fine record", str(e.exception))

    def test_each_roles_phase_at_a_position(self):
        view = overlay.view_of(stamp_record([(200, 500)] * 3))
        roles, ph = overlay.phases(view, 8)
        self.assertEqual(roles, ["front", "x"])
        self.assertEqual(ph[2][0], ("main", 1.0))       # 0.25..0.375 is inside phase.main (0.05..0.6)
        self.assertEqual(ph[7][0], ("-", 1.0))          # 0.875..1 is in none


class Alignment(Setup):
    def test_a_known_shift_is_recovered(self):
        view = overlay.view_of(stamp_record([(200, 500)] * 4))
        for origin in (0.0, 0.375, 0.75):
            # the samples' first live sample sits at window position `origin`; the series sees step(0.2, 0.5) there
            f = self.fold(lambda pos, o=origin: step(0.2, 0.5)((pos + o) % 1.0), 0.0)
            a = overlay.align(view, f)
            self.assertTrue(a.accepted, a.shift.why)
            self.assertAlmostEqual(a.origin, origin, delta=0.125 + 1e-9, msg=origin)
            self.assertGreater(a.shift.corr, 0.9)
            self.assertLess(a.shift.runner_corr, a.shift.corr - 0.05)

    def with_fold(self, view, bins):
        f = self.fold(step(0.2, 0.5), 0.0)
        name = next(n for n in f.names if n.endswith("stalled_barrier.avg"))
        f.bins[name] = [sampling.Bin(v, 0.1, 50) for v in bins]
        return overlay.align(view, f)

    def test_two_equal_humps_half_a_window_apart_are_refused(self):
        # the stamps and the samples both have two equal humps half a window apart: shift s and s + 4 fit alike
        view = overlay.view_of(stamp_record([(100, 300)] * 3 + [(600, 800)] * 3))
        prof = overlay.wait_profile(view, 8, 1 / P)
        self.assertAlmostEqual(prof[0], prof[4], places=6)
        a = self.with_fold(view, prof)
        self.assertFalse(a.accepted)
        self.assertTrue(a.shift.why.startswith("AMBIGUOUS"), a.shift.why)
        self.assertAlmostEqual(a.shift.corr, 1.0)
        self.assertAlmostEqual(a.shift.runner_corr, 1.0)
        text = overlay.format_overlay(a, "rec", ["s.csv"])
        self.assertIn("ALIGNMENT REFUSED", text)
        self.assertNotIn("OVERLAY:", text)
        self.assertIsNone(overlay.to_dict(a, "rec")["overlay"])

    def test_a_flat_profile_is_refused(self):
        sh = sampling.best_shift([2.0] * 8, [1, 5, 9, 5, 1, 1, 1, 1])
        self.assertFalse(sh.accepted)
        self.assertTrue(sh.why.startswith("FLAT"))

    def test_samples_without_variation_give_no_period_and_no_alignment(self):
        flat = write_csv(os.path.join(self.dir.name, "f.csv"), {"barrier": [0.0] * 40 + [3.0] * 200 + [0.0] * 30})
        with self.assertRaises(sampling.SamplingError):
            sampling.fold([sampling.read_export(flat)])

    def test_a_weak_correlation_is_refused(self):
        view = overlay.view_of(stamp_record([(200, 500)] * 4))
        a = self.with_fold(view, [1, 3, 2, 4, 2, 5, 1, 0])
        self.assertFalse(a.accepted)
        self.assertTrue(a.shift.why.startswith("WEAK"), a.shift.why)

    def test_the_overlay_json_and_perfetto_carry_the_roles_and_the_reasons(self):
        view = overlay.view_of(stamp_record([(200, 500)] * 4))
        f = self.fold(step(0.2, 0.5), 0.0)
        a = overlay.align(view, f)
        self.assertTrue(a.accepted)
        text = overlay.format_overlay(a, "rec.json", ["s.csv"])
        for needle in ("SHIFT", "ACCEPTED", "OVERLAY:", "front", "main", "THIS PAIR of launches"):
            self.assertIn(needle, text)
        d = json.loads(json.dumps(overlay.to_dict(a, "rec.json")))
        self.assertEqual(len(d["overlay"]["bins"]), 8)
        self.assertIn("barrier", d["overlay"]["bins"][0]["reasons"])
        self.assertIn("main", {b["phase"]["front"]["phase"] for b in d["overlay"]["bins"]})
        doc = overlay.perfetto_doc(a)
        counters = [e for e in doc["traceEvents"] if e["ph"] == "C"]
        self.assertEqual({e["name"] for e in counters}, {x[len("smsp__warps_issue_stalled_"):-4]
                                                         for x in f.varying()})
        self.assertEqual(len([e for e in counters if e["name"] == "barrier"]), 9)
        self.assertTrue(any(e["ph"] == "X" for e in doc["traceEvents"]))     # the stamp lanes as today
        self.assertTrue(max(e["ts"] for e in counters) <= 1000 + 1e-6)       # one average window


if __name__ == "__main__":
    unittest.main()
