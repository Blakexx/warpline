"""The two clocks: ties, the rate between a record's cycle counter and time, placement between ties, and where the
ties and a variant's samples meet. The synthetic records hold a KNOWN rate that changes between two regions."""
import json
import os
import tempfile
import unittest

from warpline import check, overlay, rates, record, variants, window
from warpline.importers.stamped_clocks import convert_row
from warpline.record import RecordError, from_dict
from tests.helpers import hump, make, regular, synth, write_csv

# a window of 20,000 cycles: region A the first 10,000 at 2,000 cycles a us (5,000 ns), region B the next 10,000 at
# 1,000 cycles a us (10,000 ns). The window is 20,000 cycles, 15,000 ns, 1,333.33 cycles a us.
W, A_NS, B_NS, T0 = 20000, 5000, 10000, 7000


def tied(windows=8, b_scale=1.0, back_at=None):
    """Lanes a (role front) and b (role back), one scope `k`. Per window: `win.begin` and `A.begin` at t0, `mid` at
    t0 + 4,000 (untied: placed by interpolation at 2,000 ns), `A.end` and `B.begin` at t0 + 10,000, `midb` at
    t0 + 14,000 (untied, 9,000 ns after t0), `B.end` at the next t0. Ties at t0 and t0 + 10,000 on each lane; lane b's
    nanosecond readings scaled by `b_scale` (a lane whose clock disagrees with its scope's)."""
    lanes = {"k/a": "front", "k/b": "back"}
    stamps, ties = [], []
    for w in range(windows + 1):
        t0 = W * w
        ns0 = T0 + (A_NS + B_NS) * w
        for ln in lanes:
            if ln == "k/a":
                stamps.append((ln, "win.begin", t0))
            stamps += [(ln, "A.begin", t0)]
            if w < windows:
                stamps += [(ln, "mid", t0 + 4000), (ln, "A.end", t0 + 10000), (ln, "B.begin", t0 + 10000),
                           (ln, "midb", t0 + 14000)]
            if w > 0:
                stamps.append((ln, "B.end", t0))
            sc = b_scale if ln == "k/b" else 1.0
            ties.append((ln, t0, round(ns0 * sc)))
            if w < windows:
                ties.append((ln, t0 + 10000, round((ns0 + A_NS) * sc)))
    stamps.sort(key=lambda s: (s[2], s[1] != "win.begin"))
    return make(lanes, stamps, ties=ties)


class Ties(unittest.TestCase):
    def test_the_rates_are_recovered_per_region_and_the_window(self):
        res = rates.analyze(tied(), anchor="win.begin")
        self.assertEqual(res.findings, [])
        s = res.scopes["k"]
        by = {(a.role, a.name): a for a in s.by_role}
        self.assertAlmostEqual(by[("front", "A")].rate, 2000.0)
        self.assertAlmostEqual(by[("front", "B")].rate, 1000.0)
        self.assertEqual((by[("front", "A")].cycles, by[("front", "A")].ns), (10000, 5000))
        self.assertEqual((by[("back", "B")].cycles, by[("back", "B")].ns), (10000, 10000))
        for p in s.periods:
            self.assertEqual((p.cycles, p.ns), (20000, 15000))
            self.assertAlmostEqual(p.rate, 20000 / 15.0)
        self.assertEqual(s.windows, 6)  # 8 begins + the trailing one: 9, minus 2 skipped, minus the partial

    def test_the_rate_between_consecutive_ties_and_the_grain(self):
        res = rates.analyze(tied(), anchor="win.begin")
        ps = res.lanes["k/a"]["pairs"]
        self.assertEqual([round(p.rate, 6) for p in ps[:3]], [2000.0, 1000.0, 2000.0])
        self.assertEqual(res.grain, 5000)  # the counter's step: gcd of the differences (5,000 and 10,000)
        self.assertEqual(res.smallest, 5000)
        self.assertAlmostEqual(ps[0].unc, 5000 / 5000)
        self.assertAlmostEqual(ps[1].unc, 5000 / 10000)

    def test_the_grain_is_the_step_not_the_shortest_interval(self):
        # readings on a 32 ns step, ties 4,992 and 5,024 ns apart (156 and 157 steps): the step is 32, the smallest
        # difference 4,992
        rec = make({"k/w": "x"}, [("k/w", "a", 0)],
                   ties=[("k/w", 0, 64), ("k/w", 9000, 64 + 4992), ("k/w", 20880, 64 + 4992 + 5024)])
        self.assertEqual((rates.grain(rec), rates.smallest_difference(rec)), (32, 4992))

    def test_a_stamp_between_ties_is_placed_exactly_for_a_piecewise_constant_rate(self):
        rec = tied()
        at = rates.placer(rec)
        self.assertEqual(at("k/a", W * 3 + 4000) - at("k/a", W * 3), 2000)         # 4,000 cycles at 2 cycles a ns
        self.assertEqual(at("k/b", W * 3 + 14000) - at("k/b", W * 3), 9000)        # 5,000 ns + 4,000 ns
        self.assertEqual(at("k/a", W * 3 + 10000), T0 + 3 * 15000 + 5000)          # a tie: its own reading
        self.assertIsNone(at("k/a", -5))                                           # outside the tied span
        self.assertIsNone(at("k/a", W * 8 + 1))
        av = window.average_window(rec, "win.begin", 2, clock_of=at)
        sw = av.scopes["k"]
        self.assertEqual(sw.period, 15000)
        rows = {(r.lane, r.name): r for r in sw.rows}
        self.assertEqual(rows[("k/a", "mid")].mean, 2000)
        self.assertEqual(rows[("k/a", "mid")].std, 0)
        self.assertEqual(rows[("k/b", "midb")].mean, 9000)

    def test_a_lane_without_ties_places_nothing(self):
        rec = make({"k/a": "x", "k/b": "y"}, [("k/a", "a", 5), ("k/b", "b", 6)], ties=[("k/a", 0, 0), ("k/a", 10, 5)])
        at = rates.placer(rec)
        self.assertEqual(at("k/a", 5), 2.5)
        self.assertIsNone(at("k/b", 6))

    def test_a_backwards_tie_is_a_named_finding(self):
        L = {"k/w": "x"}
        cyc = make(L, [("k/w", "a", 0)], ties=[("k/w", 100, 10), ("k/w", 90, 20)])
        ns = make(L, [("k/w", "a", 0)], ties=[("k/w", 100, 30), ("k/w", 200, 20)])
        ok = make(L, [("k/w", "a", 0)], ties=[("k/w", 100, 30), ("k/w", 200, 30)])
        self.assertEqual([f.code for f in check.ties_ordered(cyc)], ["tie_backwards"])
        self.assertEqual([f.code for f in check.ties_ordered(ns)], ["tie_backwards"])
        self.assertEqual(check.ties_ordered(ok), [])
        self.assertIn("tie_backwards", [f.code for f in check.all_findings(cyc)])
        self.assertIn("tie_backwards", [f.code for f in rates.analyze(ns).findings])
        with self.assertRaises(ValueError):
            rates.placer(ns)
        self.assertEqual(rates.analyze(ns).lanes["k/w"]["pairs"], [])   # no pair advances: no rate from a bad pair

    def test_a_tie_needs_its_lane_and_a_time_axis(self):
        def code(mutate):
            d = make({"k/w": "x"}, [("k/w", "a", 0)], ties=[("k/w", 1, 1)]).to_dict()
            mutate(d)
            with self.assertRaises(RecordError) as cm:
                from_dict(d)
            return cm.exception.code
        self.assertEqual(code(lambda d: d.update(ties=[["k/z", 1, 1]])), "unknown_lane")
        self.assertEqual(code(lambda d: d.update(ties=[["k/w", 1]])), "bad_tie")
        self.assertEqual(code(lambda d: d.update(ties=[["k/w", 1.5, 2]])), "bad_tie")
        self.assertEqual(code(lambda d: d["clock"].pop("time_unit")), "bad_clock")
        d = make({"k/w": "x"}, [("k/w", "a", 0)]).to_dict()
        d["ties"] = []
        self.assertEqual(from_dict(d).ties, {})   # absent or empty is valid
        rec = tied(3)
        self.assertEqual(record.loads(json.dumps(rec.to_dict())).ties, rec.ties)

    def test_a_lane_whose_rate_disagrees_with_its_scope_is_named(self):
        self.assertEqual([f for f in rates.analyze(tied(), anchor="win.begin").findings], [])
        res = rates.analyze(tied(b_scale=1.02), anchor="win.begin")
        bad = [f for f in res.findings if f.code == "lane_rate_disagrees"]
        self.assertEqual([f.lane for f in bad], ["k/a", "k/b"])   # two lanes: neither is the majority; both named
        res = rates.analyze(tied(b_scale=1.02), anchor="win.begin", tolerance=0.03)
        self.assertEqual(res.findings, [])

    def test_no_ties_says_so(self):
        text = rates.format_rates(rates.analyze(regular()))
        self.assertIn("carries no ties", text)


def ties_for(rec, ns_per_window, lane="k/a"):
    """The record with a tie at each window start of `lane` (window.begin: the record's `phase.x.begin` at 400 into
    each window), the windows `ns_per_window` apart."""
    d = rec.to_dict()
    d["clock"] = {"unit": "cycles", "hz": None, "time_unit": "ns", "time_origin": "the test's own"}
    d["ties"] = [[lane, c, 100 + round(c * ns_per_window / 1000)] for ln, nid, c, *_ in d["stamps"]
                 if ln == lane and d["names"][str(nid)] == "window.begin"]
    return from_dict(d)


class Overlay(unittest.TestCase):
    def test_an_overlay_places_the_window_by_ties_and_says_so(self):
        rec = tied()
        v = overlay.view_of(rec)
        self.assertEqual(v.axis, "ns")
        self.assertEqual(v.period_ns, 15000)
        self.assertEqual(v.span, 15000)
        self.assertIn("nanoseconds", v.axis_note)
        self.assertIn("NOT used", v.axis_note)
        mid = [r for r in v.rows if r.lane == "k/a" and r.name == "mid"][0]
        self.assertEqual(mid.mean, 2000)   # by the ties, not 4,000 / 20,000 of the window
        self.assertEqual(overlay.view_of(regular()).axis, "cycles")

    def test_ties_that_go_backwards_refuse_the_overlay(self):
        rec = make({"k/w": "x"}, [("k/w", "a.begin", i * 10) for i in range(8)],
                   ties=[("k/w", 0, 10), ("k/w", 10, 5)])
        with self.assertRaises(overlay.OverlayError):
            overlay.view_of(rec)


class Variants(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        p = lambda n: os.path.join(self.d.name, n)
        P = 2.6
        waits = [("k/b", "barrier.Q.wait.begin", 100), ("k/b", "barrier.Q.wait.end", 400)]
        for name, per, seed in (("plain", P, 1), ("a", P, 10), ("d", P, 20)):
            s = {"barrier": synth(hump(0.3, 0.12, 8.0, 1.0), per, 200, seed=seed, noise=0.2),
                 "wait": synth(hump(0.6, 0.12, 5.0, 1.0), per, 200, seed=seed + 1, noise=0.2),
                 "mio": synth(hump(0.8, 0.12, 4.0, 1.0), per, 200, seed=seed + 2, noise=0.2)}
            write_csv(p(name + ".csv"), s)
        # a window of 2.6 samples x 1.5 us = 3.9 us = 3,900 ns; `a`'s ties say so, `d`'s say 4,500 ns (+15%)
        record.dump(ties_for(regular(extra=waits), 3900), p("a.json"))
        record.dump(ties_for(regular(extra=waits), 4500), p("d.json"))
        self.manifest = {"kernel": "k", "variants": [
            {"name": "plain", "plain": True, "stamps": None, "samples": ["plain.csv"]},
            {"name": "a", "plain": False, "stamps": "a.json", "samples": ["a.csv"]},
            {"name": "d", "plain": False, "stamps": "d.json", "samples": ["d.csv"]}]}
        with open(p("m.json"), "w") as f:
            json.dump(self.manifest, f)
        self.path = p("m.json")

    def test_ties_and_samples_that_disagree_are_a_named_finding(self):
        res = variants.run(self.path)
        by = {v.name: v for v in res.variants}
        self.assertAlmostEqual(by["a"].ties_us, 3.9, places=2)
        self.assertEqual(by["a"].ties_findings, [])
        self.assertIn("agree", by["a"].ties_note)
        self.assertEqual(len(by["d"].ties_findings), 1)
        self.assertIn("time_bases_disagree", by["d"].ties_findings[0])
        self.assertIn("disagree for this variant", by["d"].ties_findings[0])
        text = variants.format_set(res)
        self.assertIn("FINDING time_bases_disagree", text)
        self.assertIn("ties: window 3.900 us from its own ties", text)
        self.assertEqual(variants.to_dict(res)["variants"][2]["ties"]["findings"], by["d"].ties_findings)
        res = variants.run(self.path, ties_tolerance=0.2)
        self.assertEqual({v.name: v.ties_findings for v in res.variants}, {"plain": [], "a": [], "d": []})

    def test_a_record_without_ties_says_time_is_the_samples(self):
        with open(os.path.join(self.d.name, "n.json"), "w") as f:
            json.dump(regular().to_dict(), f)
        self.manifest["variants"][1]["stamps"] = "n.json"
        with open(self.path, "w") as f:
            json.dump(self.manifest, f)
        res = variants.run(self.path)
        a = [v for v in res.variants if v.name == "a"][0]
        self.assertIsNone(a.ties_us)
        self.assertIn("no ties", a.ties_note)
        self.assertIn("time comes from the samples alone", variants.format_set(res))


PHASE, WAITS, WAITED, ARRIVES, ISSUES, TIME = 1, 2, 3, 4, 5, 6


def ev(t, kind, site):
    return [t, kind << 16 | site]


class ImporterTime(unittest.TestCase):
    def row(self, w0):
        return {"variant": "v", "roles": ["a"], "legend": {"49152": "work", "49153": "rest", "101": "tile ready"},
                "ctas": [{"cta": 0, "warps": [{"count": len(w0), "stamps": sum(w0, [])}]}]}

    def test_a_time_entry_becomes_a_tie_and_no_stamp(self):
        w0 = [ev(100, PHASE, 49152), ev(0, TIME, 49152), ev(160, ARRIVES, 101), ev(300, PHASE, 49153),
              ev(500, TIME, 49153), ev(900, PHASE, 49152), ev(2000, TIME, 49152)]
        doc, unmapped = convert_row(self.row(w0))
        rec = from_dict(doc)
        self.assertEqual(unmapped, [])
        self.assertEqual(rec.ties, {"cta0/warp0": [(100, 0), (300, 500), (900, 2000)]})
        self.assertEqual(rec.clock["time_unit"], "ns")
        self.assertEqual(len(rec.stamps), 7 - 3 + 2)   # 4 events -> 4 stamps, + the phase ends synthesized twice
        self.assertNotIn(6, [n for n in rec.names])    # a time entry has no name
        self.assertEqual(check.stamps_accounted(rec) + check.ties_ordered(rec) + check.stamps_ordered(rec), [])
        self.assertEqual(rec.lanes["cta0/warp0"].issued, 6)

    def test_a_time_entry_that_took_a_slot_counts_as_stored(self):
        w0 = [ev(100, PHASE, 49152), ev(0, TIME, 49152), ev(300, PHASE, 49153), ev(500, TIME, 49153)]
        row = self.row(w0)
        row["ctas"][0]["warps"][0]["count"] = 6     # two entries issued after the ring filled
        doc, _ = convert_row(row)
        rec = from_dict(doc)
        self.assertEqual(rec.dropped, {"cta0/warp0": 2})
        self.assertEqual(check.stamps_accounted(rec), [])

    def test_a_time_entry_with_no_stamp_of_its_site_is_a_named_error(self):
        for w0 in ([ev(5, TIME, 49152)],                                        # first entry
                   [ev(100, PHASE, 49152), ev(0, TIME, 49153)],                 # another site's stamp
                   [ev(100, PHASE, 49152), ev(0, TIME, 49152), ev(1, TIME, 49152)]):  # the stamp was used once
            with self.assertRaises(RecordError) as cm:
                convert_row(self.row(w0))
            self.assertEqual(cm.exception.code, "time_without_stamp")

    def test_readings_that_go_back_are_reported_not_assumed_ordered(self):
        w0 = [ev(100, PHASE, 49152), ev(900, TIME, 49152), ev(300, PHASE, 49153), ev(100, TIME, 49153)]
        rec = from_dict(convert_row(self.row(w0))[0])
        self.assertEqual([f.code for f in check.ties_ordered(rec)], ["tie_backwards"])


if __name__ == "__main__":
    unittest.main()
