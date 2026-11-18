"""The launch's two edges place the samples under the stamps. The synthetic launch (`launch_gen`) has a KNOWN entry
position inside a sample, setup, period and window count, and a periodic profile integrated over each sampling
interval: what the edges recover is checked against what was put in."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from warpline import check, launch, overlay, record, sampling, variants
from warpline.__main__ import main
from warpline.importers.stamped_clocks import convert_row
from tests import launch_gen
from tests.helpers import make

ENTRY, EXIT = 1, 2
PHASE = 1


def ev(t, kind, site):
    return [t, kind << 16 | site]


class Format(unittest.TestCase):
    L = {"k/a": "r", "k/b": "r"}

    def rec(self, stamps, ties):
        return make(self.L, stamps, ties=ties)

    def good(self):
        return self.rec([("k/a", "launch.begin", 10), ("k/a", "x.begin", 12), ("k/a", "x.end", 20),
                         ("k/a", "launch.end", 20), ("k/b", "launch.begin", 11), ("k/b", "launch.end", 22)],
                        [("k/a", 10, 100), ("k/a", 20, 180), ("k/b", 11, 102), ("k/b", 22, 190)])

    def codes(self, rec):
        return [f.code for f in check.launch_edges(rec)]

    def test_the_two_edges_are_instant_events_not_a_region(self):
        self.assertEqual(record.classify("launch.begin"), ("launch_begin", ""))
        self.assertEqual(record.classify("launch.end"), ("launch_end", ""))
        self.assertEqual(self.codes(self.good()), [])
        self.assertEqual(check.all_findings(self.good()), [])

    def test_a_record_without_edges_holds(self):
        self.assertEqual(self.codes(self.rec([("k/a", "x.begin", 1), ("k/a", "x.end", 2)], None)), [])

    def test_a_lane_with_one_edge_and_not_the_other_is_a_finding(self):
        rec = self.rec([("k/a", "launch.begin", 10), ("k/a", "z", 11)], [("k/a", 10, 100)])
        self.assertEqual(self.codes(rec), ["launch_edge_missing"])

    def test_an_edge_without_a_tie_is_a_finding(self):
        rec = self.rec([("k/a", "launch.begin", 10), ("k/a", "launch.end", 20)], [("k/a", 10, 100), ("k/a", 21, 110)])
        self.assertEqual(self.codes(rec), ["launch_edge_untied"])
        rec = self.rec([("k/a", "launch.begin", 10), ("k/a", "launch.end", 20)], None)
        self.assertEqual(self.codes(rec), ["launch_edge_untied"] * 2)

    def test_a_stamp_outside_the_edges_and_a_repeated_edge_are_findings(self):
        rec = self.rec([("k/a", "z", 5), ("k/a", "launch.begin", 10), ("k/a", "launch.end", 20), ("k/a", "y", 30)],
                       [("k/a", 10, 100), ("k/a", 20, 110)])
        self.assertEqual(self.codes(rec), ["stamp_outside_launch", "stamp_outside_launch"])
        rec = self.rec([("k/a", "launch.begin", 10), ("k/a", "launch.begin", 12), ("k/a", "launch.end", 20)],
                       [("k/a", 10, 100), ("k/a", 12, 105), ("k/a", 20, 110)])
        self.assertIn("launch_edge_repeated", self.codes(rec))

    def test_check_runs_it(self):
        self.assertIn(check.launch_edges, check.CHECKS)


class Importer(unittest.TestCase):
    def row(self, legend_extra=None):
        legend = {"100": "work", "101": "entry", "102": "exit", "103": "start", "104": "stop"}
        legend.update(legend_extra or {})
        # the nanosecond reading (kind 6) of a tie immediately follows its stamp, as the importer's tie convention says
        w = [ev(0, PHASE, 101), ev(500, 6, 101), ev(2, PHASE, 100), ev(30, PHASE, 102), ev(900, 6, 102)]
        return {"variant": "v", "roles": ["a"], "legend": legend,
                "ctas": [{"cta": 0, "warps": [{"count": len(w), "stamps": sum(w, [])}]}]}

    def test_the_two_legend_names_become_the_launch_edges_and_close_the_phase(self):
        doc, unmapped = convert_row(self.row())
        rec = record.from_dict(doc)
        self.assertEqual(unmapped, [])
        names = [rec.names[n] for _l, n, _t, _p in rec.stamps]
        self.assertEqual(names, ["launch.begin", "phase.work.begin", "phase.work.end", "launch.end"])
        self.assertNotIn("phase.entry.begin", names)                  # no phase region opens at the entry
        self.assertEqual(rec.ties["cta0/warp0"], [(0, 500), (30, 900)])   # each edge carries its tie
        self.assertEqual(check.launch_edges(rec), [])

    def test_other_legend_names_by_option(self):
        doc, _ = convert_row(self.row(), launch_begin="entry", launch_end="exit")
        row = self.row({"101": "start", "102": "stop"})
        doc2, _ = convert_row(row, launch_begin="start", launch_end="stop")
        self.assertEqual(doc["stamps"], doc2["stamps"])
        doc3, _ = convert_row(row)       # the defaults do not match: those sites are phases as before
        self.assertIn("phase.start.begin", doc3["names"].values())
        self.assertNotIn("launch.begin", doc3["names"].values())

    def test_an_input_without_them_imports_as_today(self):
        row = self.row({"101": "a", "102": "b"})
        doc, _ = convert_row(row)
        self.assertNotIn("launch.begin", doc["names"].values())
        self.assertIn("phase.a.begin", doc["names"].values())


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.set = launch_gen.make_set(cls.dir.name)
        cls.rec = record.load(cls.set["record"])
        cls.view = overlay.view_of(cls.rec)
        cls.lc = overlay.launch_for(cls.rec)
        cls.exports = [sampling.read_export(p) for p in cls.set["stamped"]]
        cls.fold = sampling.fold(cls.exports)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def edges(self, **kw):
        return sampling.edge_placement(self.exports, self.fold, self.lc.length_us, self.lc.period_us, self.fold.cfg)

    def test_the_launch_in_time_is_read_from_the_ties(self):
        s, lc = self.set, self.lc
        self.assertEqual(lc.length_ns, round(s["length"] * 1000))
        self.assertAlmostEqual(lc.period_us, s["period"], places=6)
        self.assertEqual(lc.setup_ns, round((s["setup"] + 2 * s["period"]) * 1000))   # to the first STEADY window (skip 2)
        self.assertEqual(lc.windows, s["windows"] - 2)
        self.assertEqual((lc.begin_lane, lc.end_lane), ("cta0/warp0", "cta0/warp3"))   # the earliest begin, the latest end
        self.assertEqual(lc.lanes, 4)
        self.assertEqual(lc.findings, [])

    def test_the_interval_for_the_entry_contains_the_truth(self):
        for (run, pi), u in self.set["truth"].items():
            pp = next(p for p in self.edges().values() if p.checks and p.checks[0].i0 == launch_gen.pass_start(pi))
            c = next(c for c in pp.checks if c.run == run)
            self.assertTrue(c.entry.ok)
            self.assertLessEqual(c.entry.lo - 1e-9, u)
            self.assertLess(u, c.entry.hi + 1e-9)

    def test_the_intersection_narrows_with_runs_and_contains_the_truth(self):
        pps = self.edges()
        by_start = {pp.checks[0].i0: pp for pp in pps.values()}
        narrowed = 0
        for pi in range(3):
            pp = by_start[launch_gen.pass_start(pi)]
            self.assertTrue(pp.placed, pp.findings)
            self.assertEqual(pp.fed, [0, 1, 2], pp.findings)
            widths = [w for _n, w in pp.steps]
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(widths, widths[1:])), widths)
            narrowed += widths[-1] < widths[0] - 1e-9
            # u of the reference run (run 0): the truth is inside
            u0 = self.set["truth"][(0, pi)]
            self.assertLessEqual(pp.lo - 1e-9, u0, (pi, pp.lo, pp.hi, u0))
            self.assertLess(u0, pp.hi + 1e-9, (pi, pp.lo, pp.hi, u0))
            # the other runs' own positions are inside their own intervals and near the truth
            for r in (1, 2):
                self.assertLess(abs(pp.u[r] - self.set["truth"][(r, pi)]), pp.half + pp.slack_us + 0.45, (pi, r))
        self.assertGreaterEqual(narrowed, 2)

    def test_more_runs_narrow_it(self):
        with tempfile.TemporaryDirectory() as d:
            s = launch_gen.make_set(d, runs=5)
            ex = [sampling.read_export(p) for p in s["stamped"]]
            f = sampling.fold(ex)
            lc = overlay.launch_for(record.load(s["record"]))
            pps = sampling.edge_placement(ex, f, lc.length_us, lc.period_us)
            for pp in pps.values():
                w = [x for _n, x in pp.steps]
                self.assertEqual(len(w), len(pp.fed))
                self.assertTrue(all(b <= a + 1e-9 for a, b in zip(w, w[1:])), w)
                self.assertLess(w[-1], w[0] - 1e-9)

    def test_the_folded_profile_lands_at_the_true_window_position_within_the_half_width(self):
        pl = overlay.place(self.view, self.lc, self.exports, self.fold)
        n = pl.bins
        for name, pi in (("wait", 1), ("mio", 2)):
            full = f"smsp__warps_issue_stalled_{name}.avg"
            s = pl.series[full]
            truth = self.set["truth"][(0, pi)]
            pp = pl.edges[s.key]
            # the half-width is the placement's own; the truth is inside it, so the position error is within it
            self.assertLessEqual(abs(pp.mid - truth), pp.half + 1e-9)
            self.assertAlmostEqual(s.half_windows, pp.half / self.lc.period_us)
            prof = launch_gen.SERIES[name][1]
            vals = [b.mean for b in s.bins]
            # the noiseless fold at the TRUE entry position: the estimate's phase differs from it by the placement's
            # error only, which the half-width (and the run alignment's half bin) bounds
            ideal = noiseless_fold(prof, self.set, pi, truth, self.lc, n)
            err = circ(phase(vals), phase(ideal))
            self.assertLessEqual(err, s.half_windows + 0.5 / n + 0.01, (name, err, s.half_windows))
        # the hump of `wait` (centre 0.6) is in bin 4 or its neighbour
        vals = [b.mean for b in pl.series["smsp__warps_issue_stalled_wait.avg"].bins]
        self.assertLessEqual(abs(vals.index(max(vals)) - 4), 1)

    def test_the_old_correlation_is_the_cross_check_and_agrees(self):
        pl = overlay.place(self.view, self.lc, self.exports, self.fold)
        self.assertTrue(pl.cross.agree, pl.cross.note)
        self.assertIn("one bin", pl.cross.note)
        self.assertEqual([x for x in pl.findings if x.startswith("cross_check")], [])
        # a setup that is wrong by 0.4 of a window puts the edges away from the correlation: a named finding
        import copy
        bad = copy.copy(self.lc)
        bad.setup_ns = self.lc.setup_ns + round(0.4 * self.lc.period_ns)
        pl2 = overlay.place(self.view, bad, self.exports, self.fold)
        self.assertFalse(pl2.cross.agree)
        self.assertTrue(any(x.startswith("cross_check_disagrees") for x in pl2.findings))

    def test_a_launch_length_off_by_two_samples_is_refused_by_the_length_check(self):
        for off in (+2 * 1.5, -2 * 1.5):
            pps = sampling.edge_placement(self.exports, self.fold, self.lc.length_us + off, self.lc.period_us)
            for pp in pps.values():
                self.assertFalse(pp.placed, off)
                self.assertTrue(any("LAUNCH LENGTH MISMATCH" in x for x in pp.findings))
            text = " ".join(pps[next(iter(pps))].findings)
            for needle in ("T ", "live samples", "us", "not the same kernel in time"):
                self.assertIn(needle, text)
            pl = overlay.place(self.view, _with_length(self.lc, off), self.exports, self.fold)
            self.assertEqual(pl.series, {})                    # that series are left out
            self.assertTrue(any("LAUNCH LENGTH MISMATCH" in x for x in pl.findings))
        # within the tolerance (a quarter of a sample past one) it passes, with the interval collapsed to the edge
        pps = sampling.edge_placement(self.exports, self.fold, self.lc.length_us + 1.5 * 1.1, self.lc.period_us)
        self.assertTrue(any(pp.placed for pp in pps.values()))

    def test_a_record_without_the_launch_events_is_refused_naming_what_to_stamp(self):
        doc = _json(self.set["record"])
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
        # an edge without its tie is refused too
        doc = _json(self.set["record"])
        begin_id = next(i for i, n in doc["names"].items() if n == "launch.begin")
        cycles = {st[2] for st in doc["stamps"] if str(st[1]) == begin_id}
        doc["ties"] = [t for t in doc["ties"] if t[1] not in cycles]
        with self.assertRaises(overlay.OverlayError) as e:
            overlay.launch_for(record.from_dict(doc))
        self.assertIn("tied", str(e.exception))

    def test_the_overlay_command_prints_the_placement_json_and_perfetto(self):
        out = io.StringIO()
        pf = os.path.join(self.dir.name, "p.json")
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--perfetto", pf]), 0)
        text = out.getvalue()
        for needle in ("PLACEMENT", "half-width", "CROSS-CHECK", "OVERLAY:", "front", "LOOSE", "narrowing",
                       "ASSUMED AND STATED", "length check"):
            self.assertIn(needle, text)
        doc = _json(pf)
        self.assertTrue(any(e["ph"] == "C" for e in doc["traceEvents"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["overlay", self.set["record"]] + self.set["stamped"] + ["--json"]), 0)
        d = json.loads(out.getvalue())
        self.assertEqual(len(d["overlay"]["bins"]), 8)
        self.assertIn("wait", d["series"])


def _json(path):
    with open(path) as f:
        return json.load(f)


def _with_length(lc, off_us):
    import copy
    x = copy.copy(lc)
    x.end_ns = lc.end_ns + round(off_us * 1000)
    return x


def phase(vals):
    """The circular centre of a folded profile (a fraction of a window), above its minimum."""
    import math
    lo, n = min(vals), len(vals)
    c = sum((v - lo) * math.cos(2 * math.pi * (b + 0.5) / n) for b, v in enumerate(vals))
    s = sum((v - lo) * math.sin(2 * math.pi * (b + 0.5) / n) for b, v in enumerate(vals))
    return (math.atan2(s, c) / (2 * math.pi)) % 1.0


def circ(a, b):
    d = (a - b) % 1.0
    return min(d, 1.0 - d)


def noiseless_fold(prof, s, pass_index, u, lc, n):
    """The placed fold of a pass's samples made with no noise, placed at the TRUE entry position."""
    dt = s["dt"]
    i0 = launch_gen.pass_start(pass_index)
    i1 = int((i0 * dt + u + s["length"]) // dt)
    import random
    vals = launch_gen.sample_values(prof, dt, i0, u, s["length"], s["setup"], s["period"], i1 + 30, 0.0,
                                    random.Random(1))
    cfg = sampling.Config()
    bins = sampling.placed_bins(vals, i0, i1, dt, u, lc.setup_us, lc.period_us, cfg)
    return [b.mean for b in bins]


class ThroughTheStampedVariant(unittest.TestCase):
    def run_set(self, **kw):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name, **kw)
        return s, variants.run(s["manifest"])

    def test_the_set_places_the_plain_variant_through_the_stamped_one(self):
        s, res = self.run_set()
        by = {v.name: v for v in res.variants}
        self.assertEqual(by["stamped"].status, "ok")
        self.assertEqual(len(res.overlays), 1)
        o = res.overlays[0]
        self.assertEqual(o.findings, [])
        for ps in o.series.values():
            self.assertIsNotNone(ps.bins, ps.finding)
            self.assertGreater(ps.shift.corr, 0.7)
        text = variants.format_set(res)
        for needle in ("(a)", "(b)", "(c)", "THE SHAPE CHECK", "PLAIN", "own", "launch length against the live span",
                       "ASSUMED"):
            self.assertIn(needle, text)
        # the stamped variant's launch length by its ties against its own samples is shown for every pass and run
        self.assertEqual(text.count("launch length against the live span, pass"), 3)
        json.dumps(variants.to_dict(res))

    def test_the_plain_variants_known_rotation_is_recovered(self):
        # the plain variant's setup is the stamped one's plus `plain_rot` of a window: the recovered rotation of its
        # fold moves by that many bins (the same entry positions in both sets) and sits where the truth puts it
        s0, r0 = self.run_set(plain_rot=0.0)
        s1, r1 = self.run_set(plain_rot=0.25)
        n = 8
        for name in ("barrier", "wait", "mio"):
            full = f"smsp__warps_issue_stalled_{name}.avg"
            a, b = r0.overlays[0].series[full].shift.shift, r1.overlays[0].series[full].shift.shift
            self.assertLessEqual(min((b - a - 2) % n, (2 - b + a) % n), 1, (name, a, b))   # 0.25 of a window = 2 bins
        # and the rotation itself: the plain fold's frame is its reference run's sample grid, so the expected shift is
        # the plain entry's window phase (setup + u over the plain window, plus the rotation), less half a sample
        for name, pi in (("barrier", 0), ("wait", 1)):
            full = f"smsp__warps_issue_stalled_{name}.avg"
            pp = 3.62 * 0.98
            a_p = (s1["plain_truth"][(0, pi)] + 8.7) / pp + 0.25
            expect = ((a_p - 0.5 * 1.5 / pp) % 1.0) * n
            got = r1.overlays[0].series[full].shift.shift
            self.assertLessEqual(min((got - expect) % n, (expect - got) % n), 1.0, (name, got, expect))

    def test_a_plain_variant_with_a_different_profile_is_named_by_the_shape_check(self):
        s, res = self.run_set(plain_shape="other")
        o = res.overlays[0]
        named = [ps for ps in o.series.values() if ps.bins is None]
        self.assertTrue(named, "no series was named")
        for ps in named:
            self.assertTrue(ps.finding.startswith("SHAPE CHECK"), ps.finding)
            self.assertIn("not overlaid", ps.finding)
            self.assertTrue(any(x in ps.finding for x in ("WEAK", "AMBIGUOUS", "not folded", "flat in")), ps.finding)
        self.assertEqual(len(o.findings), len(named))
        self.assertGreaterEqual(len(named), 3)       # blurred by a sample, the other profile is flat in two series
        text = variants.format_set(res)
        self.assertIn("SHAPE CHECK", text)
        # a series that is not overlaid is not in the table's PLAIN columns
        for ps in named:
            self.assertNotIn(f"{ps.short} PLAIN", text)

    def test_a_stamped_variant_whose_samples_fail_the_length_check_is_not_laid_over(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        s = launch_gen.make_set(d.name)
        doc = _json(s["record"])
        end_id = int(next(i for i, n in doc["names"].items() if n == "launch.end"))
        cut, later = 12000 + round(s["length"] * 1000) - 200, 3000      # every exit two samples later
        doc["ties"] = [[ln, c + round(later * 1.8) if ns >= cut else c, ns + later if ns >= cut else ns]
                       for ln, c, ns in doc["ties"]]
        for st in doc["stamps"]:
            if st[1] == end_id:
                st[2] += round(later * 1.8)
        with open(s["record"], "w") as f:
            json.dump(doc, f)
        res = variants.run(s["manifest"])
        text = variants.format_set(res)
        self.assertIn("FAILS", text)                           # the length check's output is shown even so
        self.assertEqual(res.overlays[0].series and [x for x in res.overlays[0].series.values() if x.bins], [])
        self.assertTrue(res.overlays[0].findings)
        self.assertIn("LAUNCH LENGTH MISMATCH", text)


if __name__ == "__main__":
    unittest.main()
