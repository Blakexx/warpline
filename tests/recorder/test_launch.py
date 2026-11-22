"""The launch's two edges: the record's format and checks, the importer's two legend names, and a plain variant placed
through a stamped one. The synthetic launch (`launch_gen`) has a KNOWN entry position inside a sample, setup, period
and window count, and a periodic profile integrated over each sampling interval: what the placement recovers is checked
against what was put in (`test_placement` has the real profiler's stretch, jitter and varying length)."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from warpline.recorder import check, launch, overlay, record, variants
from warpline.recorder.cli import main
from warpline.recorder.importers.stamped_clocks import convert_row
from tests.recorder import launch_gen
from tests.recorder.helpers import make

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


def _json(path):
    with open(path) as f:
        return json.load(f)


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
        for needle in ("(a)", "(b)", "(c)", "THE SHAPE CHECK", "PLAIN", "own", "CYCLES A SAMPLE", "ASSUMED", "THE JOIN IS ON CYCLES",
                       "STATED ASSUMPTION"):
            self.assertIn(needle, text)
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

if __name__ == "__main__":
    unittest.main()
