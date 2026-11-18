import json
import statistics
import unittest

from warpline import record, window
from tests.helpers import make, regular

JIT = (-4, -2, 0, 2, 4)


class AverageWindow(unittest.TestCase):
    def setUp(self):
        self.rec = regular()
        self.av = window.average_window(self.rec)
        self.sw = self.av.scopes["k"]
        self.rows = {(r.lane, r.name, r.occ): r for r in self.sw.rows}

    def test_period_and_anchor(self):
        self.assertEqual(self.sw.anchor, "window.begin")
        self.assertEqual(self.sw.anchor_lane, "k/a")
        self.assertEqual((self.sw.period, self.sw.period_std), (1000, 0))
        # 13 begins -> 12 windows between them, the first 2 skipped; the last begin opens a partial window
        self.assertEqual((self.sw.windows, self.sw.skip), (10, 2))

    def test_offsets_and_spreads_are_recovered(self):
        steady = [JIT[w % 5] for w in range(2, 12)]
        y = self.rows[("k/b", "phase.y.begin", 0)]
        self.assertAlmostEqual(y.mean, 300 + statistics.mean(steady))
        self.assertAlmostEqual(y.std, statistics.pstdev(steady))
        self.assertEqual((y.min, y.max, y.n), (300 + min(steady), 300 + max(steady), 10))
        fixed = self.rows[("k/a", "phase.x.begin", 0)]
        self.assertEqual((fixed.mean, fixed.std, fixed.min, fixed.max, fixed.n), (400, 0, 400, 400, 10))
        self.assertEqual(self.rows[("k/b", "phase.y.end", 0)].mean, 600)  # the warm-up windows' lateness is skipped
        self.assertNotIn(("k/b", "phase.y.begin", 1), self.rows)          # the partial window's stamp is dropped

    def test_the_most_regular_recurring_region_is_the_default_anchor(self):
        stamps = []
        for w, t in enumerate([0, 100, 130, 400, 410, 900, 1000, 1010, 1500]):  # irregular
            stamps.append(("k/a", "wobble.begin", t))
        for w in range(9):
            stamps.append(("k/a", "steady.begin", 50 + 200 * w))
        rec = make({"k/a": "r"}, sorted(stamps, key=lambda s: s[2]))
        sw = window.average_window(rec).scopes["k"]
        self.assertEqual(sw.anchor, "steady.begin")
        self.assertIn("most regular", sw.chosen)
        self.assertEqual(window.average_window(rec, "wobble").scopes["k"].anchor, "wobble.begin")

    def test_a_name_twice_in_a_window_is_told_apart_by_occurrence(self):
        rec = regular(extra=[("k/b", "tick", 100), ("k/b", "tick", 200)])
        rows = {(r.lane, r.name, r.occ): r for r in window.average_window(rec).scopes["k"].rows}
        self.assertEqual((rows[("k/b", "tick", 0)].mean, rows[("k/b", "tick", 1)].mean), (100, 200))

    def test_a_lane_is_reported_as_it_falls_in_the_anchor_lanes_window(self):
        # lane b's own window starts 50 before the anchor's: its begin falls in the PREVIOUS anchor window, late
        stamps = []
        for w in range(8):
            stamps += [("k/a", "window.begin", 1000 * w), ("k/b", "its.begin", 1000 * w - 50 + 1000)]
        rec = make({"k/a": "front", "k/b": "back"}, sorted(stamps, key=lambda s: s[2]))
        rows = {(r.lane, r.name): r for r in window.average_window(rec).scopes["k"].rows}
        self.assertEqual(rows[("k/b", "its.begin")].mean, 950)

    def test_too_few_begins_is_a_named_error(self):
        rec = make({"k/a": "r"}, [("k/a", "w.begin", t) for t in (0, 10, 20)])
        with self.assertRaises(window.WindowError):
            window.average_window(rec)

    def test_by_role_averages_lanes_and_shows_their_spread(self):
        stamps = []
        for w in range(8):
            stamps += [("k/a", "window.begin", 1000 * w), ("k/b", "x", 1000 * w + 100), ("k/c", "x", 1000 * w + 120)]
        rec = make({"k/a": "front", "k/b": "back", "k/c": "back"}, sorted(stamps, key=lambda s: s[2]))
        sw = window.average_window(rec).scopes["k"]
        (row,) = [r for role, _n, rows in window.by_role(rec, sw) if role == "back" for r in rows]
        self.assertEqual((row.mean, row.lane_min, row.lane_max, row.lanes), (110, 100, 120, 2))

    def test_the_average_window_is_a_record(self):
        out = window.to_record(self.av)
        again = record.loads(json.dumps(out.to_dict()))
        w = again.facts["window"]
        self.assertEqual(w["scopes"]["k"]["period"], 1000)
        self.assertEqual(w["scopes"]["k"]["anchor"], "window.begin")
        self.assertEqual(len(w["stamps"]), len(again.stamps))
        by = {(e["lane"], e["name"]): e for e in w["stamps"]}
        self.assertEqual(by[("k/a", "phase.x.begin")]["mean"], 400)
        self.assertEqual(by[("k/b", "phase.y.begin")]["n"], 10)
        times = {(l, again.names[n]): t for l, n, t, _p in again.stamps}
        self.assertEqual(times[("k/a", "phase.x.end")], 700)


if __name__ == "__main__":
    unittest.main()
