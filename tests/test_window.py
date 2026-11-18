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


class RolesOwnWindow(unittest.TestCase):
    """A role's window is its own: the role's lanes carry their own anchor event, a third of a period later than the
    scope's, so the scope's window would smear the role's phases (`lanes min..max`) and straddle."""

    def record(self, back_phases=((0, "p", 300), (300, "q", 700))):
        stamps = []
        for w in range(10):
            t0 = 1000 * w
            for ln in ("k/a0", "k/a1"):
                stamps += [(ln, "win.begin", t0), (ln, "p.begin", t0), (ln, "p.end", t0 + 300),
                           (ln, "q.begin", t0 + 300), (ln, "q.end", t0 + 1000)]
            for ln in ("k/b0", "k/b1"):
                t = t0 + 333
                stamps.append((ln, "win.begin", t))
                for off, nm, du in back_phases:
                    stamps += [(ln, nm + ".begin", t + off), (ln, nm + ".end", t + off + du)]
        stamps.sort(key=lambda x: x[2])
        return make({"k/a0": "front", "k/a1": "front", "k/b0": "back", "k/b1": "back"}, stamps)

    def test_each_roles_phases_are_in_its_own_window_with_durations_that_sum_to_its_period(self):
        rws = {rw.role: rw for rw in window.role_windows(self.record())}
        front, back = rws["front"], rws["back"]
        for rw in (front, back):
            self.assertTrue(rw.own)
            self.assertEqual(rw.period, 1000)
            self.assertTrue(rw.sums, rw.sum_note())
            self.assertAlmostEqual(rw.total, 1000)
            rows = {r.name: r for r in rw.rows}
            self.assertEqual((rows["p"].start, rows["p"].dur), (0, 300))
            self.assertEqual((rows["q"].start, rows["q"].dur), (300, 700))
            self.assertEqual((rows["p"].start_min, rows["p"].start_max), (0, 0))      # no spread across lanes
        self.assertAlmostEqual(front.offset, 0.0)
        self.assertAlmostEqual(back.offset, 333.0)               # its window starts a third of a period later
        # the scope's window would have put the back lanes' phases at 333 and 633, and the last one past the window
        sw = window.average_window(self.record()).scopes["k"]
        rows = {(r.lane, r.name): r for r in sw.rows}
        self.assertEqual(rows[("k/b0", "p.begin")].mean, 333)

    def test_phases_that_do_not_tile_the_window_are_said_not_to_sum(self):
        rw = {r.role: r for r in window.role_windows(self.record(((0, "p", 300), (300, "q", 400))))}["back"]
        self.assertFalse(rw.sums)
        self.assertAlmostEqual(rw.gap, -300.0)
        self.assertIn("do NOT sum", rw.sum_note())
        text = window.format_table(window.average_window(self.record(((0, "p", 300), (300, "q", 400)))))
        self.assertIn("do NOT sum", text)
        self.assertIn("they sum to it", text)                    # the front role's do

    def test_the_table_prints_phases_with_start_duration_and_spreads_in_cycles(self):
        text = window.format_table(window.average_window(self.record()))
        for needle in ("role back", "role front", "OWN", "start", "duration", "lanes min..max", "(cycles)",
                       "its window starts +333.0 cycles from the scope's", "period 1,000.0 cycles"):
            self.assertIn(needle, text)
        self.assertNotIn("lanes min..max 0.0..", text)           # no role is smeared across the scope's window

    def test_the_overlays_role_column_reads_each_lane_in_its_own_window(self):
        from warpline import overlay
        view = overlay.view_of(self.record())
        roles, ph = overlay.role_phases(view, 8)
        self.assertEqual(sorted(roles), ["back", "front"])
        i = roles.index("front"), roles.index("back")
        # scope bin 0 (cycles 0..125): front is in p; back's own position is 62 - 333 = 729: in q
        self.assertEqual((ph[0][i[0]], ph[0][i[1]]), (("p", 1.0), ("q", 1.0)))
        # scope bin 3 (cycles 375..500): front in q; back at 42..167 of its own window: in p
        self.assertEqual((ph[3][i[0]], ph[3][i[1]]), (("q", 1.0), ("p", 1.0)))


if __name__ == "__main__":
    unittest.main()
