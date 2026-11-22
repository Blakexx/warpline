import json
import unittest

from warpline.recorder import merge, reader, record
from tests.recorder.helpers import make, regular


def run(twin=None, **kw):
    rec = regular(**kw)
    rec.twin = twin
    return rec


class Merge(unittest.TestCase):
    def three(self):
        a = run(None, extra=[("k/b", "detail.a", 500), ("k/b", "shared", 600)])                       # consistent, no twin
        b = run({"plain": 1000, "recorded": 1020}, stretch=1.2, extra=[("k/b", "detail.b", 100)])    # anchors stretched
        c = run({"plain": 1000, "recorded": 1100}, extra=[("k/b", "detail.c", 550), ("k/b", "shared", 700)])  # over twin
        return merge.merge([("A", a), ("B", b), ("C", c)])

    def test_the_stretched_run_is_flagged_by_its_anchors_and_left_out(self):
        m = self.three()
        self.assertEqual(m.reference, 0)
        by = {r.index: r for r in m.runs}
        self.assertEqual((by[0].accepted, by[1].accepted, by[2].accepted), (True, False, True))
        self.assertTrue(any("period of k: 1,200.0 against 1,000.0" in f for f in by[1].flags))
        self.assertTrue(any("anchor `phase.x.begin`#0 role front scope k: offset 480.0 against 400.0" in f
                            for f in by[1].flags))
        self.assertEqual(m.untrusted[1], ["detail.b"])
        names = {m.record.names[n] for _l, n, _t, _p in m.record.stamps}
        self.assertNotIn("detail.b", names)
        self.assertEqual(sorted(m.record.facts["runs"][1]["flags"])[0:1], sorted(by[1].flags)[0:1])

    def test_the_twin_is_checked_never_assumed(self):
        m = self.three()
        by = {r.index: r for r in m.runs}
        self.assertEqual(by[0].twin_status, "unchecked")          # no twin: unchecked, not passing
        self.assertEqual(by[1].twin_status, "ok")                 # 1.02 <= 1 + 2 x 0.03
        self.assertEqual(by[2].twin_status, "over")               # 1.10 > 1.06
        self.assertAlmostEqual(by[2].twin_ratio, 1.1)
        self.assertTrue(by[2].accepted)                           # a twin flag alone does not leave a run out
        report = merge.format_report(m)
        self.assertIn("run 0 twin: UNCHECKED", report)
        self.assertIn("run 2 twin: FLAGGED, over the bound", report)

    def test_the_merged_timeline(self):
        m = self.three()
        rec = m.record
        self.assertEqual(m.anchors, ["phase.x.begin", "phase.x.end", "phase.y.begin", "phase.y.end", "window.begin"])
        spread = rec.facts["window"]["stamps"]
        got = {(e["lane"], e["name"]): e for e in spread}
        self.assertEqual(got[("k/a", "phase.x.begin")]["runs"], [0, 2])          # an anchor: every accepted run
        self.assertEqual(got[("k/a", "phase.x.begin")]["mean"], 400)
        self.assertEqual(got[("k/b", "detail.a")]["runs"], [0])                  # detail: the run that carried it
        self.assertEqual(got[("k/b", "detail.c")]["runs"], [2])
        self.assertEqual(got[("k/b", "detail.c")]["mean"], 550)
        self.assertEqual([r["accepted"] for r in rec.facts["runs"]], [True, False, True])
        self.assertEqual(rec.facts["window"]["scopes"]["k"]["period"], 1000)
        # a name two accepted runs carry at offsets that disagree is a finding (here 600 against 700: 10% > 3%)
        self.assertEqual(got[("k/b", "shared")]["mean"], 600)
        self.assertTrue(any("`shared`#0 on k/b" in f for f in m.findings))
        again = record.loads(json.dumps(rec.to_dict()))
        self.assertEqual(len(again.facts["window"]["stamps"]), len(again.stamps))

    def test_tolerance_is_a_parameter(self):
        a, b = run(None), run(None, stretch=1.02)
        self.assertTrue(merge.merge([("A", a), ("B", b)], 0.03).runs[1].accepted)
        self.assertFalse(merge.merge([("A", a), ("B", b)], 0.01).runs[1].accepted)

    def test_no_anchors_is_an_error(self):
        a = make({"k/a": "r"}, [("k/a", "w.begin", t) for t in range(0, 100, 10)])
        b = make({"k/a": "r"}, [("k/a", "v.begin", t) for t in range(0, 100, 10)])
        with self.assertRaises(merge.MergeError):
            merge.merge([("A", a), ("B", b)])


class ReaderOnAMerge(unittest.TestCase):
    def test_the_reader_runs_on_a_merged_record(self):
        rec = merge_with_barrier()
        regions, _f = reader.regions(rec)
        self.assertTrue(regions["k/a"])
        self.assertIsNotNone(reader.critical_path(rec, "k", 0, 1000, "k/a"))

    def test_a_wait_and_an_arrival_from_different_runs_are_not_joined(self):
        rec = merge_with_barrier()
        (w,) = [x for x in reader.waits(rec) if x.target == "Q"]
        self.assertEqual(w.release.kind, "none")
        self.assertEqual(w.release.finding, "cross_run_join_refused")
        self.assertIn("not from the wait's run", w.release.note)


def merge_with_barrier():
    """Run A carries a wait on Q (lane b); run C carries Q's arrival (lane a): a merge holds both, from two runs."""
    a = run(None, extra=[("k/b", "barrier.Q.wait.begin", 200), ("k/b", "barrier.Q.wait.end", 280)])
    b = run({"plain": 1000, "recorded": 1020}, stretch=1.2)
    c = run(None, extra=[("k/a", "barrier.Q.arrive", 250)])
    m = merge.merge([("A", a), ("B", b), ("C", c)])
    m.record.facts["barriers"] = {"Q": {"count": 1}}
    m.record._cache.clear()
    return m.record


class Join(unittest.TestCase):
    """facts.runs present: a stamp's runs are facts.window.stamps[i].runs; a join stays inside a run."""

    def build(self, arrivals):
        stamps = [("k/a", "barrier.B.wait.begin", 10), ("k/a", "barrier.B.wait.end", 50)]
        stamps += [("k/b", "barrier.B.arrive", t) for t, _r in arrivals]
        order = sorted(range(len(stamps)), key=lambda i: (stamps[i][0] != "k/a", stamps[i][2]))
        stamps = [stamps[i] for i in order]
        tags = [[0], [0]] + [r for _t, r in sorted(arrivals)]
        facts = {"barriers": {"B": {"count": 1}}, "runs": [{"index": 0}, {"index": 1}],
                 "window": {"stamps": [{"runs": t} for t in tags]}}
        return make({"k/a": "w", "k/b": "w"}, stamps, facts)

    def test_an_arrival_of_another_run_is_refused(self):
        (w,) = reader.waits(self.build([(40, [1])]))
        self.assertEqual((w.release.kind, w.release.finding), ("none", "cross_run_join_refused"))

    def test_the_wait_takes_its_own_runs_arrival_and_says_the_other_was_refused(self):
        (w,) = reader.waits(self.build([(30, [0]), (45, [1])]))
        self.assertEqual((w.release.kind, w.release.time, w.release.finding), ("arrival", 30, "cross_run_join_refused"))

    def test_an_arrival_every_run_carries_matches_with_no_finding(self):
        (w,) = reader.waits(self.build([(30, [0, 1])]))
        self.assertEqual((w.release.kind, w.release.time, w.release.finding), ("arrival", 30, ""))


if __name__ == "__main__":
    unittest.main()
