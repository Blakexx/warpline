"""Tier 0: for each check, a record that passes and one that fails."""
import unittest

from warpline import check, reader
from tests.helpers import make, three_lane

L = {"k/w": "role"}


class Tier0(unittest.TestCase):
    def codes(self, fn, rec):
        return [f.code for f in fn(rec)]

    def test_all_hold_on_the_three_lane_example(self):
        self.assertEqual(check.all_findings(three_lane()), [])

    def test_stamps_ordered(self):
        ok = make(L, [("k/w", "a", 1), ("k/w", "b", 1), ("k/w", "c", 5)])
        bad = make(L, [("k/w", "a", 5), ("k/w", "b", 3)])
        self.assertEqual(self.codes(check.stamps_ordered, ok), [])
        self.assertEqual(self.codes(check.stamps_ordered, bad), ["stamp_out_of_order"])

    def test_regions_nest_and_tile(self):
        ok = make(L, [("k/w", "a.begin", 0), ("k/w", "b.begin", 1), ("k/w", "b.end", 2), ("k/w", "a.end", 3),
                      ("k/w", "c.begin", 3), ("k/w", "c.end", 9)])
        self.assertEqual(self.codes(check.regions_nest_and_tile, ok), [])
        crossing = make(L, [("k/w", "a.begin", 0), ("k/w", "b.begin", 1), ("k/w", "a.end", 2), ("k/w", "b.end", 3)])
        self.assertEqual(self.codes(check.regions_nest_and_tile, crossing),
                         ["region_crossing", "region_end_unmatched"])
        unclosed = make(L, [("k/w", "a.begin", 0), ("k/w", "launch.end", 1)])   # a lane that stamps its exit
        self.assertEqual(self.codes(check.regions_nest_and_tile, unclosed), ["region_unclosed"])
        gap = make(L, [("k/w", "a.begin", 0), ("k/w", "a.end", 2), ("k/w", "b.begin", 4), ("k/w", "b.end", 6)])
        self.assertEqual(self.codes(check.regions_nest_and_tile, gap), ["region_gap"])

    def test_wait_after_release(self):
        facts = {"barriers": {"B": {"count": 2}}}
        ok = make({"k/a": "x", "k/b": "x"}, [("k/a", "barrier.B.arrive", 1), ("k/b", "barrier.B.arrive", 2),
                                             ("k/a", "barrier.B.wait.begin", 1), ("k/a", "barrier.B.wait.end", 3)], facts)
        bad = make({"k/a": "x", "k/b": "x"}, [("k/a", "barrier.B.arrive", 1), ("k/a", "barrier.B.wait.begin", 1),
                                              ("k/a", "barrier.B.wait.end", 3), ("k/b", "barrier.B.arrive", 4)], facts)
        self.assertEqual(self.codes(check.wait_after_release, ok), [])
        self.assertEqual(self.codes(check.wait_after_release, bad), ["wait_before_release"])
        poll_bad = make(L, [("k/w", "poll.F.begin", 1), ("k/w", "poll.F.end", 3)])
        self.assertEqual(self.codes(check.wait_after_release, poll_bad), ["wait_before_release"])

    def test_a_rendezvous_is_released_by_its_last_arrival_by_count(self):
        lanes = {"k/a": "x", "k/b": "x"}
        rdv = lambda **kw: {"barriers": {"B": {"count": 2, "rendezvous": True, **kw}}}
        stamps = [("k/a", "barrier.B.arrive", 1), ("k/a", "barrier.B.wait.begin", 1), ("k/a", "barrier.B.wait.end", 6),
                  ("k/b", "barrier.B.arrive", 8), ("k/b", "barrier.B.wait.begin", 8), ("k/b", "barrier.B.wait.end", 10)]
        # a's wait ends at 6, b's begin (the last arrival, the release) is stamped at 8: 2 cycles of skew
        self.assertEqual(self.codes(check.wait_after_release, make(lanes, stamps, rdv())), ["wait_ends_before_release"])
        self.assertEqual(self.codes(check.wait_after_release, make(lanes, stamps, rdv(skew=2))), [])
        self.assertEqual(self.codes(check.wait_after_release, make(lanes, stamps, rdv(skew=1))),
                         ["wait_ends_before_release"])
        # a barrier that is not declared one keeps its check: by the arrivals at or before the end, a's wait has one of 2
        plain = {"barriers": {"B": {"count": 2}}}
        self.assertEqual(self.codes(check.wait_after_release, make(lanes, stamps, plain)),
                         ["wait_before_release"])
        # and the skew of a barrier is not a licence for another's wait
        other = {"barriers": {"B": {"count": 2}, "C": {"count": 2, "rendezvous": True, "skew": 99}}}
        self.assertEqual(self.codes(check.wait_after_release, make(lanes, stamps, other)), ["wait_before_release"])

    def test_a_rendezvous_phase_short_of_its_count_has_no_release(self):
        lanes = {"k/a": "x", "k/b": "x"}
        stamps = [("k/a", "barrier.B.arrive", 1), ("k/a", "barrier.B.wait.begin", 1), ("k/a", "barrier.B.wait.end", 6),
                  ("k/b", "barrier.B.arrive", 2), ("k/b", "barrier.B.wait.begin", 2), ("k/b", "barrier.B.wait.end", 6),
                  ("k/a", "barrier.B.arrive", 20), ("k/a", "barrier.B.wait.begin", 20),
                  ("k/a", "barrier.B.wait.end", 30)]    # a second phase that only one lane reached
        rec = make(lanes, stamps, {"barriers": {"B": {"count": 2, "rendezvous": True}}})
        ws = reader.waits(rec)
        self.assertEqual([(w.lane, w.release.kind, w.release.time) for w in ws],
                         [("k/a", "arrival", 2), ("k/a", "none", None), ("k/b", "arrival", 2)])
        self.assertEqual(self.codes(check.wait_after_release, rec), ["wait_before_release"])

    def test_a_region_open_at_the_end_of_a_lane_with_no_exit_is_counted_not_a_finding(self):
        open_ = make(L, [("k/w", "a.begin", 0), ("k/w", "b.begin", 2), ("k/w", "mark", 7)])
        self.assertEqual(check.all_findings(open_), [])
        self.assertEqual(reader.open_at_end(open_), {"k/w": ["b", "a"]})
        regions, _ = reader.regions(open_)
        self.assertEqual([(r.name, r.begin, r.end, r.basis) for r in regions["k/w"]],
                         [("a", 0, 7, "open_at_end"), ("b", 2, 7, "open_at_end")])
        closed = make(L, [("k/w", "a.begin", 0), ("k/w", "a.end", 7)])
        self.assertEqual(reader.open_at_end(closed), {})

    def test_a_region_open_in_a_lane_that_stamped_its_exit_is_still_a_finding(self):
        exits = make(L, [("k/w", "launch.begin", 0), ("k/w", "a.begin", 1), ("k/w", "launch.end", 7)],
                     ties=[("k/w", 0, 0), ("k/w", 7, 7)])
        self.assertEqual(self.codes(check.regions_nest_and_tile, exits), ["region_unclosed"])
        self.assertEqual(reader.open_at_end(exits), {})
        begin_only = make(L, [("k/w", "launch.begin", 0), ("k/w", "a.begin", 1)], ties=[("k/w", 0, 0)])
        self.assertEqual(self.codes(check.launch_edges, begin_only), ["launch_edge_missing"])   # still named

    def test_operations_ordered(self):
        ok = make(L, [("k/w", "op.R.issue", 1, 7), ("k/w", "op.R.done", 5, 7)])
        bad = make(L, [("k/w", "op.R.issue", 9, 7), ("k/w", "op.R.done", 5, 7)])
        unfinished = make(L, [("k/w", "op.R.issue", 1, 7)])
        self.assertEqual(self.codes(check.operations_ordered, ok), [])
        self.assertEqual(self.codes(check.operations_ordered, bad), ["operation_done_before_issue"])
        self.assertEqual(self.codes(check.operations_ordered, unfinished), ["operation_unfinished"])

    def test_barrier_arrivals_equal_count(self):
        facts = {"barriers": {"B": {"count": 2}}}
        ok = make({"k/a": "x", "k/b": "x"}, [("k/a", "barrier.B.arrive", 1), ("k/b", "barrier.B.arrive", 2)], facts)
        bad = make({"k/a": "x", "k/b": "x"}, [("k/a", "barrier.B.arrive", 1), ("k/b", "barrier.B.arrive", 2),
                                              ("k/a", "barrier.B.arrive", 3)], facts)
        undeclared = make(L, [("k/w", "barrier.Q.arrive", 1)])
        self.assertEqual(self.codes(check.barrier_arrivals, ok), [])
        self.assertEqual(self.codes(check.barrier_arrivals, bad), ["barrier_arrivals_not_whole_phases"])
        self.assertEqual(self.codes(check.barrier_arrivals, undeclared), ["barrier_undeclared"])

    def test_stamps_accounted(self):
        ok = make(L, [("k/w", "a", 1), ("k/w", "b", 2)], dropped={"k/w": 3}, issued={"k/w": 5})
        bad = make(L, [("k/w", "a", 1), ("k/w", "b", 2)], dropped={"k/w": 1}, issued={"k/w": 5})
        self.assertEqual(self.codes(check.stamps_accounted, ok), [])
        self.assertEqual(self.codes(check.stamps_accounted, bad), ["stamps_unaccounted"])


if __name__ == "__main__":
    unittest.main()
