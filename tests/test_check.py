"""Tier 0: for each check, a record that passes and one that fails."""
import unittest

from warpline import check
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
        unclosed = make(L, [("k/w", "a.begin", 0)])
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
