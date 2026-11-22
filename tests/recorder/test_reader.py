import json
import unittest

from warpline.recorder import perfetto, reader
from tests.recorder.helpers import make, slack_example, three_lane


def waits_of(rec, lane):
    return [w for w in reader.waits(rec) if w.lane == lane]


class Nesting(unittest.TestCase):
    def test_regions_nest_with_depth_and_parent(self):
        rec = make({"k/w": "r"}, [("k/w", "outer.begin", 0), ("k/w", "inner.begin", 2), ("k/w", "inner.end", 5),
                                  ("k/w", "outer.end", 9)])
        rs, findings = reader.regions(rec)
        self.assertEqual(findings, [])
        by = {r.name: r for r in rs["k/w"]}
        self.assertEqual((by["outer"].depth, by["outer"].parent, by["outer"].begin, by["outer"].end), (0, None, 0, 9))
        self.assertEqual((by["inner"].depth, by["inner"].parent, by["inner"].begin, by["inner"].end), (1, "outer", 2, 5))
        self.assertEqual(by["inner"].basis, "measured")

    def test_a_wait_sits_in_its_innermost_region(self):
        rec = make({"k/w": "r"}, [("k/w", "r.begin", 0), ("k/w", "barrier.B.wait.begin", 2),
                                  ("k/w", "barrier.B.wait.end", 6), ("k/w", "r.end", 7)],
                   {"barriers": {"B": {"count": 0}}})
        (w,) = reader.waits(rec)
        self.assertEqual((w.region, w.begin, w.end), ("r", 2, 6))
        self.assertEqual(w.release.kind, "own")  # count 0: its lane's own work ends it
        self.assertEqual(reader.state_at(rec, "k/w", 3), "wait:B")
        self.assertEqual(reader.state_at(rec, "k/w", 6), "r")

    def test_crossing_and_unclosed_are_named_findings(self):
        rec = make({"k/w": "r"}, [("k/w", "a.begin", 0), ("k/w", "b.begin", 1), ("k/w", "a.end", 2),
                                  ("k/w", "c.begin", 3), ("k/w", "launch.end", 4)])
        _rs, findings = reader.regions(rec)
        self.assertEqual([f.code for f in findings], ["region_crossing", "region_unclosed"])


class Join(unittest.TestCase):
    def test_several_arrivals_the_last_of_count_releases(self):
        facts = {"barriers": {"B": {"count": 3}}}
        stamps = [("k/a", "barrier.B.arrive", 10), ("k/b", "barrier.B.arrive", 30), ("k/c", "barrier.B.arrive", 20),
                  ("k/a", "barrier.B.wait.begin", 10), ("k/a", "barrier.B.wait.end", 35),
                  # the next phase's first arrival comes before the waiter's end: it must not be taken
                  ("k/c", "barrier.B.arrive", 33)]
        rec = make({"k/a": "w", "k/b": "w", "k/c": "w"}, sorted(stamps, key=lambda s: s[2]), facts)
        (w,) = reader.waits(rec)
        self.assertEqual((w.release.kind, w.release.lane, w.release.time, w.release.basis),
                         ("arrival", "k/b", 30, "measured"))

    def test_an_unstamped_arrival_names_the_nearest_issue(self):
        facts = {"barriers": {"L": {"count": 1, "stamped": 0, "issue": "copy"}}}
        stamps = [("k/a", "op.copy.issue", 5, 0), ("k/b", "op.copy.issue", 8, 1), ("k/a", "barrier.L.wait.begin", 9),
                  ("k/a", "barrier.L.wait.end", 40), ("k/a", "op.copy.issue", 50, 2)]
        rec = make({"k/a": "w", "k/b": "w"}, stamps, facts)
        (w,) = reader.waits(rec)
        r = w.release
        self.assertEqual((r.kind, r.basis), ("unstamped", "upper_bound"))
        self.assertIn("not stamped", r.note)
        self.assertEqual((r.issue.resource, r.issue.lane, r.issue.time, r.issue.op_id), ("copy", "k/b", 8, 1))

    def test_an_unstamped_arrival_with_no_issue_declared_says_so(self):
        facts = {"barriers": {"L": {"count": 2, "stamped": 1}}}
        rec = make({"k/a": "w"}, [("k/a", "barrier.L.wait.begin", 1), ("k/a", "barrier.L.wait.end", 9)], facts)
        r = reader.waits(rec)[0].release
        self.assertEqual((r.kind, r.issue), ("unstamped", None))
        self.assertIn("no issue declared", r.note)

    def test_a_poll_is_matched_to_the_last_write_at_or_before_its_end(self):
        stamps = [("k/p", "write.F", 4), ("k/p", "write.F", 12), ("k/q", "poll.F.begin", 5),
                  ("k/q", "poll.F.end", 14), ("k/p", "write.F", 20)]
        rec = make({"k/p": "producer", "k/q": "poller"}, sorted(stamps, key=lambda s: s[2]))
        (w,) = reader.waits(rec)
        self.assertEqual((w.kind, w.release.kind, w.release.time, w.release.lane), ("poll", "write", 12, "k/p"))

    def test_scopes_are_not_joined(self):
        facts = {"barriers": {"B": {"count": 1}}}
        rec = make({"s0/a": "w", "s1/a": "w"}, [("s1/a", "barrier.B.arrive", 1), ("s0/a", "barrier.B.wait.begin", 2),
                                                ("s0/a", "barrier.B.wait.end", 3)], facts)
        self.assertEqual(reader.waits(rec)[0].release.kind, "none")


class Operations(unittest.TestCase):
    def test_overlapping_operations_on_a_resource(self):
        stamps = [("k/a", "op.tensor.issue", 0, 1), ("k/a", "op.tensor.issue", 10, 2), ("k/b", "op.tensor.done", 30, 1),
                  ("k/b", "op.tensor.done", 40, 2), ("k/a", "op.tensor.issue", 50, 3), ("k/b", "op.tensor.done", 60, 3)]
        rec = make({"k/a": "issuer", "k/b": "learner"}, sorted(stamps, key=lambda s: s[2]))
        ops, findings = reader.operations(rec)
        self.assertEqual(findings, [])
        t = ops["tensor"]
        self.assertEqual([(o.op_id, o.issue_time, o.done_time, o.done_lane, o.basis) for o in t],
                         [(1, 0, 30, "k/b", "upper_bound"), (2, 10, 40, "k/b", "upper_bound"),
                          (3, 50, 60, "k/b", "upper_bound")])
        pairs = reader.overlaps(t)
        self.assertEqual([(a.op_id, b.op_id) for a, b in pairs], [(1, 2)])  # 3 overlaps neither

    def test_an_operation_carries_its_issuing_lane_and_region(self):
        rec = make({"k/a": "issuer"}, [("k/a", "fold.begin", 0), ("k/a", "op.cp.issue", 3, 9), ("k/a", "fold.end", 4),
                                       ("k/a", "op.cp.done", 8, 9)])
        (o,) = reader.operations(rec)[0]["cp"]
        self.assertEqual((o.issue_lane, o.issue_region), ("k/a", "fold"))

    def test_declared_pairing_with_a_wait_end(self):
        facts = {"operations": {"cp": {"done": "barrier.L.wait.end"}}, "barriers": {"L": {"count": 0}}}
        rec = make({"k/a": "issuer"}, [("k/a", "op.cp.issue", 1, 0), ("k/a", "barrier.L.wait.begin", 2),
                                       ("k/a", "barrier.L.wait.end", 9)], facts)
        (o,) = reader.operations(rec)[0]["cp"]
        self.assertEqual((o.issue_time, o.done_time), (1, 9))


class Path(unittest.TestCase):
    """The hand-built example (tests/recorder/helpers.three_lane): the answer, stated.

    The span is 0..100 from lane k/a. a waits on X from 10 to 90; X has count 2 (c arrives at 50, b at 80); the last,
    b's at 80, releases it. b was itself held: it waited on Y from 20 to 70, released by c's arrival at 50.
    Walking back from 100: a's own work 90..100 (10), the release's latency 80..90 on a (10), then on b: its own
    work 70..80 (10) and Y's release latency 50..70 (20), then on c: its own work 0..50 (50). Total 100.
    c's wait on X (50..90) is not on the path. Its slack is NOT its length (40): c has no later stamp, so the
    record cannot say what c's next dependency on the path would be, and the slack is unknown (see Slack).
    """

    def test_release_of_each_wait(self):
        rec = three_lane()
        (wa,) = waits_of(rec, "k/a")
        self.assertEqual((wa.release.lane, wa.release.time), ("k/b", 80))
        (wb,) = waits_of(rec, "k/b")
        self.assertEqual((wb.release.lane, wb.release.time), ("k/c", 50))
        (wc,) = waits_of(rec, "k/c")
        self.assertEqual((wc.release.lane, wc.release.time), ("k/b", 80))

    def test_the_critical_path(self):
        rec = three_lane()
        p = reader.critical_path(rec, "k", 0, 100, "k/a", blocking=5)
        got = [(x.lane, x.begin, x.end, x.kind, x.state) for x in p.pieces]
        self.assertEqual(got, [("k/c", 0, 50, "work", "load"),
                               ("k/b", 50, 70, "release", "release at Y"),
                               ("k/b", 70, 80, "work", "work"),
                               ("k/a", 80, 90, "release", "release at X"),
                               ("k/a", 90, 100, "work", "w")])
        self.assertEqual(sum(x.end - x.begin for x in p.pieces), 100)
        self.assertTrue(all(x.basis == "measured" for x in p.pieces))
        self.assertEqual([(w.lane, w.target) for w, _ in p.binding], [("k/a", "X"), ("k/b", "Y")])

    def test_slack_that_the_record_cannot_give_is_unknown_with_its_reason_never_the_waits_length(self):
        p = reader.critical_path(three_lane(), "k", 0, 100, "k/a", blocking=5)
        (row,) = p.slack
        self.assertEqual((row.wait.lane, row.wait.target, row.slack), ("k/c", "X", None))
        self.assertIn("no later stamp of k/c", row.reason)

    def test_a_short_wait_is_the_lanes_own_work(self):
        p = reader.critical_path(three_lane(), "k", 0, 100, "k/a", blocking=500)  # nothing blocks
        self.assertEqual([(x.lane, x.begin, x.end) for x in p.pieces if x.kind == "work"],
                         [("k/a", 0, 10), ("k/a", 10, 90), ("k/a", 90, 100)])
        self.assertEqual(len(p.slack), 3)

    def test_an_unstamped_release_puts_the_flight_on_the_path(self):
        facts = {"barriers": {"L": {"count": 1, "stamped": 0, "issue": "cp"}}}
        rec = make({"k/a": "consumer"}, [("k/a", "w.begin", 0), ("k/a", "op.cp.issue", 10, 0),
                                         ("k/a", "barrier.L.wait.begin", 12), ("k/a", "barrier.L.wait.end", 90),
                                         ("k/a", "w.end", 100)], facts)
        p = reader.critical_path(rec, "k", 0, 100, "k/a", blocking=5)
        flights = [x for x in p.pieces if x.kind == "flight"]
        self.assertEqual([(x.begin, x.end, x.basis) for x in flights], [(10, 90, "upper_bound")])
        self.assertEqual(sum(x.end - x.begin for x in p.pieces), 100)

    def test_steady_span(self):
        stamps = [("k/a", "tick.begin", t) for t in (0, 10, 20, 30, 40, 50)] + [("k/a", "tick.end", 55)]
        rec = make({"k/a": "r"}, sorted(stamps, key=lambda s: s[2]))
        self.assertEqual(reader.steady(rec, "tick", 1, "k"), ("k/a", 10, 40, 3))


class Slack(unittest.TestCase):
    """tests/recorder/helpers.slack_example: the path from m's end is r's work 0..85, Z's release latency 85..90, m's work to
    100. q waits on S 10..40 (30 cycles), off the path. q's next dependency on the path is its arrival at Z at 60,
    a phase the path needs at 85 (r's arrival is Z's last): q's wait could have been released 85 - 60 = 25 cycles
    later without q holding the path back. Its length, 30, is not its slack."""

    def test_the_path(self):
        p = reader.critical_path(slack_example(), "k", 0, 100, "k/m", blocking=5)
        self.assertEqual([(x.lane, x.begin, x.end, x.kind) for x in p.pieces],
                         [("k/r", 0, 30, "work"), ("k/r", 30, 85, "work"),  # r's work, split at its stamp at 30
                          ("k/m", 85, 90, "release"), ("k/m", 90, 100, "work")])

    def test_slack_is_the_paths_time_at_the_next_dependency_minus_the_time_the_lane_reached_it(self):
        p = reader.critical_path(slack_example(), "k", 0, 100, "k/m", blocking=5)
        (row,) = p.slack
        self.assertEqual((row.wait.lane, row.wait.target, row.wait.duration), ("k/q", "S", 30))
        self.assertEqual(row.slack, 25)
        self.assertIn("arrival at Z (t=60)", row.reason)

    def test_a_lane_on_the_path_has_slack_zero(self):
        # the same kernel with blocking = 500: nothing blocks, the path is m's own span, and m's wait is on it
        p = reader.critical_path(slack_example(), "k", 0, 100, "k/m", blocking=500)
        m = [r for r in p.slack if r.wait.lane == "k/m"]
        self.assertEqual([(r.wait.target, r.slack) for r in m], [("Z", 0)])


class Stepper(unittest.TestCase):
    def test_what_every_lane_is_doing(self):
        r = reader.at(three_lane(), 60, "k")
        lanes = {x["lane"]: x for x in r["lanes"]}
        self.assertEqual(lanes["k/a"]["wait"].target, "X")
        self.assertEqual(lanes["k/b"]["wait"].target, "Y")
        self.assertEqual(lanes["k/c"]["wait"].target, "X")
        self.assertEqual(lanes["k/b"]["regions"], ["work"])
        self.assertEqual(lanes["k/c"]["regions"], [])  # c's `load` ended at 50
        self.assertEqual(lanes["k/c"]["last_stamp"], ("barrier.X.wait.begin", 50))


class Output(unittest.TestCase):
    def test_perfetto_document(self):
        doc = perfetto.build(three_lane())
        json.loads(json.dumps(doc))
        self.assertEqual(perfetto.counts(doc), {"slices": 6, "async_slices": 0, "flows": 3})
        names = {e["name"] for e in doc["traceEvents"] if e["ph"] == "X"}
        self.assertEqual(names, {"w", "work", "load", "wait X", "wait Y"})

    def test_perfetto_operations_are_async_slices_on_a_resource_track(self):
        rec = make({"k/a": "r"}, [("k/a", "op.cp.issue", 1, 0), ("k/a", "op.cp.done", 5, 0)])
        doc = perfetto.build(rec)
        self.assertEqual(perfetto.counts(doc)["async_slices"], 1)
        b = [e for e in doc["traceEvents"] if e["ph"] == "b"][0]
        self.assertEqual((b["name"], b["args"]["issued_by"], b["args"]["basis"]), ("cp", "k/a", "upper_bound"))


if __name__ == "__main__":
    unittest.main()
