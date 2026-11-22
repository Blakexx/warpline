"""A role whose waits a record does not stamp is not "100% work": one detail a run, and several runs read together.
The records are synthetic; no test reads a real record.

The kernel (scope `k`, window 1,000 cycles, 12 windows and a closing begin): four roles, one lane each.
  a  waits 100 on barrier X     b  waits 300 on Y     c  waits 50 on Z     d  waits 200 on W
every lane arrives at its own barrier except `d`'s lane, which arrives nowhere (a role the record can say nothing
about unless it stamps d's wait). A DETAIL run stamps the waits (and only the waits) of the roles it is given; every run
stamps the arrivals of a, b and c. So the work of a is 900, b 700, c 950, d 800: the longest chain is c (95%).
"""
import contextlib
import io
import json
import os
import tempfile
import unittest

from warpline.recorder import query, record, summary
from warpline.recorder.cli import main
from tests.recorder.helpers import make


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class TempRecords(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, rec):
        path = os.path.join(self.dir.name, name)
        record.dump(rec, path)
        return path

P = 1000
N = 12
WAIT = {"a": ("X", 100), "b": ("Y", 300), "c": ("Z", 50), "d": ("W", 200)}


def detail(stamp=(), period=P, first="a"):
    lanes = {f"k/{r}": (first if r == "a" else r) for r in WAIT}
    stamps = []
    for w in range(N + 1):
        t0 = period * w
        for r, (bar, wt) in WAIT.items():
            ln = f"k/{r}"
            stamps += [(ln, "tick.begin", t0), (ln, "tick.end", t0 + 10)]
            if w == N:
                continue
            stamps += [(ln, "body.begin", t0 + 10), (ln, "body.end", t0 + period)]
            if r != "d":
                stamps.append((ln, f"barrier.{bar}.arrive", t0 + period - wt - 5))
            if r in stamp:
                stamps += [(ln, f"barrier.{bar}.wait.begin", t0 + period - wt), (ln, f"barrier.{bar}.wait.end", t0 + period)]
    stamps.sort(key=lambda s: s[2])
    facts = {"barriers": {WAIT[r][0]: {"count": 1} for r in stamp}}
    return make(lanes, stamps, facts)


def roles(d):
    return {r["role"]: r for r in d["roles"]}


class OneDetail(unittest.TestCase):
    def setUp(self):
        self.d = summary.digest(detail(("a",)))
        self.r = roles(self.d)

    def test_what_a_record_can_say_of_each_role(self):
        self.assertEqual(self.r["a"]["waits_state"], summary.STAMPED)
        self.assertEqual(self.r["b"]["waits_state"], summary.KNOWN_UNSTAMPED)
        self.assertEqual(self.r["b"]["arrives_at"], ["Y"])
        self.assertEqual(self.r["d"]["waits_state"], summary.UNKNOWN)

    def test_a_role_not_stamped_has_no_waits_and_no_work(self):
        for k in "bcd":
            self.assertIsNone(self.r[k]["waits"])
            self.assertIsNone(self.r[k]["work"])
            self.assertIsNone(self.r[k]["work_share"])
        self.assertAlmostEqual(self.r["a"]["work"], 900)
        self.assertAlmostEqual(self.r["a"]["waits"]["cycles"], 100)

    def test_the_chain_is_not_named_and_says_whose_waits_are_missing(self):
        ch = self.d["longest_chain"]
        self.assertFalse(ch["named"])
        self.assertIsNone(ch["role"])
        self.assertEqual(ch["unstamped_roles"], ["b", "c", "d"])
        text = summary.format_summary(self.d, "x")
        self.assertIn("LONGEST CHAIN: not named", text)
        self.assertIn("the waits of b, c, d are not stamped", text.replace("\n    ", " "))
        self.assertNotIn("work 100%", text)

    def test_the_text_does_not_call_an_unstamped_role_work(self):
        lines = {x.split()[1]: x for x in summary.format_summary(self.d, "x").splitlines() if x.startswith("role ")}
        self.assertIn("work 900 (90% of the window)", lines["a"])
        for k in "bcd":
            self.assertIn("waits: not stamped in this record", lines[k])
            self.assertNotIn("work", lines[k])
        self.assertIn("arrives at Y", lines["b"])
        self.assertIn("no barrier stamped on it", lines["d"])

    def test_a_record_that_stamps_no_wait_names_no_chain(self):
        d = summary.digest(detail(()))
        self.assertFalse(d["waits_stamped"])
        self.assertFalse(d["longest_chain"]["named"])
        self.assertIn("a coarse record", d["longest_chain"]["note"])


class SeveralDetails(unittest.TestCase):
    def setUp(self):
        self.srcs = [("a.json", detail(("a",))), ("b.json", detail(("b",))), ("cd.json", detail(("c", "d")))]

    def test_three_details_name_the_chain(self):
        d = summary.digest_many(self.srcs)
        r = roles(d)
        self.assertTrue(all(x["waits_state"] == summary.STAMPED for x in d["roles"]))
        self.assertAlmostEqual(r["c"]["work_share"], 0.95)
        self.assertEqual(d["longest_chain"]["role"], "c")
        self.assertEqual(d["longest_chain"]["next_role"], "a")
        self.assertEqual(r["b"]["waits_runs"], ["b.json"])
        self.assertEqual(r["d"]["waits_runs"], ["cd.json"])
        self.assertTrue(all(x["agrees"] for x in d["runs"]))

    def test_the_text_says_the_waits_come_from_different_runs(self):
        text = summary.format_summary(summary.digest_many(self.srcs), "a.json")
        flat = text.replace("\n  ", " ")
        self.assertIn("LONGEST CHAIN: role c: work 95%", text)
        self.assertIn("different runs, each a few percent from the plain kernel", flat)
        self.assertIn("waits a window from b.json", text)
        self.assertIn("stamps the waits of: c, d", text)
        self.assertLessEqual(len(text.splitlines()), summary.LINES)

    def test_a_missing_run_leaves_the_chain_unnamed(self):
        d = summary.digest_many(self.srcs[:2])
        self.assertFalse(d["longest_chain"]["named"])
        self.assertEqual(d["longest_chain"]["unstamped_roles"], ["c", "d"])
        self.assertEqual(roles(d)["c"]["waits_state"], summary.KNOWN_UNSTAMPED)

    def test_a_barrier_two_runs_stamp_is_counted_once(self):
        d = summary.digest_many([("a.json", detail(("a",))), ("ab.json", detail(("a", "b"))), ("cd.json", detail(("c", "d")))])
        self.assertAlmostEqual(roles(d)["a"]["waits"]["cycles"], 100)
        self.assertEqual(roles(d)["a"]["waits"]["by_barrier"][0]["run"], "a.json")

    def test_a_run_whose_window_disagrees_is_named_and_its_waits_not_taken(self):
        d = summary.digest_many([("a.json", detail(("a",))), ("b.json", detail(("b",), period=1100)),
                                 ("cd.json", detail(("c", "d")))])
        bad = {x["name"]: x for x in d["runs"]}
        self.assertTrue(bad["a.json"]["base"])
        self.assertFalse(bad["b.json"]["agrees"])
        self.assertTrue(bad["cd.json"]["agrees"])
        self.assertTrue(any(s.startswith("period +100") for s in bad["b.json"]["disagree"]))
        self.assertEqual(roles(d)["b"]["waits_state"], summary.KNOWN_UNSTAMPED)
        self.assertFalse(d["longest_chain"]["named"])
        text = summary.format_summary(d, "a.json").replace("\n  ", " ")
        self.assertIn("b.json: DISAGREES, its waits are not taken", text)
        self.assertIn("the waits of b.json were not taken", text)

    def test_a_looser_tolerance_takes_the_run(self):
        d = summary.digest_many([("a.json", detail(("a",))), ("b.json", detail(("b",), period=1100)),
                                 ("cd.json", detail(("c", "d")))], tol=0.15)
        self.assertTrue(all(x["agrees"] for x in d["runs"]))

    def test_records_of_two_kernels_are_refused(self):
        other = detail(("a",), first="z")
        with self.assertRaises(summary.SummaryError):
            summary.digest_many([("a.json", detail(("a",))), ("z.json", other)])


class DiffAndSql(TempRecords):
    def test_diff_marks_not_stamped_and_reports_no_change_from_it(self):
        a, b = summary.digest(detail(("a", "b"))), summary.digest(detail(("a", "c"), period=1000))
        d = summary.diff_digests(a, b)
        waits_rows = {r["role"] for r in d["rows"] if r["kind"] == "waits"}
        self.assertEqual(waits_rows, {"a"})
        nc = {x["role"]: x for x in d["waits_not_compared"]}
        self.assertEqual(set(nc), {"b", "c", "d"})
        self.assertEqual((nc["b"]["before"], nc["b"]["after"]), (summary.STAMPED, summary.KNOWN_UNSTAMPED))
        self.assertIsNone(nc["b"]["after_cycles"])
        text = summary.format_diff(d).replace("\n  ", " ")
        self.assertIn("b (before 300 cycles, after not stamped)", text)
        self.assertIn("d (before not stamped, after not stamped)", text)
        self.assertIn("a change from not stamped to a number is not a change", text)

    def test_sql_does_not_return_an_unstamped_role_as_zero(self):
        con = query.load(detail(("a",)))
        self.addCleanup(con.close)
        _c, rows, _ = query.run(con, "SELECT role, waits_state FROM roles ORDER BY role")
        self.assertEqual(rows, [("a", "stamped"), ("b", "known_unstamped"), ("c", "known_unstamped"), ("d", "unknown")])
        _c, rows, _ = query.run(con, query.NAMED["waits-by-role"][1])
        got = {r[0]: r[1:] for r in rows}
        self.assertEqual(got["a"], ("stamped", 12, 1200))
        for k in "bcd":
            self.assertEqual(got[k][1:], (None, None))
        _c, rows, _ = query.run(con, "SELECT DISTINCT role, waits_state FROM phases ORDER BY role")
        self.assertEqual(dict(rows)["b"], "known_unstamped")
        _c, rows, _ = query.run(con, "SELECT DISTINCT role FROM waits")
        self.assertEqual(rows, [("a",)])


class Cli(TempRecords):
    def test_summary_over_several_records(self):
        paths = [self.write("a.json", detail(("a",))), self.write("b.json", detail(("b",))),
                 self.write("cd.json", detail(("c", "d")))]
        rc, out, err = run_cli("summary", *paths)
        self.assertEqual(rc, 0, err)
        self.assertIn("LONGEST CHAIN: role c", out)
        self.assertLessEqual(len(out.splitlines()), summary.LINES)
        rc, out, err = run_cli("summary", *paths, "--json")
        j = json.loads(out)
        self.assertEqual(j["longest_chain"]["role"], "c")
        self.assertEqual([r["name"] for r in j["runs"]], ["a.json", "b.json", "cd.json"])

    def test_a_single_detail_through_the_cli(self):
        rc, out, err = run_cli("summary", self.write("a.json", detail(("a",))))
        self.assertEqual(rc, 0, err)
        self.assertIn("LONGEST CHAIN: not named", out)

    def test_the_cli_refuses_two_kernels(self):
        other = detail(("a",), first="z")
        rc, out, err = run_cli("summary", self.write("a.json", detail(("a",))), self.write("z.json", other))
        self.assertEqual(rc, 2)
        self.assertIn("not one kernel", err)


if __name__ == "__main__":
    unittest.main()
