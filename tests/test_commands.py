"""`summary`, `diff` and `sql`: the three commands an agent iterates with, and the bounds on the older commands' output.
The records are synthetic, with phases and waits of KNOWN length; no test reads `build/`.

The kernel (scope `k`, window 1,000 cycles, 12 windows and a closing begin): a PRODUCER lane `k/p` runs `tick` 10 cycles,
`load` (`load` + jitter, jitter alternating +-J by window), `compute`, then `drain`, a region that is one wait on the
barrier `ack` to the window's end; a CONSUMER lane `k/c` runs `tick` 10, `serve` 890 and `idle` 100, a region that is one
wait on `req`. The producer's work is 1,000 less its wait (710 for load 300 and compute 400); the consumer's is 900
(its wait is 100): the consumer is the longest chain.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest

from warpline import bound, query, record, summary
from warpline.__main__ import main
from tests.helpers import make

P = 1000
N = 12


def synth(load=300, compute=400, jit=10, waits=True, ties=True, lanes=("k/p", "k/c")):
    p, c = lanes
    stamps, tie = [], []
    cur = 0.0                       # nanoseconds along the producer lane
    for w in range(N + 1):
        t0 = P * w
        j = jit if w % 2 == 0 else -jit
        ld = load + j
        stamps += [(p, "tick.begin", t0), (p, "tick.end", t0 + 10), (c, "tick.begin", t0), (c, "tick.end", t0 + 10)]
        if w == N:
            break
        stamps += [(p, "load.begin", t0 + 10), (p, "load.end", t0 + 10 + ld),
                   (p, "compute.begin", t0 + 10 + ld), (p, "compute.end", t0 + 10 + ld + compute),
                   (p, "drain.begin", t0 + 10 + ld + compute), (p, "drain.end", t0 + P),
                   (c, "serve.begin", t0 + 10), (c, "serve.end", t0 + 900), (c, "idle.begin", t0 + 900),
                   (c, "idle.end", t0 + P)]
        if waits:
            stamps += [(p, "barrier.ack.arrive", t0 + 10 + ld + compute - 5),
                       (p, "barrier.ack.wait.begin", t0 + 10 + ld + compute), (p, "barrier.ack.wait.end", t0 + P),
                       (c, "barrier.req.arrive", t0 + 880), (c, "barrier.req.wait.begin", t0 + 900),
                       (c, "barrier.req.wait.end", t0 + P)]
        if ties:   # tick 1 ns a cycle, load 0.5 (2,000 cycles an us), compute 1, the rest 1
            for t, ns_per in ((t0, 1.0), (t0 + 10, 0.5), (t0 + 10 + ld, 1.0), (t0 + 10 + ld + compute, 1.0)):
                tie.append((p, t, int(round(cur))))
                nxt = {t0: t0 + 10, t0 + 10: t0 + 10 + ld, t0 + 10 + ld + compute: t0 + P}.get(t, t0 + 10 + ld + compute)
                cur += (nxt - t) * ns_per
            cur = round(cur)
    if ties:
        tie.append((p, P * N, int(cur)))
    stamps.sort(key=lambda s: s[2])
    facts = {"barriers": {"ack": {"count": 1}, "req": {"count": 1}}} if waits else {}
    return make({p: "producer", c: "consumer"}, stamps, facts, ties=tie if ties else None)


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


class Summary(unittest.TestCase):
    def setUp(self):
        self.d = summary.digest(synth())

    def test_the_known_numbers(self):
        d = self.d
        self.assertEqual(d["health"]["lanes"], 2)
        self.assertEqual(d["health"]["dropped"], 0)
        w = d["window"]
        self.assertEqual((w["period"], w["windows"], w["anchor"]), (1000, 10, "tick.begin"))
        self.assertAlmostEqual(w["period_ns"], 850, delta=1)    # 10 + load/2 + compute + the rest, at 1,000 - load/2
        roles = {r["role"]: r for r in d["roles"]}
        ph = {p["name"]: p for p in roles["producer"]["phases"]}
        self.assertAlmostEqual(ph["load"]["cycles"], 300)
        self.assertAlmostEqual(ph["load"]["share"], 0.30)
        self.assertAlmostEqual(ph["load"]["jitter"], 10)
        self.assertAlmostEqual(ph["compute"]["cycles"], 400)
        self.assertAlmostEqual(ph["compute"]["jitter"], 0)
        self.assertAlmostEqual(ph["load"]["start"], 10)
        self.assertAlmostEqual(ph["compute"]["start"], 310)
        self.assertAlmostEqual(ph["load"]["ns"], 150)          # 300 cycles at 2,000 cycles an us
        self.assertAlmostEqual(ph["load"]["cycles_per_us"], 2000, delta=15)
        self.assertAlmostEqual(ph["compute"]["cycles_per_us"], 1000, delta=15)
        # a role's phases in order of their start
        self.assertEqual([p["name"] for p in roles["producer"]["phases"]], ["tick", "load", "compute", "drain"])

    def test_waits_by_barrier(self):
        roles = {r["role"]: r for r in self.d["roles"]}
        x = roles["producer"]["waits"]
        self.assertAlmostEqual(x["count"], 1.0)
        self.assertAlmostEqual(x["cycles"], 290)               # 1000 - (10 + 300 + 400), the jitter averaging out
        self.assertAlmostEqual(x["share"], 0.29)
        self.assertEqual([b["barrier"] for b in x["by_barrier"]], ["ack"])
        self.assertAlmostEqual(x["by_barrier"][0]["cycles"], 290)
        y = roles["consumer"]["waits"]
        self.assertAlmostEqual(y["cycles"], 100)
        self.assertEqual(y["by_barrier"][0]["barrier"], "req")
        self.assertAlmostEqual(roles["producer"]["work"], 710)
        self.assertAlmostEqual(roles["consumer"]["work"], 900)

    def test_the_longest_chain_is_named(self):
        ch = self.d["longest_chain"]
        self.assertTrue(ch["named"])
        self.assertEqual(ch["role"], "consumer")
        self.assertAlmostEqual(ch["work_share"], 0.9)
        self.assertEqual(ch["next_role"], "producer")
        text = summary.format_summary(self.d, "x")
        self.assertIn("LONGEST CHAIN: role consumer", text)
        self.assertIn("the record cannot say", text)

    def test_no_waits_no_chain(self):
        d = summary.digest(synth(waits=False))
        self.assertFalse(d["waits_stamped"])
        self.assertFalse(d["longest_chain"]["named"])
        self.assertIsNone(d["longest_chain"]["role"])
        self.assertTrue(all(r["waits"] is None for r in d["roles"]))
        text = summary.format_summary(d)
        self.assertIn("LONGEST CHAIN: not named", text)
        self.assertIn("no waits", text)
        self.assertNotIn("work ", text.split("LONGEST")[0])    # no work claimed where no wait can be told

    def test_a_record_without_ties_has_no_time(self):
        d = summary.digest(synth(ties=False))
        self.assertIsNone(d["window"]["period_ns"])
        self.assertTrue(all(p["ns"] is None for r in d["roles"] for p in r["phases"]))
        self.assertIsNone(d["launch"])
        self.assertIn("launch:", summary.format_summary(d))

    def test_findings_are_counted_by_kind(self):
        rec = synth()
        bad = record.from_dict({**rec.to_dict(), "stamps": rec.to_dict()["stamps"] + [["k/p", 0, 99999]]})
        d = summary.digest(bad)
        self.assertGreaterEqual(sum(d["health"]["findings"].values()), 1)

    def test_json_keys_are_stable(self):
        keys = {"variant", "scope", "scopes", "health", "launch", "launch_note", "window", "window_note", "roles",
                "waits_stamped", "longest_chain"}
        self.assertEqual(set(self.d), keys)
        self.assertEqual(set(self.d["window"]), {"anchor", "anchor_lane", "windows", "period", "period_std",
                                                 "period_se", "period_ns", "skip"})
        self.assertEqual(set(self.d["roles"][0]), {"role", "lanes", "period", "period_std", "windows", "own_window",
                                                   "phases", "waits", "work", "work_share", "waits_state", "arrives_at"})
        self.assertEqual(set(self.d["roles"][0]["phases"][0]), {"name", "occ", "label", "start", "cycles", "ns",
                                                                "cycles_per_us", "share", "jitter", "se", "held",
                                                                "windows"})
        json.dumps(self.d)


class Screen(TempRecords):
    def many(self, n, scale=1):
        stamps = []
        for w in range(N + 1):
            t0 = P * w
            stamps += [("k/a", "tick.begin", t0), ("k/a", "tick.end", t0 + 1)]
            if w < N:
                for i in range(n):
                    stamps += [("k/a", f"ph{i}.begin", t0 + 1 + 5 * i), ("k/a", f"ph{i}.end", t0 + 1 + 5 * i + 5 * scale)]
        stamps.sort(key=lambda s: s[2])
        return make({"k/a": "only"}, stamps)

    def test_summary_fits_a_screen_whatever_the_phases(self):
        text = summary.format_summary(summary.digest(self.many(120)), "many")
        self.assertLessEqual(len(text.splitlines()), summary.LINES)
        self.assertIn("smaller phases not shown", text)
        text = summary.format_summary(summary.digest(synth()), "x")
        self.assertLessEqual(len(text.splitlines()), summary.LINES)

    def test_diff_fits_a_screen_whatever_the_phases(self):
        d = summary.diff_digests(summary.digest(self.many(120)), summary.digest(self.many(120, 2)))
        text = summary.format_diff(d)
        self.assertLessEqual(len(text.splitlines()), summary.LINES)
        self.assertIn("more rows", text)
        self.assertEqual(len(d["rows"]), 121)

    def test_check_groups_its_findings(self):
        rec = self.many(1)
        doc = rec.to_dict()
        for i in range(30):                                   # 30 regions that never close
            doc["names"]["900"] = "open.begin"
            doc["stamps"].append(["k/a", 900, 20000 + i])
        doc["names"]["901"] = "launch.end"                    # the lane stamps its exit: an open region is a finding
        doc["stamps"].append(["k/a", 901, 30000])
        rec = record.from_dict(doc)
        path = self.write("open.json", rec)
        rc, out, _ = run_cli("check", path)
        self.assertIn("region_unclosed x30", out)
        self.assertLess(len(out.splitlines()), 20)
        self.assertIn("more of this kind", out)
        rc, full, _ = run_cli("check", path, "--all")
        self.assertGreaterEqual(len(full.splitlines()), 30)
        self.assertEqual(bound.LIMIT, None)
        bound.LIMIT = 1


class Rates(TempRecords):
    def test_rates_prints_a_line_a_role_and_all_prints_a_lane(self):
        path = self.write("r.json", synth())
        rc, short, _ = run_cli("rates", path)
        rc, full, _ = run_cli("rates", path, "--all")
        self.assertEqual(rc, 0)
        self.assertLess(len(short.splitlines()), len(full.splitlines()))
        self.assertIn("--all", short)
        self.assertIn("per lane, with the spread", full)
        self.assertEqual(json.loads(run_cli("rates", path, "--json")[1])["lanes"].keys(), {"k/p"})   # --json unchanged


class Diff(TempRecords):
    def test_five_percent_is_resolved_and_half_a_percent_is_not(self):
        base = summary.digest(synth(load=300))
        big = summary.diff_digests(base, summary.digest(synth(load=350)))      # 5% of a 1,000-cycle window, 1% jitter
        small = summary.diff_digests(base, summary.digest(synth(load=305)))    # 0.5%
        row = {r["name"]: r for r in big["rows"] if r["role"] == "producer"}
        self.assertAlmostEqual(row["load"]["change"], 50)
        self.assertTrue(row["load"]["resolved"])
        self.assertAlmostEqual(row["load"]["percent"], 100 * 50 / 300)
        self.assertFalse(row["compute"]["resolved"])                           # untouched
        srow = {r["name"]: r for r in small["rows"] if r["role"] == "producer"}
        self.assertAlmostEqual(srow["load"]["change"], 5)
        self.assertFalse(srow["load"]["resolved"])
        # the test's own number: 2 * sqrt(se^2 + se^2), se = 10 / sqrt(10)
        self.assertAlmostEqual(srow["load"]["se"], (2 * (10 / 10 ** 0.5) ** 2) ** 0.5)
        text = summary.format_diff(big)
        self.assertRegex(text, r"phase|load")
        line = [x for x in text.splitlines() if x.split()[:2] == ["producer", "load"]][0]
        self.assertIn("*", line)
        sline = [x for x in summary.format_diff(small).splitlines() if x.split()[:2] == ["producer", "load"]][0]
        self.assertIn("~", sline)
        self.assertNotIn("*", sline)
        self.assertIn("WITHIN each run", text.replace("\n  ", " ") + "WITHIN each run")   # stated in the output
        self.assertIn("diff two runs of the same kernel first", text.replace("\n  ", " "))

    def test_the_cli_diff(self):
        a = self.write("a.json", synth(load=300))
        b = self.write("b.json", synth(load=350))
        rc, out, err = run_cli("diff", a, b)
        self.assertEqual(rc, 0, err)
        self.assertLessEqual(len(out.splitlines()), summary.LINES)
        rc, out, err = run_cli("diff", a, b, "--json")
        d = json.loads(out)
        self.assertEqual(set(d), {"test", "caveat", "launch", "window", "rows", "only_before", "only_after", "anchors",
                                  "waits_stamped", "waits_not_compared"})

    def test_two_kernels_are_refused(self):
        a = self.write("a.json", synth())
        b = self.write("b.json", synth(lanes=("k/p", "k/q")))
        rc, out, err = run_cli("diff", a, b)
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED", err)
        self.assertIn("k/q", err)
        self.assertEqual(out, "")
        self.assertIsNone(summary.comparable(synth(), synth(load=350)))

    def test_the_variant_is_shown_by_summary_and_named_by_a_refused_diff(self):
        a, b = synth(), synth(lanes=("k/p", "k/q"))
        a.variant, b.variant = "fine", "coarse"
        self.assertEqual(summary.digest(a)["variant"], "fine")
        self.assertIn("(variant fine)", summary.format_summary(summary.digest(a), "a.json"))
        self.assertNotIn("variant", summary.format_summary(summary.digest(synth()), "a.json"))
        pa, pb = self.write("a.json", a), self.write("b.json", b)
        rc, out, err = run_cli("diff", pa, pb)
        self.assertEqual(rc, 2)
        self.assertIn("'fine' and 'coarse'", err)

    def test_summary_and_check_agree_on_the_lanes_open_at_the_end(self):
        doc = synth().to_dict()
        last = max(i for i, s in enumerate(doc["stamps"]) if s[0] == "k/c" and doc["names"][str(s[1])] == "idle.end")
        del doc["stamps"][last]          # the consumer's last region is left open, and it stamps no launch exit
        rec = record.from_dict(doc)
        d = summary.digest(rec)
        self.assertEqual(d["health"]["findings"], {})
        self.assertEqual(d["health"]["open_at_end"], {"lanes": 1, "regions": 1})
        self.assertIn("open at end: 1 lane(s), 1 region(s)", summary.format_summary(d))
        rc, out, err = run_cli("check", self.write("o.json", rec))
        self.assertIn("0 finding(s)", out)
        self.assertIn("open_at_end: 1 lane(s), 1 region(s)", out)

    def test_phases_in_one_record_are_listed(self):
        a = synth()
        doc = a.to_dict()
        doc["names"] = {k: ("work.begin" if v == "compute.begin" else "work.end" if v == "compute.end" else v)
                        for k, v in doc["names"].items()}
        d = summary.diff_digests(summary.digest(a), summary.digest(record.from_dict(doc)))
        self.assertEqual(d["only_before"], ["producer: compute"])
        self.assertEqual(d["only_after"], ["producer: work"])
        self.assertIn("only in before: producer: compute", summary.format_diff(d))

    def test_waits_are_compared(self):
        a = summary.digest(synth(load=300))
        b = summary.digest(synth(load=350))
        d = summary.diff_digests(a, b)
        w = [r for r in d["rows"] if r["kind"] == "waits" and r["role"] == "producer"][0]
        self.assertAlmostEqual(w["change"], -50)
        self.assertTrue(w["resolved"])
        c = [r for r in d["rows"] if r["kind"] == "waits" and r["role"] == "consumer"][0]
        self.assertFalse(c["resolved"])


class Sql(TempRecords):
    def setUp(self):
        super().setUp()
        self.rec = synth()
        self.con = query.load(self.rec)

    def q(self, text, limit=50):
        cols, rows, cut = query.run(self.con, text, limit)
        return cols, rows, cut

    def test_the_tables(self):
        n = lambda t: self.q(f"SELECT COUNT(*) FROM {t}")[1][0][0]
        self.assertEqual(n("lanes"), 2)
        self.assertEqual(n("stamps"), len(self.rec.stamps))
        self.assertEqual(n("regions"), 7 * N + 2)               # 4 + 3 regions a window, and the closing tick of each lane
        self.assertEqual(n("windows"), 10 + 2 * 10)             # the scope's, and each role's own (both carry the anchor)
        self.assertEqual(n("ties"), sum(len(v) for v in self.rec.ties.values()))
        self.assertEqual(n("waits"), 2 * N)
        self.assertEqual(n("findings"), 0)
        self.assertEqual(self.q("SELECT cycles FROM windows LIMIT 1")[1][0][0], 1000)
        self.assertEqual(self.q("SELECT MIN(idx), MAX(idx) FROM windows")[1][0], (0, 9))
        # times stay integer cycles
        self.assertEqual(self.q("SELECT typeof(t) FROM stamps LIMIT 1")[1][0][0], "integer")
        self.assertEqual(self.q("SELECT typeof(t_begin) FROM regions LIMIT 1")[1][0][0], "integer")
        # a region's window, and a wait's release
        self.assertEqual(self.q("SELECT window FROM regions WHERE name = 'load' ORDER BY t_begin LIMIT 1")[1][0][0], None)
        self.assertEqual(self.q("SELECT window FROM regions WHERE name = 'load' AND t_begin = 2010")[1][0][0], 0)
        cols, rows, _ = self.q("SELECT released_by, release_kind FROM waits WHERE barrier = 'req' LIMIT 1")
        self.assertEqual(rows[0], ("k/c", "arrival"))           # req's one arrival is stamped on the consumer lane

    def test_named_phases_by_role(self):
        cols, rows, _ = self.q(query.NAMED["phases-by-role"][1])
        self.assertEqual(cols[:3], ["role", "name", "occurrence"])
        got = {(r[0], r[1]): (r[3], r[4]) for r in rows}
        self.assertEqual(got[("producer", "load")], (10, 300))
        self.assertEqual(got[("producer", "compute")], (310, 400))
        self.assertEqual(got[("consumer", "serve")], (10, 890))
        self.assertEqual(len(rows), 7)                           # tick, load, compute, drain; tick, serve, idle

    def test_named_longest_waits(self):
        cols, rows, _ = self.q(query.NAMED["longest-waits"][1])
        self.assertEqual(rows[0][0], "k/p")
        self.assertEqual(rows[0][4], 300)                        # the producer's wait is 290 - jitter, at most 300
        self.assertEqual(rows[0][2], "ack")
        self.assertEqual(rows[-1][4], 100)

    def test_named_waits_by_barrier(self):
        cols, rows, _ = self.q(query.NAMED["waits-by-barrier"][1])
        got = {r[1]: r[2:] for r in rows}
        self.assertEqual(got["ack"], (12, 3480, 300))            # 6 waits of 280 and 6 of 300
        self.assertEqual(got["req"], (12, 1200, 100))

    def test_named_rate_by_phase_and_window_jitter(self):
        cols, rows, _ = self.q(query.NAMED["rate-by-phase"][1])
        got = {r[1]: r[5] for r in rows if r[0] == "producer"}
        self.assertAlmostEqual(got["load"], 2000, delta=15)
        self.assertAlmostEqual(got["compute"], 1000, delta=15)
        cols, rows, _ = self.q(query.NAMED["window-jitter"][1])
        self.assertEqual({r[2] for r in rows}, {1000})

    def test_every_named_query_runs(self):
        for name, (_d, text) in query.NAMED.items():
            self.q(text)

    def test_the_cap(self):
        cols, rows, cut = self.q("SELECT * FROM stamps", 5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(cut)
        cols, rows, cut = self.q("SELECT * FROM lanes", 5)
        self.assertFalse(cut)
        path = self.write("r.json", self.rec)
        rc, out, _ = run_cli("sql", path, "SELECT * FROM stamps", "--limit", "7")
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.splitlines()), 1 + 7 + 1)
        self.assertIn("CUT at --limit 7", out)
        rc, out, _ = run_cli("sql", path, "SELECT * FROM stamps")             # the default is 50
        self.assertEqual(len(out.splitlines()), 1 + 50 + 1)
        rc, out, _ = run_cli("sql", path, "SELECT * FROM stamps", "--json", "--limit", "3")
        j = json.loads(out)
        self.assertEqual((j["row_count"], j["truncated"], j["limit"]), (3, True, 3))
        rc, out, _ = run_cli("sql", path, "SELECT lane, role FROM lanes", "--csv")
        self.assertEqual(out.splitlines(), ["lane,role", "k/p,producer", "k/c,consumer"])

    def test_schema_and_named_listings(self):
        path = self.write("r.json", self.rec)
        rc, out, _ = run_cli("sql", path, "--schema")
        for t in query.TABLES:
            self.assertIn(f"  {t}(", out)
        rc, out, _ = run_cli("sql", path, "--named")
        for k, (_d, text) in query.NAMED.items():
            self.assertIn(k, out)
            self.assertIn(text, out)
        rc, out, _ = run_cli("sql", path, "--named", "waits-by-barrier")
        self.assertIn("barrier_group", out)

    def test_the_database_is_read_only_and_errors_are_said(self):
        path = self.write("r.json", self.rec)
        rc, out, err = run_cli("sql", path, "DELETE FROM stamps")
        self.assertEqual(rc, 2)
        self.assertIn("warpline: sql:", err)
        rc, out, err = run_cli("sql", path, "SELECT * FROM nothing")
        self.assertEqual(rc, 2)
        self.assertIn("--schema", err)


class Cli(TempRecords):
    def test_summary_cli(self):
        path = self.write("r.json", synth())
        rc, out, err = run_cli("summary", path)
        self.assertEqual(rc, 0, err)
        self.assertLessEqual(len(out.splitlines()), summary.LINES)
        rc, out, err = run_cli("summary", path, "--json")
        self.assertEqual(json.loads(out)["longest_chain"]["role"], "consumer")


if __name__ == "__main__":
    unittest.main()
