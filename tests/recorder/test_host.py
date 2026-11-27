"""include/warpline/host.hpp: hand-made rows (no GPU) become records `check` accepts; a C++ program written for this
test (tests/recorder/host_gen.cc, a Bazel cc_binary) makes them. The 32-bit words wrap; a row is full; a mark's tie."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from warpline.recorder import check, reader, record

GEN = sys.argv.pop(1) if len(sys.argv) > 1 else os.environ.get("HOST_GEN", "host_gen")


def stamps(rec, lane):
    return [(rec.names[n], t) for ln, n, t, _ in rec.stamps if ln == lane]


class HostRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        subprocess.run([GEN, cls.tmp.name], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, name):
        return record.load(os.path.join(self.tmp.name, name))

    def test_wrapped_words_are_unwrapped_and_check_accepts(self):
        rec = self.load("wrap.json")
        self.assertEqual(check.all_findings(rec), [])
        # the consumer's first word (C - 50) is the scope's earliest: 0; the producer starts 50 cycles later
        self.assertEqual(stamps(rec, "cta0/warp0"),
                         [("launch.begin", 50), ("phase.fill.begin", 150), ("barrier.full.arrive", 400), ("ping", 410),
                          ("phase.fill.end", 1050), ("launch.end", 1050)])
        self.assertEqual(stamps(rec, "cta0/warp1"),
                         [("launch.begin", 0), ("phase.work.begin", 170), ("barrier.full.wait.begin", 180),
                          ("barrier.full.wait.end", 460), ("phase.work.end", 1040), ("launch.end", 1040)])

    def test_marks_tie_the_two_clocks_across_the_wrap(self):
        rec = self.load("wrap.json")
        # the earliest nanosecond reading (the consumer's entry, T - 30) is 0
        self.assertEqual(rec.ties["cta0/warp1"], [(0, 0), (170, 110), (1040, 720)])
        self.assertEqual(rec.ties["cta0/warp0"], [(50, 30), (150, 130), (1050, 730)])
        self.assertEqual(rec.clock["time_unit"], "ns")

    def test_a_full_row_counts_what_it_dropped(self):
        rec = self.load("full.json")
        self.assertEqual(rec.dropped, {"cta0/warp0": 2})
        lane = rec.lanes["cta0/warp0"]
        written = len(rec.stamps)
        self.assertEqual(written + 2, lane.issued)           # stamps written + dropped = issued
        self.assertEqual(check.all_findings(rec), [])         # nothing unaccounted; no edge missing
        names = [n for n, _ in stamps(rec, "cta0/warp0")]
        self.assertEqual(names[0], "launch.begin")
        self.assertEqual(names[-1], "launch.end")

    def test_a_mark_cut_by_the_capacity_has_a_stamp_and_no_tie(self):
        rec = self.load("cut.json")
        self.assertEqual(rec.ties, {"cta0/warp0": [(0, 0)]})     # the begin's tie only
        self.assertEqual(rec.dropped, {"cta0/warp0": 1})
        found = {f.code for f in check.all_findings(rec)}
        self.assertIn("launch_edge_untied", found)            # named, not hidden

    def test_a_time_entry_with_no_stamp_is_refused(self):
        self.assertEqual(subprocess.run([GEN, "--bad"]).returncode, 0)

    def test_a_row_with_a_head_of_its_own_is_the_same_record(self):
        # the kernel's head (32 words, the issued count first, a base after) is not warpline's: the entries and the
        # count issued are given apart, and the record is what the warpline layout gives, dropped count included
        with open(os.path.join(self.tmp.name, "head_a.json")) as a, open(os.path.join(self.tmp.name, "head_b.json")) as b:
            self.assertEqual(a.read(), b.read())
        rec = self.load("head_b.json")
        self.assertEqual(rec.dropped, {"cta0/warp0": 1})
        self.assertEqual(rec.lanes["cta0/warp0"].issued, len(rec.stamps) + 1)

    def test_operations_are_declared_from_the_issue_site(self):
        rec = self.load("ops.json")
        self.assertEqual(rec.facts["operations"], {"copy": {"done": "barrier.landed.wait.end"}})
        self.assertEqual(rec.facts["barriers"]["landed"], {"count": 1, "stamped": 0, "issue": "copy"})
        self.assertEqual(check.all_findings(rec), [])
        ops, found = reader.operations(rec)
        self.assertEqual(found, [])
        self.assertEqual([(o.op_id, o.issue_time, o.done_time) for o in ops["copy"]], [(0, 100, 300), (1, 500, 700)])
        self.assertIn("op.copy.issue", rec.names.values())     # named by the resource, as an importer's record

    def test_without_the_operation_the_issues_are_unfinished(self):
        found = {f.code for f in check.all_findings(self.load("ops_undeclared.json"))}
        self.assertEqual(found, {"operation_unfinished"})

    def test_a_rendezvous_has_its_arrival_written_and_its_release_the_last_begin(self):
        rec = self.load("rdv.json")
        self.assertEqual(rec.facts["barriers"]["meet"], {"count": 2, "rendezvous": True})
        self.assertEqual(stamps(rec, "cta0/warp0"), [("barrier.meet.arrive", 0), ("barrier.meet.wait.begin", 0),
                                                     ("barrier.meet.wait.end", 110)])
        self.assertEqual(check.all_findings(rec), [])
        self.assertEqual(rec.lanes["cta0/warp0"].issued, 3)
        for w in reader.waits(rec):
            self.assertEqual((w.release.kind, w.release.time, w.release.lane), ("arrival", 10, "cta0/warp1"))
            self.assertEqual(w.release.stamp, "barrier.meet.arrive")

    def test_a_rendezvous_declares_the_skew_of_a_last_begin_stamped_late(self):
        # warp0's wait ends 5 cycles before warp1's begin is stamped: held by the declared skew, a finding without it
        self.assertEqual(check.all_findings(self.load("rdv_skew.json")), [])
        found = [f.code for f in check.all_findings(self.load("rdv_noskew.json"))]
        self.assertEqual(found, ["wait_ends_before_release"])

    def test_a_barrier_not_declared_a_rendezvous_is_still_held_to_its_arrivals(self):
        found = [f.code for f in check.all_findings(self.load("rdv_undeclared.json"))]
        self.assertEqual(found, ["wait_before_release", "wait_before_release"])

    def test_an_arrive_entry_at_a_rendezvous_is_refused(self):
        self.assertEqual(subprocess.run([GEN, "--bad-rdv"]).returncode, 0)

    def test_a_lane_with_no_launch_exit_closes_at_its_last_stamp_and_is_counted(self):
        rec = self.load("noexit.json")
        self.assertEqual(check.all_findings(rec), [])
        self.assertEqual(reader.open_at_end(rec), {"cta0/warp1": ["phase.fill"]})
        regions, _ = reader.regions(rec)
        work, fill = regions["cta0/warp1"]
        self.assertEqual((work.name, work.begin, work.end, work.basis), ("phase.work", 10, 100, "measured"))
        self.assertEqual((fill.name, fill.begin, fill.end, fill.basis), ("phase.fill", 100, 200, "open_at_end"))

    def test_the_variant_is_written_and_shown(self):
        rec = self.load("variant.json")
        self.assertEqual(rec.variant, "fine")
        out = subprocess.run([sys.executable, "-m", "warpline", "summary", os.path.join(self.tmp.name, "variant.json")],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertIn("(variant fine)", out.stdout, out.stdout + out.stderr)

    def test_a_buffer_of_rows_is_the_record_its_rows_give_one_by_one(self):
        # one buffer copied back, rows kHead + cap apart (as Cursor<cap> binds them), named by a lane function
        with open(os.path.join(self.tmp.name, "rows_buffer.json")) as a, open(os.path.join(self.tmp.name, "rows_each.json")) as b:
            self.assertEqual(a.read(), b.read())
        rec = self.load("rows_buffer.json")
        self.assertEqual({k: v.role for k, v in rec.lanes.items()}, {"cta0/warp0": "producer", "cta0/warp1": "consumer"})
        self.assertEqual(check.all_findings(rec), [])

    def test_the_runs_record_is_written_where_the_runner_says_and_nowhere_else(self):
        from warpline.recorder import cli
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, cli.RECORD_DIR: d}
            self.assertEqual(subprocess.run([GEN, "--run", "tile"], env=env).returncode, 0)
            self.assertEqual(os.listdir(d), ["tile" + cli.SUFFIX])
            rec = record.load(os.path.join(d, "tile" + cli.SUFFIX))
            self.assertEqual(check.all_findings(rec), [])
            again = subprocess.run([GEN, "--run", "tile"], env=env, capture_output=True, text=True)
            self.assertEqual(again.returncode, 4)                   # one record a name a run
            self.assertIn("already has a record named `tile`", again.stderr)
            self.assertEqual(subprocess.run([GEN, "--run", "a/b"], env=env).returncode, 4)
        unset = {k: v for k, v in os.environ.items() if k != cli.RECORD_DIR}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(subprocess.run([os.path.abspath(GEN), "--run", "tile"], env=unset, cwd=d).returncode, 3)
            self.assertEqual(os.listdir(d), [])                     # not run by the runner: nothing written

    def test_the_header_and_the_runner_name_one_variable(self):
        from warpline.recorder import cli
        header = (Path(cli.__file__).parents[2] / "include" / "warpline" / "host.hpp").read_text()
        self.assertIn(f'kRecordDir = "{cli.RECORD_DIR}"', header)
        self.assertIn(f'name + "{cli.SUFFIX}"', header)

    def test_the_cli_accepts_it(self):
        path = os.path.join(self.tmp.name, "wrap.json")
        out = subprocess.run([sys.executable, "-m", "warpline", "verify", path], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertIn("0 finding(s)", out.stdout, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
