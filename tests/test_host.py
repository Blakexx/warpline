"""include/warpline/host.hpp: hand-made rows (no GPU) become records `check` accepts; a C++ program written for this
test (tests/host_gen.cc, a Bazel cc_binary) makes them. The 32-bit words wrap; a row is full; a mark's tie."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from warpline import check, record

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

    def test_the_cli_accepts_it(self):
        path = os.path.join(self.tmp.name, "wrap.json")
        out = subprocess.run([sys.executable, "-m", "warpline", "check", path], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        self.assertIn("0 finding(s)", out.stdout, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
