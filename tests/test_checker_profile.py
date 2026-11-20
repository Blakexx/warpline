"""The profiler's exports (Nsight Compute CSV) and the two profile rules on their fixtures."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline.__main__ import main
from warpline.checker import declare, ncu
from warpline.checker.rules import profile_registry, registry

FIXTURES = Path(__file__).parent / "checker_fixtures" / "profile_rules"
STEM = str(Path(__file__).parent / "checker_fixtures" / "rules" / "stack_memory" / "no_stack" / "kernel")


def texts(rule, name):
    return [p.read_text() for p in sorted((FIXTURES / rule / name).glob("*.csv"))]


def profiles(rule, name):
    return ncu.profiles_of(texts(rule, name))


class Readers(unittest.TestCase):
    def test_the_two_pages_are_told_apart(self):
        self.assertEqual(ncu.kind_of(texts("profiler_verdicts", "fired")[0]), "details")
        self.assertEqual(ncu.kind_of(texts("shared_wavefronts_excess", "conflicted")[0]), "source")
        with self.assertRaises(ncu.ReaderError):
            ncu.kind_of('"a","b"\n"1","2"\n')

    def test_rule_results_of_the_details_page(self):
        got = ncu.details_rules(texts("profiler_verdicts", "fired")[0])
        self.assertEqual([(r.name, r.type, r.speedup_type, r.speedup, r.launch) for r in got],
                         [("HighPipeUtilization", "INF", "", None, "0"),
                          ("SharedMemoryConflicts", "OPT", "global", 8.689, "0"),
                          ("SharedMemoryConflicts", "OPT", "global", 7.888, "0"),
                          ("SharedMemoryConflicts", "OPT", "global", 1.52, "1")])
        self.assertEqual(got[1].section, "MemoryWorkloadAnalysis_Tables")
        self.assertIn("1.9 - way bank conflict", got[1].description)
        self.assertEqual({r.kernel for r in got}, {"void sample::tile_kernel<0>(Params)", "void sample::edge_kernel<0, 1>(Params)"})

    def test_source_rows(self):
        sp = ncu.source_rows(texts("shared_wavefronts_excess", "conflicted")[0])
        self.assertEqual(sp.kernel, "void sample::tile_kernel<0>(Params)")
        self.assertEqual(len(sp.rows), 8)
        lds = next(r for r in sp.rows if r.source.startswith("LDS.U16"))
        self.assertEqual((lds.space, lds.executed, lds.wavefronts, lds.ideal, lds.excess), ("Shared", 32768, 65536, 32768, 32768))
        self.assertEqual(sp.rows[0].address, 0x900f1e200)
        self.assertTrue(any(r.stalls for r in sp.rows))                       # stall samples by reason

    def test_a_column_a_version_does_not_export_reads_as_zero(self):
        text = '"Kernel Name","k",""\n"Address","Source","Instructions Executed","L1 Wavefronts Shared","L1 Wavefronts Shared Ideal"\n' \
               '"0x10","LDS R0","5","8","3"\n'
        (row,) = ncu.source_rows(text).rows
        self.assertEqual((row.executed, row.excess, row.stalls), (5, 5, {}))   # the excess is the difference

    def test_a_bad_page_is_refused(self):
        for bad in ("", '"Kernel Name","k"\n"x","y"\n', '"Kernel Name","k"\n"Address","Source"\n"zz","s"\n'):
            with self.assertRaises(ncu.ReaderError, msg=bad):
                ncu.source_rows(bad)
        with self.assertRaises(ncu.ReaderError):
            ncu.details_rules('"Kernel Name"\n"k"\n')

    def test_both_pages_make_one_profile_a_kernel(self):
        both = texts("profiler_verdicts", "fired") + texts("shared_wavefronts_excess", "conflicted")
        ps = ncu.profiles_of(both)
        self.assertEqual([p.kernel for p in ps], ["void sample::tile_kernel<0>(Params)", "void sample::edge_kernel<0, 1>(Params)"])
        self.assertEqual((ps[0].has_details, ps[0].has_source, len(ps[0].rules), len(ps[0].source)), (True, True, 3, 8))
        self.assertEqual((ps[1].has_details, ps[1].has_source), (True, False))


class Rules(unittest.TestCase):
    def test_the_profile_rules_are_their_own_registry(self):
        self.assertEqual(sorted(profile_registry()), ["profiler_verdicts", "shared_wavefronts_excess"])
        self.assertFalse(set(profile_registry()) & set(registry()))

    def test_each_rule_on_each_fixture(self):
        for r in profile_registry().values():
            on_disk = sorted(p.name for p in (FIXTURES / r.name).iterdir() if p.is_dir())
            self.assertEqual(on_disk, sorted(f.name for f in r.fixtures), r.name)
            for fx in r.fixtures:
                with self.subTest(rule=r.name, fixture=fx.name):
                    self.assertEqual(sum(r.count(p) for p in profiles(r.name, fx.name)), fx.count)

    def test_wavefronts_in_excess_by_instruction(self):
        (p,) = profiles("shared_wavefronts_excess", "conflicted")
        found = profile_registry()["shared_wavefronts_excess"].check(p)
        self.assertEqual([(f.count) for f in found], [32768] * 3)
        self.assertEqual(found[0].where, f"{0x900f21d90 - 0x900f1e200:#06x}")      # the offset in the kernel
        self.assertIn("LDS.U16 R20, [R0]: 65536 wavefronts, 32768 without conflicts, 32768 in excess", found[0].what)

    def test_the_profilers_verdicts_are_findings_named_by_its_rule(self):
        ps = profiles("profiler_verdicts", "fired")
        found = profile_registry()["profiler_verdicts"].check(ps[0])
        self.assertEqual([(f.where, f.count) for f in found], [("SharedMemoryConflicts", 1)] * 2)
        self.assertIn("OPT, estimated speedup 8.689% (global): The memory access pattern for shared loads", found[0].what)
        self.assertEqual(profile_registry()["profiler_verdicts"].check(profiles("profiler_verdicts", "informational")[0]), [])


class Refusals(unittest.TestCase):
    def test_declarations_govern_a_profile_rule(self):
        rules = profile_registry()
        ps = profiles("shared_wavefronts_excess", "conflicted")
        refusals, said = declare.check_profiles(ps, [], rules, [])
        self.assertEqual([(r.rule, r.count, r.allowed, r.kernel) for r in refusals],
                         [("shared_wavefronts_excess", 98304, 0, "void sample::tile_kernel<0>(Params)")])
        self.assertTrue(any("profiler_verdicts not run" in s and "no details page" in s for s in said))
        ds = declare.load_declarations(json.dumps({"declarations": [
            {"kernel": "tile_kernel", "rule": "shared_wavefronts_excess", "allow": 98304, "reason": "measured"}]}),
            registry(), list(rules))
        self.assertEqual(declare.check_profiles(ps, [], rules, ds)[0], [])

    def test_a_profile_is_matched_to_a_kernel_by_name(self):
        from warpline.checker import runner
        (k,) = runner.kernels_of_files(STEM).values()                  # `sample::tile()`
        (p,) = profiles("shared_wavefronts_excess", "conflicted")
        self.assertIsNone(declare.match_profile(p, [k]))
        self.assertIs(declare.match_profile(ncu.Profile("void sample::tile()"), [k]), k)
        self.assertIs(declare.match_profile(ncu.Profile("void sample::tile(float*)"), [k]), k)
        refusals, said = declare.check_profiles([p], [k], profile_registry(), [])
        self.assertEqual(refusals[0].kernel, "sample::tile()")        # one profile, one kernel: paired
        self.assertEqual(said[0].startswith("note: profile rule profiler_verdicts not run"), True)


class Command(unittest.TestCase):
    def run_check(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", STEM, "--fixture", *argv])
        return code, out.getvalue(), err.getvalue()

    def empty(self):
        path = Path(tempfile.mkdtemp()) / "d.json"
        path.write_text('{"declarations": []}')
        return str(path)

    def test_without_a_profile_the_profile_rules_say_they_were_not_run(self):
        code, out, _ = self.run_check("--declare", self.empty())
        self.assertEqual(code, 0)
        self.assertIn("note: profile rules not run (no --profile): profiler_verdicts, shared_wavefronts_excess", out)
        self.assertEqual(out.splitlines()[-1], "1 kernel(s), 3 rule(s): 0 refusal(s)")

    def test_a_profile_refuses_through_the_command(self):
        details = str(FIXTURES / "profiler_verdicts" / "fired" / "details.csv")
        source = str(FIXTURES / "shared_wavefronts_excess" / "conflicted" / "source.csv")
        code, out, _ = self.run_check("--declare", self.empty(), "--profile", details, "--profile", source)
        self.assertEqual(code, 1)
        self.assertIn("void sample::tile_kernel<0>(Params): profiler_verdicts: 2 found, 0 declared", out)
        self.assertIn("at SharedMemoryConflicts x1: OPT, estimated speedup 8.689%", out)
        self.assertIn("shared_wavefronts_excess: 98304 found, 0 declared", out)
        self.assertIn("the profile of 'void sample::tile_kernel<0>(Params)' matches no kernel", out)
        self.assertEqual(out.splitlines()[-1], "1 kernel(s), 5 rule(s): 3 refusal(s)")

    def test_a_profile_that_is_not_an_export_exits_2(self):
        bad = Path(tempfile.mkdtemp()) / "x.csv"
        bad.write_text('"a","b"\n')
        code, _, err = self.run_check("--profile", str(bad))
        self.assertEqual(code, 2)
        self.assertIn("not a Nsight Compute export", err)


if __name__ == "__main__":
    unittest.main()
