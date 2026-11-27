"""The profiler's exports (Nsight Compute CSV), the two profile rules on their fixtures, and what the checker gives a run
manifest from them (never the static manifest)."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline import declare, manifest
from warpline.checker.manifest import same_kernel
from warpline.checker.cli import main
from warpline.checker import manifest as collection, ncu
from warpline.checker.rules import profile_registry, registry

FIXTURES = Path(__file__).parent / "fixtures" / "profile_rules"
STEM = str(Path(__file__).parent / "fixtures" / "rules" / "stack_memory" / "no_stack" / "kernel")


def texts(rule, name):
    return [p.read_text() for p in sorted((FIXTURES / rule / name).glob("*.csv"))]


def profiles(rule, name):
    return ncu.profiles_of(texts(rule, name))


def drop_column(text, header_row, column):
    """The CSV text with one column left out (as a profiler version that does not export it prints it): the column of the
    header at row `header_row`, from that row on."""
    rows = list(csv.reader(io.StringIO(text)))
    i = rows[header_row].index(column)
    out = io.StringIO()
    csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(
        row if n < header_row else row[:i] + row[i + 1:] for n, row in enumerate(rows))
    return out.getvalue()


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
        self.assertEqual(sp.rows[0].samples, 64)                               # `Warp Stall Sampling (All Samples)`
        self.assertEqual((sp.rows[0].opcode, lds.opcode), ("IMAD.MOV.U32", "LDS.U16"))
        self.assertEqual(ncu.SourceRow(0, "  @!P0 STS [R1], R2", 0, "Shared", 0, 0, 0, {}).opcode, "STS")
        self.assertEqual(sum(r.samples for r in sp.rows) > sum(r.stalls.get("selected", 0) for r in sp.rows), True)

    def test_a_column_a_version_does_not_export_is_refused_never_read_as_zero(self):
        page = texts("shared_wavefronts_excess", "conflicted")[0]
        for column in (*ncu.SOURCE_COLUMNS[2:], *ncu.SHARED_COLUMNS):     # the page has shared-memory instructions
            dropped = drop_column(page, 1, column)
            with self.assertRaises(ncu.ReaderError, msg=column) as e:
                ncu.source_rows(dropped)
            self.assertIn(repr(column), str(e.exception))
        with self.assertRaises(ncu.ReaderError) as e:
            ncu.source_rows(drop_column(page, 1, "Address Space"))
        self.assertIn(ncu.SOURCE_SECTION, str(e.exception))                 # the section that exports them is named
        details = texts("profiler_verdicts", "fired")[0]
        for column in ncu.DETAILS_COLUMNS:
            if column != "Rule Name":
                with self.assertRaises(ncu.ReaderError, msg=column):
                    ncu.details_rules(drop_column(details, 0, column))
        # a page of the instructions' counts alone (a profile without the source counters' section): refused, not read
        with self.assertRaises(ncu.ReaderError):
            ncu.source_rows('"Kernel Name","k",""\n"Address","Source","Instructions Executed"\n"0x10","LDS R0","5"\n')

    def test_a_kernel_with_no_shared_memory_instruction_exports_no_wavefronts_and_takes_none(self):
        page = texts("shared_wavefronts_excess", "conflict_free")[0]
        for column in ncu.SHARED_COLUMNS:
            page = drop_column(page, 1, column)
        page = page.replace('"Shared"', '"Global"')
        sp = ncu.source_rows(page)
        self.assertEqual({(r.wavefronts, r.ideal, r.excess) for r in sp.rows}, {(0, 0, 0)})

    def test_the_stall_reasons_the_page_exports_are_listed(self):
        sp = ncu.source_rows(texts("shared_wavefronts_excess", "conflicted")[0])
        self.assertIn("long_sb", sp.reasons)
        self.assertNotIn("gmma", sp.reasons)                                 # not exported: absent, not 0
        without = ncu.source_rows(drop_column(texts("shared_wavefronts_excess", "conflicted")[0], 1, "stall_long_sb"))
        self.assertNotIn("long_sb", without.reasons)
        self.assertEqual(ncu.profiles_of([texts("shared_wavefronts_excess", "conflicted")[0]])[0].reasons, sp.reasons)

    def test_a_source_page_is_one_results_and_a_repeated_block_is_read_once(self):
        page = texts("shared_wavefronts_excess", "conflicted")[0]
        self.assertEqual(ncu.source_rows(page + page), ncu.source_rows(page))
        other = texts("shared_wavefronts_excess", "conflict_free")[0]
        with self.assertRaises(ncu.ReaderError) as e:
            ncu.source_rows(page + other)
        self.assertIn("--launch-skip I --launch-count 1", str(e.exception))
        with self.assertRaises(ncu.ReaderError):                                # a row shorter than the header
            ncu.source_rows(page.rstrip("\n").rsplit(",", 3)[0] + "\n")

    def test_the_results_of_a_details_page(self):
        self.assertEqual(ncu.details_results(texts("profiler_verdicts", "fired")[0]),
                         [("0", "void sample::tile_kernel<0>(Params)"), ("1", "void sample::edge_kernel<0, 1>(Params)")])

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

    def test_the_pages_pair_as_nsight_compute_2026_names_a_kernel_on_each(self):
        # 2026 prints a bool template argument as `1` on the details page and `(bool)1` on the source page; c++filt `true`
        details, *sources = texts("profiler_verdicts", "fired")
        source = texts("shared_wavefronts_excess", "conflicted")[0]
        details = details.replace("sample::tile_kernel<0>(Params)", "sample::tile_kernel<1>(Params)")
        source = source.replace("sample::tile_kernel<0>(Params)", "sample::tile_kernel<(bool)1>(sample::Params)")
        (p, *_) = ncu.profiles_of([details, source])
        self.assertEqual((p.kernel, p.has_details, p.has_source), ("void sample::tile_kernel<1>(Params)", True, True))
        self.assertTrue(same_kernel(p.kernel, "void sample::tile_kernel<true>(sample::Params)"))
        self.assertFalse(same_kernel(p.kernel, "void sample::tile_kernel<false>(sample::Params)"))


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


TILE = "sample::tile_kernel<0>(Params)"


def run_manifest(ps, place=TILE):
    return manifest.run_manifest(place, [collection.run_rules(ps, place, profile_registry())])


class Refusals(unittest.TestCase):
    def test_declarations_govern_a_profile_rule_in_a_run_manifest(self):
        rules = profile_registry()
        made = run_manifest(profiles("shared_wavefronts_excess", "conflicted"))
        got = declare.compare(made, [])
        self.assertEqual([(r.rule, r.count, r.allowed, r.kernel) for r in got.refusals],
                         [("shared_wavefronts_excess", 98304, 0, TILE)])
        self.assertTrue(any("profiler_verdicts not run" in s and "no details page" in s for s in got.said))
        ds = declare.load_declarations(json.dumps({"declarations": [
            {"kernel": "tile_kernel", "rule": "shared_wavefronts_excess", "allow": 98304, "reason": "measured"}]}),
            registry(), list(rules))
        self.assertEqual(declare.compare(made, ds).refusals, [])

    def test_a_site_keeps_the_instructions_offset_in_the_kernel(self):
        (place,) = run_manifest(profiles("shared_wavefronts_excess", "conflicted")).places
        sites = place.rules["shared_wavefronts_excess"].sites
        self.assertEqual(sites[0].where, f"{0x900f21d90 - 0x900f1e200:#06x}")

    def test_a_profile_is_matched_to_a_place_by_name(self):
        from warpline.checker import runner
        (k,) = runner.kernels_of_files(STEM).values()                  # `sample::tile()`
        (p,) = profiles("shared_wavefronts_excess", "conflicted")
        self.assertIsNone(collection.match_profile(p, [k]))
        self.assertIs(collection.match_profile(ncu.Profile("void sample::tile()"), [k]), k)
        self.assertIs(collection.match_profile(ncu.Profile("void sample::tile(float*)"), [k]), k)
        self.assertTrue(collection.same_kernel("void sample::tile_kernel<0>(Params)", TILE))
        made = run_manifest([p], "sample::tile()")                     # one profile: paired with the place whatever its name
        self.assertEqual([(q.place, q.kernel) for q in made.places], [("sample::tile()", True)])
        self.assertTrue(made.said[0].startswith("note: profile rule profiler_verdicts not run"))

    def test_a_profile_of_another_kernel_is_a_place_of_its_own_and_said(self):
        made = run_manifest(profiles("profiler_verdicts", "fired"))   # the page holds two kernels
        self.assertEqual([(q.place, q.kernel, q.figures) for q in made.places],
                         [(TILE, True, {}), ("void sample::edge_kernel<0, 1>(Params)", False, None)])
        self.assertEqual(made.places[0].rules["profiler_verdicts"].count, 2)
        self.assertIn("note: the profile of 'void sample::edge_kernel<0, 1>(Params)' matches no place "
                      "('sample::tile_kernel<0>(Params)')", made.said)


class Command(unittest.TestCase):
    def invoke(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def empty(self):
        path = Path(tempfile.mkdtemp()) / "d.json"
        path.write_text('{"declarations": []}')
        return str(path)

    def test_the_static_manifest_takes_no_profile(self):
        source = str(FIXTURES / "shared_wavefronts_excess" / "conflicted" / "source.csv")
        for verb in (["check", STEM, "--fixture", "--declare", self.empty()], ["manifest", STEM, "--fixture"]):
            code, _, err = self.invoke(*verb, "--profile", source)
            self.assertEqual(code, 2, verb)
            self.assertIn("unrecognized arguments: --profile", err)
        code, out, _ = self.invoke("check", STEM, "--fixture", "--declare", self.empty())
        self.assertEqual(code, 0)
        self.assertNotIn("profile", out)                                  # nothing of a profile is said of a binary
        self.assertEqual(out.splitlines()[-1], f"1 kernel(s), {len(registry())} rule(s): 0 refusal(s)")
        code, out, _ = self.invoke("manifest", STEM, "--fixture")
        self.assertFalse(set(json.loads(out)["places"][0]["rules"]) & set(profile_registry()))

    def test_a_profile_that_is_not_an_export_exits_2(self):
        bad = Path(tempfile.mkdtemp()) / "x.csv"
        bad.write_text('"a","b"\n')
        code, _, err = self.invoke("run-manifest", "--profile", str(bad), "--place", TILE)
        self.assertEqual(code, 2)
        self.assertIn("not a Nsight Compute export", err)


if __name__ == "__main__":
    unittest.main()
