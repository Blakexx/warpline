"""Register liveness on the model, the reader of the listing with source-line marks, the rule `loop_held` and the verb `held`:
a hand-checkable kernel, a small marked listing, and the committed fixtures (`tests/checker/fixtures/rules/loop_held/`)."""
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from warpline.checker import cli, flow, held, model, readers, runner
from warpline.checker.model import Instruction
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures" / "rules" / "loop_held"


def instruction(text, predicate=""):
    opcode, _, operands = text.partition(" ")
    return Instruction(0, predicate, opcode, operands, "b")


def toy() -> model.Kernel:
    """Five values set up in `A`, a loop `L` that sums into R5 with the counter R7, and `X` that stores R4 and R5 through R2:
    R2 and R4 are held across the loop without being touched, R5 is carried, R6 and R7 are read by it."""
    code = [
        (0x00, "", "MOV", "R2, 0x1"), (0x10, "", "MOV", "R4, 0x2"), (0x20, "", "MOV", "R5, RZ"), (0x30, "", "MOV", "R6, 0x3"),
        (0x40, "", "MOV", "R7, RZ"),
        (0x50, "", "FADD", "R5, R5, R6"), (0x60, "", "IADD3", "R7, R7, 0x1, RZ"),
        (0x70, "", "ISETP.LT.AND", "P0, PT, R7, 0x5, PT"), (0x80, "@P0", "BRA", "0x50"),
        (0x90, "", "STG.E", "[R2], R4"), (0xa0, "", "STG.E", "[R2+0x4], R5"), (0xb0, "", "EXIT", ""),
    ]
    sass = [readers.SassLine(*c) for c in code]
    cfg = readers.Cfg("A", (readers.CfgBlock("A", (0x00, 0x10, 0x20, 0x30, 0x40)),
                            readers.CfgBlock("L", (0x50, 0x60, 0x70, 0x80)),
                            readers.CfgBlock("X", (0x90, 0xa0, 0xb0))),
                      (("A", "L"), ("L", "L"), ("L", "X")))
    return model.build_kernel("toy", sass, cfg, readers.Usage(8, 0, 0))


def nested() -> model.Kernel:
    """An outer loop (`O` through `E`) holding R2, R3 and its counter R7, around an inner loop `I` with its own counter R8."""
    code = [
        (0x00, "", "MOV", "R2, 0x1"), (0x10, "", "MOV", "R3, 0x2"), (0x20, "", "MOV", "R7, RZ"),
        (0x30, "", "MOV", "R8, RZ"),
        (0x40, "", "IADD3", "R8, R8, 0x1, RZ"), (0x50, "", "ISETP.LT.AND", "P0, PT, R8, 0x4, PT"), (0x60, "@P0", "BRA", "0x40"),
        (0x70, "", "IADD3", "R7, R7, R2, RZ"), (0x80, "", "ISETP.LT.AND", "P1, PT, R7, 0x9, PT"), (0x90, "@P1", "BRA", "0x30"),
        (0xa0, "", "STG.E", "[R3], R7"), (0xb0, "", "EXIT", ""),
    ]
    blocks = (("A", (0x00, 0x10, 0x20)), ("O", (0x30,)), ("I", (0x40, 0x50, 0x60)), ("E", (0x70, 0x80, 0x90)), ("X", (0xa0, 0xb0)))
    cfg = readers.Cfg("A", tuple(readers.CfgBlock(*b) for b in blocks),
                      (("A", "O"), ("O", "I"), ("I", "I"), ("I", "E"), ("E", "O"), ("E", "X")))
    return model.build_kernel("nested", [readers.SassLine(*c) for c in code], cfg, readers.Usage(10, 0, 0))


def kernel_of(name):
    (k,) = runner.kernels_of_files(str(FIXTURES / name / "kernel")).values()
    return k


class Operands(unittest.TestCase):
    """What the model says an instruction writes and reads, where liveness needs it exact."""

    def test_a_destination_behind_predicates(self):
        shuffle = instruction("SHFL.IDX PT, R3, R2, 0x1, 0x1f")
        self.assertEqual((shuffle.writes, shuffle.reads), ({"R3"}, {"R2"}))
        logic = instruction("LOP3.LUT P0, R6, R0, 0x1f, RZ, 0xc0, !PT")
        self.assertEqual((logic.writes, logic.reads), ({"R6"}, {"R0"}))
        discard = instruction("LOP3.LUT P5, RZ, R33, 0x1, RZ, 0xc0, !PT")
        self.assertEqual((discard.writes, discard.reads), (set(), {"R33"}))

    def test_a_comparison_writes_predicates_alone(self):
        compare = instruction("ISETP.GE.AND P0, PT, R2, R3, PT")
        self.assertEqual((compare.writes, compare.reads), (set(), {"R2", "R3"}))

    def test_a_wide_access_is_a_run(self):
        load = instruction("LDS.128 R12, [R7]")
        self.assertEqual((load.writes, load.reads), ({"R12", "R13", "R14", "R15"}, {"R7"}))
        store = instruction("STG.E.64 desc[UR6][R2.64], R4")
        self.assertEqual(store.reads, {"R2", "R3", "R4", "R5", "UR6"})
        wide = instruction("IMAD.WIDE R4, R2, 0x4, R6")
        self.assertEqual((wide.writes, wide.reads), ({"R4", "R5"}, {"R2", "R6", "R7"}))
        double = instruction("DADD R4, R6, R8")
        self.assertEqual((double.writes, double.reads), ({"R4", "R5"}, {"R6", "R7", "R8", "R9"}))

    def test_an_atomic_returns_a_value(self):
        atomic = instruction("ATOMG.E.ADD.STRONG.GPU PT, R5, [R2.64], R6")
        self.assertEqual((atomic.writes, atomic.reads), ({"R5"}, {"R2", "R3", "R6"}))


class Liveness(unittest.TestCase):
    def test_the_registers_live_at_each_blocks_entry(self):
        live = toy().live_in()
        self.assertEqual(live["A"], frozenset())                       # everything is made in A
        self.assertEqual(live["L"], {"R2", "R4", "R5", "R6", "R7"})    # R2, R4 are held across the loop, the others it reads
        self.assertEqual(live["X"], {"R2", "R4", "R5"})

    def test_the_loop_carries_what_it_reads_on_the_next_trip(self):
        # R7 is written before it is read in the trip, yet live at the top: the trip before it wrote it (the back edge)
        # and the first trip reads A's. Without the back edge nothing would hold R6 either.
        self.assertIn("R6", toy().live_in()["L"])

    def test_a_predicated_write_keeps_the_old_value(self):
        after = frozenset({"R4"})
        self.assertEqual(flow.live_step(after, instruction("MOV R4, 0x2")), frozenset())
        self.assertEqual(flow.live_step(after, instruction("MOV R4, 0x2", "@P0")), {"R4"})
        self.assertEqual(flow.live_step(frozenset(), instruction("MOV R4, R9", "@P0")), {"R4", "R9"})

    def test_code_under_the_always_false_predicate_changes_nothing(self):
        self.assertEqual(flow.live_step(frozenset({"R4"}), instruction("MOV R4, R9", "@!PT")), {"R4"})


MARKED = """\
//--------------------- .text.first --------------------------
\t.section\t.text.first,"ax",@progbits
first:
\t//## File "/work/sandbox/9/execroot/_main/src/a.cu", line 10
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
\t//## File "/work/sandbox/9/execroot/_main/src/inc.cuh", line 4 inlined at "/work/sandbox/9/execroot/_main/src/a.cu", line 11
\t//## File "/work/sandbox/9/execroot/_main/src/a.cu", line 11
        /*0020*/                   IMAD R3, R0, R0, RZ ;
        /*0030*/                   EXIT ;
\t.section\t.debug_frame,"",@progbits
        /*0000*/ \t.byte\t0xff, 0xff
\t.section\t.text.second,"ax",@progbits
\t//## File "/elsewhere/b.cu", line 2
        /*0000*/              @P0  BRA 0x20 ;
        /*0010*/                   NOP ;
"""


class SourceLines(unittest.TestCase):
    def test_each_instruction_has_the_first_mark_since_the_one_before(self):
        found = readers.source_lines(MARKED)
        self.assertEqual(list(found), ["first", "second"])
        # 0x0010 has no mark of its own: it keeps the place before it; 0x0020 is the INNERMOST line (the first mark)
        self.assertEqual(found["first"], {0x00: ("src/a.cu", 10), 0x10: ("src/a.cu", 10), 0x20: ("src/inc.cuh", 4),
                                          0x30: ("src/inc.cuh", 4)})

    def test_the_path_is_cut_after_the_execution_root_where_there_is_one(self):
        found = readers.source_lines(MARKED)
        self.assertEqual(found["second"][0x00], ("/elsewhere/b.cu", 2))

    def test_data_sections_are_not_instructions(self):
        self.assertEqual(set(readers.source_lines(MARKED)["first"]), {0x00, 0x10, 0x20, 0x30})

    def test_a_listing_with_no_marks_has_no_places(self):
        self.assertEqual(readers.source_lines("\t.section\t.text.f,\"ax\"\n        /*0000*/   EXIT ;\n"), {"f": {}})

    def test_the_fixtures_listing_is_read_against_its_own_graph(self):
        k = kernel_of("many")
        self.assertLessEqual({i.address for i in k.instructions}, set(k.lines))     # the padding the model drops has places too
        self.assertEqual(k.lines[0x00][0], "examples/checker/held_many.cu")


class Holds(unittest.TestCase):
    def test_a_loops_top_by_where_each_value_was_made(self):
        (h,) = held.holds(toy())
        self.assertEqual((h.loop.address, h.depth, h.live, h.uniform), (0x50, 0, 5, 0))
        by = {g.label: g for g in h.groups}
        self.assertEqual(set(by), {"0x0000", "0x0010", "0x0020", "0x0030", "0x0040"})
        # R2 and R4 are made before the loop and the loop never reads them; R5 is read and rewritten by it
        self.assertEqual((by["0x0000"].registers, by["0x0000"].used, by["0x0000"].carried), (("R2",), 0, 0))
        self.assertEqual((by["0x0010"].registers, by["0x0010"].used), (("R4",), 0))
        self.assertEqual((by["0x0020"].registers, by["0x0020"].used, by["0x0020"].carried), (("R5",), 1, 1))

    def test_nested_loops_outermost_first_and_indented(self):
        found = held.holds(nested())
        self.assertEqual([(h.loop.address, h.depth, h.live) for h in found], [(0x30, 0, 3), (0x40, 1, 4)])
        lines = held.report([nested()]).splitlines()
        self.assertEqual(lines[0], "nested  (10 registers, 2 loop(s))")
        self.assertEqual([l for l in lines if "loop at" in l],
                         ["  loop at 0x0030: 3 registers live at its top", "    loop at 0x0040: 4 registers live at its top"])

    def test_a_kernel_with_no_loop_holds_nothing(self):
        self.assertEqual(held.holds(kernel_of("none")), [])

    def test_the_fixtures_figures(self):
        (few,), (many,) = held.holds(kernel_of("few")), held.holds(kernel_of("many"))
        self.assertEqual((few.live, many.live), (8, 30))
        top = many.groups[0]
        self.assertEqual((len(top.registers), top.used), (25, 0))       # the loads' registers: held, never read in the loop
        self.assertEqual(top.label, "examples/checker/held_many.cu:12")


class Rule(unittest.TestCase):
    def test_the_figure_the_loop_and_the_lines_that_made_it(self):
        (f,) = registry()["loop_held"].check(kernel_of("many"))
        self.assertEqual((f.where, f.count), ("0x02a0", 30))
        self.assertIn("examples/checker/held_many.cu:12 (25)", f.what)

    def test_without_source_lines_it_names_addresses(self):
        (f,) = registry()["loop_held"].check(replace(kernel_of("many"), lines={}))
        self.assertEqual(f.count, 30)
        self.assertIn("made at 0x", f.what)
        self.assertNotIn(".cu", f.what)

    def test_no_loop_no_finding(self):
        self.assertEqual(registry()["loop_held"].check(kernel_of("none")), [])

    def test_the_most_of_any_loop_is_the_figure(self):
        (f,) = registry()["loop_held"].check(nested())
        self.assertEqual((f.where, f.count), ("0x0040", 4))      # the inner loop holds R8 beside the outer loop's three


class Verb(unittest.TestCase):
    def run_held(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = cli.main(["held", *argv])
        return status, out.getvalue(), err.getvalue()

    def test_a_fixture(self):
        status, out, err = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.splitlines()[:3], [
            "held_many(const float *, const float *, float *, int)  (40 registers, 1 loop(s))",
            "  loop at 0x02a0: 30 registers live at its top, 2 uniform",
            "      25     0 read in the loop  examples/checker/held_many.cu:12  [R0 R4 R5 R6 R7 R8 ...]"])

    def test_the_same_text_twice_and_no_path_of_the_machine(self):
        first = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture")[1]
        self.assertEqual(first, self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture")[1])
        self.assertNotIn("/home", first)
        self.assertNotIn("execroot", first)

    def test_a_kernel_with_no_loop_and_a_filter_that_matches_none(self):
        self.assertEqual(self.run_held(str(FIXTURES / "none" / "kernel"), "--fixture")[1].splitlines(),
                         ["held_none(const float *, float *)  (10 registers, 0 loop(s))"])
        self.assertEqual(self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture", "--kernel", "^zzz")[1],
                         "no kernel matches\n")

    def test_lines_names_another_listing(self):
        with tempfile.TemporaryDirectory() as scratch:
            other = Path(scratch) / "other.lines"
            other.write_text((FIXTURES / "many" / "kernel.lines").read_text().replace("held_many.cu", "renamed.cu"))
            status, out, _ = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture", "--lines", str(other))
        self.assertEqual(status, 0)
        self.assertIn("examples/checker/renamed.cu:12", out)

    def test_a_listing_of_other_functions_is_said(self):
        status, out, err = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture", "--lines", str(FIXTURES / "few" / "kernel.lines"))
        self.assertEqual(status, 0)
        self.assertIn("names none of the kernels", err)
        self.assertIn("0x0020", out)

    def test_a_missing_listing_is_a_user_error(self):
        status, out, err = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture", "--lines", str(FIXTURES / "absent.lines"))
        self.assertEqual((status, out), (2, ""))
        self.assertIn("warpline:", err)

    def test_a_bad_pattern_is_a_user_error(self):
        status, _, err = self.run_held(str(FIXTURES / "many" / "kernel"), "--fixture", "--kernel", "(")
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
