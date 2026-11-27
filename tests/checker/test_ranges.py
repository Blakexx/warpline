"""The registers' and the source's readers a consumer reads a cubin through (docs/checker-model.md#register-pressure): the line
grammar, the inline frames, the life-range marks and the pressure the flow computes over them, the opcode table, the cubin cache.
The kernel is `examples/checker/ranges.cu`, compiled by the pinned toolkit (its outputs are `tests/checker/fixtures/ranges/`,
which `//:fixtures_regenerated` holds to the toolkit); the rest are hand-made texts of the tools' forms. No toolkit, no GPU."""
import os
import re
import stat
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from warpline.checker import flow, model, readers, runner

FIXTURES = Path(__file__).parent / "fixtures"
RANGES = FIXTURES / "ranges" / "kernel"
NAME = "_Z6rangesPfPKf"


def text(suffix: str) -> str:
    return Path(str(RANGES) + suffix).read_text()


def ranges() -> model.Kernel:
    (k,) = runner.kernels_of_files(str(RANGES)).values()
    return k


def vector(found) -> int:
    return sum(1 for r in found if re.fullmatch(r"R\d+", r))


class Grammar(unittest.TestCase):
    """THE ONE LINE GRAMMAR (`readers.INSTRUCTION`): the spellings a reader of the listings differed on are read one way."""

    def test_an_address_of_six_digits_is_an_instruction(self):
        """A function past 0xfffff bytes has six-digit addresses: a grammar of four or five digits drops them, silently."""
        m = readers.INSTRUCTION.match("        /*100a30*/              @!P0 IMAD.WIDE R4, R6, 0x4, R2 ;   /* 0x000fe200078e0203 */")
        self.assertEqual((int(m.group(1), 16), m.group(2).strip(), m.group(3), m.group(4)),
                         (0x100a30, "@!P0", "IMAD.WIDE", "R4, R6, 0x4, R2"))
        body = readers.sass_functions("\t\tFunction : f\n        /*fffff0*/                   EXIT ;\n"
                                      "        /*1000000*/                   BRA 0x1000000 ;\n")["f"]
        self.assertEqual([i.address for i in body], [0xfffff0, 0x1000000])

    def test_a_data_line_of_nvdisasm_is_not_an_instruction(self):
        """nvdisasm's listings hold data sections whose lines carry an address: an opcode starts with a capital letter."""
        for line in ("        /*0000*/ \t.byte\t0x04, 0x2f", "        /*0002*/ \t.short\t(.L_1 - .L_0)",
                     "        /*0004*/ \t.word\tindex@(_Z6rangesPfPKf)"):
            self.assertIsNone(readers.INSTRUCTION.match(line), line)
        self.assertTrue(all(readers.INSTRUCTION.match(line) is None for line in text(".life").splitlines()
                            if re.match(r"\s*/\*[0-9a-f]+\*/\s+\.", line)))

    def test_an_instruction_line_ends_with_its_semicolon(self):
        self.assertIsNone(readers.INSTRUCTION.match("        /*0010*/                   EXIT"))
        self.assertIsNotNone(readers.INSTRUCTION.match("        /*0010*/                   EXIT ;"))

    def test_the_headers(self):
        self.assertEqual(readers.ARCH.match("\tarch = sm_90a").group(1), "sm_90a")
        self.assertEqual(readers.FUNCTION.match("\t\tFunction : _Z6rangesPfPKf").group(1), NAME)
        self.assertIsNotNone(readers.ENCODING.match("                                         /* 0x000fe200078e0203 */"))
        self.assertEqual(readers.TEXT_SECTION.match('\t.section\t.text._Z6rangesPfPKf,"ax",@progbits').group(1), NAME)

    def test_every_listing_of_the_kernel_reads_the_same_instructions(self):
        listed = readers.sass_functions(text(".sass"))[NAME]
        sass = [i.address for i in listed]
        life = list(readers.life_ranges(text(".life"))[NAME])
        self.assertEqual(life, sass[:len(life)])
        #: what cuobjdump prints past nvdisasm's last instruction is the padding: NOPs (and the self-branch before them)
        self.assertTrue(all(i.opcode == "NOP" or (i.opcode == "BRA" and i.operands == hex(i.address)) for i in listed[len(life):]))
        frames = readers.inline_frames(text(".inline"))[NAME]
        self.assertTrue(set(frames) <= set(sass))


class InlineFrames(unittest.TestCase):
    def test_an_inlined_helper_has_its_caller_after_it(self):
        frames = readers.inline_frames(text(".inline"))[NAME]
        chains = set(frames.values())
        #: `scaled` (line 8) is inlined at the kernel's loop (line 13): the helper's frame first, the caller's after it
        self.assertIn((("examples/checker/ranges.cu", 8), ("examples/checker/ranges.cu", 13)), chains)
        self.assertTrue(all(chain[-1][0] == "examples/checker/ranges.cu" for chain in chains))

    def test_an_instruction_with_no_mark_keeps_the_chain_above_it(self):
        listing = ('\t.section\t.text.f,"ax",@progbits\n'
                   '\t//## File "/x/execroot/_main/a.cuh", line 3 inlined at "/x/execroot/_main/k.cu", line 9\n'
                   '\t//## File "/x/execroot/_main/k.cu", line 9\n'
                   "        /*0000*/                   MOV R1, R2 ;\n"
                   "        /*0010*/                   MOV R3, R4 ;\n"
                   '\t//## File "/x/execroot/_main/k.cu", line 10\n'
                   "        /*0020*/                   EXIT ;\n"
                   '\t.section\t.text.g,"ax",@progbits\n'
                   "        /*0000*/                   EXIT ;\n")
        found = readers.inline_frames(listing)
        self.assertEqual(found["f"], {0x0: (("a.cuh", 3), ("k.cu", 9)), 0x10: (("a.cuh", 3), ("k.cu", 9)),
                                      0x20: (("k.cu", 10),)})
        self.assertEqual(found["g"], {})                  # offsets restart in every function: g's 0x0 is its own, with no chain


def listing(rows, groups=("GPR",), width=12) -> str:
    """A life-range listing of `rows` as nvdisasm prints one: `(label, text, reads, writes)` an instruction, one group of
    `width` columns."""
    names = "                    // |" + "|".join(f"   {g}   " for g in groups) + "|"
    head = "                    // |   # " + " ".join(str(r) for r in range(width)) + " |"
    base = head.index("// |") + 4
    at = [m.end() - 1 - base for m in re.finditer(r"\b\d+\b", head)]
    out = ['\t.section\t.text._Z4loopPf,"ax",@progbits', names, head, "_Z4loopPf:"]
    for k, (label, line, reads, writes) in enumerate(rows):
        if label:
            out.append(f"{label}:")
        marks = [" "] * (max(at) + 2)
        for r in set(reads) | set(writes):
            marks[at[r]] = "x" if r in reads and r in writes else "v" if r in reads else "^"
        out.append(f"        /*{16 * k:04x}*/                   {line} ;   // |{0:4d}{''.join(marks)[4:]}")
    return "\n".join(out) + "\n"


class LifeRanges(unittest.TestCase):
    def test_the_marks_of_the_kernel(self):
        marks = readers.life_ranges(text(".life"))[NAME]
        first = marks[0x0]                                                 # LDC R1, c[0x0][0x28]: the stack pointer
        self.assertEqual((first.reads, first.writes), (frozenset(), frozenset({"R1"})))
        self.assertEqual({r.rstrip("0123456789") for m in marks.values() for r in m.reads | m.writes} - {"R", "P", "UR"},
                         set())

    def test_a_wide_write_is_every_register_it_writes(self):
        """The marks know what the operands do not: `LDC.64 R4, ...` writes R4 and R5."""
        sass = readers.sass_functions(text(".sass"))[NAME]
        wide = next(i for i in sass if i.opcode == "LDC.64")
        written = readers.life_ranges(text(".life"))[NAME][wide.address].writes
        first = int(wide.operands.split(",")[0].strip()[1:])
        self.assertEqual(written, {f"R{first}", f"R{first + 1}"})

    def test_an_unknown_group_or_an_odd_header_is_refused(self):
        with self.assertRaises(readers.ReaderError):
            readers.life_ranges(listing([(None, "EXIT", (), ())]).replace("# 0 1 2", "# 0 2 1"))
        two = listing([(None, "EXIT", (), ())]).replace(" |\n_Z4", " | # 0 1 |\n_Z4")
        with self.assertRaises(readers.ReaderError):
            readers.life_ranges(two)                                       # two groups, one name

    def test_a_register_past_the_columns_is_refused(self):
        with self.assertRaises(readers.ReaderError):
            readers.life_ranges(listing([(None, "MOV R40, R1", (1,), ())]))


#: THE LOOP THAT LEAVES THROUGH A CALL TO A LABEL, as the toolkit prints one: R2 written, a loop that leaves through
#: `CALL.REL.NOINC` to a label of the function (marked, as every call is, as writing R2..R9), then R2 read; the graph takes the call
#: for one that returns, so no edge reaches the label
LOOP = [(None, "IMAD.MOV.U32 R2, RZ, RZ, 0x1", (), (2,)),
        (".L_x_0", "IADD3 R3, R3, 0x1, RZ", (3,), (3,)),
        (None, "@!P0 CALL.REL.NOINC 0x40", (), tuple(range(2, 10))),
        (None, "BRA 0x10", (), ()),
        (".L_x_1", "STG.E [R4.64], R2", (2, 4, 5), ()),
        (None, "EXIT", (), ())]
LOOP_SASS = "\t\tFunction : _Z4loopPf\n" + "".join(
    f"        /*{16 * k:04x}*/                   {line} ;\n" for k, (_, line, _, _) in enumerate(LOOP)) + \
    "        /*0060*/                   BRA 0x60 ;\n"


def _node(name, offsets):
    rows = "|".join(f"<i{o}>{o:04x}:\\ \\ \\ X\\ ;\\l" for o in offsets)
    return f'"{name}"\n[label="{{{rows}}}"]\n'


LOOP_CFG = ('digraph f {\nsubgraph "cluster__Z4loopPf" {\n' + _node("_Z4loopPf", [0x0]) + _node(".L_x_0", [0x10, 0x20, 0x30])
            + _node(".L_x_1", [0x40, 0x50]) + '"_Z4loopPf":exit0:s -> ".L_x_0":entry:n [style=solid];\n'
            + '".L_x_0":exit0:s -> ".L_x_0":entry:n [style=solid];\n}\n}\n')
LOOP_RES = " Function _Z4loopPf:\n  REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:360\n"


class Pressure(unittest.TestCase):
    def test_values_held_across_a_slow_path_call_count_before_the_call(self):
        """The listing marks the call as writing registers the kernel holds; the call's marks are its subroutine's body's, so the
        thirty-two values held across the division count at the row before it."""
        k = ranges()
        held = flow.pressure(k)
        calls = [i for i in k.instructions if i.base == "CALL" and i.block != model.OUTLINED]
        self.assertTrue(calls, "the kernel's division has no slow-path call: the example no longer has the shape it shows")
        order = [i.address for i in k.instructions]
        before = [order[order.index(c.address) - 1] for c in calls]
        self.assertGreaterEqual(min(vector(held[a]) for a in before), 32)
        body = set().union(*(k.marks[i.address].writes for i in k.instructions if i.block == model.OUTLINED))
        self.assertTrue(all(k.marks[c.address].writes <= body for c in calls))

    def test_the_outlined_body_is_not_counted(self):
        k = ranges()
        self.assertFalse(set(flow.pressure(k)) & {i.address for i in k.instructions if i.block == model.OUTLINED})

    def test_a_value_held_across_a_loop_exit_through_a_call_counts_in_the_loop(self):
        (k,) = model.kernels_from_text(LOOP_SASS, LOOP_CFG, LOOP_RES, life_text=listing(LOOP)).values()
        held = {a: vector(s) for a, s in flow.pressure(k).items()}
        #: in the loop: R2 (held to the store) and R3 (written each trip); R4 and R5 are never written, so never count
        self.assertEqual([held[a] for a in (0x10, 0x20, 0x30, 0x40)], [2, 2, 2, 1], held)
        self.assertIn(".L_x_1", k.block(".L_x_0").successors)
        self.assertEqual(k.marks[0x20], readers.Marks(frozenset(), frozenset()))

    def test_without_marks_the_registers_are_the_operands(self):
        (k,) = model.kernels_from_text(LOOP_SASS, LOOP_CFG, LOOP_RES).values()
        ins = next(i for i in k.instructions if i.address == 0x40)
        self.assertEqual(k.registers_of(ins), (ins.reads, ins.writes))


class OpcodeTable(unittest.TestCase):
    def test_the_one_table(self):
        self.assertEqual([model.instruction_class(op) for op in
                          ("HGMMA.64x16x16.F32.BF16", "QGMMA.64x8x32.F32.E4M3.E4M3", "HMMA.16816.F32.BF16", "STSM.16.M88.4",
                           "STS.128", "UTMALDG.2D", "LDGDEPBAR", "REDUX.SUM", "UIADD3", "F2FP.BF16.F32.PACK_AB", "NANOSLEEP")],
                         ["tensor", "tensor", "tensor", "shared store", "shared store", "global", "control", "shuffle",
                          "integer/address", "pack", "other"])
        self.assertEqual(model.Instruction(0, "", "ULDC.64", "UR4, c[0x0][0x208]", "b").klass, "integer/address")


def _elf(regcount: int) -> bytes:
    """A cubin's bones: an ELF64 header, a section-name table and a `.nv.info` of one EIATTR_REGCOUNT attribute."""
    names = b"\0.shstrtab\0.nv.info\0"
    info = struct.pack("<BBHII", runner.EIFMT_SVAL, runner.EIATTR_REGCOUNT, 8, 1, regcount)
    shoff = 64 + len(names) + len(info)
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HHH", header, 0x3A, 64, 3, 1)
    sections = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", 11, 0x70000000, 0, 0, 64 + len(names), len(info), 0, 0, 4, 0)]
    return bytes(header) + names + info + b"".join(sections)


class Cubins(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="warpline_cubins_"))
        tools = self.dir / "toolkit" / "bin"
        tools.mkdir(parents=True)
        #: a stand-in `cuobjdump -xelf all BINARY`: two images an architecture each, and a count of its runs
        stub = tools / "cuobjdump"
        stub.write_text(f"#!{sys.executable}\nimport pathlib, sys\n"
                        f"runs = pathlib.Path({str(self.dir / 'runs')!r})\n"
                        "runs.write_text(str(int(runs.read_text()) + 1) if runs.exists() else '1')\n"
                        "for arch in ('sm_80', 'sm_90a'):\n"
                        "    pathlib.Path(f'lib.1.{arch}.cubin').write_bytes(pathlib.Path(sys.argv[-1]).read_bytes())\n")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
        self.toolkit = str(self.dir / "toolkit")
        self.binary = self.dir / "lib.so"
        self.binary.write_bytes(b"one build")

    def test_a_build_is_extracted_once_and_named_by_its_content(self):
        cache = self.dir / "cache"
        first = runner.cubins(self.binary, cache, self.toolkit)
        again = runner.cubins(self.binary, cache, self.toolkit)
        self.assertEqual(first, again)
        self.assertEqual((self.dir / "runs").read_text(), "1")
        self.binary.write_bytes(b"another build")
        self.assertNotEqual(runner.cubins(self.binary, cache, self.toolkit)[0].parent, first[0].parent)
        self.assertEqual(len([d for d in cache.iterdir() if not d.name.startswith(".")]), 2)

    def test_the_image_the_device_ran_is_one(self):
        cache = self.dir / "cache"
        self.assertEqual(runner.cubin(self.binary, cache, "lib", "sm_90a", self.toolkit).name, "lib.1.sm_90a.cubin")
        with self.assertRaises(runner.ToolError):
            runner.cubin(self.binary, cache, "lib", "sm_", self.toolkit)               # two images: refused, never the first
        self.assertEqual(runner.cubins(self.dir / "x.cubin", cache), [self.dir / "x.cubin"])

    def test_a_function_at_255_registers_is_listed_from_a_copy_at_254(self):
        at_255, at_200 = self.dir / "k255.cubin", self.dir / "k200.cubin"
        at_255.write_bytes(_elf(255))
        at_200.write_bytes(_elf(200))
        self.assertEqual(runner.listable(at_200, self.dir / "scratch"), at_200)
        copy = runner.listable(at_255, self.dir / "scratch")
        self.assertNotEqual(copy, at_255)
        self.assertEqual(copy.read_bytes(), _elf(runner.MAX_LISTED))
        self.assertEqual(at_255.read_bytes(), _elf(255))                                # the cubin itself is untouched


if __name__ == "__main__":
    unittest.main()
