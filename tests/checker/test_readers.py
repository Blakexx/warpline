"""The checker's readers: each on a fixture of a vendor tool's text, no toolkit."""
import unittest
from pathlib import Path

from warpline.checker import readers

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = FIXTURES / "example"
SYNTH = FIXTURES / "rules" / "step_rolled" / "nested"
NAME = "_ZN7example4mainEPfPjii"


class Sass(unittest.TestCase):
    def test_a_function_from_the_toolkit(self):
        fns = readers.sass_functions((EXAMPLE / "kernel.sass").read_text())
        self.assertEqual(list(fns), [NAME])
        body = fns[NAME]
        self.assertEqual((body[0].address, body[0].opcode, body[0].operands), (0, "LDC", "R1, c[0x0][0x28]"))
        self.assertEqual(body[1].opcode, "S2R")
        branch = next(i for i in body if i.address == 0xb0)
        self.assertEqual((branch.predicate, branch.opcode, branch.operands), ("@!P0", "BRA", "0x160"))
        steps = [i for i in body if i.opcode.startswith("HGMMA")]
        self.assertEqual(len(steps), 5)
        self.assertTrue(steps[0].opcode.startswith("HGMMA.64x16x16.F32.BF16"))
        self.assertEqual([i.address for i in body], sorted(i.address for i in body))

    def test_encoding_lines_and_both_spellings(self):
        text = ("\t\tFunction : f\n"
                "        /*0000*/                   LDC R1, c[0x0][0x28] ;       /* 0x00000a00ff017b82 */\n"
                "                                                              /* 0x000fe20000000800 */\n"
                "        /*0010*/               @!P0 EXIT ;                      /* 0x000000000000794d */\n"
                "                                                              /* 0x000fea0003800000 */\n")
        body = readers.sass_functions(text)["f"]
        self.assertEqual([(i.address, i.predicate, i.opcode, i.operands) for i in body],
                         [(0, "", "LDC", "R1, c[0x0][0x28]"), (16, "@!P0", "EXIT", "")])

    def test_a_function_twice_is_refused(self):
        with self.assertRaises(readers.ReaderError):
            readers.sass_functions("Function : f\nFunction : f\n")


class Cfg(unittest.TestCase):
    def test_the_toolkits_graph(self):
        cfg = readers.cfg_functions((EXAMPLE / "kernel.cfg.dot").read_text())[NAME]
        self.assertEqual(cfg.entry, NAME)
        self.assertEqual(len(cfg.blocks), 19)
        self.assertEqual(len(cfg.edges), 30)
        first = cfg.blocks[0]
        self.assertEqual(first.id, NAME)
        self.assertEqual(first.offsets, tuple(range(0, 0xb1, 0x10)))
        self.assertEqual(cfg.blocks[1].offsets[0], 0xc0)
        self.assertEqual({b for a, b in cfg.edges if a == NAME}, {".L_x_0", "BB0"})

    def test_the_offsets_are_the_listings_addresses(self):
        cfg = readers.cfg_functions((EXAMPLE / "kernel.cfg.dot").read_text())[NAME]
        sass = readers.sass_functions((EXAMPLE / "kernel.sass").read_text())[NAME]
        owned = sorted(o for b in cfg.blocks for o in b.offsets)
        self.assertEqual(owned, [i.address for i in sass if i.address <= owned[-1]])
        self.assertGreater(len(sass), len(owned))                   # the listing's padding after the last block

    def test_a_small_graph(self):
        cfg = readers.cfg_functions((SYNTH / "kernel.cfg.dot").read_text())["_ZN6sample4tileEv"]
        self.assertEqual([(b.id, b.offsets) for b in cfg.blocks][1:3], [("BB1", (0x20,)), ("BB2", (0x30, 0x40, 0x50, 0x60, 0x70))])
        self.assertIn(("BB2", "BB2"), cfg.edges)

    def test_an_escaped_bar_in_a_label_is_not_a_row_break(self):
        text = ('digraph f {\nsubgraph "cluster_f" {\n"f"\n[label="{<entry>0000:\\ \\ \\ FADD\\ R0,\\ \\|R1\\|,\\ R2\\ ;\\l'
                '|<exit0>0010:\\ \\ \\ EXIT\\ ;\\l}"]\n}\n}\n')
        self.assertEqual(readers.cfg_functions(text)["f"].blocks[0].offsets, (0, 16))


class Usage(unittest.TestCase):
    def test_the_toolkits_report(self):
        usage = readers.res_usage((EXAMPLE / "kernel.res").read_text())
        self.assertEqual(usage, {NAME: readers.Usage(34, 0, 3088)})

    def test_a_stack_frame_with_local_zero(self):
        text = ("Resource usage:\n Common:\n  GLOBAL:0\n Function _Z5spillPfPKf:\n"
                "  REG:24 STACK:352 SHARED:0 LOCAL:0 CONSTANT[0]:544 TEXTURE:0 SURFACE:0 SAMPLER:0\n")
        self.assertEqual(readers.res_usage(text)["_Z5spillPfPKf"].stack, 352)


class Names(unittest.TestCase):
    def test_demangled(self):
        self.assertEqual(readers.demangled("a<1>\nb\n", ["_Za", "_Zb"]), {"_Za": "a<1>", "_Zb": "b"})
        with self.assertRaises(readers.ReaderError):
            readers.demangled("a\n", ["_Za", "_Zb"])


if __name__ == "__main__":
    unittest.main()
