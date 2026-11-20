"""The kernel model, and the natural-loop finder proved on the example kernel's real toolkit output."""
import unittest
from pathlib import Path

from warpline.checker import loops, model, readers, runner

FIXTURES = Path(__file__).parent / "checker_fixtures"
EXAMPLE = FIXTURES / "example" / "kernel"
NAME = "_ZN7example4mainEPfPjii"


def synth(rule, name):
    return next(iter(runner.kernels_of_files(str(FIXTURES / "rules" / rule / name / "kernel")).values()))


class Loops(unittest.TestCase):
    def test_a_back_edge_needs_a_dominating_target(self):
        # entry -> a -> b -> a (a loop), b -> c; c -> b' stub return: x jumps back to a lower block but a does not dominate it
        edges = [("e", "a"), ("a", "b"), ("b", "a"), ("b", "c"), ("e", "s"), ("s", "c"), ("c", "s2"), ("s2", "a")]
        found = {lp.header: lp for lp in loops.natural_loops("e", edges)}
        self.assertEqual(set(found), {"a"})
        self.assertEqual(found["a"].blocks, frozenset({"a", "b"}))

    def test_two_back_edges_to_one_header_are_one_loop(self):
        edges = [("e", "h"), ("h", "x"), ("x", "h"), ("h", "y"), ("y", "h"), ("h", "z")]
        (lp,) = loops.natural_loops("e", edges)
        self.assertEqual((lp.header, lp.blocks, sorted(lp.tails)), ("h", frozenset({"h", "x", "y"}), ["x", "y"]))

    def test_nested_loops_and_a_self_loop(self):
        edges = [("e", "o"), ("o", "i"), ("i", "i"), ("i", "o"), ("o", "x")]
        found = {lp.header: lp.blocks for lp in loops.natural_loops("e", edges)}
        self.assertEqual(found, {"o": frozenset({"o", "i"}), "i": frozenset({"i"})})

    def test_an_unreachable_block_closes_no_loop(self):
        self.assertEqual(loops.natural_loops("e", [("e", "a"), ("u", "v"), ("v", "u")]), [])


class ExampleGraph(unittest.TestCase):
    """The example kernel (`examples/checker/steps.cu`) as the toolkit compiled it: 19 blocks, 30 edges. 8 edges jump
    back in address; 6 are natural back edges, of which 4 are loops that do work (the counted loop, the loop around
    the steps and their wait, the rolled loop around steps without their wait, and the long loop the waits sit in)
    and 2 are one-block spins on a barrier try-wait; the other 2 are the out-of-line stubs' returns into the body,
    which close no loop."""

    @classmethod
    def setUpClass(cls):
        cls.kernel = runner.kernels_of_files(str(EXAMPLE))[NAME]
        cls.cfg = readers.cfg_functions(Path(str(EXAMPLE) + ".cfg.dot").read_text())[NAME]

    def backward(self):
        extent = {b.id: (b.start, b.end) for b in self.kernel.blocks}
        return [(a, b) for a, b in self.cfg.edges if extent[b][0] <= extent[a][1]]

    def test_the_counts(self):
        self.assertEqual((len(self.kernel.blocks), len(self.cfg.edges)), (19, 30))
        self.assertEqual(len(self.backward()), 8)
        self.assertEqual(len(self.kernel.loops), 6)
        self.assertEqual(sum(lp.spin for lp in self.kernel.loops), 2)
        self.assertEqual(sum(not lp.spin for lp in self.kernel.loops), 4)

    def test_the_stub_returns_are_not_loops(self):
        closed = {(t, lp.header) for lp in self.kernel.loops for t in lp.tails}
        stubs = [(a, b) for a, b in self.backward() if (a, b) not in closed]
        self.assertEqual(len(stubs), 2)
        start = {b.id: b.start for b in self.kernel.blocks}
        self.assertEqual(sorted(start[a] for a, _ in stubs), [0x8a0, 0x900])   # the stubs, after the kernel's EXIT
        exit_at = max(i.address for i in self.kernel.instructions if i.opcode == "EXIT")
        self.assertTrue(all(start[a] > exit_at for a, _ in stubs))
        # a stub's return jumps into the body, whose blocks are in no loop because of it
        in_loops = {bid for lp in self.kernel.loops for bid in lp.blocks}
        self.assertLess(len(in_loops), len(self.kernel.blocks))

    def test_the_four_loops_that_work(self):
        work = [(lp.address, len(lp.blocks)) for lp in self.kernel.loops if not lp.spin]
        # (a) the counted loop; the long loop of the rounds (13 blocks); (b) steps with their wait; (c) rolled steps
        self.assertEqual(work, [(0xf0, 1), (0x230, 13), (0x2f0, 1), (0x490, 1)])

    def test_the_spins_are_one_block_try_waits(self):
        spins = [lp for lp in self.kernel.loops if lp.spin]
        self.assertEqual([(lp.address, len(lp.blocks)) for lp in spins], [(0x850, 1), (0x8b0, 1)])
        for lp in spins:
            ops = [i.opcode for i in self.kernel.block(lp.header).instructions]
            # the stub's loop: the try-wait, the toolkit's own sleep, the retry of the wait, and the branch back
            self.assertEqual(ops[1:], ["SYNCS.PHASECHK.TRANS64.TRYWAIT", "NANOSLEEP.SYNCS", "SYNCS.PHASECHK.TRANS64", "BRA"])

    def test_the_try_waits_are_two_in_the_body_and_one_in_each_stub(self):
        waits = [i for i in self.kernel.barriers if i.opcode == "SYNCS.PHASECHK.TRANS64.TRYWAIT"]
        self.assertEqual(len(waits), 4)       # two in the body, one in each stub

    def test_no_convergence_instruction_in_this_kernel(self):
        self.assertEqual(self.kernel.convergence, ())

    def test_innermost_loop_of_an_instruction(self):
        by_address = {s.instruction.address: s for s in self.kernel.steps}
        self.assertEqual(self.kernel.loop_of(by_address[0x430].instruction).address, 0x2f0)   # (b), not the rounds loop
        self.assertEqual(self.kernel.loop_of(by_address[0x570].instruction).address, 0x490)   # (c)
        self.assertEqual(self.kernel.loop_of(by_address[0x670].instruction).address, 0x230)   # the bare wait's step
        self.assertIsNone(self.kernel.loop_of(self.kernel.instructions[0]))


class ExampleModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = runner.kernels_of_files(str(EXAMPLE))[NAME]

    def test_fields(self):
        k = self.kernel
        self.assertEqual(k.demangled, "example::main(float*, unsigned int*, int, int)")
        self.assertEqual((k.registers, k.stack_bytes, k.shared_bytes), (34, 0, 3088))
        self.assertEqual(len(k.instructions), 145)
        self.assertEqual(len(k.steps), 5)
        self.assertEqual(sum(s.commit for s in k.steps), 4)
        self.assertEqual([w.left for w in k.waits], [0, 0, 0, 0])       # (b), (c), bare, with work: four waits
        self.assertEqual(len(k.arrives), 5)

    def test_every_instruction_is_in_its_block(self):
        k = self.kernel
        for block in k.blocks:
            self.assertTrue(all(i.block == block.id for i in block.instructions))
        self.assertEqual(sum(len(b.instructions) for b in k.blocks), len(k.instructions))


class Classes(unittest.TestCase):
    def test_accesses_by_kind(self):
        k = synth("stack_memory", "spill")
        self.assertEqual([(a.space, a.kind, a.width) for a in k.accesses],
                         [("local", "store", 32), ("local", "load", 32), ("local", "store", 64)])
        k = synth("wait_after_issue", "work_between")
        self.assertEqual([(a.space, a.kind) for a in k.accesses], [("shared", "load"), ("shared", "store")])

    def test_a_graph_that_is_not_the_listings_is_refused(self):
        sass = readers.sass_functions((FIXTURES / "rules/step_rolled/straight/kernel.sass").read_text())
        cfg = readers.cfg_functions((FIXTURES / "rules/step_rolled/rolled/kernel.cfg.dot").read_text())
        (name, body), = sass.items()
        with self.assertRaises(model.ModelError):
            model.build_kernel(name, body, cfg[name], readers.Usage(8, 0, 0))


if __name__ == "__main__":
    unittest.main()
