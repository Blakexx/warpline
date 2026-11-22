"""Synchronous tensor steps (`mma.sync`, `HMMA`) are steps of kind `sync`, on the toolkit's own output for
`examples/checker/mma.cu` (`sm_80`, saved in `tests/fixtures/mma/`): each step rule fires or stays quiet as its
statement says, and the rules about async commit groups do not read them."""
import unittest
from pathlib import Path

from warpline.checker import model, runner
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures"
R = registry()
#: the rules that read steps, with the kinds each reads
ASYNC_ONLY = ("step_rolled", "wait_after_issue", "store_unfenced", "step_port")
BOTH = ("step_branches", "step_predicated", "step_conversion", "step_global", "step_gap", "step_work", "step_density")


def kernels():
    return {k.demangled.split("(")[0]: k for k in runner.kernels_of_files(str(FIXTURES / "mma" / "kernel")).values()}


def counts(kernel, names):
    return {n: R[n].count(kernel) for n in names}


class SyncSteps(unittest.TestCase):
    def setUp(self):
        self.k = kernels()

    def test_every_step_is_sync_with_its_shape_no_commit_and_no_descriptor_bytes(self):
        for k in self.k.values():
            self.assertTrue(k.steps)
            for s in k.steps:
                self.assertEqual((s.kind, s.commit, s.null, s.shape, s.operand_bytes), ("sync", False, False, (16, 8, 16), 0))
            self.assertEqual((k.async_steps, k.waits), ((), ()))

    def test_the_shapes_of_the_sync_opcodes(self):
        for opcode, shape in (("HMMA.16816.F32.BF16", (16, 8, 16)), ("HMMA.1688.F32.TF32", (16, 8, 8)),
                              ("IMMA.16832.S8.S8", (16, 8, 32)), ("IMMA.8816.S8.S8", (8, 8, 16)), ("DMMA.884", (8, 8, 4))):
            s = model._step(model.Instruction(0x10, "", opcode, "R4, R8, R12, R4", "BB0"))
            self.assertEqual((s.kind, s.shape), ("sync", shape), opcode)

    def test_regions_cover_sync_steps(self):
        self.assertEqual([(r.kind, r.steps) for r in self.k["mma_loop"].regions], [("loop", 2)])
        self.assertEqual([(r.kind, r.steps) for r in self.k["mma_branch"].regions], [("loop", 2)])
        self.assertEqual([(r.kind, r.steps) for r in self.k["mma_predicated"].regions], [("unrolled", 1), ("unrolled", 1)])

    def test_the_rules_about_commit_groups_and_descriptors_do_not_read_them(self):
        for name, k in self.k.items():
            self.assertEqual(counts(k, ASYNC_ONLY), dict.fromkeys(ASYNC_ONLY, 0), name)

    def test_the_straight_loop_of_steps_is_quiet_where_it_has_nothing_to_flag(self):
        c = counts(self.k["mma_loop"], BOTH)
        for quiet in ("step_branches", "step_predicated", "step_conversion", "step_global"):
            self.assertEqual(c[quiet], 0, quiet)
        self.assertGreater(c["step_gap"], 0)
        self.assertGreater(c["step_work"], 0)
        self.assertGreater(c["step_density"], 0)

    def test_the_branch_the_conversion_and_the_global_accesses_in_the_loop_fire(self):
        k = self.k["mma_branch"]
        (b,) = R["step_branches"].check(k)
        self.assertIn("@P0 BRA", b.what)
        (c,) = R["step_conversion"].check(k)
        self.assertIn("F2I", c.what)
        self.assertEqual(R["step_global"].count(k), 7)             # six operand loads and the store
        self.assertEqual(R["step_predicated"].count(k), 0)         # the toolkit kept a branch here, not a predicate
        self.assertGreater(R["step_gap"].count(k), 0)

    def test_a_predicated_step_fires_and_only_that_one(self):
        k = self.k["mma_predicated"]
        (f,) = R["step_predicated"].check(k)
        self.assertIn("@!P0 HMMA", f.what)
        for name in ("step_branches", "step_conversion", "step_global", "step_gap"):
            self.assertEqual(R[name].count(k), 0, name)

    def test_figures_are_read_over_sync_steps(self):
        k = self.k["mma_loop"]
        self.assertEqual(R["step_density"].count(k), -(-sum(not i.dead for i in k.instructions) // 2))
        (g,) = R["step_gap"].check(k)
        self.assertIn("2 between steps", g.what)

    def test_an_async_kernel_keeps_its_steps_and_reads(self):
        (k,) = [k for k in runner.kernels_of_files(str(FIXTURES / "example" / "kernel")).values()]
        self.assertEqual(len(k.async_steps), len(k.steps))
        self.assertTrue(all(s.kind == "async" for s in k.steps))


if __name__ == "__main__":
    unittest.main()
