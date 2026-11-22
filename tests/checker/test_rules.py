"""The rule interface and the three rules: every registered rule on every one of its fixtures, found by the registry,
so a rule added to `warpline/checker/rules/` is tested by writing its fixtures and nothing else."""
import unittest
from pathlib import Path

from warpline.checker import rule, runner
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures"
STAGE_ONE = ("stack_memory", "step_rolled", "wait_after_issue")


def kernel_of(path):
    (k,) = runner.kernels_of_files(str(path / "kernel")).values()
    return k


class Registry(unittest.TestCase):
    def test_the_stage_one_rules(self):
        self.assertLessEqual({"stack_memory", "step_rolled", "wait_after_issue"}, set(registry()))

    def test_every_rule_is_complete(self):
        for r in registry().values():
            self.assertTrue(r.statement and r.refusal and r.fix, r.name)
            self.assertEqual(r.statement.count("\n"), 0, r.name)

    def test_a_rule_needs_a_fixture_that_fires_and_one_that_does_not(self):
        base = dict(name="x", statement="s", check=lambda k: [], refusal="r", fix="f")
        for fixtures in ((), (rule.Fixture("a", 1),), (rule.Fixture("a", 0), rule.Fixture("b", 0)),
                         (rule.Fixture("a", 1), rule.Fixture("b", 2))):
            with self.assertRaises(ValueError, msg=str(fixtures)):
                rule.Rule(fixtures=fixtures, **base)
        rule.Rule(fixtures=(rule.Fixture("a", 1), rule.Fixture("b", 0)), **base)


class Fixtures(unittest.TestCase):
    def test_each_rule_on_each_fixture(self):
        for r in registry().values():
            for fx in r.fixtures:
                with self.subTest(rule=r.name, fixture=fx.name):
                    path = FIXTURES / "rules" / r.name / fx.name
                    self.assertTrue(path.is_dir(), f"missing fixture directory {path}")
                    self.assertEqual(r.count(kernel_of(path)), fx.count)

    def test_every_fixture_directory_is_declared_by_its_rule(self):
        for r in registry().values():
            on_disk = sorted(p.name for p in (FIXTURES / "rules" / r.name).iterdir() if p.is_dir())
            self.assertEqual(on_disk, sorted(f.name for f in r.fixtures), r.name)


class Details(unittest.TestCase):
    def test_step_rolled_names_the_loop(self):
        (f,) = registry()["step_rolled"].check(kernel_of(FIXTURES / "rules/step_rolled/rolled"))
        self.assertEqual((f.where, f.count), ("0x0020", 4))
        self.assertIn("no tensor wait", f.what)

    def test_step_rolled_flags_only_the_inner_loop_without_the_wait(self):
        (f,) = registry()["step_rolled"].check(kernel_of(FIXTURES / "rules/step_rolled/nested"))
        self.assertEqual(f.where, "0x0030")

    def test_wait_after_issue_names_the_wait_and_the_step(self):
        (f,) = registry()["wait_after_issue"].check(kernel_of(FIXTURES / "rules/wait_after_issue/bare_wait"))
        self.assertIn("0x0050", f.what)
        self.assertIn("0x0040", f.what)

    def test_step_serialized_names_each_run_and_its_waits(self):
        found = registry()["step_serialized"].check(kernel_of(FIXTURES / "rules/step_serialized/serialized"))
        self.assertEqual([f.where for f in found], ["0x05e0", "0x0cc0"])        # the two blocks of eight steps, a run each
        self.assertTrue(all("8 tensor steps" in f.what and "0x0780" in f.what for f in found[:1]))

    def test_step_serialized_counts_the_run_before_the_shape_switch_only(self):
        (f,) = registry()["step_serialized"].check(kernel_of(FIXTURES / "rules/step_serialized/mixed_shapes"))
        self.assertIn("4 tensor steps", f.what)                                   # the lone m64n64k16 step is no run

    def test_accumulator_moved_names_the_block_and_the_registers(self):
        (f,) = registry()["accumulator_moved"].check(kernel_of(FIXTURES / "rules/accumulator_moved/mixed_shapes"))
        self.assertEqual((f.where, f.count), ("0x05c0", 64))
        self.assertIn("R24 to R59", f.what)

    def test_accumulator_moved_sees_no_move_in_one_shape_and_none_in_a_zeroing(self):
        # `serialized` zeroes the accumulators before its loop (CS2R, no register read) and one shape moves nothing
        for name in ("serialized", "fenced"):
            self.assertEqual(registry()["accumulator_moved"].count(kernel_of(FIXTURES / "rules/accumulator_moved" / name)), 0)

    def test_wait_all_with_work_names_the_wait_the_step_and_the_group(self):
        (f,) = registry()["wait_all_with_work"].check(kernel_of(FIXTURES / "rules/wait_all_with_work/wait_all_work"))
        self.assertIn("0x0420", f.what)
        self.assertIn("0x04d0", f.what)
        self.assertIn("R24 to R31", f.what)

    def test_wait_all_with_work_spares_a_wait_whose_path_reads_the_group(self):
        # the last wait of both kernels is followed by the stores of the accumulators; in `wait_all_reads` the first one's work reads it too
        self.assertEqual(registry()["wait_all_with_work"].count(kernel_of(FIXTURES / "rules/wait_all_with_work/wait_all_reads")), 0)

    def test_stack_memory_carries_the_instructions(self):
        (f,) = registry()["stack_memory"].check(kernel_of(FIXTURES / "rules/stack_memory/spill"))
        self.assertEqual(f.count, 128)
        self.assertIn("1 LDL and 2 STL", f.what)

    def test_the_example_kernel_the_toolkit_compiled(self):
        k = kernel_of(FIXTURES / "example")
        self.assertEqual({n: r.count(k) for n, r in registry().items() if n in STAGE_ONE},
                         {"stack_memory": 0, "step_rolled": 1, "wait_after_issue": 2})

    def test_step_rolled_fires_on_the_example_rolled_loop_only(self):
        (f,) = registry()["step_rolled"].check(kernel_of(FIXTURES / "example"))
        self.assertEqual((f.where, f.count), ("0x0490", 1))       # (c); the loop that holds its wait (b) is not flagged

    def test_wait_after_issue_spares_the_example_wait_with_work_between(self):
        found = registry()["wait_after_issue"].check(kernel_of(FIXTURES / "example"))
        self.assertEqual([f.where for f in found], ["0x0440", "0x06d0"])    # the two bare waits; not 0x0750 (loads and stores)

if __name__ == "__main__":
    unittest.main()
