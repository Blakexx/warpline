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
