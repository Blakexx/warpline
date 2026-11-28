"""THE COMPILER'S EXPANSIONS (docs/checker-rules.md#compiler-expansions): on the toolkit's own output of
`examples/checker/expansions/`, each kind of expansion is found by its rule alone, at its source line; refused undeclared;
passing when a declaration opts it in at its site, and only there; and the clean forms (a shift, a multiply-high, a 32-bit
index, the fast intrinsics) are flagged by no rule."""
import json
import re
import unittest
from pathlib import Path

from warpline import checkers, declare
from warpline.checker import manifest, runner
from warpline.checker.rules import registry

RULES = Path(__file__).parent / "fixtures" / "rules"
EXPANSIONS = {name: r for name, r in registry().items() if name.startswith("expansion_")}
#: each kernel that expands, the rule that names it, and its findings' sites
FIRES = {
    "int_division": ("expansion_integer_division", ["examples/checker/expansions/int_division.cu:5"] * 2),
    "wide_integer": ("expansion_wide_integer", ["examples/checker/expansions/wide_integer.cu:4",
                                                "examples/checker/expansions/wide_integer.cu:5"]),
    "float_division": ("expansion_float_division", ["examples/checker/expansions/float_division.cu:4"]),
    "float_sqrt": ("expansion_float_sqrt", ["examples/checker/expansions/float_sqrt.cu:4"]),
    "float_reciprocal": ("expansion_float_reciprocal", ["examples/checker/expansions/float_reciprocal.cu:4"]),
    "transcendental": ("expansion_transcendental", ["examples/checker/expansions/transcendental.cu:5"] * 3),
    "slow_call": ("expansion_slow_call", ["examples/checker/expansions/slow_call.cu:6"]),
}
CLEAN = {"int_clean": "expansion_integer_division", "wide_clean": "expansion_wide_integer",
         "float_clean": "expansion_float_division"}


def kernel(fixture, rule):
    (k,) = runner.kernels_of_files(str(RULES / rule / fixture / "kernel")).values()
    return k


def refusals(k, entries):
    text = json.dumps({"declarations": entries})
    m = manifest.collect([k], EXPANSIONS)
    return declare.compare(m, declare.load_declarations(text, EXPANSIONS)).refusals


class Found(unittest.TestCase):
    def test_each_expansion_by_its_rule_alone_at_its_line(self):
        for fixture, (rule, sites) in FIRES.items():
            with self.subTest(fixture=fixture):
                k = kernel(fixture, rule)
                for name, r in EXPANSIONS.items():
                    found = r.check(k)
                    self.assertEqual([f.where for f in found], sites if name == rule else [], name)

    def test_the_clean_forms_are_never_flagged(self):
        for fixture, rule in CLEAN.items():
            with self.subTest(fixture=fixture):
                k = kernel(fixture, rule)
                self.assertEqual({name: r.check(k) for name, r in EXPANSIONS.items()},
                                 {name: [] for name in EXPANSIONS})

    def test_a_double_is_a_division_and_a_root(self):
        k = kernel("double_division", "expansion_float_division")
        self.assertEqual([len(EXPANSIONS[n].check(k)) for n in ("expansion_float_division", "expansion_float_sqrt")], [1, 1])

    def test_a_finding_names_the_expansion_and_its_fix(self):
        (f,) = EXPANSIONS["expansion_float_division"].check(kernel("float_division", "expansion_float_division"))
        self.assertIn("an IEEE-exact float division at 0x0130: FCHK and its slow path", f.what)
        self.assertIn("__fdividef", EXPANSIONS["expansion_float_division"].fix)
        self.assertIn("multiply-high", EXPANSIONS["expansion_integer_division"].fix)
        self.assertIn("32-bit indices", EXPANSIONS["expansion_wide_integer"].fix)
        self.assertIn("power-of-two divisor", EXPANSIONS["expansion_integer_division"].fix)


class Declared(unittest.TestCase):
    def test_refused_undeclared_and_passing_when_declared_at_its_site(self):
        for fixture, (rule, sites) in FIRES.items():
            with self.subTest(fixture=fixture):
                k = kernel(fixture, rule)
                (r,) = refusals(k, [])
                self.assertEqual((r.rule, r.count, r.allowed), (rule, len(sites), 0))
                opted = {"kernel": f"^{fixture}\\(", "rule": rule, "at": sites[0].replace(".", "\\.") + "$",
                         "allow": sites.count(sites[0]), "reason": "the example's own expansion"}
                left = refusals(k, [opted])
                self.assertEqual([x.count for x in left], [len(sites) - sites.count(sites[0])] if len(set(sites)) > 1 else [])

    def test_a_declaration_at_another_site_allows_nothing(self):
        k = kernel("float_division", "expansion_float_division")
        elsewhere = {"kernel": "float_division", "rule": "expansion_float_division", "at": r"float_division\.cu:99$",
                     "allow": 1, "reason": "another line"}
        (r,) = refusals(k, [elsewhere])
        self.assertEqual((r.count, r.allowed, r.reason), (1, 0, None))

    def test_a_site_declared_short_is_refused_with_its_reason(self):
        k = kernel("int_division", "expansion_integer_division")
        short = {"kernel": "int_division", "rule": "expansion_integer_division", "at": r"int_division\.cu:5$", "allow": 1,
                 "reason": "one division accepted"}
        (r,) = refusals(k, [short])
        self.assertEqual((r.count, r.allowed), (2, 1))
        self.assertIn("one division accepted; at", r.reason)

    def test_strict_refuses_a_site_that_has_none(self):
        k = kernel("float_division", "expansion_float_division")
        m = manifest.collect([k], EXPANSIONS)
        at = lambda site: [declare.Declaration(re.compile("float_division"), "expansion_float_division", 1, "r", re.compile(site))]
        self.assertTrue(checkers.unshown(m, at(r"\.cu:99$")))
        self.assertFalse(checkers.unshown(m, at(r"\.cu:4$")))

    def test_a_site_is_a_regex(self):
        bad = json.dumps({"declarations": [{"kernel": "k", "rule": "expansion_slow_call", "at": "(", "allow": 1, "reason": "r"}]})
        with self.assertRaises(declare.DeclarationError):
            declare.load_declarations(bad, EXPANSIONS)


if __name__ == "__main__":
    unittest.main()
