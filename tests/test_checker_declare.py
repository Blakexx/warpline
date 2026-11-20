"""Declarations, the refusals, and the command on saved tool outputs."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline.__main__ import main
from warpline.checker import declare, runner
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "checker_fixtures" / "rules"


def kernels(rule, name):
    return list(runner.kernels_of_files(str(FIXTURES / rule / name / "kernel")).values())


def decls(*entries):
    return declare.load_declarations(json.dumps({"declarations": list(entries)}), registry())


def entry(kernel="tile", rule="stack_memory", allow=128, reason="measured and accepted"):
    return {"kernel": kernel, "rule": rule, "allow": allow, "reason": reason}


class Declarations(unittest.TestCase):
    def test_undeclared_means_zero(self):
        (refusal,) = declare.check(kernels("stack_memory", "spill"), registry(), [])
        self.assertEqual((refusal.rule, refusal.count, refusal.allowed, refusal.reason), ("stack_memory", 128, 0, None))
        self.assertEqual(declare.check(kernels("stack_memory", "no_stack"), registry(), []), [])

    def test_a_declared_count_allows_up_to_it(self):
        self.assertEqual(declare.check(kernels("stack_memory", "spill"), registry(), decls(entry(allow=128))), [])
        (refusal,) = declare.check(kernels("stack_memory", "spill"), registry(), decls(entry(allow=64)))
        self.assertEqual((refusal.count, refusal.allowed, refusal.reason), (128, 64, "measured and accepted"))

    def test_the_regex_is_over_the_demangled_name(self):
        k = kernels("stack_memory", "spill")
        self.assertEqual(k[0].demangled, "sample::tile()")
        self.assertEqual(declare.check(k, registry(), decls(entry(kernel=r"^sample::tile\(\)$"))), [])
        self.assertEqual(len(declare.check(k, registry(), decls(entry(kernel="_ZN6sample")))), 1)   # not the mangled name
        self.assertEqual(len(declare.check(k, registry(), decls(entry(kernel="^other")))), 1)

    def test_a_declaration_is_for_its_rule(self):
        k = kernels("wait_after_issue", "bare_wait")
        (refusal,) = declare.check(k, registry(), decls(entry(rule="stack_memory")))
        self.assertEqual(refusal.rule, "wait_after_issue")
        self.assertEqual(declare.check(k, registry(), decls(entry(rule="wait_after_issue", allow=1))), [])

    def test_a_reason_is_required(self):
        for reason in ("", "  ", None):
            with self.assertRaises(declare.DeclarationError):
                decls(entry(reason=reason))
        with self.assertRaises(declare.DeclarationError):
            decls({"kernel": "k", "rule": "stack_memory", "allow": 1})

    def test_bad_declarations(self):
        for bad in (entry(rule="no_such_rule"), entry(allow=-1), entry(allow="1"), entry(kernel="(")):
            with self.assertRaises(declare.DeclarationError, msg=str(bad)):
                decls(bad)
        with self.assertRaises(declare.DeclarationError):
            declare.load_declarations("[]", registry())

    def test_a_stale_declaration_is_listed(self):
        ds = decls(entry(), entry(kernel="^gone$"))
        self.assertEqual([d.kernel.pattern for d in declare.unmatched(ds, kernels("stack_memory", "spill"))], ["^gone$"])

    def test_the_refusal_says_where_and_how_to_fix(self):
        (refusal,) = declare.check(kernels("step_rolled", "rolled"), registry(), [])
        text = "\n".join(refusal.lines())
        self.assertIn("sample::tile(): step_rolled: 4 found, 0 declared", text)
        self.assertIn("at 0x0020 x4", text)
        self.assertIn("fix:", text)


class Command(unittest.TestCase):
    def run_check(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", *argv])
        return code, out.getvalue(), err.getvalue()

    def stem(self, rule, name):
        return str(FIXTURES / rule / name / "kernel")

    def write(self, text):
        d = tempfile.mkdtemp()
        path = Path(d) / "declare.json"
        path.write_text(text)
        return str(path)

    def test_a_refusal_exits_1(self):
        empty = self.write('{"declarations": []}')
        code, out, _ = self.run_check(self.stem("stack_memory", "spill"), "--fixture", "--declare", empty)
        self.assertEqual(code, 1)
        self.assertIn("stack_memory: 128 found, 0 declared", out)
        self.assertIn("1 refusal(s)", out)

    def test_a_declared_allowance_exits_0(self):
        ok = self.write(json.dumps({"declarations": [entry()]}))
        code, out, _ = self.run_check(self.stem("stack_memory", "spill"), "--fixture", "--declare", ok)
        self.assertEqual((code, out.splitlines()[-1]), (0, "1 kernel(s), 3 rule(s): 0 refusal(s)"))

    def test_a_clean_kernel_exits_0(self):
        code, out, _ = self.run_check(self.stem("stack_memory", "no_stack"), "--fixture", "--declare",
                                      self.write('{"declarations": []}'))
        self.assertEqual(code, 0)

    def test_an_unusable_declaration_exits_2(self):
        code, _, err = self.run_check(self.stem("stack_memory", "spill"), "--fixture", "--declare",
                                      self.write(json.dumps({"declarations": [entry(reason="")]})))
        self.assertEqual(code, 2)
        self.assertIn("a reason is required", err)

    def test_a_missing_toolkit_exits_2(self):
        code, _, err = self.run_check("no_such_binary", "--declare", self.write('{"declarations": []}'),
                                      "--toolkit", "/no/such/dir")
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("warpline: "))


if __name__ == "__main__":
    unittest.main()
