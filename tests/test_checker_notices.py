"""The notice ledger: the reader of a build's messages, the base ledger, a project's rows over it, the matcher, and the
notices as refusals under the declarations; the note a project keeps per rule."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline.__main__ import main
from warpline.checker import declare, notices

FIXTURES = Path(__file__).parent / "checker_fixtures"
NAMES = {"_ZN6sample4tileEv": "sample::tile()"}


def read(name):
    return notices.read_notices((FIXTURES / "notices" / name).read_text())


def row(**kw):
    base = dict(tool="ptxas", code=None, text="a notice about {}", disposition="report", reason="why", fix="")
    return {**base, **kw}


def ledger(*rows):
    return notices.load_ledger(json.dumps({"rows": list(rows)}))


def decls(*entries, known=()):
    return declare.load_declarations(json.dumps({"declarations": list(entries)}), {}, [*known, "mma_serialized",
                                                                                      "wait_injected"])


def entry(rule="mma_serialized", allow=0, kernel="tile", reason="measured and accepted"):
    return {"kernel": kernel, "rule": rule, "allow": allow, "reason": reason}


class Reader(unittest.TestCase):
    def test_every_kind_of_line(self):
        got = read("mixed.log")
        self.assertEqual([(n.tool, n.code) for n in got],
                         [("nvcc", "#128-D"), ("nvcc", None), ("nvcc", "#549-D"), ("ptxas", "C7519"), ("ptxas", "C7505"),
                          ("ptxas", None), ("ptxas", None), ("ptxas", None), ("ptxas", "C7599"), ("nvlink", None)])

    def test_a_function_is_named_and_the_text_is_the_template(self):
        n = read("serialized.log")[0]
        self.assertEqual((n.tool, n.code, n.function), ("ptxas", "C7511", "_ZN6sample4tileEv"))
        self.assertTrue(n.text.endswith("for the function '{}'"))
        self.assertIn("'_ZN6sample4tileEv'", n.message)

    def test_what_follows_a_notice_belongs_to_it(self):
        got = read("mixed.log")
        self.assertEqual(got[0].where, "tile.cu(354)")           # the directory is dropped
        self.assertIn("loop is not reachable", got[0].raw)
        self.assertIn("for (int half", got[0].raw)
        self.assertNotIn("Remark", got[0].raw)
        self.assertEqual(got[2].raw.count("\n"), 2)              # its three lines, up to the blank line
        summary = [n for n in read("serialized.log") if n.text.startswith("Function properties")]
        self.assertIn("0 bytes stack frame", summary[0].raw)

    def test_a_line_nothing_reads_is_unreadable(self):
        (n,) = notices.read_notices("gcc: something unexpected\n")
        self.assertEqual((n.tool, n.text), ("unreadable", "gcc: something unexpected"))
        self.assertEqual(notices.read_notices("\n\n"), [])

    def test_the_example_kernels_own_output(self):
        got = notices.read_notices((FIXTURES / "example" / "kernel.ptxas").read_text())
        self.assertEqual([n.code for n in got], ["C7519", None, None, None, None, None])
        self.assertEqual({notices.row_of(notices.base_ledger(), n).disposition for n in got}, {"report", "ignore"})


class Schema(unittest.TestCase):
    def test_a_row_needs_its_fields(self):
        for bad in (row(reason=""), row(reason="  "), row(disposition="allow"), row(disposition="declare"),
                    row(rule="r"), row(tool=""), row(text=""), row(code=7), row(extra=1)):
            with self.assertRaises(notices.LedgerError, msg=str(bad)):
                ledger(bad)
        with self.assertRaises(notices.LedgerError):
            ledger({"tool": "ptxas", "text": "x", "disposition": "report"})
        with self.assertRaises(notices.LedgerError):
            notices.load_ledger("[]")
        (r,) = ledger(row(disposition="declare", rule="my_rule"))
        self.assertEqual((r.disposition, r.rule), ("declare", "my_rule"))

    def test_the_base_ledger(self):
        base = notices.base_ledger()
        self.assertEqual(len(base), 28)
        self.assertEqual({r.disposition for r in base}, set(notices.DISPOSITIONS))
        self.assertEqual(notices.declared_rules(base), ["mma_serialized", "wait_injected"])
        self.assertTrue(all(r.reason and (r.fix or r.disposition == "ignore") for r in base))

    def test_two_rows_may_not_match_one_notice(self):
        with self.assertRaises(notices.LedgerError):
            ledger(row(text="a notice"), row(text="a notice"))
        with self.assertRaises(notices.LedgerError):                    # a general row over a specific one
            ledger(row(text="a {} about {}"), row(text="a notice about {}"))
        with self.assertRaises(notices.LedgerError):                    # a row with no code covers a coded row's text
            ledger(row(text="a notice"), row(text="a notice", code="C1"))
        ledger(row(text="a notice", code="C2"), row(text="a notice", code="C1"))     # codes that differ: fine
        ledger(row(text="a notice"), row(text="a notice", tool="nvcc"))              # tools that differ: fine

    def test_a_project_row_overrides_the_base_by_tool_code_and_text(self):
        base = notices.base_ledger()
        spill = next(r for r in base if r.text.startswith("Local memory used"))
        self.assertEqual(spill.disposition, "refuse")
        mine = ledger(row(text=spill.text, disposition="report", reason="mine"), row(text="my notice", tool="nvcc"))
        merged = notices.merge(base, mine)
        self.assertEqual(len(merged), len(base) + 1)
        self.assertEqual(merged.index(next(r for r in merged if r.text == spill.text)), base.index(spill))   # in place
        self.assertEqual(next(r for r in merged if r.text == spill.text).reason, "mine")
        self.assertEqual(merged[-1].text, "my notice")
        # a project row that is not the same key but would match a base row's notices is refused
        with self.assertRaises(notices.LedgerError):
            notices.merge(base, ledger(row(text="Local memory used for function '{}', {}")))

    def test_a_project_may_name_a_new_declared_rule(self):
        merged = notices.merge(notices.base_ledger(), ledger(row(text="mine {}", disposition="declare", rule="mine")))
        self.assertIn("mine", notices.declared_rules(merged))


class Matcher(unittest.TestCase):
    def test_code_and_text_must_both_match(self):
        base = notices.base_ledger()
        n = read("serialized.log")[0]
        self.assertEqual(notices.row_of(base, n).code, "C7511")
        self.assertIsNone(notices.row_of(base, n._replace(code="C7599")))                  # the row names its code
        self.assertIsNone(notices.row_of(base, n._replace(text="some other words")))
        self.assertEqual(notices.row_of(base, n._replace(tool="ptxas", code=None)), None)  # code-bearing rows need theirs

    def test_a_row_with_no_code_matches_any_code(self):
        base = notices.base_ledger()
        n = next(n for n in read("serialized.log") if n.text.startswith("Compiling"))
        self.assertIsNotNone(notices.row_of(base, n._replace(code="C1234")))


class Refusals(unittest.TestCase):
    def judge(self, name, declarations=(), led=None, notes=None):
        return declare.check_notices(read(name), led or notices.base_ledger(), list(declarations), NAMES, notes)

    def test_a_notice_with_no_row_refuses_quoted_whole(self):
        refusals, _ = self.judge("mixed.log")
        unknown = [r for r in refusals if r.rule == "notice_without_row"]
        self.assertEqual(len(unknown), 2)
        text = "\n".join(unknown[0].lines())
        self.assertIn("sample::tile(): notice_without_row: 1 found, 0 declared", text)
        self.assertIn("ptxas info    : (C7599) A notice no ledger knows, about the function '_ZN6sample4tileEv'", text)
        self.assertIn("(nvlink with no code)", "\n".join(unknown[1].lines()))
        self.assertIn("fix: add a row to the project's ledger", text)

    def test_a_refuse_row_refuses_and_names_its_fix(self):
        refusals, _ = self.judge("mixed.log")
        rules = [r.rule for r in refusals]
        for rule in ("notice nvcc #549-D", "notice ptxas C7505", "notice ptxas no_code"):
            self.assertIn(rule, rules)
        (spill,) = [r for r in refusals if "spilled" in " ".join(f.what for f in r.findings)]
        self.assertIn("function 'sample::tile()'", spill.findings[0].what)      # demangled as the build named it
        self.assertTrue(spill.fix)

    def test_report_and_ignore_rows_are_never_refusals(self):
        refusals, reports = self.judge("mixed.log")
        self.assertEqual(len(reports), 2)
        self.assertTrue(any("C7519" in r and "1 x in 1 place(s)" in r for r in reports))
        self.assertTrue(any("loop is not reachable" in r for r in reports))
        _, reports = self.judge("serialized.log")
        self.assertEqual(reports, [])
        self.assertFalse([r for r in self.judge("serialized.log")[0] if r.rule.startswith("notice ")])

    def test_declare_rows_count_under_the_rule_and_the_declarations_govern(self):
        refusals, _ = self.judge("serialized.log")
        by = {r.rule: r for r in refusals}
        self.assertEqual({k: (v.count, v.allowed) for k, v in by.items()}, {"mma_serialized": (3, 0), "wait_injected": (1, 0)})
        self.assertEqual(by["mma_serialized"].kernel, "sample::tile()")
        self.assertTrue(by["mma_serialized"].fix)                 # the rows' fixes, joined
        r, _ = self.judge("serialized.log", decls(entry(allow=3), entry("wait_injected", 1)))
        self.assertEqual(r, [])
        r, _ = self.judge("serialized.log", decls(entry(allow=2), entry("wait_injected", 1)))
        self.assertEqual([(x.rule, x.count, x.allowed, x.reason) for x in r], [("mma_serialized", 3, 2, "measured and accepted")])
        r, _ = self.judge("serialized.log", decls(entry(allow=3, kernel="^other"), entry("wait_injected", 1, "^other")))
        self.assertEqual(len(r), 2)                              # a declaration is for its kernel

    def test_a_projects_ledger_changes_the_disposition(self):
        mine = notices.merge(notices.base_ledger(), ledger(row(
            tool="ptxas", code="C7517", disposition="refuse", reason="no injected waits here",
            text=next(r.text for r in notices.base_ledger() if r.code == "C7517"))))
        refusals, _ = self.judge("serialized.log", decls(entry(allow=3)), led=mine)
        self.assertEqual([r.rule for r in refusals], ["notice ptxas C7517"])
        self.assertEqual(refusals[0].message, "no injected waits here")

    def test_unmatched_places_count_for_a_stale_declaration(self):
        read_ = read("serialized.log")
        places = declare.notice_places(read_, NAMES)
        self.assertEqual(places, ["sample::tile()"])
        ds = decls(entry(kernel="^sample::tile"), entry(kernel="^gone$"))
        self.assertEqual([d.kernel.pattern for d in declare.unmatched(ds, [], places)], ["^gone$"])


class Notes(unittest.TestCase):
    def test_the_rule_names_a_consumer_lints(self):
        names = declare.rule_names()
        for rule in ("stack_memory", "step_rolled", "wait_after_issue", "mma_serialized", "wait_injected",
                     "shared_wavefronts_excess", "profiler_verdicts"):
            self.assertIn(rule, names)
        self.assertEqual(names, sorted(names))
        self.assertIn("mine", declare.rule_names(notices.merge(notices.base_ledger(), ledger(
            row(text="mine {}", disposition="declare", rule="mine")))))

    def test_notes_are_read_per_rule(self):
        names = declare.rule_names()
        self.assertEqual(declare.load_notes('{"declarations": []}', names), {})
        self.assertEqual(declare.load_notes('{"notes": {"stack_memory": "mine"}}', names), {"stack_memory": "mine"})
        for bad in ('{"notes": {"no_such_rule": "x"}}', '{"notes": {"stack_memory": ""}}', '{"notes": []}', "[", "[]"):
            with self.assertRaises(declare.DeclarationError, msg=bad):
                declare.load_notes(bad, names)

    def test_a_refusal_prints_finding_then_fix_then_the_note(self):
        refusals, _ = declare.check_notices(read("serialized.log"), notices.base_ledger(), [], NAMES,
                                            {"mma_serialized": "the project says: see the measurement"})
        lines = {r.rule: r.lines() for r in refusals}["mma_serialized"]
        self.assertEqual(lines[-1], "  note: the project says: see the measurement")
        self.assertTrue(lines[-2].startswith("  fix: "))
        self.assertTrue(lines[-3].startswith("  at sample::tile() x"))
        self.assertFalse({r.rule: r.lines() for r in refusals}["wait_injected"][-1].startswith("  note"))


class Command(unittest.TestCase):
    STEM = str(FIXTURES / "rules" / "stack_memory" / "spill" / "kernel")

    def run_check(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["check", *argv])
        return code, out.getvalue(), err.getvalue()

    def write(self, text, name="f.json"):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text)
        return str(path)

    def declarations(self, *entries, notes=None):
        return self.write(json.dumps({"declarations": [{**entry(), **e} for e in entries], "notes": notes or {}}))

    def test_notices_refuse_through_the_command(self):
        log = str(FIXTURES / "notices" / "serialized.log")
        code, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--declare",
                                      self.declarations({"rule": "stack_memory", "allow": 128},
                                                        notes={"mma_serialized": "the project's note"}))
        self.assertEqual(code, 1)
        self.assertIn("sample::tile(): mma_serialized: 3 found, 0 declared", out)
        self.assertIn("  note: the project's note", out)
        self.assertEqual(out.splitlines()[-1], "1 kernel(s), 3 rule(s), 8 notice(s): 2 refusal(s)")

    def test_declarations_govern_the_notices(self):
        log = str(FIXTURES / "notices" / "serialized.log")
        d = self.declarations({"rule": "stack_memory", "allow": 128}, {"rule": "mma_serialized", "allow": 3},
                              {"rule": "wait_injected", "allow": 1})
        code, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--declare", d)
        self.assertEqual((code, out.splitlines()[-1]), (0, "1 kernel(s), 3 rule(s), 8 notice(s): 0 refusal(s)"))

    def test_a_ledger_file_overrides_the_base(self):
        log = str(FIXTURES / "notices" / "mixed.log")
        override = self.write(json.dumps({"rows": [row(
            tool="ptxas", code=None, disposition="ignore", reason="accepted here",
            text="Local memory used for function '{}', size of stack frame: {} bytes")]}))
        d = self.declarations({"rule": "stack_memory", "allow": 128})
        _, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--declare", d)
        self.assertIn("notice ptxas no_code", out)
        self.assertIn("Local memory used", out)
        _, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--ledger", override, "--declare", d)
        self.assertNotIn("Local memory used", out)

    def test_a_ledger_that_overlaps_exits_2(self):
        bad = self.write(json.dumps({"rows": [row(text="Local memory used for function '{}', {}")]}))
        code, _, err = self.run_check(self.STEM, "--fixture", "--ledger", bad)
        self.assertEqual(code, 2)
        self.assertIn("overlap", err)

    def test_a_declaration_may_name_a_ledger_rule_only_when_it_exists(self):
        code, _, err = self.run_check(self.STEM, "--fixture", "--declare",
                                      self.declarations({"rule": "mine"}))
        self.assertEqual(code, 2)
        mine = self.write(json.dumps({"rows": [row(text="mine {}", disposition="declare", rule="mine")]}))
        code, _, _ = self.run_check(self.STEM, "--fixture", "--ledger", mine, "--declare",
                                    self.declarations({"rule": "mine"}, {"rule": "stack_memory", "allow": 128}))
        self.assertEqual(code, 0)

    def test_a_notes_file_beside_the_declarations(self):
        notes = self.write(json.dumps({"notes": {"stack_memory": "from the notes file"}}))
        _, out, _ = self.run_check(self.STEM, "--fixture", "--notes", notes, "--declare", self.declarations())
        self.assertIn("  note: from the notes file", out)
        code, _, err = self.run_check(self.STEM, "--fixture", "--notes", self.write('{"notes": {"x": "y"}}'))
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
