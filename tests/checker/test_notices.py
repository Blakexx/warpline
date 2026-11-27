"""The notice ledger: the reader of a build's messages, the base ledger, a project's rows over it, the matcher, and the
notices as refusals under the declarations; the note a project keeps per rule."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline import declare
from warpline.checker.cli import main
from warpline.checker import manifest as collection, notices
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = {"_ZN6sample4tileEv": "sample::tile()"}


def read(name):
    return notices.read_notices((FIXTURES / "notices" / name).read_text())


def row(**kw):
    base = dict(tool="ptxas", code=None, text="a notice about {}", rule="a_rule", allow=0, reason="why", fix="")
    return {**base, **kw}


def ledger(*rows):
    return notices.load_ledger(json.dumps({"rows": list(rows)}))


def decls(*entries, known=()):
    return declare.load_declarations(json.dumps({"declarations": list(entries)}), {}, [*known, *collection.rule_names()])


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
        self.assertEqual([n.code for n in got], [None, None, None, None, None])     # toolkit 13.5 injects no arrive (13.0: C7519 first)
        self.assertEqual({notices.row_of(notices.base_ledger(), n).allow for n in got}, {None})     # all accepted by default


class Toolkit135(unittest.TestCase):
    """Toolkit 13.5's own wording (the example kernels' `.ptxas`): a serialization notice ends ` in line N` and some codes moved
    (the pipeline's registers C7511, the divergent arrive C7519), the injected wait is `wgmma.wait_group`."""

    def test_each_is_a_row(self):
        text = (
            "ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient "
            "register resources for the wgmma pipeline in the function '_ZN6sample12mixed_shapesEPfPKii' in line 168\n"
            "ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program "
            "dependence on compiler-inserted WG.AR in divergent path in the function '_ZN6sample10serializedEPfPKii' in line 245\n"
            "ptxas info    : (C7517) wgmma.wait_group is injected in around line 103 by compiler to allow use of registers "
            "defined by GMMA in function '_ZN6stages9loop_joinEPfPKji'\n")
        got = notices.read_notices(text)
        self.assertEqual([notices.row_of(notices.base_ledger(), n).rule for n in got],
                         ["mma_serialized", "mma_serialized", "wait_injected"])


class Schema(unittest.TestCase):
    def test_a_row_needs_its_fields(self):
        for bad in (row(reason=""), row(reason="  "), row(rule=""), row(tool=""), row(text=""), row(code=7), row(extra=1),
                    row(allow=-1), row(allow="2"), row(allow=True), row(disposition="refuse")):
            with self.assertRaises(notices.LedgerError, msg=str(bad)):
                ledger(bad)
        with self.assertRaises(notices.LedgerError):                 # a row names its rule
            ledger({"tool": "ptxas", "text": "x", "reason": "r"})
        with self.assertRaises(notices.LedgerError):
            notices.load_ledger("[]")

    def test_a_row_is_a_rule_and_a_default_allowance(self):
        (r,) = ledger({"tool": "ptxas", "text": "x", "rule": "my_rule", "reason": "r"})
        self.assertEqual((r.rule, r.allow), ("my_rule", 0))           # the default allowance is 0
        (r,) = ledger(row(allow=2))
        self.assertEqual(r.allow, 2)
        (r,) = ledger(row(allow=None))
        self.assertIsNone(r.allow)                                    # accepted by default

    def test_the_base_ledger(self):
        base = notices.base_ledger()
        self.assertEqual(len(base), 39)       # toolkit 13.5 adds each serialization cause in its own spelling (10) and the injected wait's (1)
        self.assertEqual(notices.row_rules(base), [
            "build_info", "local_memory_used", "mma_serialized", "register_spill", "setmaxnreg_ignored", "unreachable_loop",
            "unused_variable", "variable_unset", "wait_injected", "wgmma_arrive_injected"])
        self.assertEqual({r.allow for r in base}, {0, None})
        self.assertEqual({r.rule for r in base if r.allow is None},
                         {"build_info", "unreachable_loop", "unused_variable", "wgmma_arrive_injected"})
        self.assertTrue(all(r.reason and (r.fix or r.rule in ("build_info", "unused_variable")) for r in base))

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
        self.assertEqual((spill.rule, spill.allow), ("local_memory_used", 0))
        mine = ledger(row(text=spill.text, allow=None, rule="local_memory_used", reason="mine"),
                      row(text="my notice", tool="nvcc"))
        merged = notices.merge(base, mine)
        self.assertEqual(len(merged), len(base) + 1)
        self.assertEqual(merged.index(next(r for r in merged if r.text == spill.text)), base.index(spill))   # in place
        self.assertEqual(next(r for r in merged if r.text == spill.text).reason, "mine")
        self.assertEqual(merged[-1].text, "my notice")
        # a project row that is not the same key but would match a base row's notices is refused
        with self.assertRaises(notices.LedgerError):
            notices.merge(base, ledger(row(text="Local memory used for function '{}', {}")))

    def test_a_project_may_name_a_new_rule(self):
        merged = notices.merge(notices.base_ledger(), ledger(row(text="mine {}", rule="mine")))
        self.assertIn("mine", notices.row_rules(merged))


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
        return collection.check_notices(read(name), led or notices.base_ledger(), list(declarations), NAMES, notes)

    def test_a_notice_with_no_row_refuses_quoted_whole(self):
        refusals = self.judge("mixed.log")
        unknown = [r for r in refusals if r.rule == "notice_without_row"]
        self.assertEqual(sorted(r.kernel for r in unknown), ["sample::tile()", "the build"])
        unknown.sort(key=lambda r: r.kernel != "sample::tile()")
        text = "\n".join(unknown[0].lines())
        self.assertIn("sample::tile(): notice_without_row: 1 found, 0 declared (undeclared)", text)
        self.assertIn("ptxas info    : (C7599) A notice no ledger knows, about the function '_ZN6sample4tileEv'", text)
        self.assertIn("(nvlink with no code)", "\n".join(unknown[1].lines()))
        self.assertIn("fix: add a row to the project's ledger", text)

    def test_no_declaration_lifts_an_unknown_notice(self):
        with self.assertRaises(declare.DeclarationError):         # it is no rule a declaration may name
            decls(entry("notice_without_row", 9, "."))
        every = decls(entry("notice_without_row", 9, "."), entry("notice_without_row", 9, "the build"),
                      known=["notice_without_row"])              # and a comparison ignores one that got past the loader
        self.assertEqual(len([r for r in self.judge("mixed.log", every) if r.rule == "notice_without_row"]), 2)

    def test_a_row_with_the_default_allowance_refuses_and_names_its_fix(self):
        refusals = self.judge("mixed.log")
        by = {(r.kernel, r.rule): r for r in refusals}
        for place, rule in (("tile.cu(1706)", "variable_unset"), ("the build", "setmaxnreg_ignored"),
                            ("sample::tile()", "register_spill"), ("sample::tile()", "local_memory_used")):
            self.assertIn((place, rule), by)
            self.assertEqual(by[(place, rule)].allowed, 0)
            self.assertTrue(by[(place, rule)].fix)
        self.assertEqual(by[("the build", "setmaxnreg_ignored")].count, 2)        # two rows of one rule, summed at one place
        self.assertIn("function 'sample::tile()'", by[("sample::tile()", "register_spill")].findings[0].what)   # demangled

    def test_a_notice_accepted_by_default_is_no_refusal(self):
        for name in ("mixed.log", "serialized.log"):
            rules = {r.rule for r in self.judge(name)}
            self.assertFalse(rules & {"wgmma_arrive_injected", "unreachable_loop", "build_info", "unused_variable"}, rules)

    def test_a_declaration_tightens_a_notice_accepted_by_default(self):
        refusals = self.judge("mixed.log", decls(entry("wgmma_arrive_injected", 0, "tile", "no injected arrive here"),
                                                 entry("unreachable_loop", 0, r"tile\.cu", "dead loops are bugs here")))
        by = {r.rule: r for r in refusals}
        self.assertEqual((by["wgmma_arrive_injected"].count, by["wgmma_arrive_injected"].allowed,
                          by["wgmma_arrive_injected"].reason), (1, 0, "no injected arrive here"))
        self.assertEqual((by["unreachable_loop"].kernel, by["unreachable_loop"].count), ("tile.cu(354)", 1))
        # a bound is a bound: allow 1 accepts the one injected arrive
        self.assertNotIn("wgmma_arrive_injected", {r.rule for r in self.judge("mixed.log", decls(entry("wgmma_arrive_injected", 1)))})

    def test_a_declaration_allows_a_notice_refused_by_default(self):
        """one spill notice: refused with no declaration, quiet with allow 1 and a reason, refused at 2 under allow 1"""
        one = "ptxas warning : Registers are spilled to local memory in function '_ZN6sample4tileEv', 8 bytes spill stores, 4 bytes spill loads\n"
        two = one + one
        run = lambda log, *d: collection.check_notices(notices.read_notices(log), notices.base_ledger(), list(d), NAMES)
        self.assertEqual([(r.rule, r.count, r.allowed, r.reason) for r in run(one)], [("register_spill", 1, 0, None)])
        self.assertEqual(run(one, *decls(entry("register_spill", 1, "tile"))), [])
        (r,) = run(two, *decls(entry("register_spill", 1, "tile")))
        self.assertEqual((r.rule, r.count, r.allowed, r.reason), ("register_spill", 2, 1, "measured and accepted"))
        self.assertIn("(measured and accepted)", r.lines()[0])

    def test_rows_count_under_the_rule_and_the_declarations_govern(self):
        by = {r.rule: r for r in self.judge("serialized.log")}
        self.assertEqual({k: (v.count, v.allowed) for k, v in by.items()}, {"mma_serialized": (3, 0), "wait_injected": (1, 0)})
        self.assertEqual(by["mma_serialized"].kernel, "sample::tile()")
        self.assertTrue(by["mma_serialized"].fix)                 # the rows' fixes, joined
        self.assertEqual(self.judge("serialized.log", decls(entry(allow=3), entry("wait_injected", 1))), [])
        r = self.judge("serialized.log", decls(entry(allow=2), entry("wait_injected", 1)))
        self.assertEqual([(x.rule, x.count, x.allowed, x.reason) for x in r], [("mma_serialized", 3, 2, "measured and accepted")])
        r = self.judge("serialized.log", decls(entry(allow=3, kernel="^other"), entry("wait_injected", 1, "^other")))
        self.assertEqual(len(r), 2)                              # a declaration is for its kernel

    def test_a_projects_ledger_changes_the_default(self):
        text = next(r.text for r in notices.base_ledger() if r.code == "C7517" and "warpgroup.wait" in r.text)
        mine = notices.merge(notices.base_ledger(), ledger(row(
            tool="ptxas", code="C7517", rule="wait_injected", allow=1, reason="one injected wait is the design", text=text)))
        self.assertEqual([r.rule for r in self.judge("serialized.log", decls(entry(allow=3)), led=mine)], [])
        strict = notices.merge(notices.base_ledger(), ledger(row(
            tool="ptxas", code="C7517", rule="wait_injected", allow=0, reason="no injected waits here", text=text)))
        self.assertEqual([r.rule for r in self.judge("serialized.log", decls(entry(allow=3)), led=strict)], ["wait_injected"])

    def test_unmatched_places_count_for_a_stale_declaration(self):
        read_ = read("serialized.log")
        places = collection.notice_places(read_, NAMES)
        self.assertEqual(places, ["sample::tile()"])
        ds = decls(entry(kernel="^sample::tile"), entry(kernel="^gone$"))
        self.assertEqual([d.kernel.pattern for d in collection.unmatched(ds, [], places)], ["^gone$"])


class Notes(unittest.TestCase):
    def test_the_rule_names_a_consumer_lints(self):
        names = collection.rule_names()
        for rule in ("stack_memory", "step_rolled", "wait_after_issue", "mma_serialized", "wait_injected",
                     "shared_wavefronts_excess", "profiler_verdicts", "function_size", "register_spill", "local_memory_used",
                     "setmaxnreg_ignored", "variable_unset"):
            self.assertIn(rule, names)
        self.assertEqual(names, sorted(names))
        self.assertIn("mine", collection.rule_names(notices.merge(notices.base_ledger(), ledger(row(text="mine {}", rule="mine")))))

    def test_notes_are_read_per_rule(self):
        names = collection.rule_names()
        self.assertEqual(declare.load_notes('{"declarations": []}', names), {})
        self.assertEqual(declare.load_notes('{"notes": {"stack_memory": "mine"}}', names), {"stack_memory": "mine"})
        for bad in ('{"notes": {"no_such_rule": "x"}}', '{"notes": {"stack_memory": ""}}', '{"notes": []}', "[", "[]"):
            with self.assertRaises(declare.DeclarationError, msg=bad):
                declare.load_notes(bad, names)

    def test_a_refusal_prints_finding_then_fix_then_the_note(self):
        refusals = collection.check_notices(read("serialized.log"), notices.base_ledger(), [], NAMES,
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
        self.assertEqual(out.splitlines()[-1], f"1 kernel(s), {len(registry()) + 3} rule(s), 8 notice(s): 2 refusal(s)")

    def test_several_logs_are_read_as_one_build(self):
        """A log a compile: the notices of every `--notices` are counted, as one log of them all would be."""
        log = FIXTURES / "notices" / "serialized.log"
        lines = log.read_text().splitlines(keepends=True)
        half = len(lines) // 2
        while half < len(lines) and lines[half][:1].isspace():     # a notice's indented lines stay with it
            half += 1
        first, second = self.write("".join(lines[:half]), "a.log"), self.write("".join(lines[half:]), "b.log")
        d = self.declarations({"rule": "stack_memory", "allow": 128})
        whole = self.run_check(self.STEM, "--fixture", "--notices", str(log), "--declare", d)
        split = self.run_check(self.STEM, "--fixture", "--notices", first, "--notices", second, "--declare", d)
        self.assertEqual(split, whole)
        self.assertIn("8 notice(s)", split[1])

    def test_declarations_govern_the_notices(self):
        log = str(FIXTURES / "notices" / "serialized.log")
        d = self.declarations({"rule": "stack_memory", "allow": 128}, {"rule": "mma_serialized", "allow": 3},
                              {"rule": "wait_injected", "allow": 1})
        code, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--declare", d)
        self.assertEqual((code, out.splitlines()[-1]), (0, f"1 kernel(s), {len(registry()) + 3} rule(s), 8 notice(s): 0 refusal(s)"))

    def test_a_ledger_file_overrides_the_base(self):
        log = str(FIXTURES / "notices" / "mixed.log")
        override = self.write(json.dumps({"rows": [row(
            tool="ptxas", code=None, rule="local_memory_used", allow=None, reason="accepted here",
            text="Local memory used for function '{}', size of stack frame: {} bytes")]}))
        d = self.declarations({"rule": "stack_memory", "allow": 128})
        _, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--declare", d)
        self.assertIn("local_memory_used", out)
        self.assertIn("Local memory used", out)
        _, out, _ = self.run_check(self.STEM, "--fixture", "--notices", log, "--ledger", override, "--declare", d)
        self.assertNotIn("local_memory_used", out)

    def test_a_ledger_that_overlaps_exits_2(self):
        bad = self.write(json.dumps({"rows": [row(text="Local memory used for function '{}', {}")]}))
        code, _, err = self.run_check(self.STEM, "--fixture", "--ledger", bad)
        self.assertEqual(code, 2)
        self.assertIn("overlap", err)

    def test_a_declaration_may_name_a_ledger_rule_only_when_it_exists(self):
        code, _, err = self.run_check(self.STEM, "--fixture", "--declare",
                                      self.declarations({"rule": "mine"}))
        self.assertEqual(code, 2)
        mine = self.write(json.dumps({"rows": [row(text="mine {}", rule="mine")]}))
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
