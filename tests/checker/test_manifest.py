"""The manifest (a kernel's findings and descriptive figures, in the manifest), the checkers that judge it from the
declarations (`violations`, `bound`, `hash`) and the coverage check: each fires, is quiet, and where it has one, is waived;
a hash that matches, one that does not, and what it prints. Fixtures: the committed `loop_held` kernels (compiled with
source lines) and the hand-made loops of `test_held`."""
import contextlib
import io
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from warpline.checker import checkers, declare, manifest, model, readers, runner
from warpline.checker.cli import main
from warpline.checker.rules import registry


HELD = Path(__file__).parent / "fixtures" / "rules" / "loop_held"
FEW, MANY, NONE = (str(HELD / n / "kernel") for n in ("few", "many", "none"))
HASH_LIVE_MANY = "542bb89656d16a52"        # the values of `loops/.*/live` of the `many` kernel
HASH_LIVE_FEW = "4b5af159aba37f3c"         # ... of the `few` kernel


def nested() -> model.Kernel:
    """An outer loop (`O` through `E`) holding R2, R3 and its counter R7, around an inner loop `I` with its own counter R8."""
    code = [
        (0x00, "", "MOV", "R2, 0x1"), (0x10, "", "MOV", "R3, 0x2"), (0x20, "", "MOV", "R7, RZ"),
        (0x30, "", "MOV", "R8, RZ"),
        (0x40, "", "IADD3", "R8, R8, 0x1, RZ"), (0x50, "", "ISETP.LT.AND", "P0, PT, R8, 0x4, PT"), (0x60, "@P0", "BRA", "0x40"),
        (0x70, "", "IADD3", "R7, R7, R2, RZ"), (0x80, "", "ISETP.LT.AND", "P1, PT, R7, 0x9, PT"), (0x90, "@P1", "BRA", "0x30"),
        (0xa0, "", "STG.E", "[R3], R7"), (0xb0, "", "EXIT", ""),
    ]
    blocks = (("A", (0x00, 0x10, 0x20)), ("O", (0x30,)), ("I", (0x40, 0x50, 0x60)), ("E", (0x70, 0x80, 0x90)), ("X", (0xa0, 0xb0)))
    cfg = readers.Cfg("A", tuple(readers.CfgBlock(*b) for b in blocks),
                      (("A", "O"), ("O", "I"), ("I", "I"), ("I", "E"), ("E", "O"), ("E", "X")))
    return model.build_kernel("nested", [readers.SassLine(*c) for c in code], cfg, readers.Usage(10, 0, 0))


def manifest_of(stem=MANY):
    return manifest.collect(list(runner.kernels_of_files(stem).values()), registry())


def entry(**kw):
    return {"kernel": "held_", "reason": "the design's own", **kw}


#: the `many` kernel's one finding of a rule that is not what these tests are about, allowed last (the first declaration of a rule applies)
BASE = {"kernel": "held_", "rule": "load_in_trip", "allow": 1, "reason": "the loop's one load"}


def run(stem, *entries, coverage=None):
    """The comparison of the stem's manifest with the entries (a `rule` entry is `violations`) and `BASE`."""
    doc = {"declarations": [*entries, BASE], **({"coverage": coverage} if coverage is not None else {})}
    text = json.dumps(doc)
    got = declare.compare(manifest_of(stem), declare.load_declarations(text, registry(), declare.rule_names()), None,
                          checkers.load(text))
    return got


def refusals(stem, *entries, **kw):
    return [r.lines() for r in run(stem, *entries, **kw).refusals]


class Describing(unittest.TestCase):
    def test_a_loop_is_keyed_by_its_file_and_position_with_its_figures(self):
        m = manifest.of_kernel(next(iter(runner.kernels_of_files(MANY).values())))
        self.assertEqual(m["loops_keyed_by"], "source file and position")
        self.assertEqual(list(m["loops"]), ["held_many.cu/1"])
        loop = m["loops"]["held_many.cu/1"]
        self.assertEqual((loop["copies"], loop["live"], loop["read"], loop["instructions"]), (1, 30, 4, 8))
        self.assertEqual((loop["global_loads"], loop["global_stores"], loop["shared_loads"], loop["shared_stores"]), (1, 0, 0, 0))
        self.assertEqual(loop["class"]["integer_address"], 4)
        self.assertEqual(sum(loop["class"].values()), loop["instructions"])
        self.assertEqual(set(loop["class"]), set(manifest.CLASSES))          # every class is there, zeros included
        self.assertEqual(m["kernel"], {"instructions": 102, "registers": 40, "stack_bytes": 0, "shared_bytes": 0,
                                       "barriers": 0, "tensor_steps": 0})

    def test_with_no_source_listing_loops_are_keyed_by_nesting_path_and_it_says_so(self):
        m = manifest.of_kernel(nested())
        self.assertEqual((m["loops_keyed_by"], list(m["loops"])), ("nesting path", ["loop 1", "loop 1.1"]))
        outer, inner = m["loops"]["loop 1"], m["loops"]["loop 1.1"]
        self.assertEqual((outer["instructions"], inner["instructions"]), (7, 3))      # the outer loop holds the inner one
        self.assertEqual((inner["class"]["control"], outer["class"]["control"], outer["class"]["integer_address"]), (1, 2, 3))

    def test_a_loop_with_no_line_makes_the_kernel_keyed_by_path(self):
        k = replace(nested(), lines={0x30: ("a/b.cu", 5)})            # the inner loop's header has no line
        self.assertEqual(manifest.of_kernel(k)["loops_keyed_by"], "nesting path")

    def test_copies_of_one_source_loop_are_one_entry_with_the_most_of_each_figure(self):
        k = nested()
        both = manifest.of_kernel(replace(k, lines={0x30: ("a/b.cu", 7), 0x40: ("a/b.cu", 7)}))
        self.assertEqual(list(both["loops"]), ["b.cu/1"])
        one = manifest.of_kernel(k)["loops"]
        merged = both["loops"]["b.cu/1"]
        self.assertEqual(merged["copies"], 2)
        self.assertEqual(merged["instructions"], max(l["instructions"] for l in one.values()))
        self.assertEqual(merged["live"], max(l["live"] for l in one.values()))

    def test_no_instruction_address_is_in_a_field_a_checker_can_read(self):
        for stem in (FEW, MANY, NONE):
            (place,) = manifest_of(stem).places
            for path in manifest.fields(place):
                self.assertNotRegex(path, r"0x[0-9a-f]{3,}")
            self.assertNotRegex(json.dumps(place.figures), r"0x[0-9a-f]{3,}")
        self.assertNotIn("/home", manifest.dumps(manifest_of()))

    def test_the_manifest_is_stable_json_and_reads_back_whole(self):
        a = manifest.dumps(manifest_of())
        self.assertEqual(a, manifest.dumps(manifest_of()))
        doc = json.loads(a)
        self.assertEqual(doc["format"], "warpline-manifest/1")
        self.assertIn("figures", doc["places"][0])
        self.assertEqual(manifest.dumps(manifest.loads(a)), a)
        self.assertEqual(manifest.loads(a), manifest_of())

    def test_a_manifest_must_be_one_of_this_format_with_figures_for_each_kernel(self):
        good = json.loads(manifest.dumps(manifest_of()))
        bad_count = json.loads(manifest.dumps(manifest_of()))
        bad_count["places"][0]["figures"]["kernel"]["registers"] = -1
        no_figures = json.loads(manifest.dumps(manifest_of()))
        del no_figures["places"][0]["figures"]
        text_figure = json.loads(manifest.dumps(manifest_of()))
        text_figure["places"][0]["figures"]["loops"]["held_many.cu/1"]["live"] = "30"
        for doc in ({**good, "format": "warpline-facts/2"}, {**good, "format": "warpline-facts/1"}, bad_count, no_figures, text_figure):
            with self.assertRaises(manifest.ManifestError, msg=str(doc["format"])):
                manifest.loads(json.dumps(doc))

    def test_a_loop_key_has_no_line_number_and_the_line_is_a_note_that_is_no_field(self):
        (place,) = manifest_of().places
        (key,) = place.figures["loops"]
        self.assertEqual((key, place.figures["loops"][key]["at"]), ("held_many.cu/1", "held_many.cu:15"))
        self.assertFalse([p for p in manifest.fields(place) if p.endswith("/at") or ":15" in p])

    def test_an_edit_above_a_loop_moves_no_path_and_no_hash(self):
        k = nested()
        at = lambda a, b: manifest.of_kernel(replace(k, lines={0x30: ("a/b.cu", a), 0x40: ("a/b.cu", b)}))
        before, after = at(10, 12), at(15, 17)                    # five lines inserted above both loops
        self.assertEqual(list(before["loops"]), ["b.cu/1", "b.cu/1.1"])
        strip = lambda m: {key: {n: v for n, v in loop.items() if n != "at"} for key, loop in m["loops"].items()}
        self.assertEqual(strip(before), strip(after))
        self.assertEqual((before["loops"]["b.cu/1"]["at"], after["loops"]["b.cu/1"]["at"]), ("b.cu:10", "b.cu:15"))

    def test_the_position_is_the_rank_by_line_among_the_loops_of_a_file_at_one_nesting(self):
        k = nested()
        # the inner loop above the outer one in the file (an inlined body): its own file's loop, so each ranks first of its file
        both = manifest.of_kernel(replace(k, lines={0x30: ("a/outer.cu", 40), 0x40: ("a/inner.h", 3)}))
        self.assertEqual(list(both["loops"]), ["inner.h/1", "outer.cu/1"])
        # two loops of one file side by side: ranked by line, whatever their addresses
        k2 = replace(k, loops=tuple(lp for lp in k.loops if lp.header == "I"), lines={0x40: ("a/b.cu", 9)})
        self.assertEqual(list(manifest.of_kernel(k2)["loops"]), ["b.cu/1"])

    def test_the_text_of_the_manifest_verb_with_text(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["manifest", MANY, "--fixture", "--text"]), 0)
        text = out.getvalue()
        self.assertIn("(loops keyed by source file and position)", text)
        self.assertIn("  # held_many.cu/1 at held_many.cu:15\n  loops/held_many.cu/1/copies = 1\n", text)
        self.assertIn("  loops/held_many.cu/1/live = 30\n", text)
        self.assertIn("  rules/loop_held = 30\n", text)
        self.assertNotIn("class/tensor", text)                                   # a class with none is left out of the text
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["manifest", MANY, "--fixture", "--text", "--kernel", "^nothing"])
        self.assertEqual(out.getvalue(), "no kernel matches\n")


class Violations(unittest.TestCase):
    def test_an_entry_with_a_rule_is_violations_with_or_without_the_name_of_the_check(self):
        plain = {"kernel": "held_", "rule": "load_in_trip", "allow": 0, "reason": "none"}   # before BASE: this one applies
        named = {**plain, "check": "violations"}
        a, b = (refusals(MANY, e) for e in (plain, named))
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], "held_many(const float *, const float *, float *, int): load_in_trip: 1 found, 0 declared (none)")
        self.assertEqual(refusals(MANY, {**plain, "allow": 1}), [])

    def test_an_unknown_check_is_an_error(self):
        text = json.dumps({"declarations": [entry(check="nope")]})
        with self.assertRaises(declare.DeclarationError):
            declare.load_declarations(text, registry())
        with self.assertRaises(checkers.CheckerError):
            checkers.load(text)


class Bounds(unittest.TestCase):
    def bound(self, **kw):
        return entry(check="bound", figure="loops/.*/live", **kw)

    def test_at_most_fires_with_the_figure_the_value_and_the_reason(self):
        (r,) = refusals(MANY, self.bound(at_most=8, reason="eight fit"))
        self.assertEqual(r, ["held_many(const float *, const float *, float *, int): bound: "
                             "loops/held_many.cu/1/live is 30, at most 8 declared (eight fit)"])
        self.assertEqual(refusals(FEW, self.bound(at_most=8)), [])              # the same bound is quiet on the smaller kernel
        self.assertEqual(refusals(MANY, self.bound(at_most=30)), [])

    def test_at_least_and_exactly(self):
        self.assertEqual(refusals(FEW, self.bound(at_least=8)), [])
        self.assertIn("is 8, at least 9 declared", refusals(FEW, self.bound(at_least=9))[0][0])
        self.assertEqual(refusals(FEW, self.bound(exactly=8)), [])
        self.assertIn("is 8, exactly 30 declared", refusals(FEW, self.bound(exactly=30))[0][0])

    def test_a_figure_is_a_whole_path_and_may_be_one_among_several(self):
        self.assertEqual(refusals(MANY, entry(check="bound", figure="kernel/registers", at_most=40, reason="r")), [])
        (r,) = refusals(MANY, entry(check="bound", figure="kernel/registers", at_most=39, reason="r"))
        self.assertIn("kernel/registers is 40, at most 39", r[0])
        self.assertEqual(len(refusals(MANY, entry(check="bound", figure="kernel/(registers|instructions)", at_most=1))), 2)
        self.assertEqual(len(refusals(MANY, entry(check="bound", figure="registers", at_most=0))), 1)   # names no field: refused

    def test_a_rules_count_is_a_field_too(self):
        (r,) = refusals(MANY, entry(check="bound", figure="rules/loop_held", at_most=29, reason="r"))
        self.assertIn("rules/loop_held is 30, at most 29", r[0])

    def test_every_bound_that_applies_is_enforced_and_only_at_the_kernels_named(self):
        got = refusals(MANY, self.bound(at_most=8), self.bound(at_most=9), entry(kernel="^other", check="bound", figure=".*", at_most=0))
        self.assertEqual(len(got), 2)
        self.assertEqual(run(MANY, entry(kernel="^other", check="bound", figure=".*", at_most=0)).stale_checks,
                         ("note: the bound declaration for '^other' names no kernel",))

    def test_bad_bounds_are_errors(self):
        for bad in (self.bound(), self.bound(at_most=1, at_least=0), self.bound(at_most=-1), self.bound(at_most="1"),
                    {**self.bound(at_most=1), "reason": " "}, entry(check="bound", at_most=1), entry(check="bound", figure="(", at_most=1)):
            with self.assertRaises(checkers.CheckerError, msg=str(bad)):
                checkers.load(json.dumps({"declarations": [bad]}))
        with self.assertRaises(checkers.CheckerError):
            checkers.load(json.dumps({"declarations": [{**self.bound(at_most=1), "reason": None}]}))


class Hashes(unittest.TestCase):
    def hashed(self, expected, fields="loops/.*/live", **kw):
        return entry(check="hash", fields=fields, hash=expected, **kw)

    def test_a_hash_that_matches_is_quiet(self):
        self.assertEqual(refusals(MANY, self.hashed(HASH_LIVE_MANY)), [])
        self.assertEqual(refusals(FEW, self.hashed(HASH_LIVE_FEW)), [])

    def test_a_hash_that_differs_prints_the_values_and_the_hash_to_write(self):
        (r,) = refusals(MANY, self.hashed("stale", reason="the held values are the design"))
        self.assertEqual(r[0], f"held_many(const float *, const float *, float *, int): hash: the 1 covered field(s) hash to "
                               f"{HASH_LIVE_MANY}, stale declared (the held values are the design)")
        self.assertEqual(r[1:], ["  loops/held_many.cu/1/live = 30",
                                 f'  fix: look at the values above; if they are intended, write "hash": "{HASH_LIVE_MANY}" in the declaration'])

    def test_the_printed_hash_is_the_one_that_passes(self):
        (r,) = refusals(MANY, self.hashed("x", fields="loops/.*/(live|read)|kernel/registers"))
        self.assertEqual(r[1:4], ["  kernel/registers = 40", "  loops/held_many.cu/1/live = 30", "  loops/held_many.cu/1/read = 4"])
        new = re.search(r'write "hash": "(\w+)"', r[-1]).group(1)
        self.assertEqual(refusals(MANY, self.hashed(new, fields="loops/.*/(live|read)|kernel/registers")), [])

    def test_pointed_at_another_input_it_fails(self):
        self.assertEqual(refusals(MANY, self.hashed(HASH_LIVE_MANY)), [])
        (r,) = refusals(FEW, self.hashed(HASH_LIVE_MANY))
        self.assertIn(f"hash to {HASH_LIVE_FEW}, {HASH_LIVE_MANY} declared", r[0])
        self.assertIn("  loops/held_few.cu/1/live = 8", r)

    def test_a_hash_over_no_field_is_refused_and_the_fields_cover_by_value_only(self):
        (r,) = refusals(MANY, self.hashed("x", fields="loops/nothing"))
        self.assertIn("'loops/nothing' covers no field", r[0])
        self.assertEqual(refusals(MANY, self.hashed(HASH_LIVE_MANY, fields="loops/.*/live", kernel="^held_many")), [])

    def test_bad_hashes_are_errors(self):
        for bad in (entry(check="hash", fields="x"), entry(check="hash", hash="x"), entry(check="hash", fields="x", hash=""),
                    entry(check="hash", fields="x", hash="x", reason="")):
            with self.assertRaises(checkers.CheckerError, msg=str(bad)):
                checkers.load(json.dumps({"declarations": [bad]}))


class Coverage(unittest.TestCase):
    def waive(self, fields, reason="too noisy"):
        return entry(check="waive", fields=fields, reason=reason)

    def test_off_unless_a_project_turns_it_on(self):
        self.assertEqual(refusals(MANY), [])
        self.assertEqual(refusals(MANY, coverage=False), [])
        with self.assertRaises(checkers.CheckerError):
            checkers.load(json.dumps({"declarations": [], "coverage": "yes"}))

    def test_an_uncovered_field_is_refused_by_name(self):
        (r,) = refusals(MANY, coverage=True)
        self.assertEqual(r[0], "held_many(const float *, const float *, float *, int): coverage: "
                               "33 field(s) neither covered by a checker nor waived")
        fields = r[1:-1]
        self.assertIn("  kernel/registers", fields)
        self.assertIn("  loops/held_many.cu/1/live", fields)
        self.assertIn("  loops/held_many.cu/1/class/tensor", fields)
        self.assertEqual([f for f in fields if "rules/" in f], [f"  rules/{n}" for n in (          # the bound rules with no bound:
            "function_size", "loop_held", "step_density", "step_gap", "step_port", "step_work")])   # shown, never refused
        self.assertNotIn("  rules/stack_memory", fields)                          # a rule of default 0 is covered by its default
        self.assertIn("fix:", r[-1])

    def test_a_bound_a_hash_a_violations_entry_and_a_waiver_each_cover(self):
        covering = [entry(check="bound", figure="kernel/.*", at_most=999, reason="r"),
                    entry(check="hash", fields="loops/.*/(copies|live|read|instructions)", hash="x", reason="r"),
                    entry(rule="function_size", allow=999, reason="r"), entry(rule="loop_held", allow=99, reason="r"),
                    self.waive("rules/step_.*"),
                    self.waive("loops/.*/class/.*"), self.waive("loops/.*/(global|shared)_.*")]
        got = refusals(MANY, *covering, coverage=True)
        self.assertEqual([r for r in got if r[0].split(": ")[1] == "coverage"], [])
        self.assertEqual({r[0].split(": ")[1] for r in got}, {"hash"})            # the hash still fails: coverage is not strictness
        (left,) = [r for r in refusals(MANY, *covering[:-1], coverage=True) if ": coverage: " in r[0]]
        self.assertEqual([l.strip() for l in left[1:-1]],
                         [f"loops/held_many.cu/1/{n}" for n in ("global_loads", "global_stores", "shared_loads", "shared_stores")])

    def test_waiving_the_rest_makes_it_quiet_and_the_waiver_carries_its_reason(self):
        got = run(MANY, self.waive("kernel/.*"), self.waive("loops/.*"), self.waive("rules/(function_size|loop_held|step_.*)", "a figure"),
                  coverage=True)
        self.assertEqual(got.refusals, [])
        with self.assertRaises(checkers.CheckerError):
            checkers.load(json.dumps({"declarations": [self.waive("x", reason="")]}))

    def test_a_notice_accepted_by_default_is_uncovered_until_declared(self):
        base = manifest_of()
        (place,) = base.places
        rules = {**place.rules, "wgmma_arrive_injected": manifest.RuleFindings(1, (), "m", "f", default=None)}
        f = manifest.Manifest((replace(place, rules=rules),))
        text = json.dumps({"declarations": [BASE, self.waive("kernel/.*"), self.waive("loops/.*"), self.waive("rules/(function_size|loop_held|step_.*)")],
                           "coverage": True})
        decls = declare.load_declarations(text, registry(), [*declare.rule_names()])
        got = declare.compare(f, decls, None, checkers.load(text)).refusals
        self.assertEqual([r.lines()[1] for r in got], ["  rules/wgmma_arrive_injected"])
        text2 = json.dumps({"declarations": [*json.loads(text)["declarations"],
                                             {"kernel": "held_", "rule": "wgmma_arrive_injected", "allow": 1, "reason": "r"}], "coverage": True})
        decls = declare.load_declarations(text2, registry(), [*declare.rule_names()])
        self.assertEqual(declare.compare(f, decls, None, checkers.load(text2)).refusals, [])

    def test_a_kernel_with_no_loop_still_has_its_fields_to_cover(self):
        (r,) = refusals(NONE, coverage=True)
        self.assertIn("coverage: 12 field(s)", r[0])


class Command(unittest.TestCase):
    def check(self, doc, stem=MANY, *more):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "declarations.json"
            path.write_text(json.dumps(doc))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main(["check", stem, "--fixture", "--declare", str(path), *more])
        return status, out.getvalue(), err.getvalue()

    def test_check_runs_the_new_kinds_from_the_declarations_file(self):
        status, out, _ = self.check({"declarations": [BASE, entry(check="bound", figure="loops/.*/live", at_most=8, reason="eight fit"),
                                                      entry(check="hash", fields="loops/.*/live", hash="stale")]})
        self.assertEqual(status, 1)
        self.assertIn("bound: loops/held_many.cu/1/live is 30, at most 8 declared (eight fit)", out)
        self.assertIn("hash: the 1 covered field(s) hash to " + HASH_LIVE_MANY, out)
        self.assertTrue(out.rstrip().endswith("2 refusal(s)"), out)
        status, out, _ = self.check({"declarations": [BASE, entry(check="bound", figure="loops/.*/live", at_most=30),
                                                      entry(check="hash", fields="loops/.*/live", hash=HASH_LIVE_MANY)]})
        self.assertEqual((status, out.rstrip().endswith("0 refusal(s)")), (0, True), out)

    def test_coverage_through_the_command_and_a_stale_checker(self):
        status, out, _ = self.check({"declarations": [BASE, entry(kernel="^gone", check="waive", fields=".*")], "coverage": True})
        self.assertEqual(status, 1)
        self.assertIn("coverage: 33 field(s)", out)
        self.assertIn("note: the waive declaration for '^gone' names no kernel", out)

    def test_a_bad_checker_is_exit_2(self):
        status, _, err = self.check({"declarations": [entry(check="bound", figure="x")]})
        self.assertEqual(status, 2)
        self.assertIn("a bound needs exactly one of", err)

    def test_compare_reads_a_manifest_file_with_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "k.manifest.json"
            path = Path(d) / "declarations.json"
            path.write_text(json.dumps({"declarations": [entry(check="bound", figure="kernel/registers", at_most=39, reason="r")]}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["manifest", MANY, "--fixture", "-o", str(out)]), 0)
                self.assertEqual(main(["compare", str(out), "--declare", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
