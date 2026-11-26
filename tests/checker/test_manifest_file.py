"""The manifest (what a run found, per kernel, as stable JSON), its reader, and the comparison that refuses: every
refusal names kernel, rule, count, allowed and the declaration's reason or `undeclared`; a declaration may tighten."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline.checker.cli import main
from warpline.checker import declare, manifest, notices, runner
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures"
SPILL = str(FIXTURES / "rules" / "stack_memory" / "spill" / "kernel")
LOG = str(FIXTURES / "notices" / "mixed.log")


def kernels(stem):
    return list(runner.kernels_of_files(stem).values())


def decls(*entries, known=()):
    base = {"kernel": "tile", "reason": "measured and accepted"}
    return declare.load_declarations(json.dumps({"declarations": [{**base, **e} for e in entries]}), registry(),
                                     [*known, *declare.rule_names()])


def manifest_of(stem=SPILL, log=None, ledger=None):
    read = notices.read_notices(Path(log).read_text()) if log else None
    return manifest.collect(kernels(stem), registry(), notices=read, ledger=ledger)


class Writing(unittest.TestCase):
    def test_every_rule_is_there_a_kernel_with_its_count_and_sites(self):
        (place,) = manifest_of().places
        self.assertEqual((place.place, place.kernel), ("sample::tile()", True))
        self.assertEqual(set(place.rules), set(registry()))
        stack = place.rules["stack_memory"]
        self.assertEqual((stack.count, stack.default, stack.bound, stack.declarable), (128, 0, False, True))
        self.assertEqual([(s.where, s.count) for s in stack.sites], [("0x0020", 128)])
        self.assertEqual(place.rules["load_in_trip"].count, 0)            # a rule that found nothing is there, with 0
        self.assertEqual(place.rules["load_in_trip"].sites, ())

    def test_the_file_is_stable_json_and_reads_back_whole(self):
        a, b = manifest.dumps(manifest_of(log=LOG)), manifest.dumps(manifest_of(log=LOG))
        self.assertEqual(a, b)                                              # the same inputs, the same bytes
        doc = json.loads(a)
        self.assertEqual((doc["format"], doc["notices"]), ("warpline-manifest/1", 10))
        self.assertEqual(manifest.dumps(manifest.loads(a)), a)
        self.assertEqual(manifest.loads(a), manifest_of(log=LOG))
        self.assertNotIn("/home", a)                                        # no path in it

    def test_notices_are_rule_inputs_at_their_place(self):
        got = manifest_of(log=LOG)
        by = {p.place: p for p in got.places}
        tile = by["sample::tile()"].rules
        self.assertEqual((tile["register_spill"].count, tile["local_memory_used"].count), (1, 1))
        self.assertEqual(tile["wgmma_arrive_injected"].default, None)           # accepted by default, counted all the same
        self.assertEqual(by["the build"].rules["setmaxnreg_ignored"].count, 2)
        self.assertEqual((by["tile.cu(1706)"].rules["variable_unset"].count, by["tile.cu(1706)"].kernel), (1, False))
        self.assertEqual(by["the build"].rules["notice_without_row"].declarable, False)

    def test_the_binary_with_no_notices_says_none_were_read(self):
        self.assertIsNone(json.loads(manifest.dumps(manifest_of()))["notices"])

    def test_function_size_is_in_the_manifest_without_padding(self):
        (place,) = manifest_of(str(FIXTURES / "rules" / "function_size" / "padded" / "kernel")).places
        size = place.rules["function_size"]
        self.assertEqual((size.count, size.bound), (4, True))


class Reading(unittest.TestCase):
    def test_a_file_that_is_not_manifest_is_refused(self):
        good = json.loads(manifest.dumps(manifest_of()))
        bad_count = json.loads(manifest.dumps(manifest_of()))
        bad_count["places"][0]["rules"]["stack_memory"]["count"] = 1
        no_rules = {**good, "places": [{"place": "x"}]}
        for text in ("[", "[]", "{}", json.dumps({**good, "format": "other"}), json.dumps(bad_count), json.dumps(no_rules),
                     json.dumps({**good, "notices": "many"})):
            with self.assertRaises(manifest.ManifestError, msg=text[:60]):
                manifest.loads(text)


class Comparing(unittest.TestCase):
    def test_a_refusal_names_kernel_rule_count_allowed_and_undeclared(self):
        (r,) = declare.compare(manifest_of(), []).refusals
        self.assertEqual((r.kernel, r.rule, r.count, r.allowed, r.reason), ("sample::tile()", "stack_memory", 128, 0, None))
        self.assertEqual(r.lines()[0], "sample::tile(): stack_memory: 128 found, 0 declared (undeclared)")

    def test_a_refusal_names_the_declarations_reason(self):
        (r,) = declare.compare(manifest_of(), decls({"rule": "stack_memory", "allow": 64, "reason": "the staging array"})).refusals
        self.assertEqual((r.count, r.allowed, r.reason), (128, 64, "the staging array"))
        self.assertEqual(r.lines()[0], "sample::tile(): stack_memory: 128 found, 64 declared (the staging array)")

    def test_a_declared_count_holds_and_quiet_is_quiet(self):
        self.assertEqual(declare.compare(manifest_of(), decls({"rule": "stack_memory", "allow": 128})).refusals, [])

    def test_the_first_declaration_in_file_order_applies(self):
        first_loose = decls({"rule": "stack_memory", "allow": 128}, {"rule": "stack_memory", "allow": 0})
        self.assertEqual(declare.compare(manifest_of(), first_loose).refusals, [])
        first_tight = decls({"rule": "stack_memory", "allow": 0, "reason": "tight first"}, {"rule": "stack_memory", "allow": 128})
        (r,) = declare.compare(manifest_of(), first_tight).refusals
        self.assertEqual(r.reason, "tight first")

    def test_a_bound_is_shown_with_none_and_refused_above_the_declared_bound(self):
        manifest_ = manifest_of(str(FIXTURES / "rules" / "function_size" / "large" / "kernel"))
        got = declare.compare(manifest_, [])
        self.assertFalse([r for r in got.refusals if r.rule == "function_size"])
        self.assertTrue([line for line in got.figures if "function_size: 11 (no bound declared)" in line])
        size = lambda allow, reason="r": [r for r in declare.compare(manifest_, decls(
            {"rule": "function_size", "allow": allow, "reason": reason})).refusals if r.rule == "function_size"]
        self.assertEqual(size(11), [])
        (r,) = size(10, "the design's size")
        self.assertEqual((r.kernel, r.rule, r.count, r.allowed, r.reason), ("sample::tile()", "function_size", 11, 10, "the design's size"))

    def test_a_declaration_tightens_a_notice_accepted_by_default(self):
        f = manifest_of(log=LOG)
        self.assertFalse([r for r in declare.compare(f, []).refusals if r.rule == "wgmma_arrive_injected"])
        (r,) = [r for r in declare.compare(f, decls({"rule": "wgmma_arrive_injected", "allow": 0, "reason": "no injected arrive"}
                                                    )).refusals if r.rule == "wgmma_arrive_injected"]
        self.assertEqual((r.kernel, r.count, r.allowed, r.reason), ("sample::tile()", 1, 0, "no injected arrive"))

    def test_a_declaration_may_allow_what_is_refused_by_default(self):
        f = manifest_of(log=LOG)
        self.assertIn("register_spill", {r.rule for r in declare.compare(f, []).refusals})
        got = declare.compare(f, decls({"rule": "register_spill", "allow": 1}, {"rule": "local_memory_used", "allow": 1}))
        self.assertNotIn("register_spill", {r.rule for r in got.refusals})

    def test_an_unknown_notice_is_refused_whatever_is_declared(self):
        got = declare.compare(manifest_of(log=LOG), decls({"rule": "stack_memory", "allow": 128}))
        self.assertEqual(len([r for r in got.refusals if r.rule == "notice_without_row"]), 2)

    def test_a_declaration_naming_no_place_is_stale_and_no_refusal(self):
        got = declare.compare(manifest_of(), decls({"rule": "stack_memory", "allow": 128}, {"rule": "stack_memory", "kernel": "^gone$", "allow": 0}))
        self.assertEqual([d.kernel.pattern for d in got.stale], ["^gone$"])
        self.assertEqual(got.refusals, [])


class Commands(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def write(self, text, name="f.json"):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text)
        return str(path)

    def declarations(self, *entries):
        return self.write(json.dumps({"declarations": [{"kernel": "tile", "reason": "r", **e} for e in entries]}))

    def test_manifest_writes_the_file_and_compare_refuses_from_it(self):
        path = self.write("", "manifest.json")
        code, out, err = self.run_main("manifest", SPILL, "--fixture", "-o", path)
        self.assertEqual((code, out), (0, ""), err)
        self.assertEqual(manifest.loads(Path(path).read_text()).places, manifest_of().places)
        code, out, _ = self.run_main("compare", path, "--declare", self.declarations())
        self.assertEqual(code, 1)
        self.assertIn("sample::tile(): stack_memory: 128 found, 0 declared (undeclared)", out)
        code, out, _ = self.run_main("compare", path, "--declare", self.declarations({"rule": "stack_memory", "allow": 128}))
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[-1], f"1 kernel(s), {len(registry())} rule(s): 0 refusal(s)")

    def test_compare_prints_the_declarations_reason(self):
        path = self.write(manifest.dumps(manifest_of()))
        _, out, _ = self.run_main("compare", path, "--declare",
                                  self.declarations({"rule": "stack_memory", "allow": 100, "reason": "measured once"}))
        self.assertIn("128 found, 100 declared (measured once)", out)

    def test_compare_tightens_through_the_command(self):
        path = self.write(manifest.dumps(manifest_of(log=LOG)))
        code, out, _ = self.run_main("compare", path, "--declare", self.declarations(
            {"rule": "stack_memory", "allow": 128}, {"rule": "register_spill", "allow": 1},
            {"rule": "local_memory_used", "allow": 1}, {"rule": "wgmma_arrive_injected", "allow": 0, "reason": "none here"}))
        self.assertEqual(code, 1)
        self.assertIn("sample::tile(): wgmma_arrive_injected: 1 found, 0 declared (none here)", out)
        self.assertNotIn("register_spill", out)

    def test_compare_exits_2_on_an_unusable_file(self):
        good = self.write(manifest.dumps(manifest_of()))
        for argv in (("compare", self.write("[]"), "--declare", self.declarations()),
                     ("compare", self.write("nope"), "--declare", self.declarations()),
                     ("compare", str(Path(good).with_name("missing.json")), "--declare", self.declarations()),
                     ("compare", good, "--declare", self.declarations({"rule": "no_such_rule"})),
                     ("compare", good, "--declare", self.write("{}"))):
            code, _, err = self.run_main(*argv)
            self.assertEqual(code, 2, err)

    def test_check_is_the_two_in_one_and_can_write_the_manifest(self):
        path = self.write("", "manifest.json")
        code, out, _ = self.run_main("check", SPILL, "--fixture", "--manifest", path, "--declare", self.declarations())
        self.assertEqual(code, 1)
        self.assertEqual(manifest.loads(Path(path).read_text()).places, manifest_of().places)


if __name__ == "__main__":
    unittest.main()
