"""THE RUN MANIFEST (docs/run-manifest.md): one file a run of a kernel, written by the common verb `run-manifest` from what each
module present gives (the recorder the figures of the records, the checker the profile rules on a profiler's exports), and
judged with the same checkers as the static manifest. The records are made with the recorder's own helpers; the exports are
the profile rules' fixtures."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.recorder.helpers import stamped_run
from warpline import command, declare, manifest
from warpline.checker import cli as checker_cli
from warpline.recorder import cli as recorder_cli, record

HERE = Path(__file__).parent
PROFILES = HERE / "checker" / "fixtures" / "profile_rules"
SOURCE = str(PROFILES / "shared_wavefronts_excess" / "conflicted" / "source.csv")
DETAILS = str(PROFILES / "profiler_verdicts" / "fired" / "details.csv")
SPILL = str(HERE / "checker" / "fixtures" / "rules" / "stack_memory" / "spill" / "kernel")   # `sample::tile()`
PLACE = "sample::tile_kernel<0>(Params)"                                                   # the exports' kernel


def call(argv, modules=None):
    """The one command (both modules, or `modules`) on `argv`: (exit status, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = command.main(list(argv), modules if modules is not None else [recorder_cli, checker_cli])
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class Files(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def record(self, name="run.json", **kw):
        path = self.path(name)
        record.dump(stamped_run(**kw), path)
        return path

    def write(self, name, doc):
        path = self.path(name)
        Path(path).write_text(doc if isinstance(doc, str) else json.dumps(doc))
        return path

    def run_manifest(self, *inputs, place=PLACE, modules=None):
        """`run-manifest` on the inputs: the file it wrote, read back (and its text)."""
        out = self.path("run.manifest.json")
        code, _, err = call(["run-manifest", *inputs, "--place", place, "-o", out], modules)
        self.assertEqual(code, 0, err)
        text = Path(out).read_text()
        return manifest.loads(text), text

    def declarations(self, *entries, coverage=None, name="declarations.json"):
        base = {"kernel": "tile", "reason": "the run's own budget"}
        return self.write(name, {"declarations": [{**base, **e} for e in entries],
                                                **({"coverage": coverage} if coverage is not None else {})})


class Writing(Files):
    def test_a_stamped_only_run_gives_the_records_figures(self):
        m, text = self.run_manifest("--record", self.record())
        self.assertEqual(json.loads(text)["format"], "warpline-run-manifest/1")
        self.assertEqual(m.format, manifest.RUN_FORMAT)
        (place,) = m.places
        self.assertEqual((place.place, place.kernel, place.rules), (PLACE, True, {}))
        f = place.figures
        self.assertEqual(list(f), ["health", "launch", "window", "roles"])
        self.assertEqual(f["health"], {"lanes": 2, "roles": 2, "stamps": f["health"]["stamps"], "dropped": 0, "findings": {},
                                       "open_at_end": {"lanes": 0, "regions": 0}})
        #: the grid's span is over the record's one scope: the first scope's own
        self.assertEqual(f["launch"], {"cycles": 12150, "ns": 12150, "grid_ns": 12150, "scopes": 1})
        self.assertEqual((f["window"]["anchor"], f["window"]["period"], f["window"]["windows"]), ("tick.begin", 1000, 10))
        self.assertIn("period_se", f["window"])
        producer = f["roles"]["producer"]
        self.assertAlmostEqual(producer["phases"]["load"]["cycles"], 300)
        self.assertAlmostEqual(producer["phases"]["load"]["jitter"], 10)
        self.assertIn("cycles_se", producer["phases"]["load"])
        self.assertAlmostEqual(producer["waits"]["cycles"], 290)
        self.assertIn("cycles_se", producer["waits"])
        self.assertEqual(list(producer["waits"]["by_barrier"]), ["ack"])
        self.assertAlmostEqual(producer["work"], 710)
        self.assertAlmostEqual(f["roles"]["consumer"]["work"], 900)
        self.assertEqual(m.said, ("note: profile rules not run (no --profile)",))
        fields = manifest.fields(place)
        self.assertIn("roles/producer/phases/load/cycles", fields)
        self.assertNotIn("window/anchor", fields)                            # a note, never a field

    def test_a_profiled_only_run_gives_the_profile_rules_and_no_figures(self):
        m, text = self.run_manifest("--profile", SOURCE)
        (place,) = m.places
        self.assertEqual((place.place, place.kernel, place.figures), (PLACE, True, {}))
        self.assertEqual(json.loads(text)["places"][0]["figures"], {})
        wave = place.rules["shared_wavefronts_excess"]
        self.assertEqual((wave.count, wave.default, wave.bound, wave.declarable), (98304, 0, False, True))
        self.assertEqual(wave.sites[0].where, f"{0x900f21d90 - 0x900f1e200:#06x}")   # the instruction's offset in the kernel
        doc_rule = json.loads(text)["places"][0]["rules"]["shared_wavefronts_excess"]
        self.assertEqual(list(doc_rule), ["count", "bound", "default", "declarable", "message", "fix", "sites"])
        self.assertEqual(m.said[0], "note: no record given: the run's figures are not measured")
        self.assertIn("profile rule profiler_verdicts not run", m.said[1])

    def test_a_profiled_run_gives_its_launchs_counters(self):
        counters = str(HERE / "floor" / "tile.counters.csv")
        m, text = self.run_manifest("--profile", counters, place="sample::tile()")
        found = m.places[0].figures["counters"]
        self.assertEqual((found["dram__bytes_read.sum"], found["sm__inst_executed.sum"]), (110016, 6301))
        self.assertIn("counters/lts__t_sectors_srcunit_tex.sum", manifest.fields(m.places[0]))
        #: the details page's other rows are a section's figures in words, never counters
        m, _ = self.run_manifest("--profile", DETAILS)
        self.assertNotIn("counters", m.places[0].figures)
        #: two launches of the kernel: a run manifest's counters are one launch's
        lines = Path(counters).read_text().splitlines()
        twice = self.write("twice.csv", "\n".join([*lines, *(l.replace('"0","1"', '"1","1"', 1) for l in lines[1:])]) + "\n")
        m, _ = self.run_manifest("--profile", twice, place="sample::tile()")
        self.assertNotIn("counters", m.places[0].figures)
        self.assertIn("note: counters not written: the profile holds 2 launches of 'sample::tile()'", "\n".join(m.said))

    def test_a_run_with_both_gives_both(self):
        m, _ = self.run_manifest("--record", self.record(), "--profile", SOURCE, "--profile", DETAILS)
        mine = m.places[0]
        self.assertEqual(set(mine.figures), {"health", "launch", "window", "roles"})
        self.assertEqual({n: r.count for n, r in mine.rules.items()}, {"profiler_verdicts": 2, "shared_wavefronts_excess": 98304})
        other = m.places[1]                                              # the details page's second kernel
        self.assertEqual((other.place, other.kernel, other.figures), ("void sample::edge_kernel<0, 1>(Params)", False, None))
        self.assertIn("note: the profile of 'void sample::edge_kernel<0, 1>(Params)' matches no place "
                      "('sample::tile_kernel<0>(Params)')", m.said)

    def test_the_same_inputs_write_the_same_bytes_and_it_reads_back_whole(self):
        rec = self.record()
        _, a = self.run_manifest("--record", rec, "--profile", SOURCE)
        _, b = self.run_manifest("--record", rec, "--profile", SOURCE)
        self.assertEqual(a, b)
        self.assertEqual(manifest.dumps(manifest.loads(a)), a)
        for mark in (self.dir.name, "run.json", "source.csv", os.uname().nodename):
            self.assertNotIn(mark, a)                                    # no path, no input's name, no host
        other = self.record("elsewhere.json")
        self.assertEqual(self.run_manifest("--record", other, "--profile", SOURCE)[1], a)

    def test_what_is_not_measured_is_absent_and_said(self):
        m, _ = self.run_manifest("--record", self.record(waits=("producer",), edges=False))
        f = m.places[0].figures
        self.assertNotIn("launch", f)
        self.assertTrue(any(s.startswith("note: the launch is not measured") for s in m.said), m.said)
        consumer = f["roles"]["consumer"]
        self.assertEqual(set(consumer), {"period", "phases"})            # no waits, no work: never 0
        self.assertIn("note: waits not stamped for role consumer (it arrives at req): no waits, no work", m.said)
        self.assertIn("waits", f["roles"]["producer"])

    def test_several_records_of_one_run_give_each_roles_waits_from_the_record_that_stamps_them(self):
        a = self.record("a.json", waits=("producer",))
        b = self.record("b.json", waits=("consumer",))
        m, text = self.run_manifest("--record", a, "--record", b)
        roles = m.places[0].figures["roles"]
        self.assertAlmostEqual(roles["producer"]["waits"]["cycles"], 290)
        self.assertAlmostEqual(roles["consumer"]["waits"]["cycles"], 100)
        self.assertNotIn("cycles_se", roles["consumer"]["waits"])        # several records give no standard error
        self.assertIn("note: the waits of role consumer come from several records: no standard error of them", m.said)
        self.assertNotIn("a.json", text)

    def test_the_text_lists_every_field_and_the_lines_said(self):
        out = self.path("run.txt")
        code, _, err = call(["run-manifest", "--record", self.record(), "--place", PLACE, "--text", "-o", out])
        self.assertEqual(code, 0, err)
        text = Path(out).read_text()
        self.assertIn(f"{PLACE}  (a run; windows anchored on tick.begin)", text)
        self.assertIn("  window/period = 1000", text)
        self.assertIn("  roles/producer/waits/by_barrier/ack/cycles = ", text)
        self.assertTrue(text.rstrip().endswith("note: profile rules not run (no --profile)"))


class Refusing(Files):
    def test_a_run_manifest_needs_an_input(self):
        code, _, err = call(["run-manifest", "--place", PLACE])
        self.assertEqual(code, 2)
        self.assertIn("needs at least one input", err)
        code, _, err = call(["run-manifest", "--record", self.record()])            # no place
        self.assertEqual(code, 2)
        self.assertIn("--place", err)

    def test_an_input_whose_module_is_absent_is_refused_by_name(self):
        code, _, err = call(["run-manifest", "--profile", SOURCE, "--place", PLACE], [recorder_cli])
        self.assertEqual(code, 2)
        self.assertIn("--profile needs the checker module", err)
        code, _, err = call(["run-manifest", "--record", self.record(), "--place", PLACE], [checker_cli])
        self.assertEqual(code, 2)
        self.assertIn("--record needs the recorder module", err)
        made = self.run_manifest("--record", self.record(), modules=[recorder_cli])[0]
        code, _, err = call(["compare", self.write("m.json", manifest.dumps(made)), "--declare", self.declarations(),
                             "--ledger", self.write("ledger.json", {"rows": []})], [recorder_cli])
        self.assertEqual(code, 2)
        self.assertIn("--ledger needs the checker module", err)

    def test_an_unreadable_input_exits_2(self):
        bad = self.write("bad.json", "{")
        code, _, err = call(["run-manifest", "--record", bad, "--place", PLACE])
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("warpline: "), err)

    def test_a_file_that_is_not_a_run_manifest_is_refused(self):
        good = json.loads(self.run_manifest("--record", self.record(), "--profile", SOURCE)[1])

        def changed(edit):
            doc = json.loads(json.dumps(good))
            edit(doc)
            return json.dumps(doc)

        place = lambda d: d["places"][0]
        bad = [
            changed(lambda d: place(d)["figures"]["window"].update(period="1000")),       # a figure that is text
            changed(lambda d: place(d)["figures"]["health"].update(lanes=1.5)),           # a structural figure not a count
            changed(lambda d: place(d)["figures"].update(clock={"hz": 1})),               # a part no run has
            changed(lambda d: place(d).pop("figures")),                                   # a kernel with no figures
            changed(lambda d: d.update(notices=3)),                                       # a static manifest's key
            changed(lambda d: place(d)["rules"]["shared_wavefronts_excess"].update(count=1)),   # not its sites' sum
        ]
        for text in bad:
            with self.assertRaises(manifest.ManifestError, msg=text[:120]):
                manifest.loads(text)


class Judging(Files):
    def made(self, *inputs, place=PLACE):
        m, _ = self.run_manifest(*inputs, place=place)
        return self.write("run.manifest.json", manifest.dumps(m))

    def compare(self, path, *entries, coverage=None, more=()):
        return call(["compare", path, *more, "--declare", self.declarations(*entries, coverage=coverage)])

    def test_violations_on_the_profile_rules(self):
        path = self.made("--profile", SOURCE)
        code, out, _ = self.compare(path)
        self.assertEqual(code, 1)
        self.assertIn(f"{PLACE}: shared_wavefronts_excess: 98304 found, 0 declared (undeclared)", out)
        code, out, _ = self.compare(path, {"rule": "shared_wavefronts_excess", "allow": 98304})
        self.assertEqual(code, 0, out)
        self.assertEqual(out.splitlines()[-1], "1 kernel(s), 1 rule(s): 0 refusal(s)")

    def test_a_bound_on_a_run_figure_at_most_or_at_least(self):
        path = self.made("--record", self.record())
        code, out, _ = self.compare(path, {"check": "bound", "figure": "window/period", "at_most": 999.5})
        self.assertEqual(code, 1)
        self.assertIn(f"{PLACE}: bound: window/period is 1000", out)
        self.assertIn("at most 999.5 declared", out)
        ok = [{"check": "bound", "figure": "window/period", "at_most": 1000.5},
              {"check": "bound", "figure": "window/windows", "at_least": 10, "reason": "the evidence a period rests on"},
              {"check": "bound", "figure": "roles/.*/work_share", "at_most": 0.95},
              {"check": "bound", "figure": "health/(lanes|roles)", "exactly": 2},           # structural: exactly is allowed
              {"check": "bound", "figure": "health/dropped", "exactly": 0}]
        code, out, _ = self.compare(path, *ok)
        self.assertEqual(code, 0, out)
        code, out, _ = self.compare(path, {"check": "bound", "figure": "window/windows", "at_least": 11})
        self.assertIn("window/windows is 10, at least 11 declared", out)

    def test_a_figure_not_measured_is_named_by_no_field(self):
        path = self.made("--record", self.record(waits=("producer",)))
        code, out, _ = self.compare(path, {"check": "bound", "figure": "roles/consumer/waits/cycles", "at_most": 1000})
        self.assertEqual(code, 1)
        self.assertIn("'roles/consumer/waits/cycles' names no field of the kernel", out)
        self.assertIn("note: waits not stamped for role consumer", out)

    def test_exactly_on_a_varying_run_figure_is_refused_at_load(self):
        path = self.made("--record", self.record())
        code, out, err = self.compare(path, {"check": "bound", "figure": "window/period", "exactly": 1000})
        self.assertEqual((code, out), (2, ""))
        self.assertIn("a run's figure varies from run to run", err)
        self.assertIn("health/lanes, health/roles, health/dropped", err)

    def test_a_hash_is_refused_on_a_run_manifest(self):
        path = self.made("--record", self.record())
        code, _, err = self.compare(path, {"check": "hash", "fields": "window/.*", "hash": "any"})
        self.assertEqual(code, 2)
        self.assertIn("a hash is refused on a run manifest", err)

    def test_coverage_and_waive_as_for_the_static_manifest(self):
        path = self.made("--record", self.record(), "--profile", SOURCE)
        code, out, _ = self.compare(path, {"rule": "shared_wavefronts_excess", "allow": 98304}, coverage=True)
        self.assertEqual(code, 1)
        self.assertIn(f"{PLACE}: coverage: ", out)
        self.assertIn("  window/period_se\n", out)
        self.assertNotIn("  rules/shared_wavefronts_excess\n", out)       # declared: covered
        code, out, _ = self.compare(path, {"rule": "shared_wavefronts_excess", "allow": 98304},
                                    {"check": "bound", "figure": "window/period", "at_most": 1100},
                                    {"check": "waive", "fields": "(health|launch|roles)/.*|window/(period_.*|windows)",
                                     "reason": "the period alone is held"}, coverage=True)
        self.assertEqual(code, 0, out)

    def test_check_judges_a_binary_and_its_run_against_one_declarations_file(self):
        run = self.made("--record", self.record(), "--profile", SOURCE, place="sample::tile()")   # the binary's kernel
        entries = [{"rule": "stack_memory", "allow": 128}, {"rule": "shared_wavefronts_excess", "allow": 98304},
                   {"check": "bound", "figure": "kernel/registers", "at_most": 255},
                   {"check": "bound", "figure": "window/period", "at_most": 1100}]
        declared = self.declarations(*entries)
        code, out, _ = call(["check", SPILL, "--fixture", "--declare", declared, "--run", run])
        self.assertEqual(code, 0, out)                      # neither bound names no field: the kernel's fields are both files'
        self.assertTrue(out.rstrip().endswith("0 refusal(s)"), out)
        code, out, _ = call(["check", SPILL, "--fixture", "--declare", self.declarations(entries[0], entries[2], name="short.json"),
                              "--run", run])
        self.assertEqual(code, 1)
        self.assertIn("sample::tile(): shared_wavefronts_excess: 98304 found, 0 declared", out)
        static = self.write("static.json", "")
        call(["manifest", SPILL, "--fixture", "-o", static])
        code, out, _ = call(["compare", static, run, "--declare", declared])     # the same, as two files
        self.assertEqual(code, 0, out)
        code, _, err = call(["check", SPILL, "--fixture", "--declare", declared, "--run", static])
        self.assertEqual(code, 2)
        self.assertIn("not a run manifest", err)

    def test_the_comparison_in_code(self):
        m = manifest.loads(Path(self.made("--profile", SOURCE)).read_text())
        got = declare.compare(m, [])
        self.assertEqual([(r.kernel, r.rule) for r in got.refusals], [(PLACE, "shared_wavefronts_excess")])
        self.assertEqual(got.said, m.said)


class OneModuleTree(Files):
    """A tree with one module's directory and the common files runs what it has, and writes and judges a run manifest of its
    own part."""

    def tree(self, without):
        root = Path(self.path("tree"))
        shutil.copytree(Path(command.__file__).parent, root / "warpline",
                        ignore=lambda d, names: [n for n in names if n in (without, "__pycache__")] if d == str(
                            Path(command.__file__).parent) else [n for n in names if n == "__pycache__"])
        return root

    def python(self, root, *argv):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(root)
        return subprocess.run([sys.executable, "-m", "warpline", *argv], cwd=root, env=env, capture_output=True, text=True)

    def test_the_recorder_alone_writes_and_judges_its_figures(self):
        root = self.tree("checker")
        self.assertFalse((root / "warpline" / "checker").exists())
        out = self.path("r.manifest.json")
        got = self.python(root, "run-manifest", "--record", self.record(), "--place", PLACE, "-o", out)
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertEqual(manifest.loads(Path(out).read_text()).places[0].figures["window"]["period"], 1000)
        got = self.python(root, "compare", out, "--declare",
                          self.declarations({"check": "bound", "figure": "window/period", "at_most": 1100}))
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        got = self.python(root, "run-manifest", "--profile", SOURCE, "--place", PLACE)
        self.assertEqual(got.returncode, 2)
        self.assertIn("--profile needs the checker module", got.stderr)

    def test_the_checker_alone_writes_and_judges_its_rules(self):
        root = self.tree("recorder")
        out = self.path("c.manifest.json")
        got = self.python(root, "run-manifest", "--profile", SOURCE, "--place", PLACE, "-o", out)
        self.assertEqual(got.returncode, 0, got.stderr)
        got = self.python(root, "compare", out, "--declare", self.declarations())
        self.assertEqual(got.returncode, 1)
        self.assertIn("shared_wavefronts_excess: 98304 found", got.stdout)
        got = self.python(root, "run-manifest", "--record", self.record(), "--place", PLACE)
        self.assertEqual(got.returncode, 2)
        self.assertIn("--record needs the recorder module", got.stderr)


if __name__ == "__main__":
    unittest.main()
