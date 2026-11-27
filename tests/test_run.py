"""THE RUNNER (docs/run-layer.md): `warpline run` starts the user's command once, stamped, profiled or both, collects what it
wrote into the run's directory and writes the run manifest from those files with `run-manifest`'s code. The task here is a stub
that copies records the recorder's helpers made; the profiler is a stub `ncu` on PATH that prints the profile rules' fixtures
as its pages and logs how it was called, so no test needs a GPU or Nsight Compute."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tests.recorder.helpers import stamped_run
from warpline import command, manifest
from warpline.checker import cli as checker_cli, ncu, runner
from warpline.recorder import cli as recorder_cli, record

HERE = Path(__file__).parent
STUB = HERE / "run_stub.py"
PROFILES = HERE / "checker" / "fixtures" / "profile_rules"
DETAILS = PROFILES / "profiler_verdicts" / "fired" / "details.csv"          # results 0 (tile) and 1 (edge)
SOURCE = PROFILES / "shared_wavefronts_excess" / "conflicted" / "source.csv"  # the tile kernel's
PLACE = "sample::tile_kernel<0>(Params)"
VERSION = "note: profiled with Nsight Compute 2025.3.0.0 (build 36273991) (public-release)"


def call(argv, modules=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = command.main(list(argv), modules if modules is not None else [recorder_cli, checker_cli])
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "run"
        # the stub profiler, first on PATH (no CUDA_HOME, so the runner finds it there)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "ncu").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{STUB}" ncu "$@"\n')
        (bin_dir / "ncu").chmod(0o755)
        edge = self.tmp / "edge.source.csv"
        edge.write_text((PROFILES / "shared_wavefronts_excess" / "conflict_free" / "source.csv").read_text()
                        .replace("sample::tile_kernel<0>(Params)", "sample::edge_kernel<0, 1>(Params)"))
        self.log = self.tmp / "ncu.log"
        self.env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", "STUB_NCU_LOG": str(self.log),
                    "STUB_NCU_DETAILS": str(DETAILS), "STUB_NCU_SOURCE0": str(SOURCE), "STUB_NCU_SOURCE1": str(edge)}
        saved = {k: os.environ.get(k) for k in [*self.env, "CUDA_HOME", "STUB_NCU_CAPTURE"]}
        os.environ.update(self.env)
        os.environ.pop("CUDA_HOME", None)
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])

    def record(self, name, **kw):
        path = self.tmp / f"{name}.json"
        record.dump(stamped_run(**kw), path)
        return str(path)

    def task(self, *args):
        return ["--", sys.executable, str(STUB), "task", *args]

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


class Arguments(Run):
    def test_a_run_needs_an_instrument_a_command_and_an_empty_directory(self):
        code, _, err = call(["run", "-o", str(self.out), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("a run takes an instrument", err)
        code, _, err = call(["run", "--stamps", "-o", str(self.out), "--"])
        self.assertEqual(code, 2)
        self.assertIn("no command", err)
        self.out.mkdir()
        (self.out / "old.record.json").write_text("{}")
        code, _, err = call(["run", "--stamps", "-o", str(self.out), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("is not an empty directory", err)

    def test_an_instrument_whose_module_is_absent_is_refused_by_name(self):
        code, _, err = call(["run", "--profile", "SourceCounters", "-o", str(self.out), *self.task()], [recorder_cli])
        self.assertEqual(code, 2)
        self.assertIn("--profile needs the checker module", err)
        code, _, err = call(["run", "--stamps", "-o", str(self.out), *self.task()], [checker_cli])
        self.assertEqual(code, 2)
        self.assertIn("--stamps needs the recorder module", err)
        self.assertFalse(self.out.exists())                      # refused before anything is made or run

    def test_the_command_after_the_separator_is_the_tasks_whole(self):
        note = self.tmp / "note.json"
        code, _, err = call(["run", "--stamps", "-o", str(self.out),
                             *self.task("--record", self.record("a"), "a", "--note", str(note), "-o", "--place", "x")])
        self.assertEqual(code, 0, err)
        got = json.loads(note.read_text())
        self.assertEqual(got["args"], ["-o", "--place", "x"])     # the task's own flags reach it untouched
        self.assertEqual(got["record_dir"], str(self.out.resolve() / recorder_cli.RECORDS))

    def test_a_failed_task_is_not_a_run(self):
        code, _, err = call(["run", "--stamps", "--place", PLACE, "-o", str(self.out),
                             *self.task("--record", self.record("a"), "a", "--exit", "3")])
        self.assertEqual(code, 2)
        self.assertIn("the command exited 3", err)
        self.assertFalse((self.out / "run.manifest.json").exists())

    def test_the_profile_spec_is_sections_and_metrics(self):
        self.assertEqual(runner.profile_items(["SourceCounters,sm__cycles_elapsed.avg", "SpeedOfLight"]),
                         (["SourceCounters", "SpeedOfLight"], ["sm__cycles_elapsed.avg"]))
        code, _, err = call(["run", "--profile", "SourceCounters,,x", "-o", str(self.out), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("an empty item", err)


class Collecting(Run):
    def test_a_stamped_run_collects_the_records_and_writes_what_run_manifest_writes(self):
        a, b = self.record("a"), self.record("b", waits=("consumer",))
        code, out, err = call(["run", "--stamps", "--place", PLACE, "-o", str(self.out),
                               *self.task("--record", b, "two", "--record", a, "one")])
        self.assertEqual(code, 0, err)
        records = self.out / "records"
        self.assertEqual(sorted(p.name for p in records.iterdir()), ["one.record.json", "two.record.json"])
        self.assertIn("--stamps: records/one.record.json, records/two.record.json", out)
        made = (self.out / "run.manifest.json").read_text()
        # the same bytes as `run-manifest` on the collected records, read in the order of their names
        again = self.tmp / "again.json"
        code, _, err = call(["run-manifest", "--record", str(records / "one.record.json"), "--record",
                             str(records / "two.record.json"), "--place", PLACE, "-o", str(again)])
        self.assertEqual(code, 0, err)
        self.assertEqual(made, again.read_text())
        self.assertEqual(self.calls(), [])                        # no profiler without --profile

    def test_a_profiled_run_runs_the_task_under_the_profiler_and_exports_the_pages_its_readers_read(self):
        code, out, err = call(["run", "--profile", "SourceCounters,sm__cycles_elapsed.avg", "--profile", "SpeedOfLight",
                               "--place", PLACE, "-o", str(self.out), *self.task("x")])
        self.assertEqual(code, 0, err)
        report = str(self.out.resolve() / "profile")
        start, *imports, version = self.calls()
        self.assertEqual(start[:10], ["--profile-from-start", "off", "--export", report, "--section", "SourceCounters",
                                      "--section", "SpeedOfLight", "--metrics", "sm__cycles_elapsed.avg"])
        self.assertEqual(start[10:], [sys.executable, str(STUB), "task", "x"])
        rep = report + ".ncu-rep"
        # the export flags are the readers' own (ncu.PAGES): the details page once, the source page a result
        self.assertEqual(imports, [["--import", rep, *ncu.PAGES["details"]],
                                   ["--import", rep, *ncu.PAGES["source"], "--launch-skip", "0", "--launch-count", "1"],
                                   ["--import", rep, *ncu.PAGES["source"], "--launch-skip", "1", "--launch-count", "1"]])
        self.assertEqual(version, ["--version"])
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["details.csv", "ncu-version.txt", "profile.ncu-rep", "run.manifest.json", "source.0.csv",
                          "source.1.csv"])
        self.assertEqual((self.out / "ncu-version.txt").read_text(), "2025.3.0.0 (build 36273991) (public-release)\n")
        m = manifest.loads((self.out / "run.manifest.json").read_text())
        self.assertIn(VERSION, m.said)                            # the profiler's version, in the manifest
        mine, edge = m.places
        self.assertEqual((mine.place, mine.rules["shared_wavefronts_excess"].count, mine.rules["profiler_verdicts"].count),
                         (PLACE, 98304, 2))
        self.assertEqual((edge.place, edge.rules["shared_wavefronts_excess"].count), ("void sample::edge_kernel<0, 1>(Params)", 0))
        # the same manifest as `run-manifest` on the exports, with the version said after the profile's lines
        again = self.tmp / "again.json"
        pages = [str(self.out / n) for n in ("details.csv", "source.0.csv", "source.1.csv")]
        call(["run-manifest", *[x for p in pages for x in ("--profile", p)], "--place", PLACE, "-o", str(again)])
        plain = manifest.loads(again.read_text())
        self.assertEqual(m.said, (*plain.said, VERSION))
        self.assertEqual(m.places, plain.places)

    def test_without_the_source_section_no_source_page_is_exported_and_its_rule_is_said_not_run(self):
        code, _, err = call(["run", "--profile", "SpeedOfLight", "--place", PLACE, "-o", str(self.out), *self.task()])
        self.assertEqual(code, 0, err)
        self.assertEqual([c[2:] for c in self.calls() if c[:1] == ["--import"]], [list(ncu.PAGES["details"])])
        m = manifest.loads((self.out / "run.manifest.json").read_text())
        self.assertIn(f"note: profile rule shared_wavefronts_excess not run on {PLACE!r}: the profile has no source page", m.said)

    def test_stamped_and_profiled_in_one_run(self):
        code, _, err = call(["run", "--stamps", "--profile", "SourceCounters", "--place", PLACE, "-o", str(self.out),
                             *self.task("--record", self.record("a"), "a")])
        self.assertEqual(code, 0, err)
        (place, _) = manifest.loads((self.out / "run.manifest.json").read_text()).places
        self.assertEqual(place.figures["window"]["period"], 1000)
        self.assertEqual(place.rules["shared_wavefronts_excess"].count, 98304)

    def test_without_a_place_the_files_are_collected_and_no_manifest_is_written(self):
        code, out, err = call(["run", "--stamps", "-o", str(self.out), *self.task("--record", self.record("a"), "a")])
        self.assertEqual(code, 0, err)
        self.assertFalse((self.out / "run.manifest.json").exists())
        self.assertIn(f"warpline run-manifest --record {self.out.resolve() / 'records' / 'a.record.json'} --place NAME", out)

    def test_a_run_that_wrote_no_record_or_captured_nothing_is_refused(self):
        code, _, err = call(["run", "--stamps", "--place", PLACE, "-o", str(self.out), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("the task wrote no record", err)
        os.environ["STUB_NCU_CAPTURE"] = "0"
        code, _, err = call(["run", "--profile", "SourceCounters", "--place", PLACE, "-o", str(self.tmp / "r2"), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("the profiler captured nothing", err)

    def test_a_record_that_is_not_one_is_refused(self):
        bad = self.tmp / "bad.json"
        bad.write_text('{"format": "warpline/9"}')
        code, _, err = call(["run", "--stamps", "--place", PLACE, "-o", str(self.out), *self.task("--record", str(bad), "x")])
        self.assertEqual(code, 2)
        self.assertIn("x.record.json", err)

    def test_an_export_whose_columns_a_version_dropped_is_refused(self):
        short = self.tmp / "short.source.csv"
        short.write_text('"Kernel Name","void sample::tile_kernel<0>(Params)",""\n"Address","Source","Instructions Executed"\n'
                         '"0x10","LDS R0, [R1]","5"\n')
        os.environ["STUB_NCU_SOURCE0"] = str(short)
        code, _, err = call(["run", "--profile", "SourceCounters", "--place", PLACE, "-o", str(self.out), *self.task()])
        self.assertEqual(code, 2)
        self.assertIn("lacks the column(s) ['Address Space']", err)


if __name__ == "__main__":
    unittest.main()
