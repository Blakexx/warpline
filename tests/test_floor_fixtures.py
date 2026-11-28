"""THE FLOOR ON A CARD'S RECORDED RUNS (docs/floor.md#the-fixtures): for each example of `examples/floor/floor_examples.cu`,
its floor at the card's shipped descriptor is never above its measured launch (`launch/grid_ns`), the stream it names is the
one its counters show (its family leads the counters' shares of the rows' rates), every counter its work expects holds in
its band on the profiled run, a deliberately wrong work is refused by the counters naming the stream, and the idle check
passes it as declared (the `serial` example refused without its declaration). The runs were recorded once on the card
(`bazel run //:record_floor_fixtures`); this reads them, with no GPU.

Arguments: the device's fixture directory, its descriptor (`warpline/rates/<device>.json`), then each example's work
declaration (`examples/floor/<name>.work.json`)."""
import json
import sys
import unittest
from pathlib import Path

from warpline import checkers, floor, manifest
from warpline.checker import cli as checker_cli
from warpline.checker import ncu

DIRECTORY, DESCRIPTOR = Path(sys.argv[1]), Path(sys.argv[2])
WORKS = {Path(p).name.split(".")[0]: Path(p) for p in sys.argv[3:]}
sys.argv = sys.argv[:1]
#: the stream each example's design binds
BINDS = {"stream": "hbm.read", "shared": "shared.read", "mma": "tensor m16n8k16 f16.f32", "serial": "hbm.read"}
#: the counted streams' metrics the recording predates (the H100's runs were profiled with the rows' former metrics): their
#: bounds name no field of these runs until the card is recorded again
UNRECORDED = {"lts__t_sectors_srcunit_tex.sum", "sm__inst_executed_pipe_tensor_op_imma.sum"}


def recorded(name, work=None):
    """An example's work, its floor, its stamped run manifest, its profiled run manifest (written from the details page, as
    `warpline run --profile` writes it) and its counters (the one launch's metrics)."""
    here = DIRECTORY / name
    rates = floor.load_rates(DESCRIPTOR.read_text())
    work = floor.load_work(json.dumps(work) if work else WORKS[name].read_text())
    result = floor.floor(work, rates, json.loads((here / "parameters.json").read_text()))
    stamped = manifest.loads((here / "run.manifest.json").read_text())
    place = stamped.places[0].place
    profiled = manifest.loads(manifest.dumps(manifest.run_manifest(
        place, [checker_cli.give_rules([str(here / "counters.csv")], place)])))
    results = ncu.details_metrics((here / "counters.csv").read_text())
    if len(results) != 1:
        raise AssertionError(f"{name}: the counters hold {len(results)} results, not the one launch the region captured")
    return work, result, stamped, profiled, results[0][2]


def refusals(decl, *manifests):
    """The floor's checkers' refusals over the run manifests judged as one (the profile rules' own verdicts are not the
    floor's: the examples are not tuned)."""
    return [line for r in checkers.judge(list(manifests), checkers.load(json.dumps(decl)), []) for line in r.lines()]


def held(result, kernel, recorded_only=True):
    """The floor's declarations, its target (`at_most` on the launch) left out (the examples are not tuned to a line), and,
    `recorded_only`, the counters' bounds on the metrics the recording has."""
    out = []
    for d in floor.declarations(result, kernel, 1.0)["declarations"]:
        if d.get("figure", "").startswith("launch/") and "at_most" in d:
            continue
        if recorded_only and d.get("figure", "").replace("\\", "")[len("counters/"):] in UNRECORDED:
            continue
        out.append(d)
    return {"declarations": out}


class Fixtures(unittest.TestCase):
    def test_every_example_is_recorded(self):
        self.assertEqual(sorted(WORKS), sorted(BINDS))
        for name in BINDS:
            self.assertTrue((DIRECTORY / name / "run.manifest.json").is_file(), name)
            self.assertTrue((DIRECTORY / name / "counters.csv").is_file(), name)

    def test_the_descriptor_is_measured_and_shipped(self):
        rates = floor.load_rates(DESCRIPTOR.read_text())
        self.assertEqual(rates.source["kind"], "measured")
        self.assertEqual(floor.rates_of(DESCRIPTOR.stem), rates)
        self.assertEqual(floor.rates_of(rates.device), rates)

    def test_the_floor_and_the_counters_hold(self):
        for name in BINDS:
            with self.subTest(example=name):
                work, result, stamped, profiled, _ = recorded(name)
                self.assertEqual(refusals(held(result, work.kernel), stamped, profiled), [])

    def test_the_unrecorded_metrics_are_the_known_ones(self):
        for name in BINDS:
            with self.subTest(example=name):
                work, result, stamped, profiled, counters = recorded(name)
                wanted = {e.counter.metric for e in floor.expected(result)}
                self.assertEqual(wanted - set(counters), UNRECORDED)
                lines = refusals(held(result, work.kernel, recorded_only=False), stamped, profiled)
                self.assertTrue(lines and all("names no field" in line for line in lines), lines)

    def test_a_wrong_work_is_refused_naming_the_stream(self):
        def wrong(name, change):
            doc = json.loads(WORKS[name].read_text())
            change(doc["phases"][0]["work"])
            work, result, stamped, profiled, _ = recorded(name, doc)
            return "\n".join(refusals(held(result, work.kernel), stamped, profiled))
        half = wrong("stream", lambda w: w["hbm"].update(read="ctas * 4194304"))
        self.assertIn("counters/dram__bytes_read.sum", half)
        self.assertIn("hbm.read: the work declares", half)
        self.assertIn("UNDER-DECLARED", half)
        double = wrong("stream", lambda w: w["hbm"].update(read="ctas * 16777216"))
        self.assertIn("hbm.read: the work declares", double)
        self.assertIn("OVER-DECLARED", double)
        no_issue = wrong("mma", lambda w: w.pop("issue"))
        self.assertIn("counters/sm__inst_executed.sum", no_issue)
        self.assertIn("issue: the work declares", no_issue)
        self.assertIn("UNDER-DECLARED", no_issue)
        shared = wrong("shared", lambda w: w["shared"].update(read="ctas * 512 * 6 * 16 * 1024 * 16 * 2"))
        self.assertIn("shared.read: the work declares", shared)
        self.assertIn("OVER-DECLARED", shared)

    def test_the_stream_it_names_is_the_one_the_counters_show(self):
        for name, binds in BINDS.items():
            with self.subTest(example=name):
                _, result, _, _, counters = recorded(name)
                self.assertEqual(result.binding.label, binds)
                shown = floor.shown(result, counters, counters["sm__cycles_elapsed.max"])
                self.assertTrue(shown, "no row's counters were collected")
                self.assertEqual(shown[0][0].family, result.binding.family, [(r.label, s) for r, s in shown])

    def test_a_serial_phase_undeclared_is_refused(self):
        work, result, stamped, _, _ = recorded("serial")
        bare = floor.floor(floor.load_work(json.dumps({**json.loads(WORKS["serial"].read_text()), "serial": {}})),
                           result.rates, json.loads((DIRECTORY / "serial" / "parameters.json").read_text()))
        idle = {"declarations": [d for d in floor.declarations(bare, work.kernel, 1.0)["declarations"] if d["check"] == "idle"]}
        self.assertIn("phase worker/phase.math", "\n".join(refusals(idle, stamped)))


if __name__ == "__main__":
    unittest.main()
