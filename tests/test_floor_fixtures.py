"""THE FLOOR ON A CARD'S RECORDED RUNS (docs/floor.md#the-fixtures): for each example of `examples/floor/floor_examples.cu`,
its floor at the card's own rates is never above its measured launch (`launch/grid_ns`), the stream it names is the one its
counters show (its family leads the counters' shares of the rows' rates), and
the idle check passes it as declared (the `serial` example refused without its declaration). The fixtures were recorded once
on the card (`bazel run //:record_floor_fixtures`); this reads them, with no GPU.

Arguments: the device's fixture directory, then each example's work declaration (`examples/floor/<name>.work.json`)."""
import json
import sys
import unittest
from pathlib import Path

from warpline import checkers, declare, floor, manifest
from warpline.checker import ncu

DIRECTORY = Path(sys.argv[1])
WORKS = {Path(p).name.split(".")[0]: Path(p) for p in sys.argv[2:]}
sys.argv = sys.argv[:1]
#: the stream each example's design binds
BINDS = {"stream": "hbm.read", "shared": "shared.read", "mma": "tensor m16n8k16 f16.f32", "serial": "hbm.read"}


def recorded(name):
    """An example's floor, its run manifest and its counters (one result's metrics)."""
    here = DIRECTORY / name
    rates = floor.load_rates((DIRECTORY / "rates.json").read_text())
    work = floor.load_work(WORKS[name].read_text())
    result = floor.floor(work, rates, json.loads((here / "parameters.json").read_text()))
    results = ncu.details_metrics((here / "counters.csv").read_text())
    if len(results) != 1:
        raise AssertionError(f"{name}: the counters hold {len(results)} results, not the one launch the region captured")
    return work, result, manifest.loads((here / "run.manifest.json").read_text()), results[0][2]


def refusals(decl, m):
    text = json.dumps(decl)
    out = declare.compare([m], declare.load_declarations(text, []), None, checkers.load(text))
    return [line for r in out.refusals for line in r.lines()]


class Fixtures(unittest.TestCase):
    def test_every_example_is_recorded(self):
        self.assertEqual(sorted(WORKS), sorted(BINDS))
        for name in BINDS:
            self.assertTrue((DIRECTORY / name / "run.manifest.json").is_file(), name)

    def test_the_floor_is_never_above_the_measured_launch(self):
        for name in BINDS:
            with self.subTest(example=name):
                work, result, m, _ = recorded(name)
                #: the floor's own declarations, its target (`at_most`) left out: the examples are not tuned to a line
                held = {"declarations": [d for d in floor.declarations(result, work.kernel, 1.0)["declarations"]
                                         if "at_most" not in d]}
                self.assertEqual(refusals(held, m), [])

    def test_the_stream_it_names_is_the_one_the_counters_show(self):
        for name, binds in BINDS.items():
            with self.subTest(example=name):
                _, result, _, counters = recorded(name)
                self.assertEqual(result.binding.label, binds)
                shown = floor.shown(result, counters, counters["sm__cycles_elapsed.max"])
                self.assertTrue(shown, "no row's counters were collected")
                self.assertEqual(shown[0][0].family, result.binding.family, [(r.label, s) for r, s in shown])

    def test_a_serial_phase_undeclared_is_refused(self):
        work, result, m, _ = recorded("serial")
        bare = floor.floor(floor.load_work(json.dumps({**json.loads(WORKS["serial"].read_text()), "serial": {}})),
                           result.rates, json.loads((DIRECTORY / "serial" / "parameters.json").read_text()))
        idle = {"declarations": [d for d in floor.declarations(bare, work.kernel, 1.0)["declarations"] if d["check"] == "idle"]}
        self.assertIn("phase worker/phase.math", "\n".join(refusals(idle, m)))


if __name__ == "__main__":
    unittest.main()
