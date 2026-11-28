"""THE FLOOR (docs/floor.md): a kernel's declared work over a device's measured rates, by phase, the stream that binds it, and
the bound written from it, judged over a run manifest by the common checkers (the `bound`s, the `present` and the `idle`
check). Synthetic declarations, rates and run manifests: no GPU."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline import checkers, command, declare, floor, manifest

RATES = {"format": "warpline-rates/1", "device": "test card", "sms": 4, "clock_ghz": 2.0,
         "source": {"kind": "spec", "from": "the tests' card"}, "rows": [
    {"stream": "hbm.read", "rate": 16}, {"stream": "hbm.write", "rate": 16}, {"stream": "hbm", "rate": 24},
    {"stream": "l2.own", "rate": 32}, {"stream": "l2.shared", "footprint": 131072, "rate": 64},
    {"stream": "l2.shared", "footprint": 4194304, "rate": 48},
    {"stream": "shared", "rate": 128}, {"stream": "tensor", "shape": "m16n8k16", "dtype": "f16.f32", "rate": 0.5},
    {"stream": "issue", "rate": 4}, {"stream": "issue.float", "rate": 2},
    {"stream": "hbm.read", "rate": 24, "sms": 2}]}
WORK = {"format": "warpline-work/1", "kernel": "^k\\(", "parameters": ["n"], "phases": [
    {"role": "producer", "phase": "load", "work": {"hbm": {"read": "n * 1024"}}},
    {"role": "consumer", "phase": "mma", "work": {"tensor": [{"shape": "m16n8k16", "dtype": "f16.f32", "count": "n * 4"}],
                                                  "shared": {"read": "n * 512"}}},
    {"role": "consumer", "phase": "epilogue", "work": {"issue": {"float": "n * 8"}}}],
    "overlap": [["producer/load", "consumer/mma"]]}


def result(work=None, rates=None, **given):
    return floor.floor(floor.load_work(json.dumps(work or WORK)), floor.load_rates(json.dumps(rates or RATES)),
                       given or {"n": 1000})


def counted(r=None, **change) -> dict:
    """A profiled run's counters that hold every counted stream at its work's count (`floor.expected` of `r`, the default
    floor's), by metric, `change` a stream's count (`{"issue": 99}`) in its metric's units."""
    out = {}
    for e in floor.expected(r or result()):
        out[e.counter.metric] = change.get(e.counter.stream.replace(".", "_"), e.work / e.counter.per)
    return out


def run_manifest(ns, phases=None, counters=None):
    """A run manifest of `k()` whose launch took `ns`, whose roles measured `phases` ({`role/phase`: cycles}) and whose
    profiled launch counted `counters` (the default floor's expected counts: `counted`)."""
    figures = {"health": {"lanes": 2, "roles": 2, "stamps": 10, "dropped": 0, "findings": {}, "open_at_end": {"lanes": 0, "regions": 0}},
               "launch": {"cycles": ns * 2, "ns": ns * 0.98, "grid_ns": ns, "scopes": 4}}
    if phases is not None:
        roles: dict = {}
        for key, cycles in phases.items():
            role, phase = key.split("/")
            roles.setdefault(role, {"period": 100.0, "phases": {}})["phases"][phase] = {
                "cycles": cycles, "cycles_se": 0.0, "share": 0.5, "jitter": 0.0, "held": 1.0}
        figures["window"] = {"anchor": "w.begin", "period": 100.0, "period_std": 0.0, "period_se": 0.0, "windows": 8}
        figures["roles"] = roles
    m = manifest.run_manifest("k()", [manifest.RunPart(figures=figures, counters=counted() if counters is None else counters)])
    return manifest.loads(manifest.dumps(m))


def judged(decl: dict, m) -> list[str]:
    text = json.dumps(decl)
    out = declare.compare([m], declare.load_declarations(text, []), None, checkers.load(text))
    return [line for r in out.refusals for line in r.lines()]


class Formats(unittest.TestCase):
    def refused(self, fn, text, needle):
        with self.assertRaises(floor.FloorError) as e:
            fn(json.dumps(text))
        self.assertIn(needle, str(e.exception))

    def test_work_refusals(self):
        bad = lambda **change: {**WORK, **change}
        self.refused(floor.load_work, {**WORK, "format": "x"}, "not a work declaration")
        self.refused(floor.load_work, bad(phases=[{"role": "a", "phase": "p", "work": {"hbm": {"read": "m * 2"}}}]),
                     "names m, which the declaration's parameters do not")
        self.refused(floor.load_work, bad(phases=[{"role": "a", "phase": "p", "work": {"hbm": {"read": "__import__('os')"}}}]),
                     "an expression is numbers")
        self.refused(floor.load_work, bad(phases=[{"role": "a", "phase": "p", "work": {"hbm": {"read": -1}}}]), "0 or more")
        self.refused(floor.load_work, bad(phases=[{"role": "a", "phase": "p", "work": {"dram": {}}}]), "is not one of")
        self.refused(floor.load_work, bad(overlap=[["consumer/mma", "consumer/epilogue"]]), "two phases of one role")
        self.refused(floor.load_work, bad(overlap=[["producer/load", "consumer/mma"], ["producer/load", "consumer/epilogue"]]),
                     "in two overlap sets")
        self.refused(floor.load_work, bad(overlap=[["producer/load", "x/y"]]), "no declared phase")
        self.refused(floor.load_work, bad(serial={"consumer/epilogue": " "}), "needs its reason")
        self.refused(floor.load_work, bad(phases=[]), "no phases")
        self.refused(floor.load_work, bad(phases=[{"role": "a", "phase": "p", "work": {"issue": {"Float": 1}}}]),
                     "not a class as a manifest spells it")

    def test_rates_refusals(self):
        self.refused(floor.load_rates, {**RATES, "rows": [{"stream": "dram", "rate": 1}]}, "no stream 'dram'")
        self.refused(floor.load_rates, {**RATES, "rows": [{"stream": "l2.shared", "rate": 1}]}, "footprint")
        self.refused(floor.load_rates, {**RATES, "rows": [{"stream": "hbm", "rate": 0}]}, "above 0")
        self.refused(floor.load_rates, {**RATES, "rows": [{"stream": "hbm", "rate": 1}, {"stream": "hbm", "rate": 2}]},
                     "a second time")
        self.refused(floor.load_rates, {**RATES, "rows": [{"stream": "hbm", "rate": 1, "sms": 9}]}, "1 to 4")
        self.refused(floor.load_rates, {**RATES, "clock_ghz": None}, "clock_ghz")
        self.refused(floor.load_rates, {k: v for k, v in RATES.items() if k != "source"}, "source is")
        self.refused(floor.load_rates, {**RATES, "source": {"kind": "measured", "card": "x"}}, "names card, probe")
        self.refused(floor.load_rates, {**RATES, "source": {"kind": "spec", "from": "x", "probe": "y"}}, "probe is not one")

    def test_an_issue_class_tensor_is_refused(self):
        bad = {**WORK, "phases": [{"role": "a", "phase": "p", "work": {"issue": {"tensor": 4}}}]}
        self.refused(floor.load_work, bad, "count them twice")

    def test_the_sources_said(self):
        self.assertIn("a spec (the tests' card)", floor.load_rates(json.dumps(RATES)).said)
        measured = {**RATES, "source": {"kind": "measured", "card": "a card", "probe": "abc123"}}
        self.assertEqual(floor.load_rates(json.dumps(measured)).said, "measured on a card by the probe at abc123")


class Descriptors(unittest.TestCase):
    def test_a_shipped_descriptor_by_name(self):
        known = floor.known_rates()
        self.assertIn("h100-80gb-hbm3", known)
        by_slug, by_device = floor.rates_of("h100-80gb-hbm3"), floor.rates_of("NVIDIA H100 80GB HBM3")
        self.assertEqual(by_slug, by_device)
        self.assertEqual((by_slug.sms, by_slug.source["kind"]), (132, "measured"))
        with self.assertRaises(floor.FloorError) as e:
            floor.rates_of("no such card")
        self.assertIn("h100-80gb-hbm3", str(e.exception))


class WorkFunction(unittest.TestCase):
    def test_a_work_function_is_its_files_work(self):
        tile = Path(__file__).parent / "floor" / "tile.work.json"
        rates = floor.load_rates((Path(__file__).parent / "floor" / "tile.rates.json").read_text())
        made = floor.floor(floor.work_of("tile_work:work", {"n": 10}), rates, {"n": 10})
        declared = floor.floor(floor.work_of(str(tile), {"n": 10}), rates, {"n": 10})
        self.assertEqual(floor.declarations(made, "^t\\(", 1.15), floor.declarations(declared, "^t\\(", 1.15))

    def test_refusals(self):
        for name, needle in (("no form", "neither a file nor a work function"), ("tile_work:none", "cannot be imported"),
                             ("no_such_module:work", "cannot be imported")):
            with self.assertRaises(floor.FloorError) as e:
                floor.work_of(name, {})
            self.assertIn(needle, str(e.exception))


class Floors(unittest.TestCase):
    def test_phases_groups_and_launch(self):
        r = result()
        load, mma, epilogue = r.phases
        self.assertEqual((load.cycles, load.binding.label), (16000.0, "hbm.read"))     # 1,024,000 B / (16 B x 4 SMs)
        self.assertAlmostEqual(load.by_row["hbm"], 1024000 / 96)                       # the copy's row prices a read too
        self.assertEqual((mma.cycles, mma.binding.label), (2000.0, "tensor m16n8k16 f16.f32"))
        self.assertEqual((epilogue.cycles, epilogue.binding.label), (1000.0, "issue.float"))
        self.assertEqual([g.keys for g in r.groups], [("producer/load", "consumer/mma"), ("consumer/epilogue",)])
        #: the SMs' chain: the set's tensor 2000 cycles, then the epilogue's float 1000; the device's: the launch's HBM read
        self.assertEqual((r.chain.cycles, r.chain.ns, r.chain.binding.label), (3000.0, 1500.0, "tensor m16n8k16 f16.f32"))
        self.assertEqual((r.device.cycles, r.device.ns, r.device.binding.label), (16000.0, 8000.0, "hbm.read"))
        self.assertEqual((r.cycles, r.ns, r.binding.label), (16000.0, 8000.0, "hbm.read"))
        self.assertEqual(r.busy, ("producer/load", "consumer/mma"))

    def test_a_device_stream_bounds_the_launch_and_the_sms_own_add_in_sequence(self):
        work = {**WORK, "phases": [{"role": "a", "phase": "x", "work": {"hbm": {"read": 6400}, "shared": {"read": 51200}}},
                                   {"role": "a", "phase": "y", "work": {"issue": {"float": 1600}, "hbm": {"write": 3200}}}],
                "overlap": []}
        r = result(work)
        self.assertEqual([f.cycles for f in r.phases], [100.0, 200.0])
        self.assertEqual(r.chain.cycles, 300.0)                 # shared 100, then float 200: the SM's own, in sequence
        self.assertEqual(r.device.cycles, 9600 / 96)            # the launch's 9600 HBM bytes at the copy's 24 a cycle an SM
        self.assertEqual((r.cycles, r.binding.label), (300.0, "issue.float"))

    def test_a_device_rows_nanoseconds_are_at_its_own_clock(self):
        rates = {**RATES, "rows": [{**r, "clock_ghz": 1.0} if r["stream"] in ("hbm.read", "issue.float") else r
                                   for r in RATES["rows"]]}
        r = result(None, rates)
        self.assertEqual((r.device.ns, r.device.cycles), (16000.0, 16000.0))         # HBM's 16000 cycles at its 1 GHz
        self.assertEqual(r.phases[2].ns, 500.0)                 # an SM's own stream at the table's 2 GHz, whatever its row ran at

    def test_a_row_of_two_streams_prices_at_the_best_of_it_and_its_parts(self):
        rates = {**RATES, "rows": [{**r, "rate": 8} if r["stream"] == "hbm" else r for r in RATES["rows"]]}
        self.assertEqual(result(None, rates).phases[0].by_row["hbm"], 16000.0)    # the copy's 8 a cycle, the read's 16

    def test_an_overlap_set_sums_a_stream_its_members_share(self):
        work = {**WORK, "phases": [{"role": "a", "phase": "x", "work": {"shared": {"read": 51200}}},
                                   {"role": "b", "phase": "y", "work": {"shared": {"read": 25600}}}], "overlap": [["a/x", "b/y"]]}
        r = result(work)
        self.assertEqual([f.cycles for f in r.phases], [100.0, 50.0])
        self.assertEqual(r.cycles, 150.0)                       # more than the largest member's: both read shared memory at once

    def test_the_sms_a_launch_occupies(self):
        work = {**WORK, "sms": "2", "phases": [{"role": "a", "phase": "x", "work": {"hbm": {"read": 4800}}}], "overlap": []}
        self.assertEqual(result(work).cycles, 100.0)            # the row measured on 2 SMs: 4800 / (24 x 2)
        work["phases"][0]["work"]["shared"] = {"read": 1}
        with self.assertRaises(floor.FloorError) as e:
            result(work)
        self.assertIn("priced by no row of the rates on 2 SMs", str(e.exception))

    def test_a_shared_footprint_takes_the_largest_probed_at_most_it(self):
        work = lambda f: {**WORK, "phases": [{"role": "a", "phase": "x", "work": {
            "l2": [{"pattern": "shared", "footprint": f, "bytes": 192000}]}}], "overlap": []}
        self.assertEqual(result(work(1048576)).phases[0].binding.label, "l2.shared@131072")
        self.assertEqual(result(work(16777216)).phases[0].binding.label, "l2.shared@4194304")
        with self.assertRaises(floor.FloorError):
            result(work(65536))                                 # below every probed footprint: no rate says it

    def test_work_no_row_prices_is_refused(self):
        rates = {**RATES, "rows": [r for r in RATES["rows"] if r["stream"] != "tensor"]}
        with self.assertRaises(floor.FloorError) as e:
            result(None, rates)
        self.assertIn("tensor m16n8k16 f16.f32 is priced by no row", str(e.exception))

    def test_a_parameter_not_given_is_refused(self):
        with self.assertRaises(floor.FloorError) as e:
            result(None, None, m=1)
        self.assertIn("parameter 'n' is not given", str(e.exception))


class Bound(unittest.TestCase):
    def decl(self, factor=1.15):
        return floor.declarations(result(), "^k\\(", factor)

    def test_the_declarations(self):
        d = self.decl()["declarations"]
        self.assertEqual([e["check"] for e in d[:4]], ["present", "bound", "bound", "idle"])
        self.assertTrue(all(e["check"] == "bound" and e["figure"].startswith("counters/") for e in d[4:]))
        self.assertEqual((d[1]["at_most"], d[2]["at_least"]), (9200.0, 8000.0))
        self.assertEqual(d[3]["busy"], ["producer/load", "consumer/mma"])
        checkers.load(json.dumps(self.decl()))                 # the checkers read what the floor writes

    def test_a_run_within_the_line_passes(self):
        phases = {"producer/load": 50.0, "consumer/mma": 30.0}
        work = {**WORK, "serial": {"consumer/epilogue": "the example's serial form"}}
        decl = floor.declarations(result(work), "^k\\(", 1.15)
        self.assertEqual(judged(decl, run_manifest(9000.0, {**phases, "consumer/epilogue": 20.0})), [])

    def test_above_the_line_and_below_the_floor(self):
        self.assertIn("launch/grid_ns is 9300.0, at most 9200.0", "\n".join(judged(self.decl(), run_manifest(9300.0, {}))))
        self.assertNotIn("counters/", "\n".join(judged(self.decl(), run_manifest(9300.0, {}))))
        self.assertIn("launch/grid_ns is 7900.0, at least 8000.0", "\n".join(judged(self.decl(), run_manifest(7900.0, {}))))

    def test_a_phase_that_idles_the_binding_stream(self):
        lines = "\n".join(judged(self.decl(), run_manifest(9000.0, {"producer/load": 50.0, "consumer/mma": 30.0,
                                                                     "consumer/epilogue": 20.0})))
        self.assertIn("phase consumer/epilogue (20 cycles a window) idles hbm", lines)
        self.assertEqual(lines.count(": idle: "), 1)

    def test_a_phase_twice_a_window_is_its_phase(self):
        phases = {"producer/load": 50.0, "consumer/mma": 30.0, "consumer/mma#2": 10.0}
        self.assertEqual(judged(floor.declarations(result({**WORK, "serial": {"consumer/epilogue": "r"}}), "^k\\(", 1.15),
                                run_manifest(9000.0, {**phases, "consumer/epilogue": 1.0})), [])

    def test_a_declared_phase_not_measured_and_no_phases(self):
        lines = "\n".join(judged(self.decl(), run_manifest(9000.0, {"producer/load": 50.0})))
        self.assertIn("declares phase consumer/mma, which no run manifest measures", lines)
        self.assertIn("no run manifest measures a phase", "\n".join(judged(self.decl(), run_manifest(9000.0))))

    def test_a_bound_on_no_kernel_is_refused(self):
        decl = floor.declarations(result(), "^other\\(", 1.15)
        self.assertIn("present: names 0 kernel(s)", "\n".join(judged(decl, run_manifest(9000.0, {}))))

    def test_the_figure_and_factor(self):
        self.assertEqual(floor.declarations(result(), "^k\\(", 1.0, "launch/cycles")["declarations"][1]["at_most"], 16000.0)
        self.assertEqual(floor.declarations(result(), "^k\\(", 1.0, "launch/ns")["declarations"][1]["figure"], "launch/ns")
        for bad in ((0.9, "launch/grid_ns"), (1.15, "window/period")):
            with self.assertRaises(floor.FloorError):
                floor.declarations(result(), "^k\\(", *bad)


class Counters(unittest.TestCase):
    def test_the_expected_counts_and_their_bands(self):
        r = result()                                            # the launch floor: 8000 ns, HBM's read
        got = {e.counter.stream: e for e in floor.expected(r)}
        #: every counted stream on every card is held (none declared: at none); a stream only some cards count, only declared
        self.assertEqual(sorted(got), ["hbm.read", "hbm.write", "issue", "l2", "shared.read", "shared.write", "tensor.hmma",
                                       "tensor.imma"])
        hbm = got["hbm.read"]
        #: the band: 0.05 of the floor's 8000 ns at the slower of the rows that price a read (the read's 16 bytes a cycle
        #: an SM, the copy's 24), at its clock (the table's 2 GHz here), on the 4 SMs
        self.assertEqual((hbm.work, hbm.priced), (1024000.0, "hbm.read"))
        self.assertAlmostEqual(hbm.most - hbm.work, 0.05 * 8000 * 16 * 2.0 * 4)
        #: the L2's bytes are every byte an SM moves past L1: the HBM read counts there too
        self.assertEqual(got["l2"].work, 1024000.0)
        #: the issue counts every warp instruction: the epilogue's float and the MMAs
        self.assertEqual(got["issue"].work, 8000.0 + 4000.0)
        self.assertEqual(got["tensor.hmma"].work, 4000.0)
        #: no row prices an integer MMA: held at none, exactly
        self.assertEqual((got["tensor.imma"].most, got["tensor.imma"].priced), (0.0, None))

    def test_a_warpgroup_mma_is_held_where_declared(self):
        work = {**WORK, "phases": [{"role": "a", "phase": "x", "work": {"tensor": [
            {"shape": "m64n128k16", "dtype": "f16.f32", "count": 64}]}}], "overlap": []}
        rates = {**RATES, "rows": [*RATES["rows"], {"stream": "tensor", "shape": "m64n128k16", "dtype": "f16.f32", "rate": 0.1}]}
        got = {e.counter.stream: e for e in floor.expected(result(work, rates))}
        self.assertEqual((got["tensor.gmma"].work, got["tensor.gmma"].counter.metric),
                         (64.0, "sm__inst_executed_pipe_tensor_op_gmma.sum"))
        self.assertEqual(got["tensor.hmma"].work, 0.0)
        self.assertNotIn("tensor.gmma", {e.counter.stream for e in floor.expected(result())})

    def test_a_run_whose_counters_hold_passes(self):
        self.assertEqual(judged(floor.declarations(result({**WORK, "serial": {"consumer/epilogue": "r"}}), "^k\\(", 1.15),
                                run_manifest(9000.0, {"producer/load": 50.0, "consumer/mma": 30.0, "consumer/epilogue": 1.0})),
                         [])

    def test_under_and_over_declared_name_the_stream(self):
        decl = floor.declarations(result(), "^k\\(", 1.15)
        hbm = {e.counter.stream: e for e in floor.expected(result())}["hbm.read"]
        under = "\n".join(judged(decl, run_manifest(9000.0, {}, counted(hbm_read=hbm.most * 1.01))))
        self.assertIn("counters/dram__bytes_read.sum is", under)
        self.assertIn("hbm.read: the work declares 1.024e+06 bytes", under)
        self.assertIn("UNDER-DECLARED", under)
        over = "\n".join(judged(decl, run_manifest(9000.0, {}, counted(hbm_read=hbm.least * 0.99))))
        self.assertIn("OVER-DECLARED", over)
        omitted = "\n".join(judged(decl, run_manifest(9000.0, {}, counted(shared_write=1e9))))
        self.assertIn("shared.write: the work declares 0 bytes", omitted)
        self.assertIn("UNDER-DECLARED", omitted)
        unprofiled = "\n".join(judged(decl, run_manifest(9000.0, {}, {})))
        self.assertIn("'counters/sm__inst_executed\\\\.sum' names no field", unprofiled)

    def test_the_tolerance(self):
        wide = {e.counter.stream: e for e in floor.expected(result(), 0.5)}["hbm.read"]
        self.assertAlmostEqual(wide.most - wide.work, 0.5 * 8000 * 16 * 2.0 * 4)
        for bad in (-0.1, 1.0, True):
            with self.assertRaises(floor.FloorError):
                floor.declarations(result(), "^k\\(", 1.15, tolerance=bad)

    def test_the_metrics_to_profile(self):
        self.assertEqual(floor.metrics(), sorted({c.metric for c in floor.COUNTERS if c.everywhere}))
        self.assertIn("lts__t_sectors_srcunit_tex.sum", floor.metrics(result()))

    def test_the_stream_the_counters_show(self):
        r = result(None, {**RATES, "rows": [row for row in RATES["rows"] if row.get("sms", 4) == 4]})
        counters = {"dram__bytes_read.sum": 1024000, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": 4000,
                    "sm__inst_executed_pipe_tensor_op_hmma.sum": 4000, "sm__inst_executed.sum": 12000}
        shown = [(row.label, share) for row, share in floor.shown(r, counters, 20000)]
        self.assertEqual(shown[0], ("hbm.read", 1024000 / (16 * 4 * 20000)))
        #: issue.float's stream no counter counts: left out, never read as 0
        self.assertEqual({label for label, _ in shown}, {"hbm.read", "hbm", "shared", "tensor m16n8k16 f16.f32", "issue"})


class Verb(unittest.TestCase):
    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = command.main(list(argv), [])
        return code, out.getvalue(), err.getvalue()

    def test_the_verb(self):
        with tempfile.TemporaryDirectory() as d:
            work, rates, bound, cell = (Path(d) / n for n in ("w.json", "r.json", "b.json", "cell.json"))
            work.write_text(json.dumps(WORK))
            rates.write_text(json.dumps(RATES))
            cell.write_text(json.dumps({"n": 1000, "name": "a cell", "nested": {"n": 5}}))
            code, out, _ = self.call("floor", str(work), "--rates", str(rates), "--param", "n=1000")
            self.assertEqual(code, 0)
            self.assertIn("launch: 8000 ns, 16000 cycles, hbm.read binds", out)
            self.assertIn("IDLE in consumer/epilogue", out)
            code, out, _ = self.call("floor", str(work), "--rates", str(rates), "--parameters", str(cell), "--json")
            self.assertEqual((code, json.loads(out)["cycles"]), (0, 16000.0))
            code, _, err = self.call("floor", str(work), "--rates", str(rates), "--param", "n=1000", "--bound", str(bound))
            self.assertEqual(code, 2)
            self.assertIn("--bound needs --factor", err)
            code, _, _ = self.call("floor", str(work), "--rates", str(rates), "--param", "n=1000", "--bound", str(bound),
                                   "--factor", "1.15")
            self.assertEqual((code, json.loads(bound.read_text()), ), (0, floor.declarations(result(), "^k\\(", 1.15)))
            code, _, err = self.call("floor", str(work), "--rates", str(rates), "--param", "n=lots")
            self.assertEqual(code, 2)
            self.assertIn("parameter 'n' is not given as a number", err)
            code, out, _ = self.call("floor", "tile_work:work", "--rates", "h100-80gb-hbm3", "--param", "n=10")
            self.assertEqual(code, 0)
            self.assertIn("the rates measured on NVIDIA H100 80GB HBM3 by the probe at", out)
            self.assertIn("profile with: --profile ", out)


if __name__ == "__main__":
    unittest.main()
