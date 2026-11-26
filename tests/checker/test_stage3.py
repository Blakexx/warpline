"""The model's additions for the remaining static rules, the bound rules' declarations, the flow, and the rules on the
toolkit's own output (`examples/checker/stages.cu`, `collectives.cu`): each added field on a fixture, each rule's findings,
and where a natural loop counts differently from the address span a backward branch closes."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from warpline.checker.cli import main
from warpline.checker import declare, flow, model, runner
from warpline.checker.rules import registry

FIXTURES = Path(__file__).parent / "fixtures"
STAGES = FIXTURES / "stages" / "kernel"
R = registry()


def synth(rule, name):
    (k,) = runner.kernels_of_files(str(FIXTURES / "rules" / rule / name / "kernel")).values()
    return k


def stages():
    return {k.demangled.split("(")[0].split("::")[-1]: k for k in runner.kernels_of_files(str(STAGES)).values()}


def counts(kernel):
    return {n: r.count(kernel) for n, r in R.items() if r.count(kernel)}


def span_count(kernel, wanted):
    """What a span-based count gives: the instructions wanted inside the span a backward branch closes."""
    spans = [(i.target, i.address) for i in kernel.instructions if i.base == "BRA" and i.target is not None and i.target < i.address]
    return sum(1 for i in kernel.instructions if wanted(i) and any(lo <= i.address <= hi for lo, hi in spans))


def conv(i):
    return i.base in ("BSSY", "BSYNC") or (i.base == "BRA" and "DIV" in i.opcode.split("."))


class Instructions(unittest.TestCase):
    def ins(self, opcode, operands, predicate=""):
        return model.Instruction(0x10, predicate, opcode, operands, "BB0")

    def test_a_destination_and_the_sources(self):
        i = self.ins("IADD3", "R4, R5, R6, RZ")
        self.assertEqual((i.writes, i.reads), (frozenset({"R4"}), frozenset({"R5", "R6"})))

    def test_a_wide_result_is_a_run_of_registers(self):
        self.assertEqual(self.ins("LDG.E.64", "R4, [R2.64]").writes, frozenset({"R4", "R5"}))
        self.assertEqual(self.ins("LDG.E.128", "R4, [R2.64]").writes, frozenset({"R4", "R5", "R6", "R7"}))
        self.assertEqual(self.ins("LDG.E.64", "R4, [R2.64]").reads, frozenset({"R2", "R3"}))

    def test_a_store_a_branch_and_a_tensor_step_write_none(self):
        for opcode, operands in (("STS", "[R8+0x10], R7"), ("BRA", "P0, 0x30"), ("HGMMA.64x16x16.F32.BF16", "R24, gdesc[UR8], R24")):
            self.assertEqual(self.ins(opcode, operands).writes, frozenset(), opcode)
        self.assertEqual(self.ins("STS", "[R8+0x10], R7").reads, frozenset({"R8", "R7"}))

    def test_a_predicate_destination_is_no_register(self):
        i = self.ins("ISETP.GE.AND", "P0, PT, R2, 0x8, PT")
        self.assertEqual((i.writes, i.reads), (frozenset(), frozenset({"R2"})))

    def test_a_uniform_register_is_its_own(self):
        i = self.ins("UIADD3", "UR4, UPT, UPT, UR4, UR6, URZ")
        self.assertEqual((i.writes, i.reads), (frozenset({"UR4"}), frozenset({"UR4", "UR6"})))

    def test_the_address_operand_and_its_registers(self):
        i = self.ins("LDS", "R5, [R8+0x10]")
        self.assertEqual((i.address_operand, i.address_registers), ("[R8+0x10]", frozenset({"R8"})))
        self.assertIsNone(self.ins("IADD3", "R4, R5, R6, RZ").address_operand)

    def test_a_branch_names_its_target(self):
        self.assertEqual(self.ins("BRA", "P1, 0x210", "@P0").target, 0x210)
        self.assertEqual(self.ins("BRA.DIV", "UR6, 0x270").target, 0x270)
        self.assertEqual(self.ins("CALL.REL.NOINC", "0x4c0").target, 0x4c0)
        self.assertIsNone(self.ins("IADD3", "R4, R5, 0x10, RZ").target)

    def test_the_class_and_the_uniform_datapath(self):
        for opcode, klass in (("HGMMA.64x16x16.F32.BF16", "tensor"), ("LDSM.16.M88.4", "shared load"), ("F2I.TRUNC", "conversion unit"),
                              ("F2FP.BF16.PACK_AB", "pack"), ("REDUX.MIN", "shuffle"), ("RED.E.ADD", "global"),
                              ("LDGDEPBAR", "control"), ("UIADD3", "integer/address"), ("XYZ", "other")):
            self.assertEqual(self.ins(opcode, "").klass, klass, opcode)

    def test_the_dead_predicate(self):
        self.assertTrue(self.ins("HGMMA", "", "@!PT").dead)
        self.assertFalse(self.ins("HGMMA", "", "@!P0").dead)


class Steps(unittest.TestCase):
    def test_a_step_with_both_descriptors_reads_a_and_b(self):
        (s,) = synth("step_port", "operands_only").steps
        self.assertEqual((s.shape, s.operand_bytes, s.null, s.commit), ((64, 16, 16), (64 * 16 + 16 * 16) * 2, False, True))

    def test_a_step_with_a_in_registers_reads_b_alone(self):
        (s,) = synth("step_port", "a_in_registers").steps
        self.assertEqual(s.operand_bytes, 16 * 16 * 2)

    def test_the_empty_group_is_the_null_step_and_moves_nothing(self):
        k = stages()["step_guarded"]
        self.assertEqual([s.null for s in k.steps], [False, True])
        self.assertEqual(sum(r.steps for r in k.regions), 1)

    def test_a_shared_access_moves_its_width_a_warp(self):
        k = synth("step_port", "loop_with_loads")
        self.assertEqual([a.warp_bytes for a in k.accesses], [512, 512])          # LDSM x4: four 8 x 8 matrices of 16 bits
        self.assertEqual(model.MemoryAccess(model.Instruction(0, "", "LDS.U16", "R4, [R2]", "b"), "shared", "load", 16).warp_bytes, 64)
        self.assertEqual(model.MemoryAccess(model.Instruction(0, "", "STS.128", "[R2], R4", "b"), "shared", "store", 128).warp_bytes, 512)

    def test_a_region_is_a_loop_or_an_unrolled_step(self):
        loop = synth("step_port", "loop_with_loads").regions
        self.assertEqual([(r.kind, r.steps, len(r.instructions)) for r in loop], [("loop", 2, 8)])
        unrolled = synth("step_work", "unrolled_pair").regions
        self.assertEqual([(r.kind, r.steps, len(r.instructions)) for r in unrolled], [("unrolled", 1, 2), ("unrolled", 1, 4)])

    def test_a_dead_step_makes_no_region(self):
        self.assertEqual(len(synth("step_predicated", "dead").regions), 1)


class Graph(unittest.TestCase):
    def cold(self, rule, name):
        return sorted(b.id for b in synth(rule, name).blocks if b.cold)

    def test_the_block_past_a_conv_guard_and_the_div_fallback_are_cold(self):
        self.assertEqual(self.cold("convergence_call", "guarded_conv"), ["BBc", "BBd", "BBsub"])
        self.assertEqual(self.cold("convergence_call", "guarded_div"), ["BBfb", "BBk", "BBsub"])

    def test_a_straight_kernel_has_no_cold_block(self):
        self.assertEqual(self.cold("step_rolled", "straight"), [])

    def test_a_call_into_a_subroutine_is_told_from_a_slow_path_and_a_guarded_one(self):
        shape = lambda r, n: [(c.subroutine, c.hot) for c in synth(r, n).calls]  # noqa: E731
        self.assertEqual(shape("convergence_call", "hot_call"), [(True, True)])
        self.assertEqual(shape("convergence_call", "guarded_conv"), [(True, False)])
        self.assertEqual(shape("convergence_call", "guarded_div"), [(True, False)])
        self.assertEqual(shape("convergence_call", "slow_path"), [(False, True)])

    def test_the_bodies_the_graph_does_not_hold_are_kept_in_a_block_of_their_own(self):
        (k,) = runner.kernels_of_files(str(FIXTURES / "collectives" / "kernel")).values()
        out = k.block(model.OUTLINED)
        self.assertTrue(out.cold)
        self.assertEqual(out.instructions[-1].opcode, "RET.REL.NODEC")                 # the padding after it is dropped
        self.assertEqual([c.subroutine for c in k.calls], [True] * 5)

    def test_the_forward_flow_reaches_a_loop_head_from_its_back_edge(self):
        k = synth("store_unfenced", "loop_carried")
        states = flow.entry_states(k, lambda s, i: frozenset({i.address}) if i.base == "STS" else s)
        self.assertEqual(states["BBb"], frozenset({k.block("BBb").instructions[1].address}))
        self.assertEqual(states["BBc"], states["BBb"])


class Details(unittest.TestCase):
    def one(self, rule, name):
        (f,) = R[rule].check(synth(rule, name))
        return f

    def test_loop_convergence_names_each_instruction_and_its_loop(self):
        found = R["loop_convergence"].check(synth("loop_convergence", "in_loop"))
        self.assertEqual([(f.where, f.count) for f in found], [("0x0020", 1), ("0x0060", 1)])
        self.assertIn("BSSY at 0x0020 in the loop headed at 0x0020", found[0].what)

    def test_load_in_trip_names_the_load_and_its_reader(self):
        f = self.one("load_in_trip", "read_in_trip")
        self.assertEqual(f.where, "0x0020")
        self.assertIn("0x0030", f.what)

    def test_a_read_only_in_a_cold_fallback_is_not_a_read_in_the_trip(self):
        # the real reader (0x0020) precedes the load in the trip; the only reader laid out after it is the BRA.DIV fallback
        k = synth("load_in_trip", "cold_reader")
        self.assertEqual([b.id for b in k.blocks if b.cold], ["BBfb"])
        self.assertEqual(R["load_in_trip"].count(k), 0)

    def test_a_read_in_the_hot_path_counts_whatever_the_cold_block_does(self):
        f = self.one("load_in_trip", "cold_and_hot_reader")
        self.assertIn("0x0050", f.what)                                # the hot block's reader, not the fallback's 0x0090

    def test_a_reader_before_the_load_is_the_next_trips(self):
        # the reader (0x0020) precedes the load (0x0030) in the trip and is reached from it only through the back edge
        self.assertEqual(R["load_in_trip"].count(synth("load_in_trip", "a_stage_ahead")), 0)

    def test_a_cold_reload_is_no_reload_of_the_trip(self):
        k = synth("store_reload", "cold_reload")
        self.assertEqual(R["store_reload"].count(k), 0)

    def test_a_wide_load_is_read_through_its_second_register(self):
        self.assertEqual(R["load_in_trip"].count(synth("load_in_trip", "wide_pair")), 1)

    def test_store_reload_names_the_address(self):
        self.assertIn("[R8+0x10]", self.one("store_reload", "reload").what)

    def test_a_store_in_a_later_trip_reaches_the_arrival_and_the_address_order_does_not(self):
        k = synth("store_unfenced", "loop_carried")
        arrive = next(i for i in k.instructions if i.opcode.startswith("SYNCS.ARRIVE"))
        store = next(i for i in k.instructions if i.base == "STS")
        self.assertLess(arrive.address, store.address)               # the listing's order shows no store before the arrival
        self.assertEqual(R["store_unfenced"].count(k), 1)

    def test_clock_uniform_names_the_register(self):
        self.assertIn("UR6", self.one("clock_uniform", "summed").what)

    def test_step_port_reports_the_largest_region_in_bytes(self):
        f = self.one("step_port", "loop_with_loads")
        self.assertEqual(f.count, (2 * 3072 + 4 * 2 * 512) // 2)       # two steps' operands, two matrix loads by four warps
        self.assertIn("operands 3,072 + shared 2,048 (x 4 warps)", f.what)
        self.assertIn("40 wavefronts", f.what)

    def test_step_work_reports_the_classes(self):
        f = self.one("step_work", "loop")
        self.assertEqual(f.count, 5)
        self.assertIn("control 2", f.what)
        self.assertIn("tensor 1", f.what)

    def test_step_density_is_the_whole_function_over_its_steps(self):
        self.assertEqual(self.one("step_density", "dense").what, "16 instructions for 4 tensor step(s)")

    def test_step_gap_takes_the_larger_of_inside_and_around(self):
        self.assertIn("5 between steps, 4 around the back edge", self.one("step_gap", "gapped").what)

    def test_step_branches_counts_the_barrier_and_the_forward_branch_not_the_back_edge(self):
        found = R["step_branches"].check(synth("step_branches", "in_step"))
        self.assertEqual([f.what.split(" at ")[0] for f in found], ["BSSY", "@P1 BRA"])        # not the `@!P0 BRA` that closes the loop

    def test_the_exit_laid_out_as_the_latch_is_not_a_branch_inside_the_step(self):
        self.assertEqual(R["step_branches"].count(synth("step_branches", "latch_exit")), 0)

    def test_step_predicated_names_the_predicate(self):
        self.assertIn("@P0 HGMMA", self.one("step_predicated", "predicated").what)

    def test_a_hot_call_names_the_subroutine(self):
        self.assertIn("convergence subroutine at", self.one("convergence_call", "hot_call").what)


class Bounds(unittest.TestCase):
    def check(self, rule, name, allow=None):
        found = list(runner.kernels_of_files(str(FIXTURES / "rules" / rule / name / "kernel")).values())
        entries = [] if allow is None else [{"kernel": "tile", "rule": rule, "allow": allow, "reason": "the design's bound"}]
        declarations = declare.load_declarations(json.dumps({"declarations": entries}), R)
        refusals = [r for r in declare.check(found, R, declarations) if r.rule == rule]      # the fixtures' waits are bare
        return refusals, [line for line in declare.figures(found, R, declarations) if f": {rule}: " in line]

    def test_a_figure_with_no_bound_is_reported_and_never_refused(self):
        refusals, lines = self.check("step_port", "operands_only")
        self.assertEqual(refusals, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("step_port: 2560 (no bound declared)", lines[0])

    def test_a_declared_bound_holds_the_figure(self):
        self.assertEqual(self.check("step_port", "operands_only", 2560)[0], [])
        (refusal,), lines = self.check("step_port", "operands_only", 2559)
        self.assertEqual((refusal.count, refusal.allowed, refusal.reason), (2560, 2559, "the design's bound"))
        self.assertEqual(lines, [])

    def test_the_six_bound_rules(self):
        self.assertEqual(sorted(n for n, r in R.items() if r.bound),
                         ["function_size", "loop_held", "step_density", "step_gap", "step_port", "step_work"])

    def test_a_count_rule_is_refused_with_no_declaration(self):
        (refusal,), _ = self.check("load_in_trip", "read_in_trip")
        self.assertEqual((refusal.rule, refusal.count, refusal.allowed), ("load_in_trip", 1, 0))

    def test_the_command_prints_the_figures_and_exits_0(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            path = Path(d) / "d.json"
            path.write_text('{"declarations": []}')
            code = main(["check", str(FIXTURES / "rules/step_work/loop/kernel"), "--fixture", "--declare", str(path)])
        self.assertIn(code, (0, 1), out.getvalue() + err.getvalue())         # 1: the fixture's bare wait, `wait_after_issue`
        self.assertIn("6 figure(s) of a bound rule, with no bound declared:", out.getvalue())
        self.assertIn("step_work: 5 (no bound declared)", out.getvalue())


class Toolkit(unittest.TestCase):
    """The rules on the toolkit's output for `examples/checker/stages.cu` (nvcc, sm_90a): a count each, from the saved text."""

    def test_the_stages_kernels(self):
        got = {name: counts(k) for name, k in stages().items()}
        keep = ("loop_convergence", "load_in_trip", "store_reload", "store_unfenced", "clock_uniform", "convergence_call",
                "step_predicated", "step_branches", "step_conversion", "step_global")
        self.assertEqual({n: {r: c for r, c in d.items() if r in keep} for n, d in got.items()}, {
            "loop_join": {"loop_convergence": 2, "step_branches": 2, "step_conversion": 1, "step_global": 1},
            "trip_load": {"load_in_trip": 1},
            "trip_reload": {"store_reload": 1},
            "arrive_unfenced": {"store_unfenced": 1},
            "arrive_fenced": {},
            "step_costs": {"load_in_trip": 1, "step_conversion": 1, "step_global": 2},
            "clock_sum": {},
            "step_guarded": {},
            "stub_join": {},
        })

    def test_the_figures_of_a_step(self):
        k = stages()["arrive_unfenced"]
        self.assertEqual((R["step_port"].count(k), R["step_work"].count(k), R["step_density"].count(k)), (2560, 7, 35))
        # a step's operands, 2,560 bytes; toolkit 13.5 issues the shared store before the arrive, outside the step's region (13.0
        # kept it inside: 512 B by the 4 warps, 3,072 B a step, 25 instructions a step, density 36), so the region holds no shared bytes
        (f,) = R["step_port"].check(k)
        self.assertIn("operands 2,560 + shared 0", f.what)

    def test_the_toolkit_branches_around_a_guarded_step_and_issues_the_empty_group(self):
        k = stages()["step_guarded"]
        self.assertEqual(R["step_predicated"].count(k), 0)
        self.assertEqual([s.null for s in k.steps], [False, True])

    def test_the_toolkit_reads_the_clock_into_a_uniform_register_and_the_sum_is_not_there(self):
        k = stages()["clock_sum"]
        self.assertTrue(any(i.opcode == "S2UR" and "SR_CLOCKLO" in i.operands for i in k.instructions))
        self.assertEqual(R["clock_uniform"].count(k), 0)

    def test_the_fence_is_what_separates_the_two_arrivals(self):
        s = stages()
        self.assertEqual(len(s["arrive_fenced"].fences), 1)
        self.assertEqual((R["store_unfenced"].count(s["arrive_fenced"]), R["store_unfenced"].count(s["arrive_unfenced"])), (0, 1))

    def test_the_slow_path_call_is_no_convergence_subroutine(self):
        k = stages()["stub_join"]
        self.assertEqual([(c.subroutine, c.hot) for c in k.calls], [(False, True), (False, True)])
        self.assertEqual(R["convergence_call"].count(k), 0)

    def test_the_collectives_the_toolkit_outlines_are_guarded_calls(self):
        (k,) = runner.kernels_of_files(str(FIXTURES / "collectives" / "kernel")).values()
        self.assertEqual([(c.subroutine, c.hot) for c in k.calls], [(True, False)] * 5)
        self.assertEqual(R["convergence_call"].count(k), 0)

    def test_the_stage_one_example_kernel_under_the_new_rules(self):
        # `load_in_trip` 1: its counted loop (a) loads a word and sums it in the same trip; `wait_all_with_work` 1: the group
        # the wait at 0x0700 waits for is not read by the arrive and the two shared loads up to the next step
        (k,) = runner.kernels_of_files(str(FIXTURES / "example" / "kernel")).values()
        new = {n: r.count(k) for n, r in R.items() if n not in ("stack_memory", "step_rolled", "wait_after_issue")}
        self.assertEqual(new, {"accumulator_moved": 0, "clock_uniform": 0, "convergence_call": 0, "function_size": 148, "load_behind_load": 0, "load_in_trip": 1, "loop_convergence": 0, "loop_held": 16, "step_branches": 0,
                               "step_conversion": 0, "step_density": 30, "step_gap": 16, "step_global": 0, "step_port": 3584,
                               "step_predicated": 0, "step_serialized": 0, "step_work": 17, "store_reload": 0, "store_unfenced": 0,
                               "wait_all_with_work": 1})


class NaturalLoopsAgainstSpans(unittest.TestCase):
    """Where a loop is the natural loop of the graph, not the span a backward branch closes."""

    def test_a_stub_that_returns_into_the_body_closes_no_loop(self):
        k = synth("loop_convergence", "stub_return")
        self.assertEqual((R["loop_convergence"].count(k), span_count(k, conv)), (0, 2))

    def test_the_toolkits_own_stub_after_a_suspended_wait(self):
        k = stages()["stub_join"]
        self.assertEqual(R["loop_convergence"].count(k), 0)
        self.assertEqual(span_count(k, conv), 6)                       # four barriers and two syncs, laid out inside the stub's span

    def test_the_example_kernel_has_eight_backward_branches_and_six_loops(self):
        (k,) = runner.kernels_of_files(str(FIXTURES / "example" / "kernel")).values()
        self.assertEqual((len([i for i in k.instructions if i.base == "BRA" and i.target is not None and i.target < i.address]), len(k.loops)), (8, 6))

    def test_an_inner_loop_without_steps_is_the_steps_loops_body_and_a_trip_is_the_inner_loops(self):
        k = stages()["loop_join"]
        self.assertEqual(R["load_in_trip"].count(k), 0)                # the flags load sits in the outer trip, the inner loop is the trip
        self.assertEqual(len(k.innermost()), 1)
        self.assertEqual([r.kind for r in k.regions], ["loop"])


if __name__ == "__main__":
    unittest.main()
