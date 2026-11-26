"""accumulator_moved (reads ASYNC steps only): an instruction that is not a tensor step writes a register a step chain accumulates into, between two steps of the chain or between a step and its wait.

The toolkit moves an accumulator when two blocks of steps want it in different registers (steps of different shapes over
the same accumulators: `m64n16k16` into groups of 8 registers in one block, `m64n64k16` into 32 consecutive registers in
the other): the copies sit at the switch between the blocks, and the pipeline of steps that spans it is serialized. The rule
reads each async step's destination registers (`_accumulators.destination`: N/2 consecutive registers for an `m64nNk16` step
with a 32-bit accumulator, N/4 with a 16-bit one) and two flows over the graph: FORWARD, the registers a step wrote that may
still be live here, and BACKWARD, the registers a step later reaches. A non-step instruction is a MOVE when it writes a register
that a step reached before it and that a step reads as its accumulator input after it, with no other write between (a step writes its destination too) (between two
steps of the chain; a step that starts a fresh accumulation, `RZ` as its input, reads none, so a register reused for the
epilogue's packing is no move), or that a step wrote and no wait has covered since (between a step and its wait; any wait ends that, a partial one included), and it READS a register that is
itself a step's destination: a copy or a rescale of an accumulator, not a zeroing or a load into registers a step happens to reuse.
A step whose destination the model cannot read (no shape, no accumulator type, the null group) is not a chain. One finding a
basic block that holds moves, counting its instructions."""
from __future__ import annotations

from .. import flow
from ..model import Instruction, Kernel
from ...manifest import Finding
from ..rule import Fixture, Rule
from ._accumulators import carried, destination


def _steps(kernel: Kernel) -> dict[int, frozenset[str]]:
    return {s.instruction.address: destination(s) for s in kernel.async_steps if not s.instruction.dead}


def check(kernel: Kernel) -> list[Finding]:
    dest = _steps(kernel)
    reads = {s.instruction.address: carried(s) for s in kernel.async_steps if not s.instruction.dead}
    every = frozenset(r for regs in dest.values() for r in regs)
    if not every:
        return []

    def forward(state, ins: Instruction):
        # "w:R24": a step wrote it (still may be live); "f:R24": a step wrote it and no wait has covered it since
        regs = dest.get(ins.address)
        if regs:
            return state | {f"{k}:{r}" for r in regs for k in "wf"}
        if ins.opcode.startswith("WARPGROUP.DEPBAR"):
            return frozenset(s for s in state if s[0] == "w")
        return state

    def backward(state, ins: Instruction):
        # the registers a step later reads as its accumulator input, until a write of the register: a step writes its
        # destination (a step that starts a fresh accumulation reads none of it), as does any other instruction
        if ins.address in dest:
            return (state - dest[ins.address]) | reads[ins.address]
        return state - ins.writes if ins.writes else state

    before = dict(((ins.address), state) for state, ins in flow.walk(kernel, forward))
    after = dict(((ins.address), state) for state, ins in flow.walk_backward(kernel, backward))
    per_block: dict[str, list[Instruction]] = {}
    for block in kernel.blocks:
        for ins in block.instructions:
            if ins.address in dest or ins.dead or not (written := ins.writes & every) or not (ins.reads & every):
                continue
            seen = before[ins.address]
            between = {r for r in written if f"w:{r}" in seen and r in after[ins.address]}
            in_flight = {r for r in written if f"f:{r}" in seen}
            if between or in_flight:
                per_block.setdefault(block.id, []).append(ins)
    found = []
    for moves in per_block.values():
        regs = sorted({r for ins in moves for r in ins.writes & every}, key=lambda r: int(r[1:]))
        found.append(Finding(f"{moves[0].address:#06x}", len(moves),
                             f"{len(moves)} instruction(s) from {moves[0].address:#06x} to {moves[-1].address:#06x} write "
                             f"accumulator registers ({regs[0]} to {regs[-1]}) between steps of one chain"))
    return found


RULE = Rule(
    name="accumulator_moved",
    statement="an instruction that is not a tensor step writes a register in a step chain's destination range, between two steps of the chain or between a step and its wait (async steps only)",
    check=check,
    refusal="accumulator registers are copied between tensor steps: the toolkit moved an accumulator (steps of different shapes over the same accumulators)",
    fix="give the steps that share accumulators one shape, or lay the accumulators out the way each shape's steps want them "
        "(one array of consecutive registers per shape) so the toolkit has nothing to move at the switch",
    fixtures=(Fixture("mixed_shapes", 64), Fixture("serialized", 0), Fixture("fenced", 0)),
)
