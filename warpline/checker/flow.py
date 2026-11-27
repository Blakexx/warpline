"""A forward flow over a kernel's graph, for the rules that ask what may hold at a point (a shared store not yet fenced,
a register that holds a clock reading): the state at the entry of each basic block, a set that joins by union, from the
function's entry to a fixed point. A rule walks a block's instructions from its entry state with the same step function.
The step function must be monotone (a gen/kill function of a set is), or the iteration need not end."""
from __future__ import annotations

from collections.abc import Callable

from .model import OUTLINED, Instruction, Kernel

State = frozenset


def entry_states(kernel: Kernel, step: Callable[[State, Instruction], State], start: State = frozenset()) -> dict[str, State]:
    """Block id -> the union, over every path from the entry, of the states at the block's start. A block no path reaches
    has the empty state."""
    states = {b.id: frozenset() for b in kernel.blocks}
    if not kernel.blocks:
        return states
    states[kernel.blocks[0].id] = start
    work, seen = [kernel.blocks[0].id], {kernel.blocks[0].id}
    while work:
        block = kernel.block(work.pop())
        state = states[block.id]
        for ins in block.instructions:
            state = step(state, ins)
        for nxt in block.successors:
            if nxt not in seen or not state <= states[nxt]:
                seen.add(nxt)
                states[nxt] = states[nxt] | state
                work.append(nxt)
    return states


def walk(kernel: Kernel, step: Callable[[State, Instruction], State], start: State = frozenset()):
    """Every instruction of every block with the state just before it (`entry_states`, then each block's own
    instructions in order): `(state, instruction)` pairs, in block order."""
    states = entry_states(kernel, step, start)
    for block in kernel.blocks:
        state = states[block.id]
        for ins in block.instructions:
            yield state, ins
            state = step(state, ins)


def exit_states(kernel: Kernel, step: Callable[[State, Instruction], State]) -> dict[str, State]:
    """The same flow run backward, for the rules that ask what may still come (a step that a path reaches after a point):
    block id -> the union, over every path from the block's end to the function's end, of the states there. `step` is
    applied from a block's last instruction to its first, so its state is what holds just before the instruction, from what
    holds just after. A block with no successor starts empty."""
    ends = {b.id: frozenset() for b in kernel.blocks}
    work = [b.id for b in reversed(kernel.blocks)]
    preds: dict[str, list[str]] = {b.id: [] for b in kernel.blocks}
    for b in kernel.blocks:
        for nxt in b.successors:
            preds[nxt].append(b.id)
    while work:
        block = kernel.block(work.pop())
        state = ends[block.id]
        for ins in reversed(block.instructions):
            state = step(state, ins)
        for before in preds[block.id]:
            if not state <= ends[before]:
                ends[before] = ends[before] | state
                work.append(before)
    return ends


def walk_backward(kernel: Kernel, step: Callable[[State, Instruction], State]):
    """Every instruction of every block with the state just after it (`exit_states`, then each block's instructions from
    the last): `(state, instruction)` pairs, in block order, each block's instructions last first."""
    ends = exit_states(kernel, step)
    for block in kernel.blocks:
        state = ends[block.id]
        for ins in reversed(block.instructions):
            yield state, ins
            state = step(state, ins)


def live_step(state: State, ins: Instruction, registers: tuple[frozenset[str], frozenset[str]] | None = None) -> State:
    """The backward step of register liveness: the registers live before an instruction from those live after it. An
    instruction reads its `reads` and kills its `writes` (`registers`, where given: a kernel's `Kernel.registers_of`); one under
    a predicate (`@P0 IADD3 R4, ...`) keeps the old value where the predicate is false, so it kills nothing and reads what it
    writes. Code under `@!PT` never runs and changes nothing."""
    if ins.dead:
        return state
    reads, writes = registers if registers is not None else (ins.reads, ins.writes)
    if ins.predicate not in ("", "@PT", "@UPT"):
        return state | reads | writes
    return (state - writes) | reads


def liveness(kernel: Kernel) -> Callable[[State, Instruction], State]:
    """`live_step` over the kernel's own registers of each instruction (`Kernel.registers_of`: its life-range marks where it
    has them, else its operands)."""
    return lambda state, ins: live_step(state, ins, kernel.registers_of(ins))


def live_in(kernel: Kernel) -> dict[str, State]:
    """Block id -> the registers live at the block's entry (vector `R5` and uniform `UR5` alike): the backward flow of
    `exit_states` with `liveness`, to a fixed point over the graph's edges (a register is live where some path from there
    reads it before writing it), then each block's own instructions from its end. A call's callee, which the graph does not
    hold, is read only through the kernel's marks (`Kernel.marks`: a subroutine's body); without them what it reads and writes
    is unknown."""
    step = liveness(kernel)
    ends = exit_states(kernel, step)
    found = {}
    for block in kernel.blocks:
        state = ends[block.id]
        for ins in reversed(block.instructions):
            state = step(state, ins)
        found[block.id] = state
    return found


def pressure(kernel: Kernel) -> dict[int, State]:
    """Instruction address -> the registers HELD across the instruction: those live into it or written by it (`liveness`) that
    some write reaches (one on a path from the entry, `entry_states` of the writes, or its own). A register whose value no write
    has made is not held (one written and read under one predicate does not count from the kernel's entry), so the count of a
    vector register's set is the program point's register pressure. The instructions of the block no edge reaches (the outlined
    code after the graph, `model.OUTLINED`) are not in it: what they read and write is their caller's (`Kernel.marks`)."""
    step = liveness(kernel)
    after = {ins.address: state for state, ins in walk_backward(kernel, step)}
    written = lambda state, ins: state if ins.dead else state | kernel.registers_of(ins)[1]
    found = {}
    for reached, ins in walk(kernel, written):
        if ins.block == OUTLINED:
            continue
        own = frozenset() if ins.dead else kernel.registers_of(ins)[1]
        found[ins.address] = (step(after[ins.address], ins) | own) & (reached | own)
    return found
