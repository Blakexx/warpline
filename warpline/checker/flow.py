"""A forward flow over a kernel's graph, for the rules that ask what may hold at a point (a shared store not yet fenced,
a register that holds a clock reading): the state at the entry of each basic block, a set that joins by union, from the
function's entry to a fixed point. A rule walks a block's instructions from its entry state with the same step function.
The step function must be monotone (a gen/kill function of a set is), or the iteration need not end."""
from __future__ import annotations

from collections.abc import Callable

from .model import Instruction, Kernel

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
