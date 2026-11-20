"""Natural loops of a control-flow graph, from its dominators. A loop is a BACK EDGE: an edge whose target dominates its
source. It is not "a backward branch": an edge that jumps to a lower address but whose target does not dominate its
source (the return from an out-of-line stub, say) closes no loop. Plain graph algorithms on block ids, no SASS."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalLoop:
    """A loop: its header, every block of its body (the header included) and the sources of its back edges."""
    header: str
    blocks: frozenset[str]
    tails: tuple[str, ...]


def _reverse_postorder(entry: str, succ: dict[str, list[str]]) -> list[str]:
    seen, order = {entry}, []
    stack = [(entry, iter(succ[entry]))]
    while stack:
        node, children = stack[-1]
        for child in children:
            if child not in seen:
                seen.add(child)
                stack.append((child, iter(succ[child])))
                break
        else:
            order.append(node)
            stack.pop()
    return order[::-1]


def immediate_dominators(entry: str, edges: list[tuple[str, str]]) -> dict[str, str]:
    """Each block reachable from `entry` -> its immediate dominator (the entry -> itself): Cooper, Harvey and Kennedy's
    iteration over the reverse postorder."""
    succ: dict[str, list[str]] = {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        succ.setdefault(b, [])
    succ.setdefault(entry, [])
    order = _reverse_postorder(entry, succ)
    index = {node: i for i, node in enumerate(order)}
    pred: dict[str, list[str]] = {node: [] for node in order}
    for a, b in edges:
        if a in index and b in index:
            pred[b].append(a)
    idom = {entry: entry}

    def meet(a: str, b: str) -> str:
        while a != b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for node in order[1:]:
            done = [p for p in pred[node] if p in idom]
            new = done[0]
            for p in done[1:]:
                new = meet(p, new)
            if idom.get(node) != new:
                idom[node] = new
                changed = True
    return idom


def dominates(idom: dict[str, str], a: str, b: str) -> bool:
    """Whether `a` dominates `b` (every block dominates itself); False for a block the entry does not reach."""
    if b not in idom:
        return False
    while True:
        if a == b:
            return True
        if idom[b] == b:
            return False
        b = idom[b]


def natural_loops(entry: str, edges: list[tuple[str, str]]) -> list[NaturalLoop]:
    """The natural loops of the graph, one a header (the back edges to one header make one loop), in no order."""
    idom = immediate_dominators(entry, edges)
    pred: dict[str, list[str]] = {}
    for a, b in edges:
        pred.setdefault(b, []).append(a)
    tails: dict[str, list[str]] = {}
    for a, b in edges:
        if a in idom and dominates(idom, b, a):
            tails.setdefault(b, []).append(a)
    loops = []
    for header, sources in tails.items():
        body, stack = {header}, list(sources)
        while stack:
            node = stack.pop()
            if node not in body:
                body.add(node)
                stack.extend(pred.get(node, []))
        loops.append(NaturalLoop(header, frozenset(body), tuple(sources)))
    return loops
