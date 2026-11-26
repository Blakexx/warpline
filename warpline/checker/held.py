"""WHAT A LOOP HOLDS: the registers live at the top of each natural loop of a kernel (`Kernel.live_in` at the loop's header),
grouped by where the value was last made before the loop, and how many of them the loop reads at all. Under a small register
count the values a loop holds that do not fit are spilled (stack stores and reloads, rule `stack_memory`); this says which
source lines made them. The rule `loop_held` reports the largest of the figures; `warpline held` prints them all
(`report`).

Only vector registers (`R5`) are counted: they are what the kernel's register count limits. A uniform register (`UR5`) is a
separate small file, named in the report beside the count."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import flow
from .model import Instruction, Kernel, Loop

_VECTOR = re.compile(r"R\d+")


@dataclass(frozen=True)
class Group:
    """The live vector registers of a loop's top made at one place before the loop. `place` is the source `(file, line)` of the
    instruction that last made them, or its address where no source line is known, or `None` for a register no instruction
    made (live from the kernel's entry). `registers`: those registers (sorted by number); `used`: how many of them an
    instruction in the loop reads; `carried`: how many an instruction in the loop also writes."""
    place: tuple[str, int] | str | None
    registers: tuple[str, ...]
    used: int
    carried: int

    @property
    def label(self) -> str:
        if self.place is None:
            return "(made before any instruction)"
        return self.place if isinstance(self.place, str) else f"{self.place[0]}:{self.place[1]}"


@dataclass(frozen=True)
class Hold:
    """What one loop holds at its top: `live` vector registers (`uniform` uniform ones beside them), the `groups` by place
    most first (ties by place), and the loop's `depth` (0 outermost)."""
    loop: Loop
    depth: int
    live: int
    uniform: int
    groups: tuple[Group, ...]


def _reaching(kernel: Kernel) -> dict[str, dict[str, frozenset[int]]]:
    """Block id -> register -> the addresses of the definitions that reach the block's entry: a forward flow over the graph (the
    back edges' definitions included), a write kills the register's earlier definitions and a predicated write kills none. Each
    block is summarized once (the definitions of each register it writes that reach its end, and whether an unpredicated write kills the ones before)."""
    summary = {}
    for block in kernel.blocks:
        last: dict[str, tuple[frozenset[int], bool]] = {}
        for ins in block.instructions:
            if ins.dead:
                continue
            kills = ins.predicate in ("", "@PT", "@UPT")
            for r in ins.writes:
                before, killed = last.get(r, (frozenset(), False))
                last[r] = (frozenset({ins.address}), True) if kills else (before | {ins.address}, killed)
        summary[block.id] = last
    entry: dict[str, dict[str, frozenset[int]]] = {b.id: {} for b in kernel.blocks}
    if not kernel.blocks:
        return entry
    work, seen = [kernel.blocks[0].id], {kernel.blocks[0].id}
    while work:
        block = kernel.block(work.pop())
        out = dict(entry[block.id])
        for r, (addresses, kills) in summary[block.id].items():
            out[r] = addresses if kills else out.get(r, frozenset()) | addresses
        for nxt in block.successors:
            into, grew = entry[nxt], nxt not in seen
            for r, addresses in out.items():
                if not addresses <= into.get(r, frozenset()):
                    into[r] = into.get(r, frozenset()) | addresses
                    grew = True
            if grew:
                seen.add(nxt)
                work.append(nxt)
    return entry


def holds(kernel: Kernel) -> list[Hold]:
    """The hold of every loop of the kernel, outermost first (nested loops after the loop around them, each level in address
    order)."""
    if not kernel.loops:
        return []
    live, reaching = kernel.live_in(), _reaching(kernel)
    depth = {lp.header: sum(o is not lp and lp.blocks < o.blocks for o in kernel.loops) for lp in kernel.loops}
    order = []

    def visit(parent: Loop | None):
        for lp in kernel.loops:
            inside = [o for o in kernel.loops if o is not lp and lp.blocks < o.blocks]
            owner = min(inside, key=lambda o: len(o.blocks), default=None)
            if owner is parent:
                order.append(lp)
                visit(lp)

    visit(None)
    return [_hold(kernel, lp, depth[lp.header], live[lp.header], reaching[lp.header]) for lp in order]


def _hold(kernel: Kernel, lp: Loop, depth: int, live: frozenset[str], reaching: dict[str, frozenset[int]]) -> Hold:
    body = [i for b in sorted(lp.blocks) for i in kernel.block(b).instructions if not i.dead]
    read = set().union(*(i.reads for i in body)) if body else set()
    written = set().union(*(i.writes for i in body)) if body else set()
    inside = {i.address for i in body}
    vector = sorted((r for r in live if _VECTOR.fullmatch(r)), key=lambda r: int(r[1:]))
    made: dict[str, int | None] = {}
    for r in vector:
        every = reaching.get(r, frozenset())
        outside = every - inside
        made[r] = max(outside) if outside else (max(every) if every else None)
    by_place: dict = {}
    for r in vector:
        a = made[r]
        place = None if a is None else kernel.lines.get(a, f"{a:#06x}")
        by_place.setdefault(place, []).append(r)
    groups = tuple(sorted(
        (Group(place, tuple(rs), sum(r in read for r in rs), sum(r in written for r in rs)) for place, rs in by_place.items()),
        key=lambda g: (-len(g.registers), g.label)))
    return Hold(lp, depth, len(vector), len(live) - len(vector), groups)


def report(kernels, pattern: str | None = None) -> str:
    """The text of `warpline held`: for each kernel (those whose demangled name matches the regular expression `pattern`, all
    if none), each loop outermost first with nested loops indented, its live registers at the top and, most first, each place
    that made them with how many of its registers the loop reads. No times and no absolute paths: the same text on every run."""
    out = []
    for kernel in kernels:
        if pattern and not re.search(pattern, kernel.demangled):
            continue
        found = holds(kernel)
        out.append(f"{kernel.demangled}  ({kernel.registers} registers, {len(found)} loop(s))")
        for h in found:
            pad = "  " * (h.depth + 1)
            extra = f", {h.uniform} uniform" if h.uniform else ""
            out.append(f"{pad}loop at {h.loop.address:#06x}: {h.live} registers live at its top{extra}")
            for g in h.groups:
                regs = " ".join(g.registers[:6]) + (" ..." if len(g.registers) > 6 else "")
                carried = f", {g.carried} rewritten in it" if g.carried else ""
                out.append(f"{pad}  {len(g.registers):4d}  {g.used:4d} read in the loop{carried}  {g.label}  [{regs}]")
    return "\n".join(out) + ("\n" if out else "")
