"""LAYER 2, THE KERNEL MODEL: one plain data structure a kernel, built from the readers' data. Rules read this and
never text. See docs/checker.md for each field.

A kernel has its instructions, its basic blocks (from the control-flow graph), its NATURAL LOOPS (`loops.py`, from the
graph's back edges and dominators), its tensor steps (`HGMMA`/`IGMMA`) with their commit (the `gsb0` flag on a group's
last step) and the waits on them (`WARPGROUP.DEPBAR`), the `WARPGROUP.ARRIVE`s, its shared-, global- and local-memory
accesses by kind, its barriers and convergence instructions, and its registers, stack and shared bytes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import loops as loops_mod
from . import readers


class ModelError(ValueError):
    """Readers' data that do not describe one kernel (a graph whose instructions are not the listing's)."""


@dataclass(frozen=True)
class Instruction:
    address: int
    predicate: str
    opcode: str
    operands: str
    block: str

    @property
    def base(self) -> str:
        """The opcode without its modifiers (`HGMMA.64x16x16.F32.BF16` -> `HGMMA`)."""
        return self.opcode.split(".")[0]


@dataclass(frozen=True)
class Block:
    """A basic block: its id, address extent (first and last instruction), instructions, successors' ids, and the
    header of the innermost natural loop that holds it (None outside every loop)."""
    id: str
    start: int
    end: int
    instructions: tuple[Instruction, ...]
    successors: tuple[str, ...]
    loop: str | None


@dataclass(frozen=True)
class Loop:
    """A natural loop. `spin`: one block that holds a barrier try-wait: the wait's retry, not work."""
    header: str
    address: int
    blocks: frozenset[str]
    tails: tuple[str, ...]
    spin: bool


@dataclass(frozen=True)
class TensorStep:
    """A tensor-core step. `commit`: the last step of its group (it carries `gsb0`)."""
    instruction: Instruction
    commit: bool


@dataclass(frozen=True)
class TensorWait:
    """`WARPGROUP.DEPBAR.LE gsb0, N`: waits until at most `left` groups are in flight."""
    instruction: Instruction
    left: int


@dataclass(frozen=True)
class MemoryAccess:
    """`space` is shared, global, local or generic; `kind` is load, store, atomic or async (a copy issued without a
    register result: `LDGSTS`, the bulk and tensor-map copies); `width` is bits."""
    instruction: Instruction
    space: str
    kind: str
    width: int


@dataclass(frozen=True)
class Kernel:
    name: str
    demangled: str
    instructions: tuple[Instruction, ...]
    blocks: tuple[Block, ...]
    loops: tuple[Loop, ...]
    steps: tuple[TensorStep, ...]
    waits: tuple[TensorWait, ...]
    arrives: tuple[Instruction, ...]
    accesses: tuple[MemoryAccess, ...]
    barriers: tuple[Instruction, ...]
    convergence: tuple[Instruction, ...]
    registers: int
    stack_bytes: int
    shared_bytes: int

    def block(self, id: str) -> Block:
        return next(b for b in self.blocks if b.id == id)

    def loop_of(self, instruction: Instruction) -> Loop | None:
        """The innermost natural loop around an instruction."""
        header = self.block(instruction.block).loop
        return next((lp for lp in self.loops if lp.header == header), None) if header else None


#: base opcode -> (space, kind)
_ACCESS = {
    "LDS": ("shared", "load"), "LDSM": ("shared", "load"), "STS": ("shared", "store"), "ATOMS": ("shared", "atomic"),
    "LDG": ("global", "load"), "STG": ("global", "store"), "ATOMG": ("global", "atomic"), "RED": ("global", "atomic"),
    "LDL": ("local", "load"), "STL": ("local", "store"), "LD": ("generic", "load"), "ST": ("generic", "store"),
    "ATOM": ("generic", "atomic"), "LDGSTS": ("global", "async"), "UBLKCP": ("global", "async"),
    "UBLKRED": ("global", "async"), "UTMALDG": ("global", "async"), "UTMASTG": ("global", "async"),
    "UTMAREDG": ("global", "async"), "UTMACP": ("global", "async"),
}
_WIDTH = {"U8": 8, "S8": 8, "U16": 16, "S16": 16, "64": 64, "128": 128}
_TENSOR = ("HGMMA", "IGMMA", "QGMMA", "BGMMA")
_BARRIER = ("BAR", "SYNCS", "MEMBAR")
_CONVERGENCE = ("BSSY", "BSYNC", "BREAK", "WARPSYNC")
_WAIT = re.compile(r"gsb0\s*,\s*(0x[0-9a-f]+)")
_TRYWAIT = "SYNCS.PHASECHK.TRANS64.TRYWAIT"


def _access(ins: Instruction) -> MemoryAccess | None:
    if (found := _ACCESS.get(ins.base)) is None:
        return None
    width = next((_WIDTH[m] for m in ins.opcode.split(".")[1:] if m in _WIDTH), 32)
    return MemoryAccess(ins, found[0], found[1], width)


def build_kernel(name: str, sass: list[readers.SassLine], cfg: readers.Cfg, usage: readers.Usage,
                 demangled: str | None = None) -> Kernel:
    """One kernel from its listing's instructions, its graph and its resource report. The graph's instruction offsets
    must be the listing's addresses, no more and no fewer, except the padding after the last of them (a self-branch and
    `NOP`s the graph does not hold) which is dropped."""
    owner = {}
    for block in cfg.blocks:
        for offset in block.offsets:
            if offset in owner:
                raise ModelError(f"{name}: offset {offset:#x} is in two blocks")
            owner[offset] = block.id
    last = max(owner, default=-1)
    sass = [line for line in sass if line.address <= last]   # what follows the graph's last instruction is padding
    addresses = {line.address for line in sass}
    if set(owner) != addresses:
        odd = sorted(set(owner) ^ addresses)[:4]
        raise ModelError(f"{name}: the graph and the listing disagree at {[hex(a) for a in odd]}")
    instructions = tuple(Instruction(s.address, s.predicate, s.opcode, s.operands, owner[s.address]) for s in sass)
    by_block: dict[str, list[Instruction]] = {b.id: [] for b in cfg.blocks}
    for ins in instructions:
        by_block[ins.block].append(ins)
    successors: dict[str, list[str]] = {b.id: [] for b in cfg.blocks}
    for a, b in cfg.edges:
        if a not in successors or b not in successors:
            raise ModelError(f"{name}: edge {a} -> {b} names a block the graph does not hold")
        successors[a].append(b)

    found = loops_mod.natural_loops(cfg.entry, list(cfg.edges))
    first = {bid: (body[0].address if body else -1) for bid, body in by_block.items()}
    spin = {lp.header: len(lp.blocks) == 1 and any(i.opcode == _TRYWAIT for i in by_block[lp.header]) for lp in found}
    model_loops = tuple(sorted((Loop(lp.header, first[lp.header], lp.blocks, lp.tails, spin[lp.header]) for lp in found),
                               key=lambda lp: lp.address))
    innermost: dict[str, str] = {}
    for lp in sorted(model_loops, key=lambda lp: -len(lp.blocks)):   # larger first, so a smaller loop overwrites
        for bid in lp.blocks:
            innermost[bid] = lp.header
    blocks = tuple(Block(b.id, by_block[b.id][0].address if by_block[b.id] else -1,
                         by_block[b.id][-1].address if by_block[b.id] else -1, tuple(by_block[b.id]),
                         tuple(successors[b.id]), innermost.get(b.id)) for b in cfg.blocks)

    steps = tuple(TensorStep(i, "gsb0" in i.operands) for i in instructions if i.base in _TENSOR)
    waits = tuple(TensorWait(i, int(m.group(1), 16)) for i in instructions
                  if i.opcode.startswith("WARPGROUP.DEPBAR") and (m := _WAIT.search(i.operands)))
    return Kernel(
        name=name, demangled=demangled or name, instructions=instructions, blocks=blocks, loops=model_loops,
        steps=steps, waits=waits, arrives=tuple(i for i in instructions if i.opcode.startswith("WARPGROUP.ARRIVE")),
        accesses=tuple(a for i in instructions if (a := _access(i))),
        barriers=tuple(i for i in instructions if i.base in _BARRIER),
        convergence=tuple(i for i in instructions
                          if i.base in _CONVERGENCE or (i.base == "BRA" and "DIV" in i.opcode.split(".")[1:])),
        registers=usage.registers, stack_bytes=usage.stack, shared_bytes=usage.shared)


def kernels_from_text(sass: str, cfg: str, res: str, names: str | None = None) -> dict[str, Kernel]:
    """Every kernel of the tools' texts (`names`: `c++filt`'s answer, a line for each function of the listing in its
    order; none: the mangled name stands for the demangled one). A function the listing holds and the graph or the
    resource report lacks is an error."""
    lines, graphs, usage = readers.sass_functions(sass), readers.cfg_functions(cfg), readers.res_usage(res)
    pretty = readers.demangled(names, list(lines)) if names is not None else {}
    kernels = {}
    for name, body in lines.items():
        if name not in graphs or name not in usage:
            raise ModelError(f"{name}: the listing has it, the graph or the resource report does not")
        kernels[name] = build_kernel(name, body, graphs[name], usage[name], pretty.get(name))
    return kernels
