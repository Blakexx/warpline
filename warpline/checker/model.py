"""LAYER 2, THE KERNEL MODEL: one plain data structure a kernel, built from the readers' data. Rules read this and
never text. See docs/checker.md for each field.

A kernel has its instructions (each with its class, the registers it writes and reads, its address operand and its
branch target), its basic blocks (from the control-flow graph, each with whether a converged warp reaches it), its
NATURAL LOOPS (`loops.py`, from the graph's back edges and dominators), its tensor steps, of two kinds
(`async`: `HGMMA`/`IGMMA`/`QGMMA`/`BGMMA`; `sync`: `mma.sync`, `HMMA`/`IMMA`/`DMMA`) with their shape, operand bytes and
commit (the `gsb0` flag on an async group's last step) and the waits on the async ones (`WARPGROUP.DEPBAR`),
the regions a step's cost is read over, the `WARPGROUP.ARRIVE`s, its shared-, global- and local-memory accesses by kind
and bytes, its barriers, fences, convergence instructions and calls, and its registers, stack and shared bytes.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from . import loops as loops_mod
from . import readers


#: the id of the block that holds code after the graph's last instruction which the graph does not hold (a convergence
#: subroutine's body): no edge reaches it
OUTLINED = "outlined"


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

    @property
    def dead(self) -> bool:
        """Under the always-false predicate `@!PT`: code the toolkit keeps and a thread never executes."""
        return self.predicate == "@!PT"

    @property
    def klass(self) -> str:
        """The unit class the instruction is counted in (`INSTRUCTION_CLASSES`), `other` for none. A uniform-datapath
        instruction (`UIADD3`, `ULDC`) has its vector instruction's class."""
        found = next((name for name, prefixes in INSTRUCTION_CLASSES.items() if self.opcode.startswith(prefixes)), "other")
        if found == "other" and self.opcode.startswith("U") and len(self.opcode) > 1:
            return Instruction(self.address, self.predicate, self.opcode[1:], self.operands, self.block).klass
        return found

    def _split(self) -> tuple[frozenset[str], int]:
        """The destination registers and how many leading operands the destination takes: an opcode with a destination
        names it first, behind the predicates some of them write (`SHFL.IDX PT, R3, R2, ...`, `LOP3.LUT P0, R6, ...`); a
        discarded destination (`RZ`) takes its place and writes none. A store, a branch, a tensor step and a comparison
        that writes predicates alone have none."""
        base = self.base
        if base in _NO_DESTINATION or _PREDICATE_ONLY.fullmatch(base):
            return frozenset(), 0
        ops = [o.strip() for o in self.operands.split(",")]
        at = 0
        while at < len(ops) and _PREDICATE.fullmatch(ops[at]):
            at += 1
        if at == len(ops):
            return frozenset(), 0
        if ops[at] in ("RZ", "URZ"):
            return frozenset(), at + 1
        if (m := _REGISTER_ONLY.fullmatch(ops[at])) is None:
            return frozenset(), 0
        width = 4 if ".128" in self.opcode else 2 if (".64" in self.opcode or "WIDE" in self.opcode or base in _DOUBLE) else 1
        return frozenset(f"{m.group(1)}{int(m.group(2)) + j}" for j in range(width)), at + 1

    @property
    def writes(self) -> frozenset[str]:
        """The registers the instruction writes: its destination operand where the opcode has one, widened to the run its
        width reaches (`.64` and `.WIDE` two registers, `.128` four, a double-precision operation two). A uniform register
        is `UR5`, a vector register `R5`; `RZ` is none."""
        return self._split()[0]

    @property
    def reads(self) -> frozenset[str]:
        """The registers the instruction reads: every register its operands name, after the destination where it has
        one. A register written `R2.64` is the pair it starts; the data of a `.64` or `.128` store or atomic, the
        addend of an `IMAD.WIDE` and every operand of a double-precision operation are runs too."""
        ops = [o.strip() for o in self.operands.split(",")]
        first = self._split()[1]
        wide = 4 if ".128" in self.opcode else 2 if ".64" in self.opcode else 1
        found: set[str] = set()
        for k, op in enumerate(ops):
            if k < first:
                continue
            span = 1
            if self.base in _DOUBLE or (self.base == "IMAD" and "WIDE" in self.opcode and k == 3):
                span = 2
            elif self.base in _WIDE_DATA and "[" not in op:
                span = wide
            for kind, n, pair in _REGISTER.findall(op):
                found.update(f"{kind}{int(n) + j}" for j in range(max(span, 2 if pair else 1)))
        return frozenset(found)

    @property
    def address_operand(self) -> str | None:
        """The bracketed address operand (`[R8+0x10]`), if the instruction has one."""
        m = _ADDRESS.search(self.operands)
        return m.group(0) if m else None

    @property
    def address_registers(self) -> frozenset[str]:
        """The registers inside the address operand (`[R8+0x10]` -> `R8`; `[R2.64]` -> `R2`, `R3`)."""
        text = self.address_operand or ""
        return frozenset(f"{kind}{int(n) + j}" for kind, n, pair in _REGISTER.findall(text) for j in range(2 if pair else 1))

    @property
    def target(self) -> int | None:
        """The address a branch or call names (`BRA P0, 0x210` -> 0x210), None for any other instruction."""
        if self.base not in ("BRA", "CALL", "BSSY", "JMP"):
            return None
        found = re.findall(r"0x[0-9a-f]+", self.operands)
        return int(found[-1], 16) if found else None


@dataclass(frozen=True)
class Block:
    """A basic block: its id, address extent (first and last instruction), instructions, successors' ids, and the
    header of the innermost natural loop that holds it (None outside every loop). `cold`: no path from the entry reaches it
    but through a fallback a converged warp does not take (the branch `BRA.DIV` enters, the code past a `BRA.CONV`, a
    branch under the always-false predicate): code a converged warp never runs."""
    id: str
    start: int
    end: int
    instructions: tuple[Instruction, ...]
    successors: tuple[str, ...]
    loop: str | None
    cold: bool = False


@dataclass(frozen=True)
class Loop:
    """A natural loop. `spin`: a loop that holds a barrier try-wait and nothing but its retry (the integer move that forms the
    barrier's address, the try-wait, the toolkit's sleep, the retry's probe and the branch back), in one block or two: the wait's
    retry, not work."""
    header: str
    address: int
    blocks: frozenset[str]
    tails: tuple[str, ...]
    spin: bool


@dataclass(frozen=True)
class TensorStep:
    """A tensor-core step. `kind`: `async` (Hopper's warpgroup steps `HGMMA`, `IGMMA`, `QGMMA`, `BGMMA`: committed in groups,
    waited on with `WARPGROUP.DEPBAR`) or `sync` (`mma.sync`: `HMMA`, `IMMA`, `DMMA`, issued by a warp from registers, complete
    when the next instruction that reads its result issues: no commit group, no wait). `commit`: the last step of its async
    group (it carries `gsb0`); always false for a sync step. `shape`: (m, n, k) from the
    opcode (`HGMMA.64x16x16.F32.BF16`), None if it names none. `operand_bytes`: the shared-memory bytes the step reads
    through its descriptors (A is m x k and B is k x n elements, each only where the listing gives a descriptor, at the
    input type's size), None for a step this does not read; 0 for a sync step, which reads registers, not descriptors. `null`: the empty group the toolkit issues to hold a wait
    (`RZ, gdesc[URZ]`): it moves no operand and computes nothing."""
    instruction: Instruction
    commit: bool
    shape: tuple[int, int, int] | None = None
    operand_bytes: int | None = None
    null: bool = False
    kind: str = "async"


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

    @property
    def warp_bytes(self) -> int:
        """The bytes a warp moves, all 32 lanes (an `LDS.U16` is 64); a matrix access (`LDSM.16.M88.4`) is 4 matrices of
        8 x 8 sixteen-bit elements, 128 bytes each. 0 for an asynchronous copy, whose size is its descriptor's."""
        if self.kind == "async":
            return 0
        m = _MATRIX.match(self.instruction.opcode)
        return 128 * int(m.group(1)) if m else 32 * self.width // 8


@dataclass(frozen=True)
class Call:
    """A `CALL`: its target address (None if it names none), whether the target is a CONVERGENCE SUBROUTINE (a short
    body, `_SUBROUTINE` instructions at most before its return, that syncs the warp and runs one collective) and
    whether a converged warp executes the call (`hot`: its block is not cold)."""
    instruction: Instruction
    target: int | None
    subroutine: bool
    hot: bool


@dataclass(frozen=True)
class StepRegion:
    """What one step's cost is read over: the instructions of an innermost loop that holds steps (`kind` loop; the
    blocks a converged warp reaches, the dead ones left out), or the instructions of one unrolled step outside any such
    loop, from the one after the previous step in its basic block, or the block's start, through the step (`unrolled`).
    `steps` counts the region's steps of either kind (the null group excluded); `address` is its first instruction's."""
    kind: str
    address: int
    steps: int
    instructions: tuple[Instruction, ...]
    accesses: tuple["MemoryAccess", ...]
    tensor: tuple[TensorStep, ...]


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
    fences: tuple[Instruction, ...]
    calls: tuple[Call, ...]
    regions: tuple[StepRegion, ...]
    registers: int
    stack_bytes: int
    shared_bytes: int
    #: instruction address -> (source file, line), from the disassembly with source-line marks (`--lines`); empty where
    #: none was given. Not part of what makes two kernels equal.
    lines: Mapping[int, tuple[str, int]] = field(default_factory=dict, compare=False)

    def live_in(self) -> dict[str, frozenset[str]]:
        """Block id -> the registers live at the block's entry (`flow.live_in`)."""
        from . import flow
        return flow.live_in(self)

    @property
    def async_steps(self) -> tuple[TensorStep, ...]:
        """The steps committed in groups and waited on (kind `async`); a rule about commit groups reads these."""
        return tuple(s for s in self.steps if s.kind == "async")

    def block(self, id: str) -> Block:
        return next(b for b in self.blocks if b.id == id)

    def loop_of(self, instruction: Instruction) -> Loop | None:
        """The innermost natural loop around an instruction."""
        header = self.block(instruction.block).loop
        return next((lp for lp in self.loops if lp.header == header), None) if header else None

    def innermost(self, holding=None) -> tuple[Loop, ...]:
        """The loops with no other loop inside them; with `holding`, a predicate over an instruction, the loops that hold
        an instruction it accepts and no inner loop that does (a non-step loop inside a loop of steps is part of its body)."""
        if holding is None:
            return tuple(lp for lp in self.loops if not any(o is not lp and o.blocks < lp.blocks for o in self.loops))
        holds = {lp.header for lp in self.loops
                 if any(holding(i) for b in lp.blocks for i in self.block(b).instructions)}
        return tuple(lp for lp in self.loops if lp.header in holds
                     and not any(o is not lp and o.blocks < lp.blocks and o.header in holds for o in self.loops))


#: base opcode -> (space, kind)
_ACCESS = {
    "LDS": ("shared", "load"), "LDSM": ("shared", "load"), "STS": ("shared", "store"), "ATOMS": ("shared", "atomic"),
    "LDG": ("global", "load"), "STG": ("global", "store"), "ATOMG": ("global", "atomic"), "RED": ("global", "atomic"),
    "LDL": ("local", "load"), "STL": ("local", "store"), "LD": ("generic", "load"), "ST": ("generic", "store"),
    "ATOM": ("generic", "atomic"), "LDGSTS": ("global", "async"), "UBLKCP": ("global", "async"),
    "UBLKRED": ("global", "async"), "UTMALDG": ("global", "async"), "UTMASTG": ("global", "async"),
    "UTMAREDG": ("global", "async"), "UTMACP": ("global", "async"),
}
#: base opcode -> the unit class it is counted in; matched by opcode prefix, the first match wins (so `F2FP`'s pack is
#: matched before the conversion unit, `REDUX`, the warp reduction, is a shuffle before `RED` would make it global, and
#: `LDGDEPBAR`, the `cp.async` commit, is control before `LDG` would)
INSTRUCTION_CLASSES = {
    "tensor": ("HGMMA", "IGMMA", "QGMMA", "BGMMA", "HMMA", "IMMA", "DMMA"),
    "shared load": ("LDSM", "LDS"),
    "shared store": ("STSM", "STS"),
    "shared atomic": ("ATOMS",),
    "control": ("BRA", "BSSY", "BSYNC", "BAR", "WARPSYNC", "EXIT", "CALL", "RET", "NOP", "DEPBAR", "YIELD", "LDGDEPBAR",
                "WARPGROUP", "SETMAXREG"),
    "shuffle": ("SHFL", "REDUX", "MATCH"),
    "global": ("LDGSTS", "LDG", "STG", "RED", "ATOMG", "ATOM", "UBLKCP", "UBLKRED", "UTMA"),
    "pack": ("PRMT", "F2FP"),
    "conversion unit": ("F2I", "I2F", "F2F", "MUFU"),
    "float": ("FFMA", "FMUL", "FADD", "FMNMX", "HFMA2", "HMUL2", "HADD2", "HMNMX2"),
    "select/compare": ("SEL", "FSEL", "ISETP", "FSETP", "HSETP2", "PLOP3", "VOTE"),
    "integer/address": ("IMAD", "IADD3", "LEA", "LOP3", "SHF", "IMNMX", "IABS", "MOV", "S2R", "CS2R", "POPC", "FLO", "LDC",
                        "S2UR", "R2UR"),
}
#: opcodes whose first operand is a source, not a destination (a store, a branch, a tensor step: its accumulator is read)
_NO_DESTINATION = {"ST", "STG", "STS", "STL", "STSM", "RED", "BRA", "BSSY", "BSYNC", "BREAK", "CALL",
                   "EXIT", "RET", "NOP", "BAR", "WARPSYNC", "MEMBAR", "FENCE", "DEPBAR", "LDGDEPBAR", "HGMMA", "IGMMA",
                   "QGMMA", "BGMMA", "UBLKCP", "UBLKRED", "UTMALDG", "UTMASTG", "UTMAREDG", "UTMACP", "LDGSTS", "JMP",
                   "YIELD", "WARPGROUP", "SETMAXREG"}
#: a comparison or a logic op that writes predicates alone, the register-count setters
_PREDICATE_ONLY = re.compile(r"U?(?:[IFDH]SETP2?|PSETP|PLOP3|FCHK|SETMAXREG)")
_PREDICATE = re.compile(r"U?P(?:\d+|T)")
#: the double-precision operations: every register operand is a pair
_DOUBLE = ("DADD", "DMUL", "DFMA", "DMNMX")
#: the stores and atomics whose data operand is as wide as the access (`STG.E.64 [R2.64], R4` reads R4 and R5)
_WIDE_DATA = ("ST", "STG", "STS", "STL", "RED", "ATOM", "ATOMG", "ATOMS")
_REGISTER_ONLY = re.compile(r"(U?R)(\d+)(?:\.\w+)?")
_REGISTER = re.compile(r"\b(U?R)(\d+)(\.64)?")
_ADDRESS = re.compile(r"\[[^\]]*\]")
_MATRIX = re.compile(r"^(?:LDSM|STSM)\.\d+\.MT?88\.(\d)")
#: a convergence subroutine's length at most: its operand moves, `WARPSYNC`, the collective (the return not counted)
_SUBROUTINE = 8
_COLLECTIVE = ("SHFL", "VOTE", "MATCH", "REDUX", "BAR")
_STEP_SHAPE = re.compile(r"^(\d+)x(\d+)x(\d+)$")
_ELEMENT_BYTES = {"F16": 2, "BF16": 2, "TF32": 4, "E4M3": 1, "E5M2": 1, "S8": 1, "U8": 1}
_WIDTH = {"U8": 8, "S8": 8, "U16": 16, "S16": 16, "64": 64, "128": 128}
_ASYNC = ("HGMMA", "IGMMA", "QGMMA", "BGMMA")
_SYNC = ("HMMA", "IMMA", "DMMA")
#: a sync step's opcode names its shape as digits (`HMMA.16816.F32.BF16`: m16 n8 k16; `DMMA.884`; `IMMA.8816.S8.S8`)
_SYNC_SHAPE = re.compile(r"^(16|8)(8)(\d+)$")
_BARRIER = ("BAR", "SYNCS", "MEMBAR")
_CONVERGENCE = ("BSSY", "BSYNC", "BREAK", "WARPSYNC")
_WAIT = re.compile(r"gsb0\s*,\s*(0x[0-9a-f]+)")
_TRYWAIT = "SYNCS.PHASECHK.TRANS64.TRYWAIT"
#: the opcodes a barrier's retry loop is made of: the integer move that forms the barrier's address, the try-wait, the toolkit's
#: sleep, the retry's probe and the branch back (toolkit 13.5 puts the try-wait in a block of its own, behind the move)
_SPIN_OPS = frozenset({"IMAD.U32", "IMAD.MOV.U32", "MOV", _TRYWAIT, "NANOSLEEP.SYNCS", "SYNCS.PHASECHK.TRANS64", "BRA"})


def _access(ins: Instruction) -> MemoryAccess | None:
    if (found := _ACCESS.get(ins.base)) is None:
        return None
    width = next((_WIDTH[m] for m in ins.opcode.split(".")[1:] if m in _WIDTH), 32)
    return MemoryAccess(ins, found[0], found[1], width)


def build_kernel(name: str, sass: list[readers.SassLine], cfg: readers.Cfg, usage: readers.Usage,
                 demangled: str | None = None, lines: Mapping[int, tuple[str, int]] | None = None) -> Kernel:
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
    tail = [line for line in sass if line.address > last]
    sass = [line for line in sass if line.address <= last]
    # What follows the graph's last instruction: the bodies the toolkit's graph does not hold (a convergence subroutine a
    # `CALL` reaches) and then the padding (a self-branch and `NOP`s). The bodies are kept, in a block of their own; the
    # padding is dropped.
    stop = next((k for k, line in enumerate(tail) if line.opcode == "BRA" and line.operands.strip() == hex(line.address)), len(tail))
    outlined = tail[:stop]
    while outlined and outlined[-1].opcode == "NOP":
        outlined.pop()
    for line in outlined:
        owner[line.address] = OUTLINED
    sass += outlined
    addresses = {line.address for line in sass}
    if set(owner) != addresses:
        odd = sorted(set(owner) ^ addresses)[:4]
        raise ModelError(f"{name}: the graph and the listing disagree at {[hex(a) for a in odd]}")
    instructions = tuple(Instruction(s.address, s.predicate, s.opcode, s.operands, owner[s.address]) for s in sass)
    by_block: dict[str, list[Instruction]] = {b.id: [] for b in cfg.blocks}
    if outlined:
        by_block[OUTLINED] = []
    for ins in instructions:
        by_block[ins.block].append(ins)
    successors: dict[str, list[str]] = {b.id: [] for b in cfg.blocks}
    if outlined:
        successors[OUTLINED] = []
    for a, b in cfg.edges:
        if a not in successors or b not in successors:
            raise ModelError(f"{name}: edge {a} -> {b} names a block the graph does not hold")
        successors[a].append(b)

    found = loops_mod.natural_loops(cfg.entry, list(cfg.edges))
    first = {bid: (body[0].address if body else -1) for bid, body in by_block.items()}
    spin = {lp.header: any(i.opcode == _TRYWAIT for b in lp.blocks for i in by_block[b])
            and all(i.opcode in _SPIN_OPS for b in lp.blocks for i in by_block[b]) for lp in found}
    model_loops = tuple(sorted((Loop(lp.header, first[lp.header], lp.blocks, lp.tails, spin[lp.header]) for lp in found),
                               key=lambda lp: lp.address))
    innermost: dict[str, str] = {}
    for lp in sorted(model_loops, key=lambda lp: -len(lp.blocks)):   # larger first, so a smaller loop overwrites
        for bid in lp.blocks:
            innermost[bid] = lp.header
    cold = _cold_blocks(cfg.entry, by_block, successors)
    blocks = tuple(Block(b.id, by_block[b.id][0].address if by_block[b.id] else -1,
                         by_block[b.id][-1].address if by_block[b.id] else -1, tuple(by_block[b.id]),
                         tuple(successors[b.id]), innermost.get(b.id), b.id in cold) for b in cfg.blocks)
    if outlined:
        blocks += (Block(OUTLINED, outlined[0].address, outlined[-1].address, tuple(by_block[OUTLINED]), (), None, True),)

    steps = tuple(_step(i) for i in instructions if i.base in _ASYNC + _SYNC)
    waits = tuple(TensorWait(i, int(m.group(1), 16)) for i in instructions
                  if i.opcode.startswith("WARPGROUP.DEPBAR") and (m := _WAIT.search(i.operands)))
    accesses = tuple(a for i in instructions if (a := _access(i)))
    return Kernel(
        name=name, demangled=demangled or name, instructions=instructions, blocks=blocks, loops=model_loops,
        steps=steps, waits=waits, arrives=tuple(i for i in instructions if i.opcode.startswith("WARPGROUP.ARRIVE")),
        accesses=accesses,
        barriers=tuple(i for i in instructions if i.base in _BARRIER),
        convergence=tuple(i for i in instructions
                          if i.base in _CONVERGENCE or (i.base == "BRA" and "DIV" in i.opcode.split(".")[1:])),
        fences=tuple(i for i in instructions if i.base == "FENCE"),
        calls=_calls(instructions, {b.id: b for b in blocks}),
        regions=_regions(blocks, model_loops, steps, accesses),
        registers=usage.registers, stack_bytes=usage.stack, shared_bytes=usage.shared,
        lines=dict(lines or {}))


def _step(ins: Instruction) -> TensorStep:
    """A tensor step with its kind, shape, the bytes it reads through descriptors, and whether it is the empty group."""
    mods = ins.opcode.split(".")[1:]
    if ins.base in _SYNC:
        shape = next(((int(m.group(1)), int(m.group(2)), int(m.group(3))) for x in mods if (m := _SYNC_SHAPE.match(x))), None)
        return TensorStep(ins, False, shape, 0, False, "sync")
    shape = next(((int(m.group(1)), int(m.group(2)), int(m.group(3))) for x in mods if (m := _STEP_SHAPE.match(x))), None)
    operands = [o.strip() for o in ins.operands.split(",")]
    null = len(operands) > 1 and operands[0] == "RZ" and operands[1] == "gdesc[URZ]"
    size = next((_ELEMENT_BYTES[x] for x in reversed(mods) if x in _ELEMENT_BYTES), None)
    own = None
    if shape and size and len(operands) >= 3:
        (m_, n, k) = shape
        if operands[1].startswith("gdesc["):
            own = (m_ * k + k * n) * size                         # A and B both in shared memory
        elif operands[2].startswith("gdesc[") and re.fullmatch(r"R(?:\d+|Z)", operands[1]):
            own = k * n * size                                    # A in registers, B alone shared
    return TensorStep(ins, "gsb0" in ins.operands, shape, own, null, "async")


def _cold_blocks(entry: str, by_block: dict[str, list[Instruction]], successors: dict[str, list[str]]) -> set[str]:
    """The blocks no path from the entry reaches without a fallback a converged warp does not take: the target of a
    `BRA.DIV`, the fall-through of a `BRA.CONV`, the target of a branch under the always-false predicate `@!PT`. A block
    no edge reaches is cold too."""
    starts = {body[0].address: bid for bid, body in by_block.items() if body}
    live: dict[str, list[str]] = {b: [] for b in successors}
    for bid, succ in successors.items():
        last = by_block[bid][-1] if by_block[bid] else None
        for target in succ:
            taken = last is not None and last.target is not None and starts.get(last.target) == target
            fall = last is not None and starts.get(last.address + 0x10) == target and not (taken and last.target == last.address + 0x10)
            if last is not None and last.base == "BRA":
                mods = last.opcode.split(".")[1:]
                if (last.dead and taken) or ("DIV" in mods and taken) or ("CONV" in mods and fall and not taken):
                    continue
            live[bid].append(target)
    seen, work = {entry}, [entry]
    while work:
        for nxt in live.get(work.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                work.append(nxt)
    return set(successors) - seen


def _calls(instructions: tuple[Instruction, ...], blocks: dict[str, Block]) -> tuple[Call, ...]:
    """Each `CALL`, and whether its target is a convergence subroutine."""
    at = {ins.address: k for k, ins in enumerate(instructions)}

    def subroutine(target: int | None) -> bool:
        if target not in at:
            return False
        body = []
        for ins in instructions[at[target]:at[target] + _SUBROUTINE + 1]:
            if ins.base == "RET":
                return any(i.base == "WARPSYNC" for i in body) and any(i.base in _COLLECTIVE for i in body)
            body.append(ins)
        return False

    return tuple(Call(i, i.target, subroutine(i.target), not blocks[i.block].cold and not i.dead)
                 for i in instructions if i.base == "CALL")


def _regions(blocks: tuple[Block, ...], loops: tuple[Loop, ...], steps: tuple[TensorStep, ...],
             accesses: tuple[MemoryAccess, ...]) -> tuple[StepRegion, ...]:
    """The step regions (`StepRegion`): each innermost natural loop holding a step, then each step outside them."""
    by_ins = {s.instruction.address: s for s in steps}
    access = {a.instruction.address: a for a in accesses}
    by_id = {b.id: b for b in blocks}

    def holds(lp: Loop) -> bool:
        return any(i.address in by_ins and not by_ins[i.address].null and not i.dead
                   for b in lp.blocks for i in by_id[b].instructions)

    holding = [lp for lp in loops if holds(lp)]
    inner = [lp for lp in holding if not any(o is not lp and o.blocks < lp.blocks for o in holding)]

    def region(kind: str, body: list[Instruction]) -> StepRegion:
        tensor = tuple(by_ins[i.address] for i in body if i.address in by_ins)
        return StepRegion(kind, body[0].address, sum(not t.null for t in tensor), tuple(body),
                          tuple(access[i.address] for i in body if i.address in access), tensor)

    found, inside = [], set()
    for lp in inner:
        body = sorted((i for b in lp.blocks if not by_id[b].cold for i in by_id[b].instructions if not i.dead),
                      key=lambda i: i.address)
        inside |= {b for b in lp.blocks}
        found.append(region("loop", body))
    for block in blocks:
        if block.id in inside or block.cold:
            continue
        start = 0
        for k, ins in enumerate(block.instructions):
            if ins.address in by_ins and not by_ins[ins.address].null and not ins.dead:
                found.append(region("unrolled", [i for i in block.instructions[start:k + 1] if not i.dead]))
                start = k + 1
    return tuple(sorted(found, key=lambda r: r.address))


def kernels_from_text(sass: str, cfg: str, res: str, names: str | None = None, lines_text: str | None = None) -> dict[str, Kernel]:
    """Every kernel of the tools' texts (`names`: `c++filt`'s answer, a line for each function of the listing in its
    order; none: the mangled name stands for the demangled one; `lines_text`: the disassembly with source-line marks, none: the
    kernels know no source lines). A function the listing holds and the graph or the resource report lacks is an error."""
    lines, graphs, usage = readers.sass_functions(sass), readers.cfg_functions(cfg), readers.res_usage(res)
    pretty = readers.demangled(names, list(lines)) if names is not None else {}
    marks = readers.source_lines(lines_text) if lines_text is not None else {}
    kernels = {}
    for name, body in lines.items():
        if name not in graphs or name not in usage:
            raise ModelError(f"{name}: the listing has it, the graph or the resource report does not")
        kernels[name] = build_kernel(name, body, graphs[name], usage[name], pretty.get(name), marks.get(name))
    return kernels
