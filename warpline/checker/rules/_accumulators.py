"""What the rules over async steps' accumulators share: the registers a step accumulates into, and the registers an
instruction touches. Not a rule (it defines no `RULE`)."""
from __future__ import annotations

import re

from ..model import Instruction, TensorStep

_REGISTER = re.compile(r"R(\d+)(?:\.\w+)?$")
#: the accumulator's type, from the opcode (`HGMMA.64x16x16.F32.BF16`): 32-bit types take n/2 registers a thread, 16-bit n/4
_ACCUMULATOR_REGISTERS = {"F32": 2, "S32": 2, "F16": 4}
_DOUBLE = ("DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "DSET")


def destination(step: TensorStep) -> frozenset[str]:
    """The registers an async step accumulates into: from its first operand, `m64nNk16` with a 32-bit accumulator N/2
    consecutive registers and with a 16-bit one N/4. Empty for the null group, a step with no shape or no accumulator
    type in its opcode, or a first operand that is no register: a step this cannot read."""
    ins = step.instruction
    first = ins.operands.split(",")[0].strip()
    m = _REGISTER.match(first)
    if step.null or step.shape is None or m is None:
        return frozenset()
    divisor = next((_ACCUMULATOR_REGISTERS[x] for x in ins.opcode.split(".")[1:] if x in _ACCUMULATOR_REGISTERS), None)
    if divisor is None:
        return frozenset()
    return frozenset(f"R{int(m.group(1)) + j}" for j in range(step.shape[1] // divisor))


def carried(step: TensorStep) -> frozenset[str]:
    """The registers an async step reads as its accumulator input: `destination` when the operand after the descriptor
    (`R24, gdesc[UR8], R24`, or after the second operand where A is in registers) is a register, empty when it is `RZ` (a step
    that starts a fresh accumulation: its first step of a tile, `R24, gdesc[UR8], RZ, !UPT`), and empty where the model cannot
    tell (no descriptor operand)."""
    operands = [o.strip() for o in step.instruction.operands.split(",")]
    where = [k for k, o in enumerate(operands) if o.startswith("gdesc[")]
    if not where or where[-1] + 1 >= len(operands) or _REGISTER.match(operands[where[-1] + 1]) is None:
        return frozenset()
    return destination(step)


def width(ins: Instruction) -> int:
    """How many consecutive registers one operand of the instruction may name: 4 for a `.128` access, 2 for a `.64` or
    `.WIDE` one or a double-precision operation, else 1. The model's `reads` widens only a written pair (`R2.64`), so a rule
    that must not miss a read widens every operand by this."""
    mods = ins.opcode.split(".")[1:]
    if "128" in mods:
        return 4
    if "64" in mods or "WIDE" in mods or "F64" in mods or ins.base in _DOUBLE:
        return 2
    return 1


def touched(ins: Instruction) -> frozenset[str]:
    """The registers the instruction reads or writes, every operand widened by `width`: a superset of the truth where the
    opcode's operand widths are not all known (a rule that wants to be sure a register is untouched asks this)."""
    span = width(ins)
    found = set()
    for reg in ins.reads | ins.writes:
        kind, n = ("UR", int(reg[2:])) if reg.startswith("UR") else ("R", int(reg[1:]))
        found.update(f"{kind}{n + j}" for j in range(span))
    return frozenset(found)
