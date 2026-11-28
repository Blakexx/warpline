"""THE COMPILER'S EXPANSIONS, shared by the `expansion_*` rules: each place a one-line operation of the source became a hidden
sequence, found by the signature the toolkit's output gives it (13.5: its `sm_90a` output of the fixtures, the 64-bit
multiply also its `sm_80` output; docs/checker-rules.md#compiler-expansions), anywhere in a kernel (dead `@!PT` code aside),
one EXPANSION an operation:

- A SLOW-PATH CALL, a `CALL` that is no convergence subroutine (`model.Call.subroutine`): the expansion is the call, the
  instructions of the convergence region around it (its `BSSY` to the `BSYNC` it names) and the body it calls (to the body's
  `RET`). Its kind is the first that its instructions show: `FCHK` (the division check) a float division, `I2F.U64.RP` or
  `I2F.S64.RP` (a 64-bit reciprocal estimate) a 64-bit integer division, `MUFU.RSQ` (or `RSQ64H`) a square root, `MUFU.RCP64H`
  a double's division, `MUFU.RCP` a reciprocal, else a call no other kind names.
- Outside those, inline: `I2F.RP` (32-bit, unsigned or signed: the reciprocal estimate of a division by a runtime value) an
  integer division or modulo; a 64-bit multiply, 64 by 64 or 32 by 64 (an `IMAD.WIDE` of two words, registers or a
  constant's, and an `IMAD` of one and the other's next word: a high word's cross product); the precise transcendentals by
  their range reductions: a multiply by 2/pi (sine, cosine, tangent), a round-to-integer `FFMA.RM` by 12582913 (the
  exponentials and powers), an add of -0x3f2aaaab to the bits (the logarithms).

The clean forms have none of these: a shift or a mask (a power-of-two divisor), `IMAD.HI` and a shift (a precomputed
inverse), a 32-bit by 32-bit `IMAD.WIDE` (a 32-bit index's address), and the fast intrinsics (`MUFU.RCP`, `MUFU.RSQ`,
`MUFU.SQRT`, `MUFU.EX2`, `MUFU.LG2`, `MUFU.SIN` with no slow path behind them)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ...manifest import Finding
from ..model import Instruction, Kernel

#: the kinds, each the rule that counts it, with the fix its finding names
KINDS = {
    "integer_division": ("an integer division or modulo by a runtime value",
                         "a power-of-two divisor (a shift and a mask), or a precomputed inverse: multiply-high and a shift "
                         "(`__umulhi(x, m) >> s`, the inverse made once outside the loop)"),
    "wide_integer": ("a 64-bit integer division or multiply",
                     "32-bit indices and sizes (the address a 32-by-32 wide multiply-add), or a precomputed inverse"),
    "float_division": ("an IEEE-exact float division",
                       "a precomputed reciprocal and a multiply, or `__fdividef` where its accuracy (2 ulp) allows"),
    "float_sqrt": ("an IEEE-exact square root", "`sqrt.approx.f32` (`__fsqrt` under fast math) or `rsqrtf` where the "
                                                "accuracy allows"),
    "float_reciprocal": ("an IEEE-exact reciprocal", "`rcp.approx.f32` where the accuracy allows, or the reciprocal made "
                                                     "once outside the loop"),
    "transcendental": ("a precise transcendental", "the fast intrinsic where its accuracy allows (`__sinf`, `__cosf`, "
                                                   "`__expf`, `exp2f` on `MUFU.EX2`, `__logf`)"),
    "slow_call": ("a call into an outlined body", "inline it (`__forceinline__`), or remove the operation that needs it"),
}
#: the precise transcendentals' range reductions, by the constant the toolkit writes
_TRANSCENDENTAL = (
    (re.compile(r"^FMUL$"), re.compile(r"\b0\.6366197466850280761"), "sine, cosine or tangent (a reduction by 2/pi)"),
    (re.compile(r"^FFMA\.RM$"), re.compile(r"\b12582913\b"), "an exponential or a power (a round to integer)"),
    (re.compile(r"^IADD3$"), re.compile(r"-0x3f2aaaab\b"), "a logarithm (the mantissa's split at 2/3)"),
)
_INT_RP = re.compile(r"^I2F(\.U32|\.S32)?\.RP$")
_WIDE_RP = re.compile(r"^I2F\.[US]64\.RP$")


@dataclass(frozen=True)
class Expansion:
    kind: str
    address: int
    detail: str


def _site(kernel: Kernel, address: int) -> str:
    """A finding's site: its source line where the kernel has its lines (`file:line`), else its address."""
    found = kernel.lines.get(address)
    return f"{found[0]}:{found[1]}" if found else f"{address:#06x}"


def _body(kernel: Kernel, target: int | None) -> list[Instruction]:
    """The instructions of a called body: from its target to its first `RET`."""
    out = []
    for ins in kernel.instructions:
        if target is not None and ins.address >= target:
            out.append(ins)
            if ins.base == "RET":
                break
    return out


def _region(kernel: Kernel, at: int) -> list[Instruction]:
    """The convergence region around an address: from the nearest `BSSY` before it whose `BSYNC` is after it to that `BSYNC`
    (none: the instruction alone)."""
    around = [ins for ins in kernel.instructions if ins.base == "BSSY" and ins.address < at and (ins.target or 0) > at]
    if not around:
        return [ins for ins in kernel.instructions if ins.address == at]
    start = max(around, key=lambda i: i.address)
    return [ins for ins in kernel.instructions if start.address <= ins.address < (start.target or 0)]


def _call_kind(region: list[Instruction], body: list[Instruction]) -> tuple[str, str]:
    seen = [*region, *body]
    ops = {ins.opcode for ins in seen if not ins.dead}
    if any(o.startswith("FCHK") for o in ops):
        return "float_division", "FCHK and its slow path"
    if any(_WIDE_RP.match(o) for o in ops):
        return "wide_integer", "a 64-bit division's reciprocal estimate (I2F.64.RP) and its slow path"
    if any(o.startswith("MUFU.RSQ") for o in ops):
        return "float_sqrt", "MUFU.RSQ, its refinement and its slow path"
    if any(o.startswith("MUFU.RCP64H") for o in ops):
        return "float_division", "a double's division: MUFU.RCP64H, its refinement and its slow path"
    if any(o.startswith("MUFU.RCP") for o in ops):
        return "float_reciprocal", "MUFU.RCP, its refinement and its slow path"
    return "slow_call", "a call no other expansion names"


def _register(op: str) -> tuple[str, int, int] | None:
    """A multiply's operand that holds a word of a value: a register (`R5`, `UR6`: its next is the next register) or a
    constant's word (`c[0x0][0x168]`: its next is the next word, 4 bytes on), as (space, index, the step to the next)."""
    op = op.strip()
    if m := re.fullmatch(r"(U?R)(\d+)(?:\.\w+)?", op):
        return m.group(1), int(m.group(2)), 1
    if m := re.fullmatch(r"(c\[0x[0-9a-f]+\])\[(0x[0-9a-f]+)\]", op):
        return m.group(1), int(m.group(2), 16), 4
    return None


def _wide_multiplies(instructions: list[Instruction]) -> list[Instruction]:
    """Each `IMAD.WIDE` of two registers that an `IMAD` of one of them and the other's next register follows: a 64-bit
    multiply's low product and a cross product of its high words."""
    out = []
    plain = [i for i in instructions if i.base == "IMAD" and ".WIDE" not in i.opcode and not i.dead]
    for ins in instructions:
        if ins.base != "IMAD" or ".WIDE" not in ins.opcode or ins.dead:
            continue
        ops = [o.strip() for o in ins.operands.split(",")]
        if len(ops) < 3 or not (a := _register(ops[1])) or not (b := _register(ops[2])):
            continue
        crossed = {(a, (b[0], b[1] + b[2], b[2])), ((a[0], a[1] + a[2], a[2]), b)}
        for other in plain:
            o = [x.strip() for x in other.operands.split(",")]
            if len(o) >= 3 and (x := _register(o[1])) and (y := _register(o[2])):
                if (x, y) in crossed or (y, x) in crossed:
                    out.append(ins)
                    break
    return out


def expansions(kernel: Kernel) -> list[Expansion]:
    """Every expansion in the kernel, in address order (the module's text)."""
    found: list[Expansion] = []
    claimed: set[int] = set()
    for call in kernel.calls:
        if call.subroutine or call.instruction.dead:
            continue
        region, body = _region(kernel, call.instruction.address), _body(kernel, call.target)
        claimed |= {i.address for i in (*region, *body)}
        kind, what = _call_kind(region, body)
        found.append(Expansion(kind, call.instruction.address, f"{what} (the call at {call.instruction.address:#06x})"))
    free = [i for i in kernel.instructions if i.address not in claimed and not i.dead]
    for ins in free:
        if _INT_RP.match(ins.opcode):
            found.append(Expansion("integer_division", ins.address, f"{ins.opcode} (a division's reciprocal estimate)"))
        elif _WIDE_RP.match(ins.opcode):
            found.append(Expansion("wide_integer", ins.address, f"{ins.opcode} (a 64-bit division's reciprocal estimate)"))
        elif ins.base == "FCHK":
            found.append(Expansion("float_division", ins.address, "FCHK (a division's check)"))
        for opcode, constant, what in _TRANSCENDENTAL:
            if opcode.match(ins.opcode) and constant.search(ins.operands):
                found.append(Expansion("transcendental", ins.address, what))
    found += [Expansion("wide_integer", i.address, f"{i.opcode} {i.operands.strip()} and the high words' cross products "
                                                   "(a 64-bit multiply)") for i in _wide_multiplies(free)]
    return sorted(found, key=lambda e: e.address)


def check(kind: str):
    """A rule's check: the kernel's expansions of `kind`, a finding each, at its site (`file:line`, else its address)."""
    name = KINDS[kind][0]

    def run(kernel: Kernel) -> list[Finding]:
        return [Finding(_site(kernel, e.address), 1, f"{name} at {e.address:#06x}: {e.detail}")
                for e in expansions(kernel) if e.kind == kind]
    return run
