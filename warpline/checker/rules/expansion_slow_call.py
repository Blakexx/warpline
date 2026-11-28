"""expansion_slow_call: a call into an outlined body (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_slow_call",
    statement="a call into an outlined body that no other expansion names (no convergence subroutine)",
    check=_expansions.check("slow_call"),
    refusal="the kernel calls an outlined body: a call, its argument moves and a return the source does not show",
    fix=_expansions.KINDS["slow_call"][1],
    fixtures=(Fixture("slow_call", 1), Fixture("float_division", 0)),
)
