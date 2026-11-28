"""expansion_transcendental: a precise transcendental (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_transcendental",
    statement="a precise sine, cosine, tangent, exponential, power or logarithm: its range reduction and polynomial, where the fast intrinsic is one MUFU",
    check=_expansions.check("transcendental"),
    refusal="a precise transcendental is a range reduction and a polynomial (a sine's slow reduction tens of instructions more) the source writes as one call",
    fix=_expansions.KINDS["transcendental"][1],
    fixtures=(Fixture("transcendental", 3), Fixture("float_clean", 0)),
)
