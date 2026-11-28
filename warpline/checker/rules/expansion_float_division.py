"""expansion_float_division: an IEEE-exact float division (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_float_division",
    statement="an IEEE-exact float or double division: the reciprocal's refinement, the check (FCHK) and the slow-path call behind it",
    check=_expansions.check("float_division"),
    refusal="an IEEE-exact division is a refinement, a check and a slow path the source writes as one operator",
    fix=_expansions.KINDS["float_division"][1],
    fixtures=(Fixture("float_division", 1), Fixture("double_division", 1), Fixture("float_clean", 0)),
)
