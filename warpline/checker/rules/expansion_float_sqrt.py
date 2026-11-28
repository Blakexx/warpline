"""expansion_float_sqrt: an IEEE-exact square root (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_float_sqrt",
    statement="an IEEE-exact square root, float or double: MUFU.RSQ, its refinement and the slow-path call behind it",
    check=_expansions.check("float_sqrt"),
    refusal="an IEEE-exact square root is a refinement and a slow path the source writes as one call",
    fix=_expansions.KINDS["float_sqrt"][1],
    fixtures=(Fixture("float_sqrt", 1), Fixture("double_division", 1), Fixture("float_clean", 0)),
)
