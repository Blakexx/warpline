"""expansion_integer_division: an integer division or modulo by a runtime value (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_integer_division",
    statement="an integer division or modulo by a runtime value, expanded into a reciprocal-estimate sequence (I2F.RP, MUFU.RCP, IMAD.HI and corrections)",
    check=_expansions.check("integer_division"),
    refusal="an integer division or modulo by a runtime value is about twenty instructions the source does not show",
    fix=_expansions.KINDS["integer_division"][1],
    fixtures=(Fixture("int_division", 2), Fixture("int_clean", 0)),
)
