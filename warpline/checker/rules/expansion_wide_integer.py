"""expansion_wide_integer: a 64-bit integer division or multiply (the compiler's expansions: `_expansions.py`, docs/checker-rules.md#compiler-expansions). A finding an
expansion, at its source line where the listing's lines are given; refused unless a declaration opts it in at its site
(`"at"`), with the reason."""
from __future__ import annotations

from ..rule import Fixture, Rule
from . import _expansions

RULE = Rule(
    name="expansion_wide_integer",
    statement="a 64-bit integer division (a slow-path call and its 32-bit fast path) or a 64-bit by 64-bit multiply (IMAD.WIDE and the high words' cross products)",
    check=_expansions.check("wide_integer"),
    refusal="a 64-bit integer division is a call into a body of about a hundred instructions, a 64-bit multiply three where one would do",
    fix=_expansions.KINDS["wide_integer"][1],
    fixtures=(Fixture("wide_integer", 2), Fixture("wide_clean", 0)),
)
