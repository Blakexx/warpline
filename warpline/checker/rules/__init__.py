"""The registry: every module of this package that defines `RULE` is a rule. A rule is added by writing that module
(and its fixtures) and nothing else."""
from __future__ import annotations

import importlib
import pkgutil

from ..rule import Rule


def registry() -> dict[str, Rule]:
    """Rule name -> rule, in module-name order."""
    found = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        rule = getattr(importlib.import_module(f"{__name__}.{info.name}"), "RULE", None)
        if rule is not None:
            if rule.name in found:
                raise ValueError(f"two rules named {rule.name}")
            found[rule.name] = rule
    return found
