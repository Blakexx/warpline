"""LAYER 4, DECLARATIONS AND THE RUNNER'S CORE. A project declares, per kernel (a regex over the demangled name), how
many of a rule's findings it allows and WHY. Undeclared means zero. The module holds no threshold and no reason: they
are the project's data (a JSON file). `check` returns the refusals.

    {"declarations": [{"kernel": "^example::main\\(", "rule": "stack_memory", "allow": 128,
                       "reason": "the staging array is indexed by a runtime value; measured and accepted"}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .model import Kernel
from .rule import Finding, Rule


class DeclarationError(ValueError):
    """A declaration file that is not valid: bad JSON, an unknown rule, a missing or empty reason."""


@dataclass(frozen=True)
class Declaration:
    kernel: re.Pattern
    rule: str
    allow: int
    reason: str


@dataclass(frozen=True)
class Refusal:
    """A kernel a rule refuses: its findings' count above what is declared."""
    kernel: str
    rule: str
    count: int
    allowed: int
    reason: str | None
    findings: tuple[Finding, ...]
    message: str
    fix: str

    def lines(self) -> list[str]:
        head = f"{self.kernel}: {self.rule}: {self.count} found, {self.allowed} declared"
        if self.reason:
            head += f" ({self.reason})"
        return [head, f"  {self.message}", *[f"  at {f.where} x{f.count}: {f.what}" for f in self.findings],
                f"  fix: {self.fix}"]


def load_declarations(text: str, rules: dict[str, Rule]) -> list[Declaration]:
    """The declarations of a JSON text, in file order. A reason is required; a rule must exist."""
    try:
        doc = json.loads(text)
        entries = doc["declarations"]
    except (ValueError, KeyError, TypeError) as e:
        raise DeclarationError(f'the file must be {{"declarations": [...]}}: {e}') from e
    found = []
    for i, entry in enumerate(entries):
        try:
            kernel, rule, allow, reason = entry["kernel"], entry["rule"], entry["allow"], entry["reason"]
        except (KeyError, TypeError) as e:
            raise DeclarationError(f"declaration {i}: needs kernel, rule, allow and reason ({e})") from e
        if rule not in rules:
            raise DeclarationError(f"declaration {i}: no rule {rule!r} (the rules: {sorted(rules)})")
        if not isinstance(allow, int) or isinstance(allow, bool) or allow < 0:
            raise DeclarationError(f"declaration {i}: allow must be a count, 0 or more")
        if not isinstance(reason, str) or not reason.strip():
            raise DeclarationError(f"declaration {i} ({rule} on {kernel!r}): a reason is required")
        try:
            pattern = re.compile(kernel)
        except re.error as e:
            raise DeclarationError(f"declaration {i}: kernel is not a regex ({e})") from e
        found.append(Declaration(pattern, rule, allow, reason))
    return found


def unmatched(declarations: list[Declaration], kernels: list[Kernel]) -> list[Declaration]:
    """The declarations whose regex names no kernel: stale, reported and never a refusal."""
    return [d for d in declarations if not any(d.kernel.search(k.demangled) for k in kernels)]


def check(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[Declaration]) -> list[Refusal]:
    """Every rule on every kernel; a refusal where a rule's count is above the declared allowance. A kernel's
    allowance for a rule is the first declaration (file order) whose regex matches (`re.search`) its demangled name
    and names the rule; none declared is 0."""
    refusals = []
    for kernel in kernels:
        for rule in rules.values():
            findings = rule.check(kernel)
            count = sum(f.count for f in findings)
            declared = next((d for d in declarations if d.rule == rule.name and d.kernel.search(kernel.demangled)), None)
            allowed = declared.allow if declared else 0
            if count > allowed:
                refusals.append(Refusal(kernel.demangled, rule.name, count, allowed, declared.reason if declared else None,
                                        tuple(findings), rule.refusal, rule.fix))
    return refusals
