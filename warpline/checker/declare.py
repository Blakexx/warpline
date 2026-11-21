"""LAYER 4, DECLARATIONS AND THE RUNNER'S CORE. A project declares, per kernel (a regex over the demangled name), how
many of a rule's findings it allows and WHY. Undeclared means zero. The module holds no threshold and no reason: they
are the project's data (a JSON file). `check` returns the refusals.

    {"declarations": [{"kernel": "^example::main\\(", "rule": "stack_memory", "allow": 128,
                       "reason": "the staging array is indexed by a runtime value; measured and accepted"}],
     "notes": {"stack_memory": "the project's own words on this rule, printed after a refusal's fix"}}

A rule is a model rule (`rules/`), a profile rule (`ncu`-backed, `rules/`) or a rule a row of the notice ledger counts
its notices under (`notices.py`): `rule_names` lists them all. A declaration may allow more than a rule's default or
tighten it (`allow: 0` where a notice is accepted by default).

`compare` is the one place a verdict is made: it takes the FACTS of a run (`facts.py`) and the declarations and refuses. `notes` gives, per rule name, a text of the project's own: the module prints
it after a refusal's fix and attaches no meaning to it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from . import facts as facts_mod
from . import notices as notices_mod
from .model import Kernel
from .ncu import Profile
from .rule import Finding, ProfileRule, Rule


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
    allowed: int | None
    reason: str | None
    findings: tuple[Finding, ...]
    message: str
    fix: str
    note: str | None = None

    def lines(self) -> list[str]:
        head = f"{self.kernel}: {self.rule}: {self.count} found, {self.allowed} declared ({self.reason or 'undeclared'})"
        return [head, f"  {self.message}", *[f"  at {f.where} x{f.count}: {f.what}" for f in self.findings],
                f"  fix: {self.fix}", *([f"  note: {self.note}"] if self.note else [])]


def rule_names(ledger: list[notices_mod.Row] | None = None) -> list[str]:
    """Every rule name a declaration (and a note) may name, sorted: the registry's rules, the profile rules, and the
    rules the notice ledger's rows count under (the base ledger's, or `ledger`'s). A consumer lints that each
    has a note by comparing this list with the keys of its notes."""
    from .rules import profile_registry, registry
    return sorted({*registry(), *profile_registry(), *notices_mod.row_rules(ledger or notices_mod.base_ledger())})


def load_notes(text: str, known: Collection[str]) -> dict[str, str]:
    """The `notes` of a JSON text (`{"notes": {"<rule>": "<text>"}}`; none when the key is absent): a text the project
    keeps per rule name. A name must be one of `known` and a text must not be empty."""
    try:
        doc = json.loads(text)
        notes = doc.get("notes", {})
    except (ValueError, AttributeError) as e:
        raise DeclarationError(f'the notes must be {{"notes": {{rule: text}}}}: {e}') from e
    if not isinstance(notes, dict):
        raise DeclarationError('"notes" must map rule names to texts')
    for name, note in notes.items():
        if name not in known:
            raise DeclarationError(f"a note for {name!r}, which is no rule (the rules: {sorted(known)})")
        if not isinstance(note, str) or not note.strip():
            raise DeclarationError(f"the note for {name!r} must be a text")
    return dict(notes)


def load_declarations(text: str, rules: Collection[str], extra_names: Collection[str] = ()) -> list[Declaration]:
    """The declarations of a JSON text, in file order. A reason is required; a rule must exist: one of `rules` (the
    registry, or just its names) or `extra_names` (the ledger's and the profile rules')."""
    known = {*rules, *extra_names}
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
        if rule not in known:
            raise DeclarationError(f"declaration {i}: no rule {rule!r} (the rules: {sorted(known)})")
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


def unmatched(declarations: list[Declaration], kernels: list[Kernel], places: Collection[str] = ()) -> list[Declaration]:
    """The declarations whose regex names no kernel (nor any of `places`, where a notice or a profile was read): stale,
    reported and never a refusal."""
    return [d for d in declarations
            if not any(d.kernel.search(k.demangled) for k in kernels) and not any(d.kernel.search(p) for p in places)]


def judge(place: str, rule: str, findings: list[Finding], message: str, fix: str,
          declarations: list[Declaration], notes: dict[str, str] | None = None, default: int | None = 0,
          declarable: bool = True) -> Refusal | None:
    """A refusal if the findings' count is above what is allowed for `rule` at `place`, else None. The allowance is the
    first declaration (file order) whose regex matches (`re.search`) `place` and names the rule, else `default` (0, a
    notice row's count, or None: accepted); a declaration may be tighter than the default. A rule that is not `declarable`
    takes no declaration."""
    count = sum(f.count for f in findings)
    declared = next((d for d in declarations if declarable and d.rule == rule and d.kernel.search(place)), None)
    allowed = declared.allow if declared else default
    if allowed is None or count <= allowed:
        return None
    return Refusal(place, rule, count, allowed, declared.reason if declared else None, tuple(findings), message, fix,
                   (notes or {}).get(rule))


@dataclass(frozen=True)
class Comparison:
    """What a comparison found: the refusals, the lines for the bound rules with no bound declared (a figure, never a
    refusal), the declarations whose regex names no place of the facts (stale: reported, never a refusal), and the lines
    the facts say about what was not run."""
    refusals: list[Refusal]
    figures: list[str]
    stale: list[Declaration]
    said: tuple[str, ...]


def _declared(declarations: list[Declaration], rule: str, place: str) -> bool:
    return any(d.rule == rule and d.kernel.search(place) for d in declarations)


def compare(facts: facts_mod.Facts, declarations: list[Declaration], notes: dict[str, str] | None = None) -> Comparison:
    """The facts against the declarations. For every place and rule: a bound rule with no declaration at the place is a
    figure; any other rule is judged (`judge`): refused where its count is above the declaration's allowance, else the
    rule's default. A refusal names the place (kernel), rule, count, allowed and the declaration's reason or `undeclared`."""
    refusals, figures = [], []
    for place in facts.places:
        for name, r in place.rules.items():
            if r.bound and not _declared(declarations, name, place.place):
                if r.count:
                    figures.append(f"  {place.place}: {name}: {r.count} (no bound declared): " + "; ".join(s.what for s in r.sites))
                continue
            refusal = judge(place.place, name, list(r.sites), r.message, r.fix, declarations, notes, r.default, r.declarable)
            if refusal:
                refusals.append(refusal)
    places = [p.place for p in facts.places]
    stale = [d for d in declarations if not any(d.kernel.search(p) for p in places)]
    return Comparison(refusals, figures, stale, facts.said)


def check(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[Declaration],
          notes: dict[str, str] | None = None) -> list[Refusal]:
    """Every rule on every kernel, compared (`facts.collect`, `compare`): the refusals."""
    return compare(facts_mod.collect(kernels, rules), declarations, notes).refusals


def figures(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[Declaration]) -> list[str]:
    """The lines for the bound rules that found something on a kernel where no bound is declared: the figure and what it
    is of. Never a refusal: a project declares a bound (`allow`) to have the figure held."""
    return compare(facts_mod.collect(kernels, rules), declarations).figures


def match_profile(profile: Profile, kernels: list[Kernel]) -> Kernel | None:
    return facts_mod.match_profile(profile, kernels)


def check_profiles(profiles: list[Profile], kernels: list[Kernel], rules: dict[str, ProfileRule],
                   declarations: list[Declaration], notes: dict[str, str] | None = None) -> tuple[list[Refusal], list[str]]:
    """Every profile rule on every profile: the refusals, and the lines that say what was not run or not matched."""
    result = compare(facts_mod.collect(kernels, {}, profiles=profiles, profile_rules=rules), declarations, notes)
    return result.refusals, list(result.said)


def check_notices(read: list[notices_mod.Notice], ledger: list[notices_mod.Row], declarations: list[Declaration],
                  names: dict[str, str] | None = None, notes: dict[str, str] | None = None) -> list[Refusal]:
    """A build's notices against the ledger and the declarations: the refusals. Each notice is counted under its row's
    rule at its place (`facts.collect`), a notice with no row is refused whole; `names` maps a function as the toolkit
    printed it to its demangled name."""
    by_place = facts_mod._notice_rules(read, ledger, names or {})
    facts = facts_mod.Facts(tuple(facts_mod.Place(p, False, rules) for p, rules in by_place.items()))
    return compare(facts, declarations, notes).refusals


def notice_places(read: list[notices_mod.Notice], names: dict[str, str]) -> list[str]:
    """Where the notices are (a function's demangled name, or `file(line)`): what a declaration's regex may name."""
    return sorted({names.get(n.function, n.function) if n.function else n.where for n in read if n.function or n.where})
