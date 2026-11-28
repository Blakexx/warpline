"""THE DECLARATIONS AND THE ONE VERDICT. A project declares, per kernel (a regex over the demangled name), how many of a rule's
findings it allows and WHY. Undeclared means the rule's default. The module holds no threshold and no reason: they are the
project's data (a JSON file).

    {"declarations": [{"kernel": "^example::main\\(", "rule": "stack_memory", "allow": 128,
                       "reason": "the staging array is indexed by a runtime value; measured and accepted"}],
     "notes": {"stack_memory": "the project's own words on this rule, printed after a refusal's fix"}}

A rule is one a module names (the checker's rules, profile rules and notice rows: `warpline.checker.manifest.rule_names`) or one a
manifest being judged holds. A declaration may allow more than a rule's default or tighten it (`allow: 0` where a notice is
accepted by default).

`compare` is the one place a verdict is made: it takes a MANIFEST (`manifest.py`: a static one, a run one, or several judged as
one) and the declarations and refuses. `notes` gives, per rule name, a text of the project's own: it is printed after a
refusal's fix and has no meaning here.
"""
from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass

from . import checkers as checkers_mod
from .manifest import Finding


class DeclarationError(ValueError):
    """A declaration file that is not valid: bad JSON, an unknown rule, a missing or empty reason."""


@dataclass(frozen=True)
class Declaration:
    kernel: re.Pattern
    rule: str
    allow: int
    reason: str
    #: the SITE it is for, a regex searched in a finding's `where` (a source line `file:line`, an address): the findings at
    #: the sites it names, and only those, are allowed by it; None, every finding of the rule at the kernel
    at: re.Pattern | None = None


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
    """The declarations of a JSON text, in file order. A reason is required; a rule must exist: one of `rules` (a registry,
    or just its names) or `extra_names` (the ledger's and the profile rules', the manifest's own)."""
    known = {*rules, *extra_names}
    try:
        doc = json.loads(text)
        entries = doc["declarations"]
    except (ValueError, KeyError, TypeError) as e:
        raise DeclarationError(f'the file must be {{"declarations": [...]}}: {e}') from e
    found = []
    for i, entry in enumerate(entries):
        kind = entry.get("check", "violations") if isinstance(entry, dict) else "violations"
        if kind in checkers_mod.KINDS[1:]:
            continue                                  # a bound, a hash or a waiver: `checkers.load` reads it
        if kind != "violations":
            raise DeclarationError(f"declaration {i}: no check {kind!r} (the checks: {', '.join(checkers_mod.KINDS)})")
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
        at = entry.get("at")
        if at is not None:
            try:
                at = re.compile(at)
            except (re.error, TypeError) as e:
                raise DeclarationError(f"declaration {i}: at is a regex of the sites it allows ({e})") from e
        found.append(Declaration(pattern, rule, allow, reason, at))
    return found


def judge(place: str, rule: str, findings: list[Finding], message: str, fix: str,
          declarations: list[Declaration], notes: dict[str, str] | None = None, default: int | None = 0,
          declarable: bool = True) -> list[Refusal]:
    """The refusals of `rule` at `place`: a group of findings whose count is above what is allowed for it. The declarations
    that name a SITE (`at`, in file order) take the findings whose `where` it matches, each its own group against its own
    `allow`; the rest are one group, allowed by the first declaration (file order) with no site whose regex matches
    (`re.search`) `place` and names the rule, else `default` (0, a notice row's count, or None: accepted); a declaration may
    be tighter than the default. A rule that is not `declarable` takes no declaration."""
    named = [d for d in declarations if declarable and d.rule == rule and d.kernel.search(place)]
    groups: list[tuple[Declaration | None, list[Finding]]] = []
    rest = list(findings)
    for d in (d for d in named if d.at is not None):
        mine = [f for f in rest if d.at.search(f.where)]
        rest = [f for f in rest if not d.at.search(f.where)]
        groups.append((d, mine))
    groups.append((next((d for d in named if d.at is None), None), rest))
    out = []
    for declared, group in groups:
        count = sum(f.count for f in group)
        allowed = declared.allow if declared else default
        if allowed is None or count <= allowed:
            continue
        reason = (f"{declared.reason}; at {declared.at.pattern!r}" if declared.at is not None else declared.reason) \
            if declared else None
        out.append(Refusal(place, rule, count, allowed, reason, tuple(group), message, fix, (notes or {}).get(rule)))
    return out


@dataclass(frozen=True)
class Comparison:
    """What a comparison found: the refusals, the lines for the bound rules with no bound declared (a figure, never a
    refusal), the declarations whose regex names no place of the manifests (stale: reported, never a refusal), and the lines
    the manifests say about what was not run or not measured. `refusals` also holds what the checkers over the manifests
    refuse (`checkers.Refused`, the same `lines()`), and `stale_checks` the lines for their stale declarations."""
    refusals: list[Refusal]
    figures: list[str]
    stale: list[Declaration]
    said: tuple[str, ...]
    stale_checks: tuple[str, ...] = ()


def _declared(declarations: list[Declaration], rule: str, place: str) -> bool:
    return any(d.rule == rule and d.kernel.search(place) for d in declarations)


def compare(manifest, declarations: list[Declaration], notes: dict[str, str] | None = None,
            checks: checkers_mod.Checks | None = None) -> Comparison:
    """The manifest (one `manifest.Manifest`, or several judged as one: a binary's static manifest and its kernels' run
    manifests) against the declarations. For every place and rule: a bound rule with no declaration at the place is a figure;
    any other rule is judged (`judge`): refused where its count is above the declaration's allowance, else the rule's
    default. A refusal names the place (kernel), rule, count, allowed and the declaration's reason or `undeclared`. With
    `checks` (`checkers.load`) the bounds, the hashes and, where on, coverage judge the manifests as well; a checker that
    cannot judge a run manifest (`checkers.refuse_on_run`) raises `checkers.CheckerError` before anything is judged."""
    manifests = checkers_mod.manifests_of(manifest)
    checks = checks or checkers_mod.Checks()
    checkers_mod.refuse_on_run(manifests, checks)
    refusals, figures = [], []
    for m in manifests:
        for place in m.places:
            for name, r in place.rules.items():
                if r.bound and not _declared(declarations, name, place.place):
                    if r.count:
                        figures.append(f"  {place.place}: {name}: {r.count} (no bound declared): " + "; ".join(s.what for s in r.sites))
                    continue
                refusals += judge(place.place, name, list(r.sites), r.message, r.fix, declarations, notes, r.default,
                                  r.declarable)
    places = list(dict.fromkeys(p.place for m in manifests for p in m.places))
    stale = [d for d in declarations if not any(d.kernel.search(p) for p in places)]
    refusals += checkers_mod.judge(manifests, checks, declarations)
    said = tuple(line for m in manifests for line in m.said)
    return Comparison(refusals, figures, stale, said, tuple(checkers_mod.stale(checks, places)))


def manifest_rules(manifest) -> list[str]:
    """The rule names the manifest(s) hold: names a declaration of them may name wherever the module that wrote them is
    absent."""
    return sorted({name for m in checkers_mod.manifests_of(manifest) for p in m.places for name in p.rules})
