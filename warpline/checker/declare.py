"""LAYER 4, DECLARATIONS AND THE RUNNER'S CORE. A project declares, per kernel (a regex over the demangled name), how
many of a rule's findings it allows and WHY. Undeclared means zero. The module holds no threshold and no reason: they
are the project's data (a JSON file). `check` returns the refusals.

    {"declarations": [{"kernel": "^example::main\\(", "rule": "stack_memory", "allow": 128,
                       "reason": "the staging array is indexed by a runtime value; measured and accepted"}],
     "notes": {"stack_memory": "the project's own words on this rule, printed after a refusal's fix"}}

A rule is a model rule (`rules/`), a profile rule (`ncu`-backed, `rules/`) or a `declare` row's rule of the notice ledger
(`notices.py`): `rule_names` lists them all. `notes` gives, per rule name, a text of the project's own: the module prints
it after a refusal's fix and attaches no meaning to it.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

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
    allowed: int
    reason: str | None
    findings: tuple[Finding, ...]
    message: str
    fix: str
    note: str | None = None

    def lines(self) -> list[str]:
        head = f"{self.kernel}: {self.rule}: {self.count} found, {self.allowed} declared"
        if self.reason:
            head += f" ({self.reason})"
        return [head, f"  {self.message}", *[f"  at {f.where} x{f.count}: {f.what}" for f in self.findings],
                f"  fix: {self.fix}", *([f"  note: {self.note}"] if self.note else [])]


def rule_names(ledger: list[notices_mod.Row] | None = None) -> list[str]:
    """Every rule name a declaration (and a note) may name, sorted: the registry's rules, the profile rules, and the
    rules the notice ledger's `declare` rows count under (the base ledger's, or `ledger`'s). A consumer lints that each
    has a note by comparing this list with the keys of its notes."""
    from .rules import profile_registry, registry
    return sorted({*registry(), *profile_registry(), *notices_mod.declared_rules(ledger or notices_mod.base_ledger())})


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
          declarations: list[Declaration], notes: dict[str, str] | None = None) -> Refusal | None:
    """A refusal if the findings' count is above what is declared for `rule` at `place`, else None. The allowance is the
    first declaration (file order) whose regex matches (`re.search`) `place` and names the rule; none declared is 0."""
    count = sum(f.count for f in findings)
    declared = next((d for d in declarations if d.rule == rule and d.kernel.search(place)), None)
    allowed = declared.allow if declared else 0
    if count <= allowed:
        return None
    return Refusal(place, rule, count, allowed, declared.reason if declared else None, tuple(findings), message, fix,
                   (notes or {}).get(rule))


def check(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[Declaration],
          notes: dict[str, str] | None = None) -> list[Refusal]:
    """Every rule on every kernel; a refusal where a rule's count is above the declared allowance (`judge`)."""
    refusals = []
    for kernel in kernels:
        for rule in rules.values():
            if rule.bound and not _declared(declarations, rule.name, kernel.demangled):
                continue
            refusal = judge(kernel.demangled, rule.name, rule.check(kernel), rule.refusal, rule.fix, declarations, notes)
            if refusal:
                refusals.append(refusal)
    return refusals


def _declared(declarations: list[Declaration], rule: str, place: str) -> bool:
    return any(d.rule == rule and d.kernel.search(place) for d in declarations)


def figures(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[Declaration]) -> list[str]:
    """The lines for the bound rules that found something on a kernel where no bound is declared: the figure and what it
    is of. Never a refusal: a project declares a bound (`allow`) to have the figure held."""
    lines = []
    for kernel in kernels:
        for rule in rules.values():
            if rule.bound and not _declared(declarations, rule.name, kernel.demangled):
                found = rule.check(kernel)
                if found:
                    lines.append(f"  {kernel.demangled}: {rule.name}: {sum(f.count for f in found)} (no bound declared): "
                                 + "; ".join(f.what for f in found))
    return lines


def match_profile(profile: Profile, kernels: list[Kernel]) -> Kernel | None:
    """The kernel a profile is of: the profiler's name (`void ns::k<0>(Params)`) against `c++filt`'s, compared by the
    name before the parameter list with the profiler's `void ` and `(bool)` decorations and spaces removed; a lone
    kernel and a lone profile are paired whatever they are called."""
    def core(name: str) -> str:
        name = name.removeprefix("void ").replace("(bool)", "").replace(" ", "")
        return name.split("(")[0]
    found = [k for k in kernels if core(k.demangled) == core(profile.kernel)]
    return found[0] if len(found) == 1 else None


def check_profiles(profiles: list[Profile], kernels: list[Kernel], rules: dict[str, ProfileRule],
                   declarations: list[Declaration], notes: dict[str, str] | None = None) -> tuple[list[Refusal], list[str]]:
    """Every profile rule on every profile: the refusals, and the lines that say what was not run or not matched. A
    refusal's place is the matched kernel's demangled name (the profiler's name where none matches)."""
    refusals, said = [], []
    lone = kernels[0] if len(kernels) == 1 and len(profiles) == 1 else None
    for profile in profiles:
        kernel = match_profile(profile, kernels) or lone
        place = kernel.demangled if kernel else profile.kernel
        if kernel is None:
            said.append(f"note: the profile of {profile.kernel!r} matches no kernel of the binary")
        for rule in rules.values():
            if not (profile.has_details if rule.needs == "details" else profile.has_source):
                said.append(f"note: profile rule {rule.name} not run on {place!r}: the profile has no "
                            f"{'details page' if rule.needs == 'details' else 'source page'}")
                continue
            refusal = judge(place, rule.name, rule.check(profile), rule.refusal, rule.fix, declarations, notes)
            if refusal:
                refusals.append(refusal)
    return refusals, said


def check_notices(read: list[notices_mod.Notice], ledger: list[notices_mod.Row], declarations: list[Declaration],
                  names: dict[str, str] | None = None, notes: dict[str, str] | None = None) -> tuple[list[Refusal], list[str]]:
    """A build's notices against the ledger: the refusals and the lines of `report` rows.

    - a notice with no row is refused whole (the toolkit's lines quoted), grouped by tool, code, text and place;
    - a `refuse` row's notices are refused, grouped by row and place;
    - a `declare` row's notices are counted under the row's rule at their place (the function's name, demangled through
      `names` where it is there), so the declarations govern them as they govern a model rule;
    - a `report` row's notices become one line a row; an `ignore` row's are silent.
    `names` maps a function as the toolkit printed it to its demangled name."""
    names = names or {}
    shown = lambda n: re.sub(r"function '([^']+)'", lambda m: f"function '{names.get(m.group(1), m.group(1))}'", n.message)
    unknown: dict[tuple, list] = {}
    refused: dict[tuple, list] = {}
    declared: dict[tuple, list] = {}
    reported: dict[int, tuple[notices_mod.Row, list]] = {}
    for n in read:
        place = names.get(n.function, n.function) if n.function else (n.where or "the build")
        row = notices_mod.row_of(ledger, n)
        if row is None:
            unknown.setdefault((n.tool, n.code, n.text, place), []).append(n)
        elif row.disposition == "refuse":
            refused.setdefault((row.key, place), [row, []])[1].append(n)
        elif row.disposition == "declare":
            declared.setdefault((row.rule, place), []).append((row, n))
        elif row.disposition == "report":
            reported.setdefault(id(row), (row, []))[1].append(n)
    refusals = []
    for (tool, code, _, place), found in unknown.items():
        whole = "\n      ".join(found[0].raw.splitlines())
        refusals.append(Refusal(place, "notice_without_row", len(found), 0, None,
                                (Finding(found[0].where or place, len(found), whole),),
                                f"the toolkit printed a notice the ledger has no row for ({tool} {code or 'with no code'})",
                                "add a row to the project's ledger (--ledger) with a disposition (refuse, declare, report "
                                "or ignore) and the reason for it", (notes or {}).get("notice_without_row")))
    for (_, place), (row, found) in refused.items():
        what = Counter(shown(n) for n in found)
        refusals.append(Refusal(place, f"notice {row.tool} {row.code or 'no_code'}", len(found), 0, None,
                                tuple(Finding(place, c, w) for w, c in what.items()), row.reason, row.fix, None))
    for (rule, place), found in declared.items():
        what = Counter((row, shown(n)) for row, n in found)
        rows = list(dict.fromkeys(row for row, _ in found))
        refusal = judge(place, rule, [Finding(place, c, f"{row.tool} {row.code or 'no code'}: {w}") for (row, w), c in what.items()],
                        " ".join(dict.fromkeys(r.reason for r in rows)),
                        "; ".join(dict.fromkeys(r.fix for r in rows if r.fix)) or "read the notice's text for its cause",
                        declarations, notes)
        if refusal:
            refusals.append(refusal)
    reports = []
    for row, found in reported.values():
        places = {n.function or n.where or "the build" for n in found}
        text = row.text.replace(" in function '{}'", "")
        reports.append(f"  {row.tool} {row.code or ''}: {len(found)} x in {len(places)} place(s): {text} -- {row.reason}")
    return refusals, reports


def notice_places(read: list[notices_mod.Notice], names: dict[str, str]) -> list[str]:
    """Where the notices are (a function's demangled name, or `file(line)`): what a declaration's regex may name."""
    return sorted({names.get(n.function, n.function) if n.function else n.where for n in read if n.function or n.where})
