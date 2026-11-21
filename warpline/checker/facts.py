"""THE FACTS: what a checker run found, written once per binary as a stable JSON, and its reader. The facts are data about a
build, not a verdict on it: per place (a kernel, or the function or file a notice or a profile names) each rule's count and
its sites. The rules of the model, the notices (each a row of the ledger, which counts it under a rule) and the profile
rules are the writers; the comparison (`declare.compare`, `warpline compare`) is the only reader that judges.

    {"format": "warpline-facts/1",
     "notices": 6,                                  # notices read from the build's messages; null where none were given
     "said": ["note: profile rules not run ..."],   # what was not run
     "places": [{"place": "sample::tile()", "kernel": true, "name": "_ZN6sample4tileEv",
                 "rules": {"stack_memory": {"count": 128, "bound": false, "default": 0, "declarable": true,
                                            "message": "...", "fix": "...",
                                            "sites": [{"where": "0x0020", "count": 128, "what": "..."}]}}}]}

- `count` is the sum of the sites' counts, what a declaration's `allow` is compared with; `sites` say where and what.
- `bound`: the rule reports a figure (`function_size`, `step_work`): with no declaration it is shown, never refused.
- `default`: the allowance when no declaration names the rule at the place (0, a count a notice row gives, or null for a
  notice accepted by default); `declarable` is false for the one count no declaration can lift (`notice_without_row`).
- A kernel lists every rule of the model that ran, counts of 0 included. A notice or profile rule is listed where it found
  something. The file is written in a fixed order (kernels in the binary's order, then other places in the order seen), with
  no time, path or run identity in it: the same binary and the same inputs write the same bytes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field

from . import notices as notices_mod
from .model import Kernel
from .ncu import Profile
from .rule import Finding, ProfileRule, Rule

FORMAT = "warpline-facts/1"
#: the one notice rule no declaration can lift: an unknown notice is a decision, never a count
NO_ROW = "notice_without_row"


class FactsError(ValueError):
    """A facts file that is not valid."""


@dataclass(frozen=True)
class RuleFacts:
    count: int
    sites: tuple[Finding, ...]
    message: str
    fix: str
    bound: bool = False
    default: int | None = 0
    declarable: bool = True


@dataclass(frozen=True)
class Place:
    place: str
    kernel: bool
    rules: dict[str, RuleFacts]
    name: str | None = None


@dataclass(frozen=True)
class Facts:
    places: tuple[Place, ...]
    said: tuple[str, ...] = ()
    notices: int | None = None


def match_profile(profile: Profile, kernels: list[Kernel]) -> Kernel | None:
    """The kernel a profile is of: the profiler's name (`void ns::k<0>(Params)`) against `c++filt`'s, compared by the
    name before the parameter list with the profiler's `void ` and `(bool)` decorations and spaces removed; a lone
    kernel and a lone profile are paired whatever they are called."""
    def core(name: str) -> str:
        name = name.removeprefix("void ").replace("(bool)", "").replace(" ", "")
        return name.split("(")[0]
    found = [k for k in kernels if core(k.demangled) == core(profile.kernel)]
    return found[0] if len(found) == 1 else None


def _lowest(defaults: list[int | None]) -> int | None:
    """The strictest of several default allowances (null, accepted, is the loosest)."""
    counts = [d for d in defaults if d is not None]
    return min(counts) if counts else None


def _notice_rules(read: list[notices_mod.Notice], ledger: list[notices_mod.Row],
                  names: dict[str, str]) -> dict[str, dict[str, RuleFacts]]:
    """A build's notices as rule facts by place: each notice is counted under its row's rule at the function it names
    (demangled through `names`), else its `file(line)`, else `the build`. A notice with no row is counted under
    `notice_without_row`, quoted whole. The notices of all of a rule's rows at one place are summed."""
    shown = lambda n: re.sub(r"function '([^']+)'", lambda m: f"function '{names.get(m.group(1), m.group(1))}'", n.message)
    known: dict[tuple[str, str], list] = {}
    unknown: dict[str, dict[tuple, list]] = {}
    for n in read:
        place = names.get(n.function, n.function) if n.function else (n.where or "the build")
        row = notices_mod.row_of(ledger, n)
        if row is None:
            unknown.setdefault(place, {}).setdefault((n.tool, n.code, n.text), []).append(n)
        else:
            known.setdefault((place, row.rule), []).append((row, n))
    by_place: dict[str, dict[str, RuleFacts]] = {}
    for (place, rule), found in known.items():
        what = Counter((row, shown(n)) for row, n in found)
        rows = list(dict.fromkeys(row for row, _ in found))
        sites = tuple(Finding(place, c, f"{row.tool} {row.code or 'no code'}: {w}") for (row, w), c in what.items())
        by_place.setdefault(place, {})[rule] = RuleFacts(
            sum(s.count for s in sites), sites, " ".join(dict.fromkeys(r.reason for r in rows)),
            "; ".join(dict.fromkeys(r.fix for r in rows if r.fix)) or "read the notice's text for its cause",
            default=_lowest([r.allow for r in rows]))
    for place, groups in unknown.items():
        sites = tuple(Finding(found[0].where or place, len(found), "\n      ".join(found[0].raw.splitlines()))
                      for found in groups.values())
        by_place.setdefault(place, {})[NO_ROW] = RuleFacts(
            sum(s.count for s in sites), sites,
            "the toolkit printed a notice the ledger has no row for (" + "; ".join(
                f"{t} {c or 'with no code'}" for t, c, _ in groups) + ")",
            "add a row to the project's ledger (--ledger) with a rule, the notice's default allowance and the reason for it",
            declarable=False)
    return by_place


def _model_facts(kernel: Kernel, rules: dict[str, Rule]) -> dict[str, RuleFacts]:
    out = {}
    for rule in rules.values():
        found = rule.check(kernel)
        out[rule.name] = RuleFacts(sum(f.count for f in found), tuple(found), rule.refusal, rule.fix, bound=rule.bound)
    return out


def collect(kernels: list[Kernel], rules: dict[str, Rule], *, notices: list[notices_mod.Notice] | None = None,
            ledger: list[notices_mod.Row] | None = None, profiles: list[Profile] | None = None,
            profile_rules: dict[str, ProfileRule] | None = None, said: Collection[str] = ()) -> Facts:
    """The facts of a binary: every rule of `rules` on every kernel; with `notices` (the build's messages, read), each
    counted under its row's rule of `ledger` (default: the base ledger); with `profiles`, every profile rule on every
    profile, at the matched kernel (the profiler's own name where none matches). `said` are lines the facts carry about
    what was not run."""
    places: dict[str, dict] = {}
    for kernel in kernels:
        places[kernel.demangled] = {"kernel": kernel, "rules": _model_facts(kernel, rules)}
    if notices is not None:
        pretty = {k.name: k.demangled for k in kernels}
        for place, found in _notice_rules(notices, ledger if ledger is not None else notices_mod.base_ledger(), pretty).items():
            places.setdefault(place, {"kernel": None, "rules": {}})["rules"].update(dict(sorted(found.items())))
    lines = list(said)
    if profiles is not None:
        lone = kernels[0] if len(kernels) == 1 and len(profiles) == 1 else None
        for profile in profiles:
            kernel = match_profile(profile, kernels) or lone
            place = kernel.demangled if kernel else profile.kernel
            if kernel is None:
                lines.append(f"note: the profile of {profile.kernel!r} matches no kernel of the binary")
            entry = places.setdefault(place, {"kernel": None, "rules": {}})
            for rule in (profile_rules or {}).values():
                if not (profile.has_details if rule.needs == "details" else profile.has_source):
                    lines.append(f"note: profile rule {rule.name} not run on {place!r}: the profile has no "
                                 f"{'details page' if rule.needs == 'details' else 'source page'}")
                    continue
                found = rule.check(profile)
                entry["rules"][rule.name] = RuleFacts(sum(f.count for f in found), tuple(found), rule.refusal, rule.fix,
                                                      bound=rule.bound)
    return Facts(tuple(Place(p, e["kernel"] is not None, e["rules"], e["kernel"].name if e["kernel"] else None)
                       for p, e in places.items()),
                 tuple(lines), len(notices) if notices is not None else None)


def to_data(facts: Facts) -> dict:
    return {
        "format": FORMAT,
        "notices": facts.notices,
        "said": list(facts.said),
        "places": [{
            "place": p.place, "kernel": p.kernel, "name": p.name,
            "rules": {name: {"count": r.count, "bound": r.bound, "default": r.default, "declarable": r.declarable,
                             "message": r.message, "fix": r.fix,
                             "sites": [{"where": s.where, "count": s.count, "what": s.what} for s in r.sites]}
                      for name, r in p.rules.items()},
        } for p in facts.places],
    }


def dumps(facts: Facts) -> str:
    """The facts as JSON text: the same facts always write the same bytes (fixed order, no timestamps)."""
    return json.dumps(to_data(facts), indent=1, ensure_ascii=False) + "\n"


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise FactsError(f"facts file: {what}")


def _count(value, what: str) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{what} must be a count, 0 or more")
    return value


def loads(text: str) -> Facts:
    """The facts of a JSON text, validated: the format, every place's rules, their counts and sites (a rule's count is its
    sites' sum)."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise FactsError(f"facts file: not JSON ({e})") from e
    _need(isinstance(doc, dict) and doc.get("format") == FORMAT, f"not a facts file (format {FORMAT!r})")
    _need(isinstance(doc.get("places"), list) and isinstance(doc.get("said", []), list), "places and said are lists")
    notices = doc.get("notices")
    _need(notices is None or isinstance(notices, int), "notices is a count or null")
    places = []
    for i, p in enumerate(doc["places"]):
        _need(isinstance(p, dict) and isinstance(p.get("place"), str) and isinstance(p.get("rules"), dict),
              f"place {i} needs a name and rules")
        rules = {}
        for name, r in p["rules"].items():
            where = f"{p['place']}: {name}"
            _need(isinstance(r, dict) and isinstance(r.get("sites"), list), f"{where}: needs sites")
            sites = tuple(Finding(str(s["where"]), _count(s["count"], f"{where}: a site's count"), str(s["what"]))
                          for s in r["sites"])
            _need(_count(r.get("count"), f"{where}: count") == sum(s.count for s in sites), f"{where}: count is not the sites' sum")
            default = r.get("default", 0)
            _need(default is None or _count(default, f"{where}: default") >= 0, f"{where}: default")
            rules[name] = RuleFacts(r["count"], sites, str(r.get("message", "")), str(r.get("fix", "")),
                                    bool(r.get("bound", False)), default, bool(r.get("declarable", True)))
        places.append(Place(p["place"], bool(p.get("kernel", False)), rules, p.get("name")))
    return Facts(tuple(places), tuple(str(s) for s in doc.get("said", [])), notices)
