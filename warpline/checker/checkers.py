"""THE CHECKERS over a manifest: named kinds whose arguments are the project's declarations, beside the declarations of rule
counts (`declare.py`), in the same spirit (a place pattern, the arguments, a reason). A manifest field is named by a path
(`manifest.py`: `loops/held_many.cu/1/live`); a pattern is a regular expression that must match the whole path.

    {"declarations": [
      {"kernel": "^k\\\\(", "rule": "stack_memory", "allow": 128, "reason": "..."},              # `violations` (the original form)
      {"kernel": "^k\\\\(", "check": "bound", "figure": "loops/.*/live", "at_most": 30, "reason": "..."},
      {"kernel": "^k\\\\(", "check": "hash", "fields": "loops/.*/live", "hash": "9f2c...", "reason": "..."},
      {"kernel": "^k\\\\(", "check": "waive", "fields": "loops/.*/class/.*", "reason": "too noisy to hold"}],
     "coverage": true}

- `violations` is `declare.judge`: the rule counts against `allow`, else the rule's default. An entry with a `rule` and no
  `check` is one.
- `bound`: every field `figure` matches, at each kernel `kernel` matches, against one of `at_most`, `at_least`, `exactly`. All
  the bounds that apply are enforced (they are not alternatives). A figure that matches no field of a kernel is refused.
- `hash`: the values of the fields `fields` matches at the kernels `kernel` matches, hashed (`digest`); refused while the hash
  differs, printing the values and the hash to write. A hash that covers no field is refused.
- `waive`: the fields it matches are not for coverage to ask about; its reason is the project's.
- `coverage` (true, a top-level key; absent is off): every numeric field of a kernel's manifest, the rule counts included, must
  be covered (a bound's figure, a hash's fields, a `violations` declaration of the rule, or the rule's default) or waived. A
  rule's default covers it unless it is null (a notice accepted by default) or the rule is a bound rule (a figure shown, never
  refused): those need a declaration. Coverage does not judge whether a covering checker is strict enough.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from . import manifest as manifest_mod

KINDS = ("violations", "bound", "hash", "waive")
_COMPARISONS = {"at_most": (lambda v, w: v <= w, "at most"), "at_least": (lambda v, w: v >= w, "at least"),
                "exactly": (lambda v, w: v == w, "exactly")}


class CheckerError(ValueError):
    """A declaration of a checker that is not valid (the declarations' own error: `declare.DeclarationError` is raised over it)."""


@dataclass(frozen=True)
class Bound:
    kernel: re.Pattern
    figure: re.Pattern
    comparison: str
    value: int
    reason: str


@dataclass(frozen=True)
class Hash:
    kernel: re.Pattern
    fields: re.Pattern
    hash: str
    reason: str


@dataclass(frozen=True)
class Waiver:
    kernel: re.Pattern
    fields: re.Pattern
    reason: str


@dataclass(frozen=True)
class Checks:
    """The checkers a declarations file names besides `violations`, and whether coverage is on."""
    bounds: tuple[Bound, ...] = ()
    hashes: tuple[Hash, ...] = ()
    waivers: tuple[Waiver, ...] = ()
    coverage: bool = False


@dataclass(frozen=True)
class Refused:
    """What a checker refuses at a kernel: `rule` is the checker's name (`bound`, `hash`, `coverage`), `message` the first line
    (without the kernel) and `detail` the lines under it."""
    kernel: str
    rule: str
    message: str
    detail: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        return [f"{self.kernel}: {self.rule}: {self.message}", *[f"  {d}" for d in self.detail]]


def _pattern(entry: dict, key: str, i: int) -> re.Pattern:
    try:
        return re.compile(entry[key])
    except KeyError as e:
        raise CheckerError(f"declaration {i}: a {entry.get('check')} needs {key}") from e
    except (re.error, TypeError) as e:
        raise CheckerError(f"declaration {i}: {key} is not a regex ({e})") from e


def load(text: str) -> Checks:
    """The checkers of a declarations text: its `bound`, `hash` and `waive` entries and its `coverage` flag. The entries of
    `violations` (`declare.load_declarations`) are not read here. An unknown `check`, a missing argument or reason, a bound
    with none or two comparisons, a value that is not a count, or a regex that does not compile is an error."""
    try:
        doc = json.loads(text)
        entries = doc["declarations"]
    except (ValueError, KeyError, TypeError) as e:
        raise CheckerError(f'the file must be {{"declarations": [...]}}: {e}') from e
    coverage = doc.get("coverage", False)
    if not isinstance(coverage, bool):
        raise CheckerError('"coverage" must be true or false')
    bounds, hashes, waivers = [], [], []
    for i, entry in enumerate(entries):
        kind = entry.get("check", "violations") if isinstance(entry, dict) else None
        if kind not in KINDS:
            raise CheckerError(f"declaration {i}: no check {kind!r} (the checks: {', '.join(KINDS)})")
        if kind == "violations":
            continue
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise CheckerError(f"declaration {i} ({kind} on {entry.get('kernel')!r}): a reason is required")
        kernel = _pattern(entry, "kernel", i)
        if kind == "bound":
            given = [k for k in _COMPARISONS if k in entry]
            if len(given) != 1:
                raise CheckerError(f"declaration {i}: a bound needs exactly one of {', '.join(_COMPARISONS)}")
            value = entry[given[0]]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckerError(f"declaration {i}: {given[0]} must be a count, 0 or more")
            bounds.append(Bound(kernel, _pattern(entry, "figure", i), given[0], value, reason))
        elif kind == "hash":
            expected = entry.get("hash")
            if not isinstance(expected, str) or not expected.strip():
                raise CheckerError(f"declaration {i}: a hash needs the expected hash (write any text to learn the current one)")
            hashes.append(Hash(kernel, _pattern(entry, "fields", i), expected.strip(), reason))
        else:
            waivers.append(Waiver(kernel, _pattern(entry, "fields", i), reason))
    return Checks(tuple(bounds), tuple(hashes), tuple(waivers), coverage)


def digest(rows: list[tuple[str, str, int]]) -> str:
    """The hash of covered values: `(kernel, path, value)` rows (the manifest' kernel order, each kernel's paths sorted), each as
    `path=value` on a line, SHA-256, its first sixteen hex digits. The kernel's name is not in it: the hash is of the values
    (a rename moves nothing; a field's value moving does)."""
    return hashlib.sha256("".join(f"{p}={v}\n" for _, p, v in rows).encode()).hexdigest()[:16]


def _kernels(manifest: manifest_mod.Manifest, pattern: re.Pattern) -> list[tuple[str, dict[str, int]]]:
    """The kernels of the manifest a pattern names, each with its fields (`manifest.fields`)."""
    return [(place.place, manifest_mod.fields(place)) for place in manifest.places if place.kernel and pattern.search(place.place)]


def judge_bounds(manifest: manifest_mod.Manifest, bounds: tuple[Bound, ...]) -> list[Refused]:
    out = []
    for b in bounds:
        ok, words = _COMPARISONS[b.comparison]
        for kernel, found in _kernels(manifest, b.kernel):
            hit = [(p, v) for p, v in found.items() if b.figure.fullmatch(p)]
            if not hit:
                out.append(Refused(kernel, "bound", f"{b.figure.pattern!r} names no field of the kernel ({b.reason})"))
            out += [Refused(kernel, "bound", f"{p} is {v}, {words} {b.value} declared ({b.reason})") for p, v in hit if not ok(v, b.value)]
    return out


def judge_hashes(manifest: manifest_mod.Manifest, hashes: tuple[Hash, ...]) -> list[Refused]:
    out = []
    for h in hashes:
        rows = [(k, p, v) for k, found in _kernels(manifest, h.kernel) for p, v in sorted(found.items()) if h.fields.fullmatch(p)]
        names = ", ".join(dict.fromkeys(k for k, _, _ in rows)) or h.kernel.pattern
        if not rows:
            out.append(Refused(names, "hash", f"{h.fields.pattern!r} covers no field ({h.reason})"))
            continue
        now = digest(rows)
        if now == h.hash:
            continue
        shown = [f"{p} = {v}" if len({k for k, _, _ in rows}) == 1 else f"{k}: {p} = {v}" for k, p, v in rows]
        out.append(Refused(names, "hash", f"the {len(rows)} covered field(s) hash to {now}, {h.hash} declared ({h.reason})",
                           (*shown, f"fix: look at the values above; if they are intended, write \"hash\": \"{now}\" in the declaration")))
    return out


def uncovered(manifest: manifest_mod.Manifest, checks: Checks, declarations) -> list[Refused]:
    """The coverage check: per kernel, the fields no checker covers and no waiver names. A rule's count is covered by a
    `violations` declaration of the rule at the kernel or by the rule's default (not a null one, not a bound rule's)."""
    out = []
    for place in manifest.places:
        if not place.kernel:
            continue
        found = manifest_mod.fields(place)
        bare = [p for p in found if not p.startswith("rules/")]
        covers = [*(b.figure for b in checks.bounds if b.kernel.search(place.place)),
                  *(h.fields for h in checks.hashes if h.kernel.search(place.place)),
                  *(w.fields for w in checks.waivers if w.kernel.search(place.place))]
        missing = [p for p in bare if not any(c.fullmatch(p) for c in covers)]
        for name, r in place.rules.items():
            path = f"rules/{name}"
            declared = any(d.rule == name and d.kernel.search(place.place) for d in declarations)
            if not (declared or (r.default is not None and not r.bound and r.declarable) or any(c.fullmatch(path) for c in covers)):
                missing.append(path)
        if missing:
            out.append(Refused(place.place, "coverage", f"{len(missing)} field(s) neither covered by a checker nor waived",
                               (*missing, "fix: cover each with a bound or a hash, or waive it with a reason "
                                          "({\"check\": \"waive\", \"fields\": ..., \"reason\": ...})")))
    return out


def stale(checks: Checks, places: list[str]) -> list[str]:
    """The checkers whose kernel pattern names no place of the manifest: a line each, never a refusal."""
    return [f"note: the {kind} declaration for {d.kernel.pattern!r} names no kernel"
            for kind, group in (("bound", checks.bounds), ("hash", checks.hashes), ("waive", checks.waivers)) for d in group
            if not any(d.kernel.search(p) for p in places)]


def judge(manifest: manifest_mod.Manifest, checks: Checks, declarations) -> list[Refused]:
    """The refusals of the bounds, then the hashes, then (where on) coverage."""
    return [*judge_bounds(manifest, checks.bounds), *judge_hashes(manifest, checks.hashes),
            *(uncovered(manifest, checks, declarations) if checks.coverage else [])]
