"""THE CHECKERS over manifests, static or run: named kinds whose arguments are the project's declarations, beside the declarations
of rule counts (`declare.py`), in the same spirit (a place pattern, the arguments, a reason). A manifest field is named by a path
(`manifest.py`: `loops/held_many.cu/1/live`, `window/period`); a pattern is a regular expression that must match the whole path.

    {"declarations": [
      {"kernel": "^k\\\\(", "rule": "stack_memory", "allow": 128, "reason": "..."},              # `violations` (the original form)
      {"kernel": "^k\\\\(", "check": "bound", "figure": "loops/.*/live", "at_most": 30, "reason": "..."},
      {"kernel": "^k\\\\(", "check": "hash", "fields": "loops/.*/live", "hash": "9f2c...", "reason": "..."},
      {"kernel": "^k\\\\(", "check": "waive", "fields": "loops/.*/class/.*", "reason": "too noisy to hold"},
      {"kernel": "^void k<1>\\\\(", "check": "present", "exactly": 1, "reason": "the build carries this instance"},
      {"kernel": "^k\\\\(", "check": "idle", "stream": "hbm", "busy": ["producer/load", "consumer/mma"],
       "serial": {"consumer/epilogue": "..."}, "reason": "..."}],
     "coverage": true, "strict": true}

- `violations` is `declare.judge`: the rule counts against `allow`, else the rule's default. An entry with a `rule` and no
  `check` is one.
- `bound`: every field `figure` matches, at each kernel `kernel` matches, against one of `at_most`, `at_least` (a number) or
  `exactly` (a count). All the bounds that apply are enforced (they are not alternatives). A figure that matches no field of a
  kernel is refused. A run's figure varies from run to run: `exactly` on one is an error, but for the structural figures
  (`manifest.STRUCTURAL`).
- `hash`: the values of the fields `fields` matches at the kernels `kernel` matches, hashed (`digest`); refused while the hash
  differs, printing the values and the hash to write. A hash that covers no field is refused; one that covers a run's figure
  is an error.
- `waive`: the fields it matches are not for coverage to ask about; its reason is the project's.
- `present`: the number of kernels `kernel` matches, over the manifests judged as one, against one of `at_most`, `at_least` or
  `exactly` (a count): a project says which kernels its binary must carry (an instance it built, every kernel its source
  defines). It reads no field and covers none, and a pattern that names no kernel is its finding, never stale.
- `idle`: every PHASE a run manifest measures at the kernels `kernel` matches (`roles/<role>/phases/<phase>/cycles` above 0;
  `phase#2`, a phase a window holds twice, is `phase`'s) is one of `busy`, the phases the binding stream `stream` is busy in,
  or of `serial`, the phases declared to run alone, each with its reason; a phase `busy` or `serial` names that no run
  manifest measures is refused, and so is a kernel whose run manifests measure no phase. The floor writes it from the work
  declared (`floor.declarations`, docs/floor.md): no phase idles the binding stream, over the DECLARED overlap (a run
  manifest's phases carry no start on one frame, so whether two roles' phases ran at once is not measured here). It covers
  no field.
- `strict` (true, a top-level key; absent is off): a `violations` declaration that allows findings (`allow` above 0, of a rule
  that is no bound rule) where no place it names has one is refused: the declarations say only what is so.
- `coverage` (true, a top-level key; absent is off): every numeric field of a kernel's manifest, the rule counts included, must
  be covered (a bound's figure, a hash's fields, a `violations` declaration of the rule, or the rule's default) or waived. A
  rule's default covers it unless it is null (a notice accepted by default) or the rule is a bound rule (a figure shown, never
  refused): those need a declaration. Coverage does not judge whether a covering checker is strict enough.

Several manifests are judged as one (a binary's static manifest and its kernels' run manifests, against one declarations
file): a kernel is its place name, and its fields are those of every manifest that has the place.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from . import manifest as manifest_mod

KINDS = ("violations", "bound", "hash", "waive", "present", "idle")
#: a run manifest's phase field: `roles/<role>/phases/<phase>/cycles` (a phase a window holds twice is `<phase>#2`)
_PHASE = re.compile(r"^roles/([^/]+)/phases/([^/]+?)(?:#\d+)?/cycles$")
_COMPARISONS = {"at_most": (lambda v, w: v <= w, "at most"), "at_least": (lambda v, w: v >= w, "at least"),
                "exactly": (lambda v, w: v == w, "exactly")}


class CheckerError(ValueError):
    """A declaration of a checker that is not valid (the declarations' own error: `declare.DeclarationError` is raised over it),
    or one that cannot judge the manifest it is given (`exactly` on a run's figure, a hash over one)."""


@dataclass(frozen=True)
class Bound:
    kernel: re.Pattern
    figure: re.Pattern
    comparison: str
    value: int | float
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
class Presence:
    kernel: re.Pattern
    comparison: str
    value: int
    reason: str


@dataclass(frozen=True)
class Idle:
    """No phase idles the binding stream: the phases (`role/phase`) `stream` is busy in, and those declared serial with their
    reasons."""
    kernel: re.Pattern
    stream: str
    busy: tuple[str, ...]
    serial: tuple[tuple[str, str], ...]
    reason: str


@dataclass(frozen=True)
class Checks:
    """The checkers a declarations file names besides `violations`, and whether coverage and strictness are on."""
    bounds: tuple[Bound, ...] = ()
    hashes: tuple[Hash, ...] = ()
    waivers: tuple[Waiver, ...] = ()
    coverage: bool = False
    presences: tuple[Presence, ...] = ()
    strict: bool = False
    idles: tuple[Idle, ...] = ()


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


def _value_ok(comparison: str, value) -> bool:
    """A bound's value: a count for `exactly`, a finite number for `at_most` and `at_least` (a run's figures are not counts);
    0 or more."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return False
    return isinstance(value, int) or comparison != "exactly"


def _comparison(entry: dict, i: int, kind: str, count: bool) -> tuple[str, int | float]:
    """The one comparison of a bound or a presence and its value: a count for `exactly` (and for every presence, `count`), a
    finite number for `at_most` and `at_least`; 0 or more."""
    given = [k for k in _COMPARISONS if k in entry]
    if len(given) != 1:
        raise CheckerError(f"declaration {i}: a {kind} needs exactly one of {', '.join(_COMPARISONS)}")
    value = entry[given[0]]
    if not _value_ok("exactly" if count else given[0], value):
        raise CheckerError(f"declaration {i}: {given[0]} must be "
                           f"{'a count' if count or given[0] == 'exactly' else 'a number'}, 0 or more")
    return given[0], value


def load(text: str) -> Checks:
    """The checkers of a declarations text: its `bound`, `hash`, `waive`, `present` and `idle` entries and its `coverage` and
    `strict` flags. The entries of `violations` (`declare.load_declarations`) are not read here. An unknown `check`, a missing argument
    or reason, a bound or a presence with none or two comparisons, a value that is not a number 0 or more (a count for `exactly`
    and for a presence), or a regex that does not compile is an error."""
    try:
        doc = json.loads(text)
        entries = doc["declarations"]
    except (ValueError, KeyError, TypeError) as e:
        raise CheckerError(f'the file must be {{"declarations": [...]}}: {e}') from e
    coverage, strict = doc.get("coverage", False), doc.get("strict", False)
    if not isinstance(coverage, bool):
        raise CheckerError('"coverage" must be true or false')
    if not isinstance(strict, bool):
        raise CheckerError('"strict" must be true or false')
    bounds, hashes, waivers, presences, idles = [], [], [], [], []
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
            comparison, value = _comparison(entry, i, "bound", count=False)
            bounds.append(Bound(kernel, _pattern(entry, "figure", i), comparison, value, reason))
        elif kind == "present":
            presences.append(Presence(kernel, *_comparison(entry, i, "presence", count=True), reason))
        elif kind == "hash":
            expected = entry.get("hash")
            if not isinstance(expected, str) or not expected.strip():
                raise CheckerError(f"declaration {i}: a hash needs the expected hash (write any text to learn the current one)")
            hashes.append(Hash(kernel, _pattern(entry, "fields", i), expected.strip(), reason))
        elif kind == "idle":
            idles.append(_idle(entry, i, kernel, reason))
        else:
            waivers.append(Waiver(kernel, _pattern(entry, "fields", i), reason))
    return Checks(tuple(bounds), tuple(hashes), tuple(waivers), coverage, tuple(presences), strict, tuple(idles))


def _idle(entry: dict, i: int, kernel: re.Pattern, reason: str) -> Idle:
    """An `idle` entry: its stream, its busy phases (a list of `role/phase`) and its serial phases (each with a reason)."""
    stream, busy, serial = entry.get("stream"), entry.get("busy"), entry.get("serial", {})
    if not isinstance(stream, str) or not stream:
        raise CheckerError(f"declaration {i}: an idle check names its binding stream")
    phase = re.compile(r"^[^/]+/[^/]+$")
    if not isinstance(busy, list) or not all(isinstance(k, str) and phase.match(k) for k in busy):
        raise CheckerError(f"declaration {i}: an idle check's busy is a list of phases, `role/phase`")
    if not isinstance(serial, dict) or not all(phase.match(k) and isinstance(v, str) and v.strip() for k, v in serial.items()):
        raise CheckerError(f"declaration {i}: an idle check's serial maps a phase (`role/phase`) to its reason")
    return Idle(kernel, stream, tuple(busy), tuple(sorted(serial.items())), reason)


def digest(rows: list[tuple[str, str, int]]) -> str:
    """The hash of covered values: `(kernel, path, value)` rows (the manifest' kernel order, each kernel's paths sorted), each as
    `path=value` on a line, SHA-256, its first sixteen hex digits. The kernel's name is not in it: the hash is of the values
    (a rename moves nothing; a field's value moving does)."""
    return hashlib.sha256("".join(f"{p}={v}\n" for _, p, v in rows).encode()).hexdigest()[:16]


def manifests_of(manifests) -> tuple[manifest_mod.Manifest, ...]:
    """One manifest or several, as a tuple."""
    return (manifests,) if isinstance(manifests, manifest_mod.Manifest) else tuple(manifests)


def kernels(manifests) -> list[tuple[str, dict[str, int | float], list[manifest_mod.Place]]]:
    """Each kernel of the manifests, once a place name (in the order first seen), with its fields over every manifest that has
    the place (`manifest.fields`) and those places."""
    found: dict[str, tuple[dict, list]] = {}
    for m in manifests_of(manifests):
        for place in m.places:
            if place.kernel:
                fields, places = found.setdefault(place.place, ({}, []))
                fields.update(manifest_mod.fields(place))
                places.append(place)
    return [(name, fields, places) for name, (fields, places) in found.items()]


def _kernels(manifests, pattern: re.Pattern) -> list[tuple[str, dict[str, int | float]]]:
    """The kernels a pattern names, each with its fields."""
    return [(name, fields) for name, fields, _ in kernels(manifests) if pattern.search(name)]


def refuse_on_run(manifests, checks: Checks) -> None:
    """The checkers that cannot judge a run manifest, as an error (`CheckerError`) before anything is judged: a bound
    `exactly` on a run's figure that is not structural (`manifest.STRUCTURAL`), and a hash over a run's figure. A run's figures
    vary from run to run: a project bounds them with `at_most` and `at_least`."""
    for m in manifests_of(manifests):
        if m.format != manifest_mod.RUN_FORMAT:
            continue
        for place in m.places:
            if not place.kernel:
                continue
            found = manifest_mod.fields(place)
            for b in checks.bounds:
                if b.comparison != "exactly" or not b.kernel.search(place.place):
                    continue
                varying = [p for p in found if b.figure.fullmatch(p) and p not in manifest_mod.STRUCTURAL]
                if varying:
                    raise CheckerError(
                        f"the bound {b.figure.pattern!r} on {b.kernel.pattern!r} holds {varying[0]} of a run manifest exactly: "
                        f"a run's figure varies from run to run; bound it with at_most or at_least (exactly is for the "
                        f"structural figures {', '.join(manifest_mod.STRUCTURAL)})")
            for h in checks.hashes:
                covered = [p for p in found if h.kernel.search(place.place) and h.fields.fullmatch(p)]
                if covered:
                    raise CheckerError(
                        f"the hash {h.fields.pattern!r} on {h.kernel.pattern!r} covers {covered[0]} of a run manifest: a hash "
                        f"is refused on a run manifest, whose figures vary from run to run; bound them with at_most or at_least")


def judge_bounds(manifests, bounds: tuple[Bound, ...]) -> list[Refused]:
    out = []
    for b in bounds:
        ok, words = _COMPARISONS[b.comparison]
        for kernel, found in _kernels(manifests, b.kernel):
            hit = [(p, v) for p, v in found.items() if b.figure.fullmatch(p)]
            if not hit:
                out.append(Refused(kernel, "bound", f"{b.figure.pattern!r} names no field of the kernel ({b.reason})"))
            out += [Refused(kernel, "bound", f"{p} is {v}, {words} {b.value} declared ({b.reason})") for p, v in hit if not ok(v, b.value)]
    return out


def judge_hashes(manifests, hashes: tuple[Hash, ...]) -> list[Refused]:
    out = []
    for h in hashes:
        rows = [(k, p, v) for k, found in _kernels(manifests, h.kernel) for p, v in sorted(found.items()) if h.fields.fullmatch(p)]
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


def uncovered(manifests, checks: Checks, declarations) -> list[Refused]:
    """The coverage check: per kernel, the fields no checker covers and no waiver names. A rule's count is covered by a
    `violations` declaration of the rule at the kernel or by the rule's default (not a null one, not a bound rule's)."""
    out = []
    for name, found, places in kernels(manifests):
        bare = [p for p in found if not p.startswith("rules/")]
        covers = [*(b.figure for b in checks.bounds if b.kernel.search(name)),
                  *(h.fields for h in checks.hashes if h.kernel.search(name)),
                  *(w.fields for w in checks.waivers if w.kernel.search(name))]
        missing = [p for p in bare if not any(c.fullmatch(p) for c in covers)]
        for place in places:
            for rule, r in place.rules.items():
                path = f"rules/{rule}"
                declared = any(d.rule == rule and d.kernel.search(name) for d in declarations)
                if not (declared or (r.default is not None and not r.bound and r.declarable) or any(c.fullmatch(path) for c in covers)):
                    missing.append(path)
        if missing:
            out.append(Refused(name, "coverage", f"{len(missing)} field(s) neither covered by a checker nor waived",
                               (*missing, "fix: cover each with a bound or a hash, or waive it with a reason "
                                          "({\"check\": \"waive\", \"fields\": ..., \"reason\": ...})")))
    return out


def judge_presences(manifests, presences: tuple[Presence, ...]) -> list[Refused]:
    """Each presence whose pattern names a number of kernels its comparison refuses, with the kernels it names."""
    names = [name for name, _, _ in kernels(manifests)]
    out = []
    for p in presences:
        ok, words = _COMPARISONS[p.comparison]
        named = [n for n in names if p.kernel.search(n)]
        if not ok(len(named), p.value):
            out.append(Refused(p.kernel.pattern, "present", f"names {len(named)} kernel(s), {words} {p.value} declared ({p.reason})",
                               tuple(named)))
    return out


def judge_idles(manifests, idles: tuple[Idle, ...]) -> list[Refused]:
    """Each measured phase of a kernel an idle check names that is neither busy nor serial, each declared phase no run manifest
    measures, and each kernel with no measured phase at all."""
    out = []
    for c in idles:
        serial = dict(c.serial)
        for kernel, found in _kernels(manifests, c.kernel):
            measured: dict[str, float] = {}
            for path, value in found.items():
                m = _PHASE.match(path)
                if m:
                    key = f"{m.group(1)}/{m.group(2)}"
                    measured[key] = measured.get(key, 0) + value
            if not measured:
                out.append(Refused(kernel, "idle", f"no run manifest measures a phase of the kernel (a window's roles/<role>/phases): "
                                                   f"nothing to judge ({c.reason})"))
                continue
            for key, cycles in measured.items():
                if cycles > 0 and key not in c.busy and key not in serial:
                    out.append(Refused(kernel, "idle", f"phase {key} ({cycles:.6g} cycles a window) idles {c.stream}: no work on it is "
                                                       f"declared there, nor in a phase declared to run beside it ({c.reason})",
                                       (f"fix: overlap it with a phase that moves {c.stream} (an overlap set of the work declared), "
                                        "or declare it serial with the reason no overlapped form exists",)))
            out += [Refused(kernel, "idle", f"declares phase {key}, which no run manifest measures ({c.reason})")
                    for key in (*c.busy, *serial) if key not in measured]
    return out


def unshown(manifests, declarations) -> list[Refused]:
    """The strict check: each `violations` declaration that allows findings (`allow` above 0, of a rule no place holds as a bound
    rule) where no place of the manifests it names has one of the rule, refused by its pattern."""
    places = [p for m in manifests_of(manifests) for p in m.places]
    bound = {name for p in places for name, r in p.rules.items() if r.bound}
    out = []
    for d in declarations:
        if d.allow == 0 or d.rule in bound:
            continue
        named = [p for p in places if d.kernel.search(p.place)]
        if not any(p.rules[d.rule].count for p in named if d.rule in p.rules):
            out.append(Refused(d.kernel.pattern, "strict", f"declares {d.rule} (allow {d.allow}), which no place it names has "
                                                           f"({d.reason})" + ("" if named else "; it names no place")))
    return out


def stale(checks: Checks, places: Iterable[str]) -> list[str]:
    """The checkers whose kernel pattern names no place of the manifests: a line each, never a refusal. A presence is not
    among them: a pattern naming no kernel is what it counts."""
    places = list(places)
    return [f"note: the {kind} declaration for {d.kernel.pattern!r} names no kernel"
            for kind, group in (("bound", checks.bounds), ("hash", checks.hashes), ("waive", checks.waivers),
                                ("idle", checks.idles)) for d in group
            if not any(d.kernel.search(p) for p in places)]


def judge(manifests, checks: Checks, declarations) -> list[Refused]:
    """The refusals of the bounds, then the hashes, then the presences, then the idle checks, then (where on) coverage and
    strictness, over one manifest or several judged as one."""
    return [*judge_bounds(manifests, checks.bounds), *judge_hashes(manifests, checks.hashes),
            *judge_presences(manifests, checks.presences), *judge_idles(manifests, checks.idles),
            *(uncovered(manifests, checks, declarations) if checks.coverage else []),
            *(unshown(manifests, declarations) if checks.strict else [])]
