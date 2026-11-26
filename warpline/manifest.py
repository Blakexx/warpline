"""THE MANIFESTS: what warpline learned of a kernel, as one stable JSON file, in two formats of one family, and their reader.
A manifest is data, not a verdict: the checkers (`checkers.py`, `declare.py`, `warpline compare`) are the only readers that
judge. This file holds what is about the files and not about where they came from: their form, their reading and writing,
their validation and their fields. It imports neither module; the checker writes the static manifest, and a run manifest's
parts are given by the module that reads each input (docs/run-manifest.md).

- THE STATIC MANIFEST, `warpline-manifest/1`, a build output: per PLACE (a kernel, or the function or file a notice names)
  each rule's count and sites and, for a kernel, its FIGURES: totals and, for each loop, what it holds and does
  (docs/checker-manifest.md).

      {"format": "warpline-manifest/1", "notices": 6, "said": [],
       "places": [{"place": "sample::tile()", "kernel": true, "name": "_ZN6sample4tileEv",
                   "figures": {"kernel": {...}, "loops_keyed_by": "source file and position", "loops": {...}},
                   "rules": {"stack_memory": {"count": 128, "bound": false, "default": 0, "declarable": true,
                                              "message": "...", "fix": "...",
                                              "sites": [{"where": "0x0020", "count": 128, "what": "..."}]}}}]}

- THE RUN MANIFEST, `warpline-run-manifest/1`, one file a run of a kernel: the same shape where the meaning is the same.
  The place is the static manifest's (the demangled kernel name, given by the caller); its `figures` are the run's (the
  recorder's `summary` of the record(s)), its `rules` the profile rules' findings; `said` says what was not run or not
  measured. A figure that was not measured is absent, never 0.

      {"format": "warpline-run-manifest/1", "said": ["note: waits not stamped for role consumer ..."],
       "places": [{"place": "sample::tile()", "kernel": true,
                   "figures": {"health": {...}, "launch": {...}, "window": {...}, "roles": {...}},
                   "rules": {"shared_wavefronts_excess": {...}}}]}

Neither holds a time, a path, a host or a run's identity, and both are written in a fixed order: the same inputs write the
same bytes. A FIELD is a number of a place's figures or the count of a rule, named by its path (`kernel/registers`,
`loops/held_many.cu/1/live`, `window/period`, `roles/producer/phases/load/cycles`, `rules/stack_memory`): what a bound or a
hash names and what coverage lists (`fields`). Text in the figures (a loop's `at`, the window's `anchor`) is a note, never a
field.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

FORMAT = "warpline-manifest/1"
RUN_FORMAT = "warpline-run-manifest/1"
#: how a static manifest's loops are keyed (docs/checker-manifest.md)
KEYS = ("source file and position", "nesting path")
#: the run figures that are integers of the kernel's structure, not measurements that vary from run to run: the only run
#: figures a bound may hold `exactly` (the lanes and roles a launch has, the stamps a ring dropped)
STRUCTURAL = ("health/lanes", "health/roles", "health/dropped")
#: the parts of a run's figures, in the file's order
RUN_PARTS = ("health", "launch", "window", "roles")


class ManifestError(ValueError):
    """A manifest file that is not valid."""


class InputError(ValueError):
    """An input that the module reading it could not read (a record, a profiler's export, a notice ledger), in words."""


@dataclass(frozen=True)
class Finding:
    """A SITE of a rule's findings: `where` (an address, an offset, the kernel, a profiler's rule name), `count` (how many sites
    or units it stands for: what a declaration allows is the sum over a rule's sites) and `what` (the instance in a line)."""
    where: str
    count: int
    what: str


@dataclass(frozen=True)
class RuleFindings:
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
    rules: dict[str, RuleFindings]
    #: the mangled name (a static manifest's kernel); None for any other place
    name: str | None = None
    #: a kernel's figures (a static manifest's, or a run's); None for any other place
    figures: dict | None = None


@dataclass(frozen=True)
class Manifest:
    places: tuple[Place, ...]
    said: tuple[str, ...] = ()
    #: the notices read from the build's messages (a static manifest's); None where none were given
    notices: int | None = None
    format: str = FORMAT


@dataclass(frozen=True)
class RunPart:
    """What a module gives a run manifest from its inputs: the place's figures (the recorder's, from the records), its rules
    (the checker's, from a profiler's exports), the places a profile names that are not the place, and the lines on what was
    not run or not measured."""
    figures: dict | None = None
    rules: dict[str, RuleFindings] = field(default_factory=dict)
    others: tuple[Place, ...] = ()
    said: tuple[str, ...] = ()


def run_manifest(place: str, parts: list[RunPart | str]) -> Manifest:
    """The run manifest of one run of the kernel `place`, from the parts the modules gave, in order (a text in place of a part
    is a line said: an input not given). The place's figures are `{}` where no part gives any."""
    figures: dict = {}
    rules: dict[str, RuleFindings] = {}
    others: list[Place] = []
    said: list[str] = []
    for part in parts:
        if isinstance(part, str):
            said.append(part)
            continue
        figures.update(part.figures or {})
        rules.update(part.rules)
        others += part.others
        said += part.said
    return Manifest((Place(place, True, rules, None, figures), *others), tuple(said), None, RUN_FORMAT)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def valid(figures, where: str) -> str | None:
    """Why a static manifest's figures read from a file are not figures (None where they are): the shape of the form above,
    every figure a count and the only text a loop's `at`."""
    if not (isinstance(figures, dict) and isinstance(figures.get("kernel"), dict) and isinstance(figures.get("loops"), dict)
            and figures.get("loops_keyed_by") in KEYS):
        return f"{where}: figures need kernel, loops and loops_keyed_by ({' or '.join(KEYS)})"
    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                if (bad := walk(v, f"{path}/{k}")):
                    return bad
            elif not (_count(v) or (k == "at" and isinstance(v, str) and path.startswith("loops/"))):
                return f"{where}: {path}/{k} must be a count, 0 or more"
    return walk(figures["kernel"], "kernel") or walk(figures["loops"], "loops")


def valid_run(figures, where: str) -> str | None:
    """Why a run's figures read from a file are not figures (None where they are): an object of the parts `health`, `launch`,
    `window` and `roles` (each absent where not measured), every figure a finite number, the structural ones counts, and the
    only text the window's `anchor`."""
    if not isinstance(figures, dict) or set(figures) - set(RUN_PARTS):
        return f"{where}: a run's figures are an object of {', '.join(RUN_PARTS)}"
    def walk(node, path):
        if not isinstance(node, dict):
            return f"{where}: {path} must be an object"
        for k, v in node.items():
            p = f"{path}/{k}"
            if isinstance(v, dict):
                if (bad := walk(v, p)):
                    return bad
            elif p == "window/anchor":
                if not isinstance(v, str):
                    return f"{where}: {p} is the anchor's name"
            elif not _number(v) or (p in STRUCTURAL and not _count(v)):
                return f"{where}: {p} must be {'a count, 0 or more' if p in STRUCTURAL else 'a number'}"
    for part, node in figures.items():
        if (bad := walk(node, part)):
            return bad
    return None


def fields(place: Place) -> dict[str, int | float]:
    """Every numeric field of a place's figures and every rule's count, by path (`kernel/registers`, `loops/<key>/live`,
    `window/period`, `rules/<rule>`): in the figures' own order, then the rules'. Text (a loop's `at`, the window's `anchor`)
    is not a field."""
    out: dict[str, int | float] = {}
    def walk(node, path):
        for k, v in node.items():
            p = f"{path}/{k}" if path else k
            if isinstance(v, dict):
                walk(v, p)
            elif _number(v):
                out[p] = v
    walk(place.figures or {}, "")
    out.update({f"rules/{name}": r.count for name, r in place.rules.items()})
    return out


def report(manifest: Manifest, pattern: str | None = None) -> str:
    """The text of `--text`: for each kernel (those whose name matches `pattern`, all if none) every field, one `path = value` a
    line: what a bound or a hash names, what coverage lists. A static manifest's kernel says how its loops are keyed and each
    loop's `at` note follows its key (its class counts that are zero are left out of this text; they are in the file); a run's
    says its window's anchor. The lines the manifest says close the text."""
    out = []
    for place in manifest.places:
        if place.figures is None or (pattern and not re.search(pattern, place.place)):
            continue
        figures = place.figures
        if "loops_keyed_by" in figures:
            out.append(f"{place.place}  (loops keyed by {figures['loops_keyed_by']})")
        else:
            anchor = figures.get("window", {}).get("anchor")
            out.append(f"{place.place}  (a run" + (f"; windows anchored on {anchor}" if anchor else "") + ")")
        notes = {f"loops/{k}/": v["at"] for k, v in figures.get("loops", {}).items() if "at" in v}
        last = None
        for path, value in fields(place).items():
            if not value and "/class/" in path:
                continue
            prefix = next((p for p in notes if path.startswith(p)), None)
            if prefix and prefix != last:
                out.append(f"  # {prefix[6:-1]} at {notes[prefix]}")
            last = prefix
            out.append(f"  {path} = {value}")
    if out:
        out += list(manifest.said)
    return "\n".join(out) + ("\n" if out else "")


def _rule_data(r: RuleFindings) -> dict:
    return {"count": r.count, "bound": r.bound, "default": r.default, "declarable": r.declarable, "message": r.message,
            "fix": r.fix, "sites": [{"where": s.where, "count": s.count, "what": s.what} for s in r.sites]}


def to_data(manifest: Manifest) -> dict:
    if manifest.format == RUN_FORMAT:
        return {"format": RUN_FORMAT, "said": list(manifest.said), "places": [{
            "place": p.place, "kernel": p.kernel, **({"figures": p.figures} if p.figures is not None else {}),
            "rules": {name: _rule_data(r) for name, r in p.rules.items()}} for p in manifest.places]}
    return {
        "format": FORMAT,
        "notices": manifest.notices,
        "said": list(manifest.said),
        "places": [{
            "place": p.place, "kernel": p.kernel, "name": p.name,
            **({"figures": p.figures} if p.figures is not None else {}),
            "rules": {name: _rule_data(r) for name, r in p.rules.items()},
        } for p in manifest.places],
    }


def dumps(manifest: Manifest) -> str:
    """The manifest as JSON text: the same manifest always writes the same bytes (fixed order, no timestamps)."""
    return json.dumps(to_data(manifest), indent=1, ensure_ascii=False) + "\n"


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise ManifestError(f"manifest: {what}")


def _site_count(value, what: str) -> int:
    _need(_count(value), f"{what} must be a count, 0 or more")
    return value


def _rules(p: dict) -> dict[str, RuleFindings]:
    rules = {}
    for name, r in p["rules"].items():
        where = f"{p['place']}: {name}"
        _need(isinstance(r, dict) and isinstance(r.get("sites"), list), f"{where}: needs sites")
        sites = tuple(Finding(str(s["where"]), _site_count(s["count"], f"{where}: a site's count"), str(s["what"]))
                      for s in r["sites"])
        _need(_site_count(r.get("count"), f"{where}: count") == sum(s.count for s in sites), f"{where}: count is not the sites' sum")
        default = r.get("default", 0)
        _need(default is None or _site_count(default, f"{where}: default") >= 0, f"{where}: default")
        rules[name] = RuleFindings(r["count"], sites, str(r.get("message", "")), str(r.get("fix", "")),
                                   bool(r.get("bound", False)), default, bool(r.get("declarable", True)))
    return rules


def loads(text: str) -> Manifest:
    """The manifest of a JSON text, of either format, validated: the format, every place's rules, their counts and sites (a
    rule's count is its sites' sum), a kernel's figures in its format's form (a kernel has figures, no other place has)."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ManifestError(f"manifest: not JSON ({e})") from e
    _need(isinstance(doc, dict) and doc.get("format") in (FORMAT, RUN_FORMAT),
          f"not a manifest (format {FORMAT!r} or {RUN_FORMAT!r})")
    run = doc["format"] == RUN_FORMAT
    _need(isinstance(doc.get("places"), list) and isinstance(doc.get("said", []), list), "places and said are lists")
    notices = doc.get("notices")
    _need(notices is None or (not run and isinstance(notices, int)), "notices is a count or null (a run manifest has none)")
    places = []
    for i, p in enumerate(doc["places"]):
        _need(isinstance(p, dict) and isinstance(p.get("place"), str) and isinstance(p.get("rules"), dict),
              f"place {i} needs a name and rules")
        rules = _rules(p)
        found = p.get("figures")
        _need(bool(p.get("kernel", False)) == (found is not None), f"{p['place']}: a kernel has figures and no other place does")
        why = None if found is None else (valid_run if run else valid)(found, p["place"])
        _need(why is None, why or "")
        places.append(Place(p["place"], bool(p.get("kernel", False)), rules, None if run else p.get("name"), found))
    return Manifest(tuple(places), tuple(str(s) for s in doc.get("said", [])), notices, doc["format"])
