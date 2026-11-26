"""THE MANIFEST: what a checker run found and measured, written once per binary as one stable JSON file, and its reader. It is data
about a build, not a verdict on it. Per PLACE (a kernel, or the function or file a notice or a profile names) it holds each rule's
count and sites (the detectors' findings) and, for a kernel, DESCRIPTIVE FIGURES: totals and, for each loop, what it holds and does.
The rules of the model, the notices (each a row of the ledger, which counts it under a rule) and the profile rules are the writers;
the checkers (`declare.compare`, `checkers.py`, `warpline compare`) are the only readers that judge.

    {"format": "warpline-manifest/1",
     "notices": 6,                                  # notices read from the build's messages; null where none were given
     "said": ["note: profile rules not run ..."],   # what was not run
     "places": [{"place": "sample::tile()", "kernel": true, "name": "_ZN6sample4tileEv",
                 "figures": {...},                  # a kernel's figures (below); a place that is not a kernel has none
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

A kernel's FIGURES have no instruction address in any field a checker can read (addresses move with every edit):

    {"kernel": {"instructions": 212, "registers": 40, "stack_bytes": 0, "shared_bytes": 0, "barriers": 0, "tensor_steps": 0},
     "loops_keyed_by": "source file and position",         # or "nesting path"
     "loops": {"held_many.cu/1": {"at": "held_many.cu:15", "copies": 1, "live": 30, "read": 5, "instructions": 9,
                                  "class": {"tensor": 0, "shared_load": 0, ..., "other": 1},
                                  "global_loads": 1, "global_stores": 0, "shared_loads": 0, "shared_stores": 0}}}

A LOOP's key must not move when code above it is edited, so it holds no line number. The disassembly's source marks give a
file and a line (and, for inlined code, the file and line it was inlined at) and NO function name, so where a listing with source
marks was given the key is the innermost source file of the loop's header and the loop's POSITION: its rank, by line, among the
loops of that file at the same nesting (`held_many.cu/2.1`: the first loop inside the second loop of the file's loops). Compiled
copies of one source loop (the same file and line: an unrolled or inlined one) are one entry holding the most of each figure and
the number of copies. `at` (`held_many.cu:15`) is a note for a person: text, never a field, never hashed. A rank moves only when a
loop is added or removed before it in its file, a line number moves with every edit above. Without a listing (or if a loop of the
kernel has no line) every loop of the kernel is keyed by its nesting path (`loop 2.1`), and `loops_keyed_by` says so.
`live` is the vector registers live at the loop's top, `read` how many of those an instruction of the loop reads (`held.py`),
`instructions` the loop's instructions (the dead ones excluded, inner loops included) and `class` their counts by the model's unit
classes (`model.INSTRUCTION_CLASSES`, names with `_` for spaces and slashes, `other` for none). The loads and stores are those of a
global or shared instruction (atomics and asynchronous copies excluded).

A FIELD is a number of the figures or the count of a rule, named by a path of the names above joined by `/`:
`kernel/registers`, `loops/held_many.cu/1/live`, `loops/loop 2/class/tensor`, `rules/stack_memory`. `fields` flattens a place to
`{path: number}`: what the checkers (`checkers.py`) judge and the coverage check lists.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

from . import held as held_mod
from . import notices as notices_mod
from .model import INSTRUCTION_CLASSES, Kernel, Loop
from .ncu import Profile
from .rule import Finding, ProfileRule, Rule

KEYS = ("source file and position", "nesting path")
_KIND = ("load", "store")


def _class(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name)


#: the classes a loop's instructions are counted in, in the model's order, then `other`
CLASSES = tuple(_class(n) for n in INSTRUCTION_CLASSES) + ("other",)


def _enclosing(kernel: Kernel) -> dict[str, Loop | None]:
    """Loop header -> the smallest loop around it (None for an outermost one)."""
    out = {}
    for lp in kernel.loops:
        inside = [o for o in kernel.loops if o is not lp and lp.blocks < o.blocks]
        out[lp.header] = min(inside, key=lambda o: len(o.blocks), default=None)
    return out


def _paths(kernel: Kernel) -> dict[str, str]:
    """Loop header -> its nesting path (`loop 2.1`): the 1-based place among the loops that share its enclosing loop, in
    address order, behind the enclosing loop's own."""
    owner = _enclosing(kernel)
    names, counter = {}, {}
    for lp in kernel.loops:                                   # address order
        parent = owner[lp.header]
        up = names[parent.header] if parent else ""
        counter[up] = counter.get(up, 0) + 1
        names[lp.header] = f"{up}.{counter[up]}" if up else str(counter[up])
    return {header: "loop " + path for header, path in names.items()}


def _positions(kernel: Kernel) -> dict[str, tuple[str, str]] | None:
    """Loop header -> (its key, its note) where every loop has a source line, else None. A SOURCE LOOP is a file and line; its
    parent is the source loop of the nearest compiled loop around its first copy that is of the same file and another line; its
    position is its rank, by line, among the source loops of the same file and parent."""
    owner = _enclosing(kernel)
    marks = {lp.header: kernel.lines.get(lp.address) for lp in kernel.loops}
    if not kernel.lines or not all(marks.values()):
        return None
    ident = {h: (m[0].rsplit("/", 1)[-1], m[1]) for h, m in marks.items()}
    parent: dict[tuple, tuple | None] = {}
    for lp in kernel.loops:                                   # the first copy (lowest address) of a source loop decides
        me = ident[lp.header]
        if me in parent:
            continue
        up = owner[lp.header]
        while up is not None and (ident[up.header][0] != me[0] or ident[up.header] == me):
            up = owner[up.header]
        parent[me] = ident[up.header] if up is not None else None
    rank = {}
    for me, up in parent.items():
        rank[me] = 1 + sum(o[0] == me[0] and parent[o] == up and o[1] < me[1] for o in parent)
    path: dict[tuple, str] = {}

    def of(me: tuple, seen=()) -> str:
        if me not in path:
            up = parent[me]
            path[me] = f"{of(up, (*seen, me))}.{rank[me]}" if up is not None and up not in seen else str(rank[me])
        return path[me]

    return {h: (f"{me[0]}/{of(me)}", f"{me[0]}:{me[1]}") for h, me in ident.items()}


def _figures(kernel: Kernel, lp: Loop, hold) -> dict:
    body = [i for b in sorted(lp.blocks) for i in kernel.block(b).instructions if not i.dead]
    inside = {i.address for i in body}
    by_class = dict.fromkeys(CLASSES, 0)
    for i in body:
        by_class[_class(i.klass)] += 1
    seen = [a for a in kernel.accesses if a.instruction.address in inside and a.kind in _KIND]
    count = lambda space, kind: sum(a.space == space and a.kind == kind for a in seen)
    return {"copies": 1, "live": hold.live, "read": sum(g.used for g in hold.groups), "instructions": len(body),
            "class": by_class, "global_loads": count("global", "load"), "global_stores": count("global", "store"),
            "shared_loads": count("shared", "load"), "shared_stores": count("shared", "store")}


def _merge(a: dict, b: dict) -> dict:
    """Two compiled copies of one source loop: the most of each figure, the copies added, the note kept."""
    return {k: (_merge(v, b[k]) if isinstance(v, dict) else v if isinstance(v, str) else a[k] + b[k] if k == "copies"
                else max(v, b[k])) for k, v in a.items()}


def of_kernel(kernel: Kernel) -> dict:
    """The figures of a kernel (the nested form above)."""
    held = {h.loop.header: h for h in held_mod.holds(kernel)}
    keyed = _positions(kernel)
    paths = _paths(kernel)
    loops: dict[str, dict] = {}
    for lp in kernel.loops:
        key, note = keyed[lp.header] if keyed else (paths[lp.header], None)
        figures = {**({"at": note} if note else {}), **_figures(kernel, lp, held[lp.header])}
        loops[key] = _merge(loops[key], figures) if key in loops else figures
    return {
        "kernel": {"instructions": len(kernel.instructions), "registers": kernel.registers, "stack_bytes": kernel.stack_bytes,
                   "shared_bytes": kernel.shared_bytes, "barriers": len(kernel.barriers), "tensor_steps": len(kernel.steps)},
        "loops_keyed_by": KEYS[0] if keyed is not None else KEYS[1],
        "loops": dict(sorted(loops.items())),
    }


def valid(figures, where: str) -> str | None:
    """Why figures read from a file are not figures (None where they are): the shape of the form above, every figure a count and
    the only text a loop's `at`."""
    count = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0
    if not (isinstance(figures, dict) and isinstance(figures.get("kernel"), dict) and isinstance(figures.get("loops"), dict)
            and figures.get("loops_keyed_by") in KEYS):
        return f"{where}: figures need kernel, loops and loops_keyed_by ({' or '.join(KEYS)})"
    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                if (bad := walk(v, f"{path}/{k}")):
                    return bad
            elif not (count(v) or (k == "at" and isinstance(v, str) and path.startswith("loops/"))):
                return f"{where}: {path}/{k} must be a count, 0 or more"
    return walk(figures["kernel"], "kernel") or walk(figures["loops"], "loops")


def fields(place: "Place") -> dict[str, int]:
    """Every numeric field of a place's figures and every rule's count, by path: `kernel/registers`, `loops/<key>/live`,
    `loops/<key>/class/<class>`, `rules/<rule>`. In the figures' own order, then the rules' (the registry's). The loops' `at`
    notes are text, not fields."""
    out: dict[str, int] = {}
    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, f"{path}/{k}")
            elif isinstance(v, int):
                out[f"{path}/{k}"] = v
    walk(place.figures["kernel"], "kernel")
    walk(place.figures["loops"], "loops")
    out.update({f"rules/{name}": r.count for name, r in place.rules.items()})
    return out


def report(manifest: "Manifest", pattern: str | None = None) -> str:
    """The text of `warpline manifest --text`: for each kernel (those whose demangled name matches `pattern`, all if none) how its
    loops are keyed and every field, one `path = value` a line: what a bound or a hash names, what coverage lists; each loop's
    `at` note follows its key. The class counts that are zero are left out of this text (they are in the file)."""
    out = []
    for place in manifest.places:
        if place.figures is None or (pattern and not re.search(pattern, place.place)):
            continue
        out.append(f"{place.place}  (loops keyed by {place.figures['loops_keyed_by']})")
        notes = {f"loops/{k}/": v["at"] for k, v in place.figures["loops"].items() if "at" in v}
        last = None
        for path, value in fields(place).items():
            if not value and "/class/" in path:
                continue
            prefix = next((p for p in notes if path.startswith(p)), None)
            if prefix and prefix != last:
                out.append(f"  # {prefix[6:-1]} at {notes[prefix]}")
            last = prefix
            out.append(f"  {path} = {value}")
    return "\n".join(out) + ("\n" if out else "")


FORMAT = "warpline-manifest/1"
#: the one notice rule no declaration can lift: an unknown notice is a decision, never a count
NO_ROW = "notice_without_row"


class ManifestError(ValueError):
    """A manifest file that is not valid."""


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
    name: str | None = None
    #: a kernel's figures (`of_kernel`); None for any other place
    figures: dict | None = None


@dataclass(frozen=True)
class Manifest:
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
                  names: dict[str, str]) -> dict[str, dict[str, RuleFindings]]:
    """A build's notices as rule manifest by place: each notice is counted under its row's rule at the function it names
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
    by_place: dict[str, dict[str, RuleFindings]] = {}
    for (place, rule), found in known.items():
        #: a row matching under any code names the notice's own code (13.5 renumbers the causes 13.0 coded)
        what = Counter((row, row.code or n.code, shown(n)) for row, n in found)
        rows = list(dict.fromkeys(row for row, _ in found))
        sites = tuple(Finding(place, c, f"{row.tool} {code or 'no code'}: {w}") for (row, code, w), c in what.items())
        by_place.setdefault(place, {})[rule] = RuleFindings(
            sum(s.count for s in sites), sites, " ".join(dict.fromkeys(r.reason for r in rows)),
            "; ".join(dict.fromkeys(r.fix for r in rows if r.fix)) or "read the notice's text for its cause",
            default=_lowest([r.allow for r in rows]))
    for place, groups in unknown.items():
        sites = tuple(Finding(found[0].where or place, len(found), "\n      ".join(found[0].raw.splitlines()))
                      for found in groups.values())
        by_place.setdefault(place, {})[NO_ROW] = RuleFindings(
            sum(s.count for s in sites), sites,
            "the toolkit printed a notice the ledger has no row for (" + "; ".join(
                f"{t} {c or 'with no code'}" for t, c, _ in groups) + ")",
            "add a row to the project's ledger (--ledger) with a rule, the notice's default allowance and the reason for it",
            declarable=False)
    return by_place


def _model_findings(kernel: Kernel, rules: dict[str, Rule]) -> dict[str, RuleFindings]:
    out = {}
    for rule in rules.values():
        found = rule.check(kernel)
        out[rule.name] = RuleFindings(sum(f.count for f in found), tuple(found), rule.refusal, rule.fix, bound=rule.bound)
    return out


def collect(kernels: list[Kernel], rules: dict[str, Rule], *, notices: list[notices_mod.Notice] | None = None,
            ledger: list[notices_mod.Row] | None = None, profiles: list[Profile] | None = None,
            profile_rules: dict[str, ProfileRule] | None = None, said: Collection[str] = ()) -> Manifest:
    """The manifest of a binary: every rule of `rules` on every kernel; with `notices` (the build's messages, read), each
    counted under its row's rule of `ledger` (default: the base ledger); with `profiles`, every profile rule on every
    profile, at the matched kernel (the profiler's own name where none matches). `said` are lines the manifest carry about
    what was not run."""
    places: dict[str, dict] = {}
    for kernel in kernels:
        places[kernel.demangled] = {"kernel": kernel, "rules": _model_findings(kernel, rules), "figures": of_kernel(kernel)}
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
                entry["rules"][rule.name] = RuleFindings(sum(f.count for f in found), tuple(found), rule.refusal, rule.fix,
                                                      bound=rule.bound)
    return Manifest(tuple(Place(p, e["kernel"] is not None, e["rules"], e["kernel"].name if e["kernel"] else None, e.get("figures"))
                       for p, e in places.items()),
                 tuple(lines), len(notices) if notices is not None else None)


def to_data(manifest: Manifest) -> dict:
    return {
        "format": FORMAT,
        "notices": manifest.notices,
        "said": list(manifest.said),
        "places": [{
            "place": p.place, "kernel": p.kernel, "name": p.name,
            **({"figures": p.figures} if p.figures is not None else {}),
            "rules": {name: {"count": r.count, "bound": r.bound, "default": r.default, "declarable": r.declarable,
                             "message": r.message, "fix": r.fix,
                             "sites": [{"where": s.where, "count": s.count, "what": s.what} for s in r.sites]}
                      for name, r in p.rules.items()},
        } for p in manifest.places],
    }


def dumps(manifest: Manifest) -> str:
    """The manifest as JSON text: the same manifest always write the same bytes (fixed order, no timestamps)."""
    return json.dumps(to_data(manifest), indent=1, ensure_ascii=False) + "\n"


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise ManifestError(f"manifest: {what}")


def _count(value, what: str) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{what} must be a count, 0 or more")
    return value


def loads(text: str) -> Manifest:
    """The manifest of a JSON text, validated: the format, every place's rules, their counts and sites (a rule's count is its
    sites' sum)."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ManifestError(f"manifest: not JSON ({e})") from e
    _need(isinstance(doc, dict) and doc.get("format") == FORMAT, f"not a manifest (format {FORMAT!r})")
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
            rules[name] = RuleFindings(r["count"], sites, str(r.get("message", "")), str(r.get("fix", "")),
                                    bool(r.get("bound", False)), default, bool(r.get("declarable", True)))
        found = p.get("figures")
        _need(bool(p.get("kernel", False)) == (found is not None), f"{p['place']}: a kernel has figures and no other place does")
        why = None if found is None else valid(found, p["place"])
        _need(why is None, why or "")
        places.append(Place(p["place"], bool(p.get("kernel", False)), rules, p.get("name"), found))
    return Manifest(tuple(places), tuple(str(s) for s in doc.get("said", [])), notices)
