"""THE STATIC MANIFEST'S COLLECTION, and a run manifest's rules: what the checker writes into the manifests (`warpline/manifest.py`,
the common module, holds their form, their reading and writing). It is data about a build, not a verdict on it.

`collect` makes the static manifest (`warpline-manifest/1`) of a binary: per PLACE (a kernel, or the function or file a notice
names) each rule's count and sites, and for a kernel its FIGURES (`of_kernel`). The rules of the model and the notices (each a row
of the ledger, which counts it under a rule) are the writers. `run_rules` gives a run manifest (`warpline-run-manifest/1`) its
`rules`: the profile rules on a profiler's exports, at the place the caller names. A profile rule is never in the static manifest.

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

Beside them, the checker's side of the declarations: the rule names a declaration may name (`rule_names`) and the collection
and comparison of a binary's kernels in one call (`check`, `figures`, `check_notices`).
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection

from .. import declare
from ..manifest import KEYS, Finding, Manifest, Place, RuleFindings, RunPart
from . import held as held_mod
from . import notices as notices_mod
from .model import INSTRUCTION_CLASSES, Kernel, Loop
from .ncu import Profile
from .rule import ProfileRule, Rule

_KIND = ("load", "store")
#: the one notice rule no declaration can lift: an unknown notice is a decision, never a count
NO_ROW = "notice_without_row"


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


def same_kernel(profiled: str, demangled: str) -> bool:
    """Whether the profiler's name of a kernel (`void ns::k<0>(Params)`) names the kernel `c++filt` calls `demangled`: the two
    compared by the name before the parameter list, with the profiler's `void ` and `(bool)` decorations and spaces removed."""
    def core(name: str) -> str:
        name = name.removeprefix("void ").replace("(bool)", "").replace(" ", "")
        return name.split("(")[0]
    return core(profiled) == core(demangled)


def match_profile(profile: Profile, kernels: list[Kernel]) -> Kernel | None:
    """The kernel a profile is of (`same_kernel`), where exactly one is."""
    found = [k for k in kernels if same_kernel(profile.kernel, k.demangled)]
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
            ledger: list[notices_mod.Row] | None = None, said: Collection[str] = ()) -> Manifest:
    """The static manifest of a binary: every rule of `rules` on every kernel; with `notices` (the build's messages, read),
    each counted under its row's rule of `ledger` (default: the base ledger). `said` are lines the manifest carries about what
    was not run."""
    places: dict[str, dict] = {}
    for kernel in kernels:
        places[kernel.demangled] = {"kernel": kernel, "rules": _model_findings(kernel, rules), "figures": of_kernel(kernel)}
    if notices is not None:
        pretty = {k.name: k.demangled for k in kernels}
        for place, found in _notice_rules(notices, ledger if ledger is not None else notices_mod.base_ledger(), pretty).items():
            places.setdefault(place, {"kernel": None, "rules": {}})["rules"].update(dict(sorted(found.items())))
    return Manifest(tuple(Place(p, e["kernel"] is not None, e["rules"], e["kernel"].name if e["kernel"] else None, e.get("figures"))
                          for p, e in places.items()),
                    tuple(said), len(notices) if notices is not None else None)


def run_rules(profiles: list[Profile], place: str, profile_rules: dict[str, ProfileRule]) -> RunPart:
    """A run manifest's rules: every profile rule on every profile. A profile of the kernel `place` (`same_kernel`; a lone
    profile is paired whatever it is called) gives the rules at the place; any other profile gives a place of its own, under
    the profiler's kernel name, said to match no place. A rule whose page a profile has not got is not run on it, and said.
    A site keeps its rule's `where` (an instruction's offset in the kernel, a profiler's rule name)."""
    mine: dict[str, RuleFindings] = {}
    others: dict[str, dict[str, RuleFindings]] = {}
    said = []
    for profile in profiles:
        matched = len(profiles) == 1 or same_kernel(profile.kernel, place)
        where = place if matched else profile.kernel
        if not matched:
            said.append(f"note: the profile of {profile.kernel!r} matches no place ({place!r})")
        entry = mine if matched else others.setdefault(profile.kernel, {})
        for rule in profile_rules.values():
            if not (profile.has_details if rule.needs == "details" else profile.has_source):
                said.append(f"note: profile rule {rule.name} not run on {where!r}: the profile has no "
                            f"{'details page' if rule.needs == 'details' else 'source page'}")
                continue
            found = rule.check(profile)
            entry[rule.name] = RuleFindings(sum(f.count for f in found), tuple(found), rule.refusal, rule.fix, bound=rule.bound)
    return RunPart(rules=mine, others=tuple(Place(name, False, rules) for name, rules in others.items()), said=tuple(said))


def rule_names(ledger: list[notices_mod.Row] | None = None) -> list[str]:
    """Every rule name a declaration (and a note) may name, sorted: the registry's rules, the profile rules, and the
    rules the notice ledger's rows count under (the base ledger's, or `ledger`'s). A consumer lints that each
    has a note by comparing this list with the keys of its notes."""
    from .rules import profile_registry, registry
    return sorted({*registry(), *profile_registry(), *notices_mod.row_rules(ledger or notices_mod.base_ledger())})


def unmatched(declarations: list[declare.Declaration], kernels: list[Kernel],
              places: Collection[str] = ()) -> list[declare.Declaration]:
    """The declarations whose regex names no kernel (nor any of `places`, where a notice was read): stale, reported and never a
    refusal."""
    return [d for d in declarations
            if not any(d.kernel.search(k.demangled) for k in kernels) and not any(d.kernel.search(p) for p in places)]


def check(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[declare.Declaration],
          notes: dict[str, str] | None = None) -> list[declare.Refusal]:
    """Every rule on every kernel, compared (`collect`, `declare.compare`): the refusals."""
    return declare.compare(collect(kernels, rules), declarations, notes).refusals


def figures(kernels: list[Kernel], rules: dict[str, Rule], declarations: list[declare.Declaration]) -> list[str]:
    """The lines for the bound rules that found something on a kernel where no bound is declared: the figure and what it
    is of. Never a refusal: a project declares a bound (`allow`) to have the figure held."""
    return declare.compare(collect(kernels, rules), declarations).figures


def check_notices(read: list[notices_mod.Notice], ledger: list[notices_mod.Row], declarations: list[declare.Declaration],
                  names: dict[str, str] | None = None, notes: dict[str, str] | None = None) -> list[declare.Refusal]:
    """A build's notices against the ledger and the declarations: the refusals. Each notice is counted under its row's
    rule at its place (`collect`), a notice with no row is refused whole; `names` maps a function as the toolkit
    printed it to its demangled name."""
    by_place = _notice_rules(read, ledger, names or {})
    manifest = Manifest(tuple(Place(p, False, rules) for p, rules in by_place.items()))
    return declare.compare(manifest, declarations, notes).refusals


def notice_places(read: list[notices_mod.Notice], names: dict[str, str]) -> list[str]:
    """Where the notices are (a function's demangled name, or `file(line)`): what a declaration's regex may name."""
    return sorted({names.get(n.function, n.function) if n.function else n.where for n in read if n.function or n.where})
