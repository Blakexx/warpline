"""LAYER 1, THE PROFILER'S EXPORTS: Nsight Compute's CSV text in, plain data out. A reader never runs the profiler.

    details_rules(text)   `ncu --import R --page details --csv`                   -> [RuleResult]
                          (the built-in rules' results: Rule Name / Type / Description / Estimated Speedup)
    source_rows(text)     `ncu --import R --page source --print-source sass --csv` -> SourceProfile
                          (one row an instruction: executed count, shared wavefronts, stall samples by reason)
    profiles_of(texts)    the two exports of one profile, matched by kernel name  -> [Profile]

The two exports are told apart by their header: the source page starts with a `Kernel Name` line, the details page has a
`Rule Name` column. A column a profiler version does not export reads as 0 (a count) or absent (a share).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from .readers import ReaderError


@dataclass(frozen=True)
class RuleResult:
    """One result of a built-in rule: `type` is `OPT` (an optimization), `WRN` (a warning) or `INF` (information);
    `speedup` is the estimated speedup in percent, None where the rule gives none, `speedup_type` `global` or `local`."""
    kernel: str
    launch: str
    section: str
    name: str
    type: str
    description: str
    speedup_type: str
    speedup: float | None


@dataclass(frozen=True)
class SourceRow:
    """One instruction of the source page. `address` is the profiler's; `executed` the warp-level count of its executions;
    `wavefronts`, `ideal` and `excess` the shared-memory wavefronts the instruction took, the conflict-free count, and
    the difference as the profiler reports it; `stalls` the stall samples by reason (`long_sb`, `barrier`, ...)."""
    address: int
    source: str
    executed: int
    space: str
    wavefronts: int
    ideal: int
    excess: int
    stalls: dict[str, int]


@dataclass(frozen=True)
class SourceProfile:
    kernel: str
    rows: tuple[SourceRow, ...]


@dataclass(frozen=True)
class Profile:
    """What a profile says of one kernel: the rule results (all launches of the name) and the source rows."""
    kernel: str
    rules: tuple[RuleResult, ...] = ()
    source: tuple[SourceRow, ...] = ()
    has_details: bool = False
    has_source: bool = False


def _count(value: str) -> int:
    value = value.replace(",", "").strip()
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError as e:
        raise ReaderError(f"a count was expected, found {value!r}") from e


def _table(text: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(text)) if row]


def kind_of(text: str) -> str:
    """`source` or `details`: which export a CSV text is."""
    table = _table(text)
    if table and table[0][0] == "Kernel Name":
        return "source"
    if table and "Rule Name" in table[0]:
        return "details"
    raise ReaderError("not a Nsight Compute export this reader knows (the details page needs a Rule Name column; "
                      "the source page starts with a Kernel Name line)")


def details_rules(text: str) -> list[RuleResult]:
    """The rule results of the details page: every row with a Rule Name, in file order."""
    table = _table(text)
    if not table or "Rule Name" not in table[0]:
        raise ReaderError("the details page needs a header with a Rule Name column (`--page details --csv`)")
    head = {name: i for i, name in enumerate(table[0])}
    need = ("Kernel Name", "Rule Name", "Rule Type", "Rule Description")
    if missing := [n for n in need if n not in head]:
        raise ReaderError(f"the details page lacks the column(s) {missing}")

    def cell(row, name):
        i = head.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    found = []
    for row in table[1:]:
        if cell(row, "Rule Name"):
            speed = cell(row, "Estimated Speedup")
            found.append(RuleResult(cell(row, "Kernel Name"), cell(row, "ID"), cell(row, "Section Name"),
                                    cell(row, "Rule Name"), cell(row, "Rule Type"), cell(row, "Rule Description"),
                                    cell(row, "Estimated Speedup Type"), float(speed) if speed else None))
    return found


def source_rows(text: str) -> SourceProfile:
    """The source page of one kernel (`--page source --print-source sass --csv`): its name and a row an instruction."""
    table = _table(text)
    if len(table) < 2 or table[0][0] != "Kernel Name" or table[1][:2] != ["Address", "Source"]:
        raise ReaderError("the source page starts with a `Kernel Name` line and an `Address,Source,...` header")
    head = {name: i for i, name in enumerate(table[1])}

    def num(row, name):
        i = head.get(name)
        return _count(row[i]) if i is not None and i < len(row) else 0

    def word(row, name):
        i = head.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    reasons = {name[len("stall_"):]: i for name, i in head.items() if name.startswith("stall_") and "(" not in name}
    rows = []
    for row in table[2:]:
        try:
            address = int(row[0], 16)
        except ValueError as e:
            raise ReaderError(f"an address was expected, found {row[0]!r}") from e
        wavefronts, ideal = num(row, "L1 Wavefronts Shared"), num(row, "L1 Wavefronts Shared Ideal")
        excess = num(row, "L1 Wavefronts Shared Excessive") if "L1 Wavefronts Shared Excessive" in head \
            else max(wavefronts - ideal, 0)
        rows.append(SourceRow(address, row[1].strip(), num(row, "Instructions Executed"), word(row, "Address Space"),
                              wavefronts, ideal, excess,
                              {r: _count(row[i]) for r, i in reasons.items() if i < len(row) and _count(row[i])}))
    return SourceProfile(table[0][1], tuple(rows))


def details_kernels(text: str) -> list[str]:
    """The kernel names a details page holds, first appearance first (a kernel no rule fired on is still named)."""
    table = _table(text)
    i = table[0].index("Kernel Name") if table and "Kernel Name" in table[0] else None
    return [] if i is None else list(dict.fromkeys(row[i] for row in table[1:] if i < len(row)))


def profiles_of(texts: list[str]) -> list[Profile]:
    """The exports of a profile, any number of either page: a Profile a kernel name, in the order names first appear."""
    rules: dict[str, list[RuleResult]] = {}
    sources: dict[str, tuple[SourceRow, ...]] = {}
    detailed: list[str] = []
    for text in texts:
        if kind_of(text) == "details":
            detailed += details_kernels(text)
            for result in details_rules(text):
                rules.setdefault(result.kernel, []).append(result)
        else:
            sp = source_rows(text)
            sources[sp.kernel] = sources.get(sp.kernel, ()) + sp.rows
    return [Profile(name, tuple(rules.get(name, ())), sources.get(name, ()), name in detailed, name in sources)
            for name in dict.fromkeys([*detailed, *sources])]
