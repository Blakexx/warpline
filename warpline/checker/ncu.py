"""LAYER 1, THE PROFILER'S EXPORTS: Nsight Compute's CSV text in, plain data out. A reader never runs the profiler.

    details_rules(text)   `ncu --import R --page details --csv`                   -> [RuleResult]
                          (the built-in rules' results: Rule Name / Type / Description / Estimated Speedup)
    source_rows(text)     `ncu --import R --page source --print-source sass --csv` -> SourceProfile
                          (one row an instruction: executed count, shared wavefronts, stall samples by reason)
    profiles_of(texts)    the two exports of one profile, matched by kernel name  -> [Profile]

The two exports are told apart by their header: the source page starts with a `Kernel Name` line, the details page has a
`Rule Name` column. THE EXPORTS ARE DECLARED HERE ONCE: `PAGES` holds the flags each page is printed with, the one place the
runner (`warpline run --profile`, `runner.export`) and these readers agree on. A COLUMN A READER READS IS REQUIRED
(`DETAILS_COLUMNS`, `SOURCE_COLUMNS`): a page without one is refused, naming it, never read as 0. The source page's columns are
the metrics of the sections collected (a profile without `SOURCE_SECTION` exports no address space and no wavefronts), so the
refusal names that section. The shared-memory wavefronts (`SHARED_COLUMNS`) are required where an instruction's address space
is shared; a kernel with no shared-memory instruction has none to export, and its wavefronts are 0 because no instruction
takes one. A stall reason is a column too: the reasons a page exports are listed (`SourceProfile.reasons`), so a reason it
does not export is told apart from a reason that sampled nothing.
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
    """One result's source page: the kernel, a row an instruction, and the stall reasons the page exports (a row's `stalls`
    leaves out a reason that sampled nothing; a reason not in `reasons` was not exported)."""
    kernel: str
    rows: tuple[SourceRow, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Profile:
    """What a profile says of one kernel: the rule results (all launches of the name) and the source rows."""
    kernel: str
    rules: tuple[RuleResult, ...] = ()
    source: tuple[SourceRow, ...] = ()
    has_details: bool = False
    has_source: bool = False
    #: the stall reasons the source page(s) export
    reasons: tuple[str, ...] = ()


#: the flags `ncu --import REPORT` prints each page with: the one place the runner and the readers agree on the exports
PAGES = {
    "details": ("--page", "details", "--csv"),
    "source": ("--page", "source", "--print-source", "sass", "--csv"),
}
#: the section whose metrics are the source page's columns: a profile that did not collect it has no source page to read
SOURCE_SECTION = "SourceCounters"
#: the columns the details page must have: every one `details_rules` reads
DETAILS_COLUMNS = ("ID", "Kernel Name", "Section Name", "Rule Name", "Rule Type", "Rule Description", "Estimated Speedup Type",
                   "Estimated Speedup")
#: the columns the source page must have (the stall reasons are the `stall_*` columns present)
SOURCE_COLUMNS = ("Address", "Source", "Instructions Executed", "Address Space")
#: the columns of the shared-memory wavefronts, which the page must have where an instruction's address space is shared:
#: Nsight Compute exports none of them for a kernel with no shared-memory instruction (seen with 2025.3 and the full set)
SHARED_COLUMNS = ("L1 Wavefronts Shared", "L1 Wavefronts Shared Ideal", "L1 Wavefronts Shared Excessive")


def _missing(head, need) -> list[str]:
    return [n for n in need if n not in head]


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
    if missing := _missing(head, DETAILS_COLUMNS):
        raise ReaderError(f"the details page lacks the column(s) {missing}: a profiler version that does not export them is "
                          f"refused, never read as empty")

    def cell(row, name):
        i = head[name]
        return row[i].strip() if i < len(row) else ""    # a metric's row ends before the rule's columns

    found = []
    for row in table[1:]:
        if cell(row, "Rule Name"):
            speed = cell(row, "Estimated Speedup")
            found.append(RuleResult(cell(row, "Kernel Name"), cell(row, "ID"), cell(row, "Section Name"),
                                    cell(row, "Rule Name"), cell(row, "Rule Type"), cell(row, "Rule Description"),
                                    cell(row, "Estimated Speedup Type"), float(speed) if speed else None))
    return found


def _blocks(table: list[list[str]]) -> list[list[list[str]]]:
    """The page's blocks: each from a `Kernel Name` line to the next."""
    blocks: list[list[list[str]]] = []
    for row in table:
        if row[0] == "Kernel Name":
            blocks.append([])
        if not blocks:
            raise ReaderError("the source page starts with a `Kernel Name` line")
        blocks[-1].append(row)
    return blocks


def source_rows(text: str) -> SourceProfile:
    """The source page of ONE result (`--page source --print-source sass --csv`, `--launch-skip I --launch-count 1` for
    result I of a report of several): its kernel's name and a row an instruction. Nsight Compute may print a result's block
    more than once: identical copies are read once, blocks that differ are refused (a page of several results)."""
    blocks = _blocks(_table(text))
    if not blocks:
        raise ReaderError("the source page starts with a `Kernel Name` line and an `Address,Source,...` header")
    if any(b != blocks[0] for b in blocks[1:]):
        raise ReaderError(f"the source page holds {len(blocks)} blocks that differ: a page is one result's (export result I "
                          f"with `--launch-skip I --launch-count 1`)")
    table = blocks[0]
    if len(table) < 2 or len(table[0]) < 2 or table[1][:2] != ["Address", "Source"]:
        raise ReaderError("the source page starts with a `Kernel Name` line and an `Address,Source,...` header")
    head = {name: i for i, name in enumerate(table[1])}
    if missing := _missing(head, SOURCE_COLUMNS):
        raise ReaderError(f"the source page lacks the column(s) {missing}: they are the metrics of the {SOURCE_SECTION} "
                          f"section (profile with it), and a page without them is refused, never read as 0")
    if short := [row[:2] for row in table[2:] if len(row) < len(head)]:
        raise ReaderError(f"a source row has fewer cells than the header's {len(head)}: {short[0]}")
    shared = any(row[head["Address Space"]].strip() == "Shared" for row in table[2:])
    if shared and (missing := _missing(head, SHARED_COLUMNS)):
        raise ReaderError(f"the source page has shared-memory instructions and lacks the column(s) {missing}: a page "
                          f"without them is refused, never read as 0")
    reasons = {name[len("stall_"):]: i for name, i in head.items() if name.startswith("stall_") and "(" not in name}
    rows = []
    for row in table[2:]:
        try:
            address = int(row[0], 16)
        except ValueError as e:
            raise ReaderError(f"an address was expected, found {row[0]!r}") from e
        # no shared-memory instruction: none takes a wavefront, and the page exports no wavefront column
        wave = [_count(row[head[name]]) if name in head else 0 for name in SHARED_COLUMNS]
        rows.append(SourceRow(address, row[1].strip(), _count(row[head["Instructions Executed"]]),
                              row[head["Address Space"]].strip(), *wave,
                              {r: _count(row[i]) for r, i in reasons.items() if _count(row[i])}))
    return SourceProfile(table[0][1], tuple(rows), tuple(reasons))


def details_results(text: str) -> list[tuple[str, str]]:
    """The results a details page holds, `(ID, kernel name)` in the order they first appear (a result no rule fired on is
    still named): result I of the report is the I-th."""
    table = _table(text)
    if not table or _missing(table[0], ("ID", "Kernel Name")):
        raise ReaderError("the details page needs a header with ID and Kernel Name columns (`--page details --csv`)")
    i, k = table[0].index("ID"), table[0].index("Kernel Name")
    return list(dict.fromkeys((row[i], row[k]) for row in table[1:] if len(row) > max(i, k)))


def profiles_of(texts: list[str]) -> list[Profile]:
    """The exports of a profile, any number of either page: a Profile a kernel name, in the order names first appear."""
    rules: dict[str, list[RuleResult]] = {}
    sources: dict[str, tuple[SourceRow, ...]] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    detailed: list[str] = []
    for text in texts:
        if kind_of(text) == "details":
            detailed += [kernel for _, kernel in details_results(text)]
            for result in details_rules(text):
                rules.setdefault(result.kernel, []).append(result)
        else:
            sp = source_rows(text)
            sources[sp.kernel] = sources.get(sp.kernel, ()) + sp.rows
            reasons[sp.kernel] = tuple(dict.fromkeys([*reasons.get(sp.kernel, ()), *sp.reasons]))
    return [Profile(name, tuple(rules.get(name, ())), sources.get(name, ()), name in detailed, name in sources,
                    reasons.get(name, ()))
            for name in dict.fromkeys([*detailed, *sources])]
