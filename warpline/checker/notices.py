"""THE NOTICE LEDGER. A build prints notices: the assembler's `info` and `warning` lines, the front end's warnings, bare
performance lines. Each is a fact about the code the toolkit made (a step it serialized, a wait it injected, a register
request it ignored). This module reads a build's messages (text in, data out) and matches each notice to a ROW of a
ledger that says what to do with it. A notice with no row is REFUSED, quoted whole: a new notice is a decision, never
a silence.

    row = tool, code, text, rule, allow, reason, fix

- `tool`: `ptxas`, `nvcc`, `nvlink`, `cicc`, `fatbinary`. `code`: the notice's own (`C7512`, `#549-D`), or null for a
  row that matches the text under any code (and none).
- `text`: the notice's fixed words, `{}` standing for a number or a name; a function's own name is written `'{}'`.
- `rule`: the rule the notice is counted under, at the notice's place (the function, demangled): the project's
  declarations say how many a kernel may have, exactly as for a rule of the model.
- `allow`: the row's default allowance, what a kernel may have with no declaration: 0 when absent (a refusal), a count,
  or null (accepted by default: counted, never refused until a declaration gives a number, which may be 0).
- `reason`: what the notice means for the kernel, one sentence; `fix`: what to change (may be empty).

The module ships a BASE ledger (`base_ledger.py`); a project's own ledger file, `{"rows": [...]}`, overrides or adds to
it by (tool, code, text). A notice matches a row by tool, then code (when the row names one), then text; no two rows
may match one notice.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class LedgerError(ValueError):
    """A ledger that is not valid: a bad row, or two rows that match one notice."""


class Notice(NamedTuple):
    """One notice of a build: its tool (`unreadable` for a line nothing reads), its code, its text with a function's name
    as `'{}'` (what the ledger matches), the function it names (as printed, usually mangled), `file(line)` for the front
    end's warnings, the lines it printed whole, and its message as printed."""
    tool: str
    code: str | None
    text: str
    function: str | None
    where: str | None
    raw: str
    message: str


_TOOL_LINE = re.compile(r"^(ptxas|nvcc|nvlink|cicc|fatbinary) (?:info|warning)\s*: (?:\((C\d+)\) )?(.*)$")
_NVCC_LINE = re.compile(r"^(\S.*?)\((\d+)\): (?:warning|remark)(?: (#\d+-D))?: (.*)$")
_BARE_LINE = re.compile(r"^(Potential Performance Loss: .*)$")
_REMARK_LINE = re.compile(r"^Remark: (.*)$")
_FUNCTION = re.compile(r"function '([^']+)'")


def _notice(tool: str, code: str | None, message: str, where: str | None, raw: str) -> Notice:
    text = " ".join(message.split())
    found = _FUNCTION.search(text)
    return Notice(tool, code, _FUNCTION.sub("function '{}'", text), found.group(1) if found else None, where, raw, text)


def read_notices(text: str) -> list[Notice]:
    """Every notice of a build's messages, in order: ptxas's `info`/`warning` lines with or without a `(Cnnnn)` code and
    a function name, nvcc's `file(line): warning #N-D: ...`, and bare performance lines. What follows a notice belongs to
    it: the indented lines of ptxas's per-function summary, and an nvcc warning's source excerpt and instantiations up
    to the blank line. A line nothing reads is a notice of tool `unreadable` (the ledger has no row for it)."""
    found: list[Notice] = []
    in_block = False
    for line in text.splitlines():
        if (in_block and line.strip()) or (line[:1].isspace() and found):
            found[-1] = found[-1]._replace(raw=found[-1].raw + "\n" + line)
            continue
        in_block = False
        if not line.strip():
            continue
        if m := _TOOL_LINE.match(line):
            found.append(_notice(m.group(1), m.group(2), m.group(3), None, line))
        elif m := _NVCC_LINE.match(line):
            found.append(_notice("nvcc", m.group(3), m.group(4), f"{Path(m.group(1)).name}({m.group(2)})", line))
            in_block = True
        elif m := _BARE_LINE.match(line):
            found.append(_notice("ptxas", None, m.group(1), None, line))
        elif m := _REMARK_LINE.match(line):
            found.append(_notice("nvcc", None, m.group(1), None, line))
        else:
            found.append(Notice("unreadable", None, line, None, None, line, line))
    return found


@dataclass(frozen=True)
class Row:
    tool: str
    code: str | None
    text: str
    rule: str
    allow: int | None
    reason: str
    fix: str

    @property
    def key(self) -> tuple[str, str | None, str]:
        return (self.tool, self.code, self.text)


def _row(entry: dict, where: str) -> Row:
    if not isinstance(entry, dict):
        raise LedgerError(f"{where}: a row is an object")
    unknown = set(entry) - {"tool", "code", "text", "rule", "allow", "reason", "fix"}
    if unknown:
        raise LedgerError(f"{where}: unknown field(s) {sorted(unknown)}")
    try:
        tool, text, rule, reason = entry["tool"], entry["text"], entry["rule"], entry["reason"]
    except KeyError as e:
        raise LedgerError(f"{where}: needs tool, text, rule and reason (missing {e})") from e
    code, fix, allow = entry.get("code"), entry.get("fix", ""), entry.get("allow", 0)
    for name, value in (("tool", tool), ("text", text), ("rule", rule), ("reason", reason), ("fix", fix)):
        if not isinstance(value, str):
            raise LedgerError(f"{where}: {name} must be a string")
    if not tool.strip() or not text.strip() or not rule.strip():
        raise LedgerError(f"{where}: tool, text and rule must not be empty")
    if not reason.strip():
        raise LedgerError(f"{where}: a reason is required")
    if code is not None and (not isinstance(code, str) or not code.strip()):
        raise LedgerError(f"{where}: code is a string or null")
    if allow is not None and (not isinstance(allow, int) or isinstance(allow, bool) or allow < 0):
        raise LedgerError(f"{where}: allow is a count (0 or more), or null for a notice accepted by default")
    return Row(tool, code, " ".join(text.split()), rule.strip(), allow, reason.strip(), fix.strip())


def load_ledger(text: str) -> list[Row]:
    """The rows of a ledger file's JSON text, in file order. A row's key (tool, code, text) must be unique in the file."""
    try:
        entries = json.loads(text)["rows"]
    except (ValueError, KeyError, TypeError) as e:
        raise LedgerError(f'a ledger file must be {{"rows": [...]}}: {e}') from e
    if not isinstance(entries, list):
        raise LedgerError('a ledger file must be {"rows": [...]}')
    rows = [_row(entry, f"row {i}") for i, entry in enumerate(entries)]
    return _checked(rows)


@functools.cache
def _template(text: str) -> re.Pattern:
    return re.compile(".*?".join(re.escape(part) for part in text.split("{}")), re.S)


def _checked(rows: list[Row]) -> list[Row]:
    """The rows, if no key repeats and no two rows can match one notice (rows of one tool, with codes that agree, one
    row's template matching a notice built from the other's)."""
    seen: dict[tuple, Row] = {}
    for row in rows:
        if row.key in seen:
            raise LedgerError(f"two rows for {row.tool} {row.code or '(no code)'} {row.text!r}")
        seen[row.key] = row
    for a in rows:
        for b in rows:
            if a is not b and a.tool == b.tool and (a.code is None or b.code is None or a.code == b.code):
                if _template(a.text).fullmatch(b.text.replace("{}", "x")):
                    raise LedgerError(f"rows overlap: {a.tool} {a.code or '(no code)'} {a.text!r} matches what "
                                      f"{b.code or '(no code)'} {b.text!r} matches")
    return rows


def base_ledger() -> list[Row]:
    """The vendor's templates the module knows, with their default allowances (`base_ledger.py`)."""
    from .base_ledger import ROWS
    return _checked([_row(entry, f"base row {i}") for i, entry in enumerate(ROWS)])


def merge(base: list[Row], project: list[Row]) -> list[Row]:
    """The base with the project's rows: a project row replaces the base row of the same (tool, code, text) in place;
    the others are added after. A project row that would match what a different base row matches is an error."""
    replaced = {row.key: row for row in project}
    out = [replaced.pop(row.key, row) for row in base] + list(replaced.values())
    return _checked(out)


def row_of(ledger: list[Row], notice: Notice) -> Row | None:
    """The ledger's row for a notice: its tool, its code where the row names one, and the template's words."""
    found = [r for r in ledger if r.tool == notice.tool and r.code in (None, notice.code)
             and _template(r.text).fullmatch(notice.text)]
    if len(found) > 1:
        raise LedgerError(f"{len(found)} rows match one notice: {notice.raw.splitlines()[0]}")
    return found[0] if found else None


def row_rules(ledger: list[Row]) -> list[str]:
    """The rule names the ledger's rows count their notices under, sorted."""
    return sorted({r.rule for r in ledger})
