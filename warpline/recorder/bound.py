"""Bounded output: a command that has many findings of one kind prints the kind, its count and the worst case, and
every one of them only when asked (`--all`).

`LIMIT` is how many of a kind are printed; `None` prints all. Commands set it from their `--all` flag.
"""
from __future__ import annotations

import re
from collections import defaultdict

LIMIT: int | None = 1


def code_of(item) -> str:
    """The kind of a finding: a `Finding`'s code, or the first word of a finding's text."""
    code = getattr(item, "code", None)
    if code is not None:
        return code
    m = re.match(r"[A-Za-z0-9_.\-]+", str(item))
    return m.group(0) if m else "finding"


def _worst_number(item) -> float:
    """The largest percentage a finding's text calls `worst N%` (0 where it has none)."""
    m = re.search(r"worst ([0-9.]+)%", str(item))
    return float(m.group(1)) if m else 0.0


def group_lines(items, limit="default", indent="  ", worst=None) -> list[str]:
    """One line a kind: `code xN`, the worst case (by `worst(item)`, a number; else the first one), and `limit` of the
    kind in all (None: every one). The lines past the limit are counted, never silently dropped."""
    if limit == "default":
        limit = LIMIT
    groups: dict[str, list] = defaultdict(list)
    for it in items:
        groups[code_of(it)].append(it)
    out = []
    for code, fs in groups.items():
        if limit is None:
            out.append(f"{indent}{code} x{len(fs)}")
            out += [f"{indent}  {f}" for f in fs]
            continue
        pick = max(fs, key=worst) if worst is not None else fs[0]
        label = "worst" if worst is not None and len(fs) > 1 else "e.g."
        out.append(f"{indent}{code} x{len(fs)}  {label}: {pick}")
        extra = [f for f in fs if f is not pick][: max(0, limit - 1)]
        out += [f"{indent}  {f}" for f in extra]
        rest = len(fs) - 1 - len(extra)
        if rest > 0:
            out.append(f"{indent}  ... {rest} more of this kind (--all prints every one)")
    return out
