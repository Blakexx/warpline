"""Importer: a record of per-lane (clock, event) stamps with named kinds, written as a warpline record.

The input (JSON): `rows`, each with `variant`, `roles` (a role a warp), `legend` (site id -> name) and `ctas`, each
`{cta, warps: [{count, stamps: [clock, kind << 16 | site, ...]}]}`. `count` is how many stamps the warp issued; its
`stamps` are those that fit. The event kinds, and how they map onto the naming convention:

    1 phase starts   -> `phase.P.end` for the previous phase and `phase.P.begin` (a lane does one phase at a time)
    2 wait begins    -> `barrier.B.wait.begin`
    3 wait ends      -> `barrier.B.wait.end`
    4 arrives        -> `barrier.B.arrive`
    5 copy issued    -> `op.<B>_copy.issue` (payload: the lane's running count of issues at that site), with the
                        barrier declared in `facts` as released by an unstamped arrival, the copy's landing
    a wait on a site named `... steps` is the warp's own asynchronous steps: a barrier declared with count 0
    (nothing arrives; the lane's own operations end it)
    a wait on a hardware-named barrier (site 0x8000..0xBFFF) is also its warp's ARRIVAL there: one
    `barrier.B.arrive` is synthesized at the wait's begin (recorded in `source.synthesized`)

Barrier counts are not in the input. They are INFERRED: a barrier's count is the number of distinct lanes that arrive
at it in a scope (a hardware-named barrier: that wait on it), each once a phase; recorded as `basis: inferred`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from ..record import FORMAT, from_dict

PHASE, WAITS, WAITED, ARRIVES, ISSUES = 1, 2, 3, 4, 5
NAMED, MARKS = 0x8000, 0xC000
KINDS = {PHASE, WAITS, WAITED, ARRIVES, ISSUES}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def convert_row(row: dict, index: int = 0, source_name: str = "") -> tuple[dict, list[str]]:
    """One row's record as a document (dict), and the event kinds met that could not be mapped."""
    legend = {int(k): v for k, v in row["legend"].items()}

    def site_name(site: int) -> str:
        return slug(legend.get(site, f"site_{site:#x}"))

    names: dict[str, int] = {}

    def nid(name: str) -> int:
        return names.setdefault(name, len(names))

    lanes, stamps, dropped = [], [], {}
    unmapped: dict[int, int] = defaultdict(int)
    arriving: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    waiting: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    barrier_kind: dict[str, str] = {}
    issuing: set[str] = set()
    for cta in row["ctas"]:
        scope = f"cta{cta['cta']}"
        for w, warp in enumerate(cta["warps"]):
            lane = f"{scope}/warp{w}"
            role = row["roles"][w]
            raw = warp["stamps"]
            events = [(raw[k], raw[k + 1] >> 16 & 15, raw[k + 1] & 0xFFFF) for k in range(0, len(raw), 2)]
            lane_entry = {"id": lane, "role": role, "group": slug(role)}
            lanes.append(lane_entry)
            written_before = len(stamps)
            phase = None
            opened = None
            issues_at: dict[int, int] = defaultdict(int)
            for t, kind, site in events:
                name = site_name(site)
                if kind == PHASE:
                    if phase is not None:
                        stamps.append([lane, nid(f"phase.{phase}.end"), t])
                    phase = name
                    stamps.append([lane, nid(f"phase.{phase}.begin"), t])
                elif kind in (WAITS, WAITED):
                    steps = legend.get(site, "").endswith(" steps")
                    named = NAMED <= site < MARKS
                    barrier_kind.setdefault(name, "steps" if steps else "named" if named else "barrier")
                    if kind == WAITS:
                        opened = site
                        if named:
                            stamps.append([lane, nid(f"barrier.{name}.arrive"), t])
                            waiting[(scope, name)][lane] += 1
                        stamps.append([lane, nid(f"barrier.{name}.wait.begin"), t])
                    else:
                        stamps.append([lane, nid(f"barrier.{name}.wait.end"), t])
                        opened = None
                elif kind == ARRIVES:
                    barrier_kind.setdefault(name, "barrier")
                    arriving[(scope, name)][lane] += 1
                    stamps.append([lane, nid(f"barrier.{name}.arrive"), t])
                elif kind == ISSUES:
                    barrier_kind.setdefault(name, "copy")
                    issuing.add(name)
                    stamps.append([lane, nid(f"op.{name}_copy.issue"), t, issues_at[site]])
                    issues_at[site] += 1
                else:
                    unmapped[kind] += 1
            # issued is in the record's units: the source's count, plus the stamps the importer added (a phase's end,
            # a named barrier's arrival), which the source's count does not include
            added = (len(stamps) - written_before) - len(events)
            lane_entry["issued"] = warp["count"] + added
            if warp["count"] > len(events):
                dropped[lane] = warp["count"] - len(events)
    facts_b: dict[str, dict] = {}
    facts_o: dict[str, dict] = {}
    for name, kind in sorted(barrier_kind.items()):
        if kind == "steps":
            facts_b[name] = {"count": 0, "basis": "declared: the warp's own asynchronous steps end it"}
        elif name in issuing:
            facts_b[name] = {"count": 1, "stamped": 0, "issue": f"{name}_copy",
                             "basis": "inferred: the copy's landing is the one arrival, not stamped"}
            facts_o[f"{name}_copy"] = {"done": f"barrier.{name}.wait.end"}
        else:
            per_scope = {}
            for (scope, b), lanes_n in list(arriving.items()) + list(waiting.items()):
                if b == name:
                    per_scope[scope] = len(lanes_n)
            counts = set(per_scope.values())
            facts_b[name] = {"count": min(counts) if counts else 1,
                             "basis": "inferred: distinct lanes that arrive, one arrival a lane a phase"
                                      + ("" if len(counts) <= 1 else f"; the scopes differ: {per_scope}")}
    doc = {
        "format": FORMAT,
        "clock": {"unit": "cycles", "hz": None},
        "names": {str(i): n for n, i in names.items()},
        "lanes": lanes,
        "stamps": stamps,
        "dropped": dropped,
        "facts": {"barriers": facts_b, "operations": facts_o},
        "source": {
            "importer": "stamped_clocks", "input": source_name, "row": index, "variant": row.get("variant", ""),
            "synthesized": ["barrier.B.arrive for a hardware-named barrier, at the begin of its wait"],
            "unmapped_kinds": {str(k): v for k, v in sorted(unmapped.items())},
        },
    }
    return doc, [f"kind {k}: {v} event(s)" for k, v in sorted(unmapped.items())]


def convert_file(path: Path, out_dir: Path, rows: list[int] | None = None) -> list[Path]:
    data = json.loads(path.read_text())["rows"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.name.removesuffix(".json").removesuffix(".trace")
    written = []
    for i, row in enumerate(data):
        if rows is not None and i not in rows:
            continue
        doc, unmapped = convert_row(row, i, path.name)
        from_dict(doc)  # validates what is written
        target = out_dir / f"{stem}.row{i}.json"
        target.write_text(json.dumps(doc, separators=(",", ":")))
        print(f"{target}: {len(doc['lanes'])} lanes, {len(doc['stamps']):,} stamps, "
              f"{sum(doc['dropped'].values())} dropped" + (f"; UNMAPPED {unmapped}" if unmapped else ""))
        written.append(target)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m warpline.importers.stamped_clocks")
    ap.add_argument("input", nargs="+", help="a record of per-lane stamps with named kinds (JSON)")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--row", type=int, action="append", help="only this row (repeatable)")
    a = ap.parse_args(argv)
    for p in a.input:
        convert_file(Path(p), Path(a.out_dir), a.row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
