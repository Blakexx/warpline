"""Importer: a record of per-lane (clock, event) stamps with named kinds, written as a warpline record.

The input (JSON): `rows`, each with `variant`, `roles` (a role a warp), `legend` (site id -> name) and `ctas`, each
`{cta, warps: [{count, stamps: [clock, kind << 16 | site, ...]}]}`. `count` is how many stamps the warp issued; its
`stamps` are those that fit. The event kinds, and how they map onto the naming convention:

    1 phase starts   -> `phase.P.end` for the previous phase and `phase.P.begin` (a lane does one phase at a time);
                        EXCEPT at a site whose legend name is the launch's begin (`launch_begin`, default `entry`) or
                        end (`launch_end`, default `exit`): that stamp becomes the instant event `launch.begin` /
                        `launch.end` (the launch's two edges, each with its tie, one of each a lane), opens no phase
                        region, and `launch.end` closes the lane's open phase region
    2 wait begins    -> `barrier.B.wait.begin`
    3 wait ends      -> `barrier.B.wait.end`
    4 arrives        -> `barrier.B.arrive`
    5 copy issued    -> `op.<B>_copy.issue` (payload: the lane's running count of issues at that site), with the
                        barrier declared in `facts` as released by an unstamped arrival, the copy's landing
    6 time           -> a TIE, not a stamp: `[t, 6 << 16 | site]` immediately FOLLOWS a stamp of the same site on the
                        same lane, and `t` is that same moment's nanosecond reading, relative to the scope's first such
                        reading. It becomes the tie `[lane, <the preceding stamp's time>, t]`. A 32-bit nanosecond
                        counter wraps in about 4.3 s (a signed offset, in about 2.1 s): a recording longer than about
                        two seconds of a 32-bit word must be UNWRAPPED by the writer into a signed offset before
                        it is written here; the importer does not unwrap, it CHECKS monotonicity (a lane's readings
                        that go back are reported, and `check` names them: `tie_backwards`) instead of assuming it.
                        A time entry with no preceding stamp of its site is an error (`time_without_stamp`). It took
                        a slot: `count` and the dropped arithmetic count it as issued and stored, and the record's
                        `issued` (written stamps + dropped) leaves it out, since it is no stamp.
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

from ..record import FORMAT, RecordError, from_dict

PHASE, WAITS, WAITED, ARRIVES, ISSUES, TIME = 1, 2, 3, 4, 5, 6
NAMED, MARKS = 0x8000, 0xC000
KINDS = {PHASE, WAITS, WAITED, ARRIVES, ISSUES, TIME}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def convert_row(row: dict, index: int = 0, source_name: str = "", launch_begin: str = "entry",
                launch_end: str = "exit") -> tuple[dict, list[str]]:
    """One row's record as a document (dict), and the event kinds met that could not be mapped. A phase-kind stamp
    whose site's legend name is `launch_begin` / `launch_end` becomes `launch.begin` / `launch.end`."""
    legend = {int(k): v for k, v in row["legend"].items()}
    edge_sites = {s: ("launch.begin" if slug(n) == slug(launch_begin) else "launch.end")
                  for s, n in legend.items() if slug(n) in (slug(launch_begin), slug(launch_end))}

    def site_name(site: int) -> str:
        return slug(legend.get(site, f"site_{site:#x}"))

    names: dict[str, int] = {}

    def nid(name: str) -> int:
        return names.setdefault(name, len(names))

    lanes, stamps, dropped = [], [], {}
    ties: list[list] = []
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
            last = None   # (site, time) of the stamp the previous entry wrote, for a time entry that follows it
            for e, (t, kind, site) in enumerate(events):
                name = site_name(site)
                if kind == TIME:
                    if last is None or last[0] != site:
                        raise RecordError("time_without_stamp",
                                          f"{lane} entry {e}: a time entry (site {site:#x}) must immediately follow "
                                          f"a stamp of the same site, and the entry before it is "
                                          f"{'not a stamp' if last is None else f'a stamp of site {last[0]:#x}'}")
                    ties.append([lane, last[1], t])
                    last = None
                    continue
                before = len(stamps)
                if kind == PHASE and site in edge_sites:
                    if edge_sites[site] == "launch.end" and phase is not None:
                        stamps.append([lane, nid(f"phase.{phase}.end"), t])
                        phase = None
                    stamps.append([lane, nid(edge_sites[site]), t])
                elif kind == PHASE:
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
                last = (site, t) if len(stamps) > before else None
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
    clock = {"unit": "cycles", "hz": None}
    if ties:
        clock.update(time_unit="ns", time_origin="the scope's first nanosecond reading, as the input gives it")
    doc = {
        "format": FORMAT,
        "clock": clock,
        "names": {str(i): n for n, i in names.items()},
        "lanes": lanes,
        "stamps": stamps,
        "dropped": dropped,
        "facts": {"barriers": facts_b, "operations": facts_o},
        **({"ties": ties} if ties else {}),
        "source": {
            "importer": "stamped_clocks", "input": source_name, "row": index, "variant": row.get("variant", ""),
            "synthesized": ["barrier.B.arrive for a hardware-named barrier, at the begin of its wait"],
            "unmapped_kinds": {str(k): v for k, v in sorted(unmapped.items())},
        },
    }
    return doc, [f"kind {k}: {v} event(s)" for k, v in sorted(unmapped.items())]


def convert_file(path: Path, out_dir: Path, rows: list[int] | None = None, launch_begin: str = "entry",
                 launch_end: str = "exit") -> list[Path]:
    data = json.loads(path.read_text())["rows"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.name.removesuffix(".json").removesuffix(".trace")
    written = []
    for i, row in enumerate(data):
        if rows is not None and i not in rows:
            continue
        doc, unmapped = convert_row(row, i, path.name, launch_begin, launch_end)
        rec = from_dict(doc)  # validates what is written
        from .. import check
        for f in check.ties_ordered(rec):
            print(f"{path.name} row {i}: FINDING {f}")
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
    ap.add_argument("--launch-begin", default="entry", metavar="NAME",
                    help="the legend name of the site whose phase-kind stamp is `launch.begin` (default: entry)")
    ap.add_argument("--launch-end", default="exit", metavar="NAME",
                    help="the legend name of the site whose phase-kind stamp is `launch.end` (default: exit)")
    a = ap.parse_args(argv)
    for p in a.input:
        convert_file(Path(p), Path(a.out_dir), a.row, a.launch_begin, a.launch_end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
