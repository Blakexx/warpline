"""The record: one JSON document of stamps and the facts declared about the kernel (see docs/format.md).

Loading validates the schema and raises `RecordError`, whose `code` names what is wrong. `classify` is the reader's
only knowledge of meaning: the naming convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "warpline/0"
LAUNCH_BEGIN, LAUNCH_END = "launch.begin", "launch.end"   # the launch's two edges: instant events, each with a tie


class RecordError(Exception):
    """A record that does not follow the format. `code` is a stable name; `str()` says where."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class StrictError(RecordError):
    """Strict loading found tier-0 findings; they are in `findings`."""

    def __init__(self, findings: list["Finding"]):
        super().__init__("tier0_findings", f"{len(findings)} finding(s); first: {findings[0]}")
        self.findings = findings


@dataclass(frozen=True)
class Finding:
    """A fact about a record that a check or the reader found. Never hidden, never fatal on its own."""

    code: str
    message: str
    lane: str | None = None
    time: int | None = None

    def __str__(self) -> str:
        where = "".join([f" lane={self.lane}" if self.lane else "", f" t={self.time}" if self.time is not None else ""])
        return f"{self.code}{where}: {self.message}"


@dataclass(frozen=True)
class Lane:
    id: str
    role: str
    group: str | None = None
    issued: int | None = None  # how many stamps the lane's warp issued, if declared (written + dropped = issued)

    @property
    def scope(self) -> str:
        """The part of the id before `/`: lanes of one scope (a CTA) share a clock, and only they are joined."""
        return scope_of(self.id)


def scope_of(lane: str) -> str:
    return lane.split("/", 1)[0] if "/" in lane else ""


@dataclass
class Record:
    format: str
    clock: dict[str, Any]
    names: dict[int, str]
    lanes: dict[str, Lane]
    stamps: list[tuple[str, int, int, int | None]]
    dropped: dict[str, int]
    facts: dict[str, Any]
    twin: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    ties: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # lane -> [(cycles, nanoseconds)]
    variant: str | None = None  # which variant of the kernel this record is: the name `set`'s manifest gives it
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def name(self, name_id: int) -> str:
        return self.names[name_id]

    def lane_stamps(self, lane: str) -> list[tuple[int, str, int, int | None]]:
        """The lane's stamps in the order written: (global index, name, time, payload)."""
        index = self._cache.get("by_lane")
        if index is None:
            index = {lid: [] for lid in self.lanes}
            for i, (lane_id, nid, t, p) in enumerate(self.stamps):
                index[lane_id].append((i, self.names[nid], t, p))
            self._cache["by_lane"] = index
        return index[lane]

    def scopes(self) -> list[str]:
        seen: dict[str, None] = {}
        for lane in self.lanes.values():
            seen[lane.scope] = None
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": self.format,
            "clock": self.clock,
            "names": {str(k): v for k, v in self.names.items()},
            "lanes": [
                {k: v for k, v in (("id", ln.id), ("role", ln.role), ("group", ln.group), ("issued", ln.issued))
                 if v is not None} for ln in self.lanes.values()],
            "stamps": [[lane, nid, t] + ([p] if p is not None else []) for lane, nid, t, p in self.stamps],
            "dropped": self.dropped,
            "facts": self.facts,
        }
        if self.twin is not None:
            out["twin"] = self.twin
        if self.source is not None:
            out["source"] = self.source
        if self.variant is not None:
            out["variant"] = self.variant
        if self.ties:
            out["ties"] = [[lane, c, ns] for lane, lst in self.ties.items() for c, ns in lst]
        return out


def classify(name: str) -> tuple[str, str]:
    """The naming convention: a stamp's name -> (kind, key).

    barrier.B.arrive | barrier.B.wait.begin | barrier.B.wait.end | op.R.issue | op.R.done | write.F |
    poll.F.begin | poll.F.end | X.begin | X.end (region X); anything else is a mark. `launch.begin` and
    `launch.end` are the launch's two instant edges (kinds `launch_begin`, `launch_end`), never a region `launch`.
    """
    if name == LAUNCH_BEGIN:
        return "launch_begin", ""
    if name == LAUNCH_END:
        return "launch_end", ""
    if name.startswith("barrier."):
        for suffix, kind in ((".wait.begin", "wait_begin"), (".wait.end", "wait_end"), (".arrive", "arrive")):
            if name.endswith(suffix) and len(name) > len("barrier.") + len(suffix):
                return kind, name[len("barrier."):-len(suffix)]
    elif name.startswith("op."):
        for suffix, kind in ((".issue", "op_issue"), (".done", "op_done")):
            if name.endswith(suffix) and len(name) > len("op.") + len(suffix):
                return kind, name[len("op."):-len(suffix)]
    elif name.startswith("write.") and len(name) > len("write."):
        return "write", name[len("write."):]
    elif name.startswith("poll."):
        for suffix, kind in ((".begin", "poll_begin"), (".end", "poll_end")):
            if name.endswith(suffix) and len(name) > len("poll.") + len(suffix):
                return kind, name[len("poll."):-len(suffix)]
    if name.endswith(".begin") and len(name) > 6:
        return "region_begin", name[:-6]
    if name.endswith(".end") and len(name) > 4:
        return "region_end", name[:-4]
    return "mark", name


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _need(doc: dict, key: str, where: str = "record") -> Any:
    if key not in doc:
        raise RecordError("missing_field", f"{where} has no `{key}`")
    return doc[key]


def from_dict(doc: Any) -> Record:
    """Validate a parsed document and build the record. Raises `RecordError`."""
    if not isinstance(doc, dict):
        raise RecordError("bad_type", "the record is not a JSON object")
    fmt = _need(doc, "format")
    if fmt != FORMAT:
        raise RecordError("format_unsupported", f"format is {fmt!r}, this reader reads {FORMAT!r}")

    clock = _need(doc, "clock")
    if not isinstance(clock, dict) or clock.get("unit") != "cycles":
        raise RecordError("bad_clock", "`clock.unit` must be \"cycles\"")
    hz = clock.get("hz", None)
    if hz is not None and (isinstance(hz, bool) or not isinstance(hz, (int, float)) or hz <= 0):
        raise RecordError("bad_clock", "`clock.hz` must be a positive number or null")

    names_raw = _need(doc, "names")
    if not isinstance(names_raw, dict):
        raise RecordError("bad_type", "`names` must map id -> name")
    names: dict[int, str] = {}
    for k, v in names_raw.items():
        try:
            nid = int(k)
        except (TypeError, ValueError):
            raise RecordError("bad_name_id", f"name id {k!r} is not an integer") from None
        if not isinstance(v, str) or not v:
            raise RecordError("bad_type", f"name {nid} is not a non-empty string")
        names[nid] = v
    if len(set(names.values())) != len(names):
        raise RecordError("duplicate_name", "two ids carry the same name")

    lanes_raw = _need(doc, "lanes")
    if not isinstance(lanes_raw, list):
        raise RecordError("bad_type", "`lanes` must be a list")
    lanes: dict[str, Lane] = {}
    for ln in lanes_raw:
        if not isinstance(ln, dict) or not isinstance(ln.get("id"), str) or not ln["id"]:
            raise RecordError("bad_lane", f"a lane needs a string `id`: {ln!r}")
        if not isinstance(ln.get("role"), str):
            raise RecordError("bad_lane", f"lane {ln['id']} needs a string `role`")
        if ln["id"] in lanes:
            raise RecordError("duplicate_lane", f"lane {ln['id']} appears twice")
        group = ln.get("group")
        issued = ln.get("issued")
        if group is not None and not isinstance(group, str):
            raise RecordError("bad_lane", f"lane {ln['id']}: `group` must be a string")
        if issued is not None and (not _int(issued) or issued < 0):
            raise RecordError("bad_lane", f"lane {ln['id']}: `issued` must be a non-negative integer")
        lanes[ln["id"]] = Lane(ln["id"], ln["role"], group, issued)

    stamps_raw = _need(doc, "stamps")
    if not isinstance(stamps_raw, list):
        raise RecordError("bad_type", "`stamps` must be a list")
    stamps: list[tuple[str, int, int, int | None]] = []
    for i, s in enumerate(stamps_raw):
        if not isinstance(s, list) or len(s) not in (3, 4):
            raise RecordError("bad_stamp", f"stamp {i} is not [lane, name_id, time, payload?]")
        lane, nid, t = s[0], s[1], s[2]
        payload = s[3] if len(s) == 4 else None
        if lane not in lanes:
            raise RecordError("unknown_lane", f"stamp {i} names lane {lane!r}, not in `lanes`")
        if not _int(nid) or nid not in names:
            raise RecordError("unknown_name_id", f"stamp {i} has name id {nid!r}, not in `names`")
        if not _int(t):
            raise RecordError("bad_time", f"stamp {i} time {t!r} is not an integer")
        if len(s) == 4 and not _int(payload):
            raise RecordError("bad_payload", f"stamp {i} payload {payload!r} is not an integer")
        stamps.append((lane, nid, t, payload))

    dropped = _need(doc, "dropped")
    if not isinstance(dropped, dict):
        raise RecordError("bad_type", "`dropped` must map lane -> count")
    for lane, n in dropped.items():
        if lane not in lanes:
            raise RecordError("unknown_lane", f"`dropped` names lane {lane!r}, not in `lanes`")
        if not _int(n) or n < 0:
            raise RecordError("bad_dropped", f"dropped count for {lane} must be a non-negative integer")

    facts = _need(doc, "facts")
    if not isinstance(facts, dict):
        raise RecordError("bad_type", "`facts` must be an object")
    _check_facts(facts)

    win = facts.get("window")
    if win is not None:
        spread = win.get("stamps") if isinstance(win, dict) else None
        if not isinstance(spread, list) or len(spread) != len(stamps):
            raise RecordError("bad_facts", "`facts.window.stamps` must hold one spread entry for each stamp")
    if "runs" in facts and not isinstance(facts["runs"], list):
        raise RecordError("bad_facts", "`facts.runs` must be a list")

    twin = doc.get("twin")
    if twin is not None and not isinstance(twin, dict):
        raise RecordError("bad_type", "`twin` must be an object")
    source = doc.get("source")
    if source is not None and not isinstance(source, dict):
        raise RecordError("bad_type", "`source` must be an object")
    ties = _read_ties(doc.get("ties"), lanes, clock)
    variant = doc.get("variant")
    if variant is not None and (not isinstance(variant, str) or not variant.strip()):
        raise RecordError("bad_variant", "`variant` must be a non-empty string naming the variant of the kernel")
    return Record(fmt, clock, names, lanes, stamps, dropped, facts, twin, source, ties, variant)


def _read_ties(raw: Any, lanes: dict[str, Lane], clock: dict) -> dict[str, list[tuple[int, int]]]:
    """`ties`: a list of [lane, cycles, nanoseconds]; absent or empty is valid. A tie's lane must exist and its two
    coordinates be integers; a record with ties names its time axis in `clock` (`time_unit` "ns", `time_origin` a free
    string). That a lane's ties do not go backwards is a FINDING (`check.ties_ordered`), not a load error."""
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise RecordError("bad_tie", "`ties` must be a list of [lane, cycles, nanoseconds]")
    out: dict[str, list[tuple[int, int]]] = {}
    for i, t in enumerate(raw):
        if not isinstance(t, list) or len(t) != 3:
            raise RecordError("bad_tie", f"tie {i} is not [lane, cycles, nanoseconds]")
        lane, c, ns = t
        if lane not in lanes:
            raise RecordError("unknown_lane", f"tie {i} names lane {lane!r}, not in `lanes`")
        if not _int(c) or not _int(ns):
            raise RecordError("bad_tie", f"tie {i}: cycles {c!r} and nanoseconds {ns!r} must be integers")
        out.setdefault(lane, []).append((c, ns))
    if out:
        if clock.get("time_unit") != "ns":
            raise RecordError("bad_clock", "a record with `ties` needs `clock.time_unit` = \"ns\"")
        origin = clock.get("time_origin")
        if origin is not None and not isinstance(origin, str):
            raise RecordError("bad_clock", "`clock.time_origin` must be a string")
    return out


def _check_facts(facts: dict) -> None:
    barriers = facts.get("barriers", {})
    if not isinstance(barriers, dict):
        raise RecordError("bad_facts", "`facts.barriers` must map barrier -> {count, ...}")
    for b, f in barriers.items():
        if not isinstance(f, dict) or not _int(f.get("count")) or f["count"] < 0:
            raise RecordError("bad_facts", f"barrier {b!r} needs an integer `count` >= 0")
        stamped = f.get("stamped", f["count"])
        if not _int(stamped) or not 0 <= stamped <= f["count"]:
            raise RecordError("bad_facts", f"barrier {b!r}: `stamped` must be an integer in 0..count")
        issue = f.get("issue")
        if issue is not None and not isinstance(issue, str):
            raise RecordError("bad_facts", f"barrier {b!r}: `issue` must name a resource")
        rendezvous = f.get("rendezvous", False)
        skew = f.get("skew")
        if not isinstance(rendezvous, bool):
            raise RecordError("bad_facts", f"barrier {b!r}: `rendezvous` is true or false")
        if skew is not None and not rendezvous:
            raise RecordError("bad_facts", f"barrier {b!r}: `skew` belongs to a rendezvous")
        if rendezvous:
            if f["count"] < 1 or stamped != f["count"] or issue is not None:
                raise RecordError("bad_facts", f"barrier {b!r}: a rendezvous has a count of at least 1, every "
                                  f"arrival stamped (the waiters' own), and no `issue`")
            if skew is not None and (not _int(skew) or skew < 0):
                raise RecordError("bad_facts", f"barrier {b!r}: `skew` is a non-negative integer (cycles)")
    ops = facts.get("operations", {})
    if not isinstance(ops, dict):
        raise RecordError("bad_facts", "`facts.operations` must map resource -> {done}")
    for r, f in ops.items():
        done = f.get("done") if isinstance(f, dict) else None
        if not isinstance(done, str) or not done.startswith("barrier.") or not done.endswith(".wait.end"):
            raise RecordError("bad_facts", f"operation {r!r}: `done` must be a `barrier.B.wait.end` name")


def loads(text: str, strict: bool = False) -> Record:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise RecordError("bad_json", str(e)) from None
    rec = from_dict(doc)
    if strict:
        from . import check
        findings = check.all_findings(rec)
        if findings:
            raise StrictError(findings)
    return rec


def load(path: str | Path, strict: bool = False) -> Record:
    return loads(Path(path).read_text(), strict=strict)


def dump(rec: Record, path: str | Path) -> None:
    Path(path).write_text(json.dumps(rec.to_dict(), separators=(",", ":")))
