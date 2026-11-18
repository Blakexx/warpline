"""The reader: pure functions over a loaded record. Every meaning is computed here, from the stamps, their names
(the convention in `record.classify`) and the facts the record declares.

Every result carries its `basis`: `measured` (an interval between two stamps), or `upper_bound` (a completion: where
a warp learned of it, so the work may have finished earlier).
"""
from __future__ import annotations

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .record import Finding, Record, classify, scope_of

BLOCKING = 64  # cycles: a wait longer than this is blocking (the release, not its warp's work, held the lane)


# ---------------------------------------------------------------------------------------------------- results

@dataclass(frozen=True)
class Region:
    lane: str
    name: str
    begin: int
    end: int | None  # None: never closed
    depth: int
    parent: str | None
    basis: str = "measured"


@dataclass(frozen=True)
class Issue:
    """An `op.R.issue` stamp, named as what may have released an unstamped arrival."""
    resource: str
    lane: str
    time: int
    op_id: int | None


@dataclass(frozen=True)
class Release:
    """What ended a wait. kind: arrival | write | unstamped | own | none."""
    kind: str
    basis: str
    time: int | None = None
    lane: str | None = None
    stamp: str | None = None
    issue: Issue | None = None  # for `unstamped`: the nearest declared issue, if any
    note: str = ""


@dataclass(frozen=True)
class Wait:
    lane: str
    kind: str  # barrier | poll
    target: str  # the barrier or the polled field
    begin: int
    end: int | None
    region: str | None  # the innermost region at its begin
    release: Release | None = None
    basis: str = "measured"

    @property
    def duration(self) -> int | None:
        return None if self.end is None else self.end - self.begin


@dataclass(frozen=True)
class Operation:
    resource: str
    op_id: int | None
    issue_lane: str
    issue_time: int
    issue_region: str | None
    done_time: int | None
    done_lane: str | None
    basis: str = "upper_bound"  # `done` is where a warp learned of completion


@dataclass(frozen=True)
class Piece:
    """A stretch of the critical path: one lane in one state, a release's latency, or a copy in flight."""
    lane: str
    role: str
    begin: int
    end: int
    state: str
    kind: str  # work | release | flight
    basis: str


@dataclass(frozen=True)
class SlackRow:
    wait: Wait
    slack: int  # the wait's end minus the time its lane arrived
    basis: str = "measured"


@dataclass
class CriticalPath:
    scope: str
    begin: int
    end: int
    lane: str
    pieces: list[Piece] = field(default_factory=list)
    binding: list[tuple[Wait, Release]] = field(default_factory=list)  # the waits the path followed, latest first
    slack: list[SlackRow] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------- scan

@dataclass
class _Lane:
    regions: list[Region] = field(default_factory=list)
    waits: list[Wait] = field(default_factory=list)
    segments: list[tuple[int, int, str]] = field(default_factory=list)  # (begin, end, state)
    context: dict[int, str | None] = field(default_factory=dict)  # stamp index -> innermost region
    findings: list[Finding] = field(default_factory=list)


def _scan(rec: Record) -> dict[str, _Lane]:
    cached = rec._cache.get("scan")
    if cached is not None:
        return cached
    out = {lane: _scan_lane(rec, lane) for lane in rec.lanes}
    rec._cache["scan"] = out
    return out


def _scan_lane(rec: Record, lane: str) -> _Lane:
    out = _Lane()
    stack: list[tuple[str, int]] = []  # (region, begin)
    wait: tuple[str, str, int, str | None] | None = None  # (kind, target, begin, region)
    prev: int | None = None

    def state() -> str:
        if wait is not None:
            return f"{'wait' if wait[0] == 'barrier' else 'poll'}:{wait[1]}"
        return stack[-1][0] if stack else "(outside)"

    def close_region(name: str, begin: int, end: int | None, depth: int) -> None:
        parent = stack[depth - 1][0] if depth > 0 else None
        out.regions.append(Region(lane, name, begin, end, depth, parent))

    for idx, name, t, _payload in rec.lane_stamps(lane):
        if prev is not None and t > prev:
            out.segments.append((prev, t, state()))
        prev = t
        out.context[idx] = stack[-1][0] if stack else None
        kind, key = classify(name)
        if kind == "region_begin":
            stack.append((key, t))
        elif kind == "region_end":
            names = [n for n, _ in stack]
            if key not in names:
                out.findings.append(Finding("region_end_unmatched", f"`{key}.end` with no open `{key}`", lane, t))
            else:
                at = len(names) - 1 - names[::-1].index(key)
                while len(stack) - 1 > at:  # regions opened inside it and never closed: crossing
                    n, b = stack[-1]
                    out.findings.append(Finding(
                        "region_crossing", f"`{key}.end` while `{n}` (opened later, t={b}) is still open", lane, t))
                    close_region(n, b, None, len(stack) - 1)
                    stack.pop()
                n, b = stack[-1]
                close_region(n, b, t, len(stack) - 1)
                stack.pop()
        elif kind in ("wait_begin", "poll_begin"):
            wk = "barrier" if kind == "wait_begin" else "poll"
            if wait is not None:
                out.findings.append(Finding(
                    "wait_nested", f"`{key}` begins while `{wait[1]}` (t={wait[2]}) is still open", lane, t))
                out.waits.append(Wait(lane, wait[0], wait[1], wait[2], None, wait[3]))
            wait = (wk, key, t, stack[-1][0] if stack else None)
        elif kind in ("wait_end", "poll_end"):
            wk = "barrier" if kind == "wait_end" else "poll"
            if wait is None or wait[0] != wk or wait[1] != key:
                out.findings.append(Finding("wait_end_unmatched", f"`{key}` ends with no matching begin", lane, t))
            else:
                out.waits.append(Wait(lane, wk, key, wait[2], t, wait[3]))
                wait = None
    if wait is not None:
        out.findings.append(Finding("wait_unclosed", f"wait on `{wait[1]}` never ends", lane, wait[2]))
        out.waits.append(Wait(lane, wait[0], wait[1], wait[2], None, wait[3]))
    for depth in range(len(stack) - 1, -1, -1):
        n, b = stack[depth]
        out.findings.append(Finding("region_unclosed", f"region `{n}` never closes", lane, b))
        close_region(n, b, None, depth)
    out.regions.sort(key=lambda r: (r.begin, r.depth))
    out.waits.sort(key=lambda w: (w.begin, w.end if w.end is not None else 1 << 62))
    return out


# ---------------------------------------------------------------------------------------------------- regions

def regions(rec: Record) -> tuple[dict[str, list[Region]], list[Finding]]:
    """Each lane's regions (a stack: nesting by depth), and the findings: unclosed, crossing, unmatched end."""
    scan = _scan(rec)
    return ({lane: s.regions for lane, s in scan.items()},
            [f for s in scan.values() for f in s.findings])


# ---------------------------------------------------------------------------------------------------- operations

def operations(rec: Record) -> tuple[dict[str, list[Operation]], list[Finding]]:
    """Operations per resource: an `op.R.issue` paired with where a warp learned of its completion.

    The pair is `op.R.done` with the same payload id, or, if `facts.operations[R].done` names a
    `barrier.B.wait.end`, the issuing lane's next such wait end (if it never waits on B: the earliest wait end on B
    in its scope at or after the issue that no other issue has taken). The completion is an upper bound."""
    scan = _scan(rec)
    findings: list[Finding] = []
    ops: dict[str, list[Operation]] = defaultdict(list)
    issues: dict[str, list[tuple[int, str, int, int | None]]] = defaultdict(list)
    dones: dict[str, list[tuple[int, str, int, int | None]]] = defaultdict(list)
    for lane in rec.lanes:
        for idx, name, t, payload in rec.lane_stamps(lane):
            kind, key = classify(name)
            if kind == "op_issue":
                issues[key].append((idx, lane, t, payload))
            elif kind == "op_done":
                dones[key].append((idx, lane, t, payload))
    declared = rec.facts.get("operations", {})
    for resource, lst in issues.items():
        lst.sort(key=lambda x: (x[2], x[0]))
        if resource in declared:
            wait_name = declared[resource]["done"][len("barrier."):-len(".wait.end")]
            consumed: dict[str, int] = defaultdict(int)
            pool: dict[str, list[tuple[int, str]]] = defaultdict(list)  # per scope: wait ends on B by any lane
            for ln in rec.lanes:
                pool[scope_of(ln)] += [(w.end, ln) for w in scan[ln].waits
                                       if w.kind == "barrier" and w.target == wait_name and w.end is not None]
            for v in pool.values():
                v.sort()
            taken: set[tuple[str, int, str]] = set()
            for idx, lane, t, payload in lst:
                ends = [w.end for w in scan[lane].waits
                        if w.kind == "barrier" and w.target == wait_name and w.end is not None]
                found = None
                if ends:  # the issuing lane waits on that barrier itself: it learns of the completion
                    k = max(consumed[lane], bisect.bisect_left(ends, t))
                    if k < len(ends):
                        consumed[lane] = k + 1
                        found = (ends[k], lane)
                else:  # another lane learns of it: the earliest wait end at or after the issue not yet taken
                    sc = scope_of(lane)
                    for end, ln in pool[sc][bisect.bisect_left(pool[sc], (t, "")):]:
                        if (sc, end, ln) not in taken:
                            taken.add((sc, end, ln))
                            found = (end, ln)
                            break
                if found:
                    ops[resource].append(Operation(resource, payload, lane, t, scan[lane].context[idx], *found))
                else:
                    findings.append(Finding("operation_unfinished", f"`{resource}` issued, never learned done", lane, t))
                    ops[resource].append(Operation(resource, payload, lane, t, scan[lane].context[idx], None, None))
        else:
            by_id = {}
            for idx, lane, t, payload in dones.get(resource, []):
                by_id.setdefault((scope_of(lane), payload), (lane, t))
            for idx, lane, t, payload in lst:
                if payload is None:
                    findings.append(Finding("operation_without_id", f"`op.{resource}.issue` carries no id", lane, t))
                done = by_id.pop((scope_of(lane), payload), None) if payload is not None else None
                if done is None and payload is not None:
                    findings.append(Finding("operation_unfinished", f"`{resource}` id {payload} never done", lane, t))
                ops[resource].append(Operation(resource, payload, lane, t, scan[lane].context[idx],
                                               done[1] if done else None, done[0] if done else None))
            for (scope, payload), (lane, t) in by_id.items():
                findings.append(Finding("operation_unissued", f"`{resource}` id {payload} done, never issued", lane, t))
    for resource in dones:
        if resource not in issues:
            for _idx, lane, t, _p in dones[resource]:
                findings.append(Finding("operation_unissued", f"`op.{resource}.done` with no issue", lane, t))
    for lst in ops.values():
        lst.sort(key=lambda o: (o.issue_time, o.issue_lane))
    return dict(ops), findings


def overlaps(ops: list[Operation]) -> list[tuple[Operation, Operation]]:
    """The pairs of one resource's finished operations in flight at the same time (within one scope)."""
    done = [o for o in ops if o.done_time is not None]
    out = []
    for i, a in enumerate(done):
        for b in done[i + 1:]:
            if scope_of(a.issue_lane) == scope_of(b.issue_lane) and b.issue_time < a.done_time and a.issue_time < b.done_time:
                out.append((a, b))
    return out


# ---------------------------------------------------------------------------------------------------- the join

def _barrier_fact(rec: Record, barrier: str) -> dict | None:
    return rec.facts.get("barriers", {}).get(barrier)


def _arrival_index(rec: Record) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    index = rec._cache.get("arrivals")
    if index is None:
        index = defaultdict(list)
        for i, (lane, nid, t, _p) in enumerate(rec.stamps):
            kind, key = classify(rec.names[nid])
            if kind == "arrive":
                index[(scope_of(lane), key)].append((t, i, lane))
            elif kind == "write":
                index[(scope_of(lane), "write." + key)].append((t, i, lane))
            elif kind == "op_issue":
                index[(scope_of(lane), "op." + key)].append((t, i, lane))
        for lst in index.values():
            lst.sort()
        rec._cache["arrivals"] = index
    return index


def _payload(rec: Record, i: int) -> int | None:
    return rec.stamps[i][3]


def join_wait(rec: Record, w: Wait) -> Release:
    """What ended wait `w`.

    A barrier: the LAST of its `count` arrivals at or before the wait's end (arrivals are taken in order, `count` to
    a phase). If fewer than `count` are stamped (`facts.barriers.B.stamped`), the release is unstamped: the result
    says so and names the nearest declared `op.R.issue` at or before the end. A poll: the last `write.F` at or
    before the end."""
    scope = scope_of(w.lane)
    index = _arrival_index(rec)
    if w.end is None:
        return Release("none", "measured", note="the wait never ends")
    if w.kind == "poll":
        found = index.get((scope, "write." + w.target), [])
        k = bisect.bisect_right(found, (w.end, 1 << 60, "")) - 1
        if k < 0:
            return Release("none", "measured", note=f"no `write.{w.target}` at or before the end")
        t, i, lane = found[k]
        return Release("write", "measured", t, lane, rec.names[rec.stamps[i][1]])
    fact = _barrier_fact(rec, w.target)
    arrivals = index.get((scope, w.target), [])
    if fact is None:
        count, stamped, issue = 1, 1, None
    else:
        count, stamped, issue = fact["count"], fact.get("stamped", fact["count"]), fact.get("issue")
    if count == 0:
        return Release("own", "measured", note="no arrivals: the lane's own work ends it")
    if stamped < count:
        near = None
        if issue is not None:
            found = index.get((scope, "op." + issue), [])
            k = bisect.bisect_right(found, (w.end, 1 << 60, "")) - 1
            if k >= 0:
                t, i, lane = found[k]
                near = Issue(issue, lane, t, _payload(rec, i))
        note = (f"{count - stamped} of {count} arrival(s) not stamped"
                + (f"; nearest `op.{issue}.issue` shown" if near else "; no issue declared on it"))
        return Release("unstamped", "upper_bound", near.time if near else None, near.lane if near else None,
                       f"op.{issue}.issue" if near else None, near, note)
    k = bisect.bisect_right(arrivals, (w.end, 1 << 60, "")) - 1  # the last arrival at or before the end
    complete = (k + 1) // count
    if complete == 0:
        return Release("none", "measured",
                       note=f"{k + 1} stamped arrival(s) at or before the end, fewer than its count {count}")
    t, i, lane = arrivals[complete * count - 1]
    return Release("arrival", "measured", t, lane, rec.names[rec.stamps[i][1]])


def waits(rec: Record) -> list[Wait]:
    """Every wait with its release joined, in lane order then time."""
    cached = rec._cache.get("joined")
    if cached is not None:
        return cached
    out = []
    for lane in rec.lanes:
        for w in _scan(rec)[lane].waits:
            out.append(Wait(w.lane, w.kind, w.target, w.begin, w.end, w.region, join_wait(rec, w)))
    rec._cache["joined"] = out
    return out


# ---------------------------------------------------------------------------------------------------- states

def state_at(rec: Record, lane: str, t: int) -> str:
    """The lane's state at time t: its wait, else its innermost region, else `(outside)`."""
    segs = _scan(rec)[lane].segments
    i = bisect.bisect_right(segs, (t, 1 << 62, "")) - 1
    return segs[i][2] if 0 <= i < len(segs) and segs[i][0] <= t < segs[i][1] else "(outside)"


def _clip(rec: Record, lane: str, a: int, b: int) -> list[tuple[int, int, str]]:
    return [(max(s0, a), min(s1, b), st) for s0, s1, st in _scan(rec)[lane].segments if min(s1, b) > max(s0, a)]


def group(name: str) -> str:
    """A name with its indices taken out: the waits of a kind read together."""
    return re.sub(r"\d+", "#", name)


# ---------------------------------------------------------------------------------------------------- the path

def steady(rec: Record, region: str, skip: int, scope: str) -> tuple[str, int, int, int] | None:
    """A span from a repeating region: the first lane of the scope carrying it gives its begins; the span runs from
    its `skip`th begin to its last-but-one begin that every lane's stamps cover. Returns (lane, begin, end, count)."""
    lanes = [ln for ln in rec.lanes if scope_of(ln) == scope]
    carrier = next((ln for ln in lanes if any(r.name == region for r in _scan(rec)[ln].regions)), None)
    if carrier is None:
        return None
    marks = [r.begin for r in _scan(rec)[carrier].regions if r.name == region]
    last = min(rec.lane_stamps(ln)[-1][2] for ln in lanes if rec.lane_stamps(ln))
    marks = [m for m in marks if m <= last]
    if len(marks) < skip + 3:
        return None
    return carrier, marks[skip], marks[-2], len(marks) - 2 - skip


def critical_path(rec: Record, scope: str, begin: int, end: int, lane: str, blocking: int = BLOCKING) -> CriticalPath:
    """From `end` on `lane`, back to `begin`: at each step the dependency that finished last. A wait that lasted
    more than `blocking` cycles and was ended by another lane's arrival or by an unstamped arrival's nearest issue
    is what its lane was held by: the path continues from the release, on the releasing lane. Otherwise the lane's
    own preceding work is last. Every cycle of the span is then one lane's state, a release's latency or a copy in
    flight. Slack: for every wait in the span not on the path, its end minus the time its lane arrived."""
    path = CriticalPath(scope, begin, end, lane)
    all_waits = [w for w in waits(rec) if scope_of(w.lane) == scope]
    by_lane: dict[str, list[Wait]] = defaultdict(list)
    for w in all_waits:
        by_lane[w.lane].append(w)

    def binding(w: Wait) -> bool:
        r = w.release
        if w.end is None or r is None or w.end - w.begin <= blocking or r.time is None or r.lane is None:
            return False
        if r.kind == "unstamped":
            return True  # the landing is unstamped; its nearest issue (any lane's) starts the flight
        # the release finished last only if it came at or after the lane arrived (a release before it did not hold it)
        return r.kind in ("arrival", "write") and r.lane != w.lane and r.time >= w.begin

    on_path: set[tuple[str, int]] = set()
    cur, t = lane, end
    while t > begin:
        cands = [w for w in by_lane[cur] if w.end is not None and w.end <= t and binding(w)]
        last = max(cands, key=lambda w: (w.end, w.begin), default=None)
        if last is None or last.end <= begin:
            _own(rec, path, cur, begin, t)
            break
        _own(rec, path, cur, last.end, t)
        rel = last.release
        at = max(rel.time, begin)
        role = rec.lanes[cur].role
        label = group(last.target)
        if rel.kind == "unstamped":
            path.pieces.append(Piece(cur, role, at, last.end, f"in flight: {label}", "flight", "upper_bound"))
        else:
            path.pieces.append(Piece(cur, role, at, last.end, f"release at {label}", "release", "measured"))
        path.binding.append((last, rel))
        on_path.add((last.lane, last.begin))
        if at >= t:
            break
        cur, t = rel.lane, at
    path.pieces.reverse()
    for w in all_waits:
        if w.end is not None and begin <= w.end <= end and (w.lane, w.begin) not in on_path:
            path.slack.append(SlackRow(w, w.end - w.begin))
    return path


def _own(rec: Record, path: CriticalPath, lane: str, a: int, b: int) -> None:
    role = rec.lanes[lane].role
    for s0, s1, state in reversed(_clip(rec, lane, a, b)):
        path.pieces.append(Piece(lane, role, s0, s1, state, "work", "measured"))


def blocking_waits(rec: Record, scope: str, begin: int, end: int, blocking: int = BLOCKING) -> list[tuple[Wait, str]]:
    """Blocking waits ending in the span, each with a description of the releaser: its role and what the releasing
    lane was doing just before the release (the join table)."""
    out = []
    for w in waits(rec):
        r = w.release
        if scope_of(w.lane) != scope or w.end is None or not begin <= w.end <= end or w.end - w.begin <= blocking:
            continue
        if r is None or r.time is None or r.lane is None or r.kind not in ("arrival", "write", "unstamped"):
            continue
        if r.kind == "unstamped":
            src = f"its copy ({r.issue.resource})" if r.issue else "an unstamped arrival"
        else:
            src = f"{rec.lanes[r.lane].role}, from {state_at(rec, r.lane, r.time - 1)}"
        out.append((w, src))
    return out


# ---------------------------------------------------------------------------------------------------- the stepper

def at(rec: Record, t: int, scope: str | None = None) -> dict:
    """What every lane and every resource is doing at time t (within one scope: scopes' clocks are unrelated)."""
    scopes = rec.scopes()
    scope = scopes[0] if scope is None else scope
    lanes = []
    for lane in rec.lanes:
        if scope_of(lane) != scope:
            continue
        s = _scan(rec)[lane]
        stack = [r.name for r in s.regions if r.begin <= t and (r.end is None or t < r.end)]
        wait = next((w for w in waits(rec) if w.lane == lane and w.begin <= t and (w.end is None or t < w.end)), None)
        prior = [x for x in rec.lane_stamps(lane) if x[2] <= t]
        lanes.append({"lane": lane, "role": rec.lanes[lane].role, "regions": stack, "wait": wait,
                      "last_stamp": (prior[-1][1], prior[-1][2]) if prior else None})
    resources = {}
    for resource, lst in operations(rec)[0].items():
        flying = [o for o in lst if scope_of(o.issue_lane) == scope and o.issue_time <= t
                  and (o.done_time is None or t < o.done_time)]
        resources[resource] = flying
    return {"scope": scope, "time": t, "lanes": lanes, "resources": resources}
