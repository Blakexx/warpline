"""Tier 0: the logical checks. Each is a function over a record returning findings (an empty list: it holds).

A warp's regions nest and tile its span; a wait ends at or after the arrival that released it; an operation
completes at or after its issue; a barrier's arrivals equal its count; no stamp is unaccounted (written + dropped =
issued). Also that a lane's stamps are in time order, which every other check assumes.
"""
from __future__ import annotations

from collections import defaultdict

from . import reader
from .record import LAUNCH_BEGIN, LAUNCH_END, Finding, Record, classify, scope_of


def stamps_ordered(rec: Record) -> list[Finding]:
    """A lane's stamps, in the order written, never go back in time."""
    out = []
    for lane in rec.lanes:
        prev = None
        for _i, name, t, _p in rec.lane_stamps(lane):
            if prev is not None and t < prev:
                out.append(Finding("stamp_out_of_order", f"`{name}` at {t} follows a stamp at {prev}", lane, t))
            prev = t
    return out


def ties_ordered(rec: Record) -> list[Finding]:
    """A lane's ties, in the order given, never go back in either coordinate: a tie that does is a wrapped or
    mis-decoded reading of one of the two clocks."""
    out = []
    for lane, lst in rec.ties.items():
        for (c0, n0), (c1, n1) in zip(lst, lst[1:]):
            if c1 < c0:
                out.append(Finding("tie_backwards", f"tie at {c1} cycles follows a tie at {c0} cycles: the cycle "
                                   f"reading went back (wrapped or mis-decoded)", lane, c1))
            if n1 < n0:
                out.append(Finding("tie_backwards", f"tie at {n1} ns follows a tie at {n0} ns: the time reading went "
                                   f"back (wrapped or mis-decoded)", lane, c1))
    return out


def regions_nest_and_tile(rec: Record) -> list[Finding]:
    """Regions of a lane nest (no crossing, none unclosed or ended unopened), waits do not overlap, and a lane's
    outermost regions tile its span: no gap between one's end and the next's begin."""
    _regions, findings = reader.regions(rec)
    out = list(findings)
    for lane, rs in _regions.items():
        top = sorted((r for r in rs if r.depth == 0 and r.end is not None), key=lambda r: r.begin)
        for a, b in zip(top, top[1:]):
            if b.begin > a.end:
                out.append(Finding("region_gap", f"`{a.name}` ends at {a.end}, `{b.name}` begins at {b.begin}",
                                   lane, a.end))
            elif b.begin < a.end:
                out.append(Finding("region_overlap", f"`{a.name}` and `{b.name}` overlap", lane, b.begin))
    return out


def wait_after_release(rec: Record) -> list[Finding]:
    """A wait ends at or after the arrival that released it: no wait ends before its barrier's phase could have
    completed, and a poll ends after the write it watches."""
    out = []
    for w in reader.waits(rec):
        r = w.release
        if r is None:
            continue
        if r.kind == "none":
            out.append(Finding("wait_before_release", f"wait on `{w.target}` {r.note}", w.lane, w.end))
        elif r.time is not None and w.end is not None and r.kind in ("arrival", "write") and w.end < r.time:
            out.append(Finding("wait_ends_before_release", f"wait on `{w.target}` ends before its release at {r.time}",
                               w.lane, w.end))
    return out


def operations_ordered(rec: Record) -> list[Finding]:
    """An operation completes at or after its issue, and every issue is paired."""
    ops, findings = reader.operations(rec)
    out = list(findings)
    for lst in ops.values():
        for o in lst:
            if o.done_time is not None and o.done_time < o.issue_time:
                out.append(Finding("operation_done_before_issue",
                                   f"`{o.resource}` id {o.op_id} done at {o.done_time}, issued at {o.issue_time}",
                                   o.issue_lane, o.issue_time))
    return out


def barrier_arrivals(rec: Record) -> list[Finding]:
    """A barrier's stamped arrivals in a scope are whole phases: a multiple of its stamped count. A barrier that
    has arrivals but no declared count is a finding too (the join assumed a count of 1)."""
    out = []
    per: dict[tuple[str, str], int] = defaultdict(int)
    for lane, nid, _t, _p in rec.stamps:
        kind, key = classify(rec.names[nid])
        if kind == "arrive":
            per[(scope_of(lane), key)] += 1
    for (scope, b), n in sorted(per.items()):
        fact = rec.facts.get("barriers", {}).get(b)
        if fact is None:
            out.append(Finding("barrier_undeclared", f"`{b}` has {n} arrivals in `{scope}` and no declared count"))
            continue
        stamped = fact.get("stamped", fact["count"])
        if stamped and n % stamped:
            out.append(Finding("barrier_arrivals_not_whole_phases",
                               f"`{b}` has {n} stamped arrivals in `{scope}`, not a multiple of {stamped}"))
    return out


def stamps_accounted(rec: Record) -> list[Finding]:
    """Written + dropped = issued, for every lane that declares what it issued."""
    out = []
    written: dict[str, int] = defaultdict(int)
    for lane, *_ in rec.stamps:
        written[lane] += 1
    for lane, ln in rec.lanes.items():
        if ln.issued is None:
            continue
        got = written[lane] + rec.dropped.get(lane, 0)
        if got != ln.issued:
            out.append(Finding("stamps_unaccounted",
                               f"{written[lane]} written + {rec.dropped.get(lane, 0)} dropped = {got}, "
                               f"issued {ln.issued}", lane))
    return out


def launch_edges(rec: Record) -> list[Finding]:
    """The launch's two edges (`launch.begin`, `launch.end`): a lane has at most one of each, a lane with one has the
    other, each has a tie (the overlay places the samples by them, on both clocks), and no stamp of the lane falls
    outside them. A record with no edge at all holds (it imports as it always did; the overlay refuses it)."""
    out = []
    for lane in rec.lanes:
        edges: dict[str, list[tuple[int, int]]] = {LAUNCH_BEGIN: [], LAUNCH_END: []}
        stamps = rec.lane_stamps(lane)
        for i, name, t, _p in stamps:
            if name in edges:
                edges[name].append((i, t))
        b, e = edges[LAUNCH_BEGIN], edges[LAUNCH_END]
        for name, lst in edges.items():
            if len(lst) > 1:
                out.append(Finding("launch_edge_repeated", f"`{name}` stamped {len(lst)} times: one a lane a launch",
                                   lane, lst[1][1]))
        if not b and not e:
            continue
        if bool(b) != bool(e):
            has, lacks = (LAUNCH_BEGIN, LAUNCH_END) if b else (LAUNCH_END, LAUNCH_BEGIN)
            out.append(Finding("launch_edge_missing", f"`{has}` without `{lacks}`: a lane carries both or neither",
                               lane, (b or e)[0][1]))
            continue
        cycles = {c for c, _ns in rec.ties.get(lane, [])}
        for name, lst in edges.items():
            for _i, t in lst:
                if t not in cycles:
                    out.append(Finding("launch_edge_untied", f"`{name}` at {t} has no tie: the edge is a stamp with "
                                       f"a tie, read on both clocks", lane, t))
        t0, t1 = b[0][1], e[0][1]
        for i, name, t, _p in stamps:
            if name not in edges and (t < t0 or t > t1):
                out.append(Finding("stamp_outside_launch", f"`{name}` at {t} is outside the launch "
                                   f"({LAUNCH_BEGIN} at {t0}, {LAUNCH_END} at {t1})", lane, t))
    return out


CHECKS = (stamps_ordered, ties_ordered, regions_nest_and_tile, wait_after_release, operations_ordered, barrier_arrivals,
          stamps_accounted, launch_edges)


def all_findings(rec: Record) -> list[Finding]:
    return [f for check in CHECKS for f in check(rec)]
