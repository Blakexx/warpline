"""Chrome trace-event JSON that Perfetto's UI opens: a track a lane (its regions as nested slices, its waits as
slices named by their barrier), a track a resource (its operations as async slices), and flow arrows from the stamp
that released a wait to the wait it ended.

Times are the clock's cycles. If `clock.hz` is declared they are written in microseconds (the format's unit);
otherwise one cycle is written as one microsecond and `otherData.time_unit` says so.
"""
from __future__ import annotations

from collections import defaultdict

from . import reader
from .record import Record, scope_of


def build(rec: Record) -> dict:
    hz = rec.clock.get("hz")
    scale = 1e6 / hz if hz else 1.0

    def ts(t: int) -> float:
        return t * scale

    scopes = rec.scopes()
    pid_of = {s: i + 1 for i, s in enumerate(scopes)}
    res_pid = len(scopes) + 1
    tid_of = {lane: i + 1 for i, lane in enumerate(rec.lanes)}
    ev: list[dict] = []
    for s, pid in pid_of.items():
        ev.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": s or "kernel"}})
    for lane, tid in tid_of.items():
        ln = rec.lanes[lane]
        ev.append({"ph": "M", "name": "thread_name", "pid": pid_of[ln.scope], "tid": tid,
                   "args": {"name": f"{lane} ({ln.role})"}})
        ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid_of[ln.scope], "tid": tid,
                   "args": {"sort_index": tid}})
    slices: list[dict] = []
    regs, _ = reader.regions(rec)
    last_time = {lane: (rec.lane_stamps(lane)[-1][2] if rec.lane_stamps(lane) else 0) for lane in rec.lanes}
    for lane, rs in regs.items():
        for r in rs:
            end = r.end if r.end is not None else last_time[lane]
            slices.append({"ph": "X", "name": r.name, "cat": "region", "pid": pid_of[scope_of(lane)],
                           "tid": tid_of[lane], "ts": ts(r.begin), "dur": ts(end) - ts(r.begin),
                           "args": {"basis": "measured", "closed": r.end is not None}})
    flows = []
    for w in reader.waits(rec):
        end = w.end if w.end is not None else last_time[w.lane]
        rel = w.release
        args: dict = {"basis": "measured", "kind": w.kind}
        if rel is not None:
            args["release"] = rel.kind
            if rel.note:
                args["note"] = rel.note
            if rel.lane:
                args["released_by"] = f"{rel.lane} {rel.stamp} t={rel.time}"
        slices.append({"ph": "X", "name": f"wait {w.target}" if w.kind == "barrier" else f"poll {w.target}",
                       "cat": "wait", "pid": pid_of[scope_of(w.lane)], "tid": tid_of[w.lane], "ts": ts(w.begin),
                       "dur": ts(end) - ts(w.begin), "args": args})
        if (w.end is not None and rel is not None and rel.time is not None and rel.lane is not None
                and rel.kind in ("arrival", "write", "unstamped")):
            fid = len(flows) + 1
            flows.append({"ph": "s", "name": "release", "cat": "release", "id": fid,
                          "pid": pid_of[scope_of(rel.lane)], "tid": tid_of[rel.lane], "ts": ts(rel.time)})
            flows.append({"ph": "f", "bp": "e", "name": "release", "cat": "release", "id": fid,
                          "pid": pid_of[scope_of(w.lane)], "tid": tid_of[w.lane], "ts": ts(w.end)})
    slices.sort(key=lambda e: (e["pid"], e["tid"], e["ts"], -e["dur"]))
    ev += slices
    ops, _ = reader.operations(rec)
    asyncs = 0
    if ops:
        ev.append({"ph": "M", "name": "process_name", "pid": res_pid, "tid": 0, "args": {"name": "resources"}})
    aid = 0
    for k, (resource, lst) in enumerate(sorted(ops.items())):
        ev.append({"ph": "M", "name": "thread_name", "pid": res_pid, "tid": k + 1, "args": {"name": resource}})
        for o in lst:
            if o.done_time is None:
                continue
            aid += 1
            args = {"op_id": o.op_id, "issued_by": o.issue_lane, "region": o.issue_region,
                    "basis": o.basis, "note": "done is where a warp learned of completion: an upper bound"}
            common = {"name": resource, "cat": "operation", "id": f"{scope_of(o.issue_lane)}:{aid}",
                      "pid": res_pid, "tid": k + 1}
            ev.append({**common, "ph": "b", "ts": ts(o.issue_time), "args": args})
            ev.append({**common, "ph": "e", "ts": ts(o.done_time)})
            asyncs += 1
    ev += flows
    return {"traceEvents": ev, "displayTimeUnit": "ns",
            "otherData": {"format": rec.format,
                          "time_unit": "microseconds" if hz else "one cycle is written as one microsecond"}}


def counts(doc: dict) -> dict[str, int]:
    evs = doc["traceEvents"]
    return {"slices": sum(e["ph"] == "X" for e in evs), "async_slices": sum(e["ph"] == "b" for e in evs),
            "flows": sum(e["ph"] == "s" for e in evs)}
