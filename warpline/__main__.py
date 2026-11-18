"""python -m warpline check|path|step|perfetto|window|rates|merge <record>; fold|overlay <csv>; set <manifest>"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import check, merge, overlay, perfetto, rates, reader, sampling, variants, window
from .record import RecordError, dump, load


def _scope(rec, wanted):
    scopes = rec.scopes()
    if wanted is None:
        return scopes[0]
    if wanted not in scopes:
        raise SystemExit(f"warpline: no scope {wanted!r}; the record has {scopes}")
    return wanted


def cmd_check(a) -> int:
    rec = load(a.record)
    n = 0
    for fn in check.CHECKS:
        found = fn(rec)
        n += len(found)
        print(f"{fn.__name__}: {'holds' if not found else str(len(found)) + ' finding(s)'}")
        shown = defaultdict(list)
        for f in found:
            shown[f.code].append(f)
        for code, fs in shown.items():
            print(f"  {code} x{len(fs)}")
            for f in fs[: a.show]:
                print(f"    {f}")
            if len(fs) > a.show:
                print(f"    ... {len(fs) - a.show} more")
    print(f"{n} finding(s)")
    return 1 if n and a.strict else 0


def cmd_path(a) -> int:
    rec = load(a.record)
    scope = _scope(rec, a.cta)
    windows = None
    lane = a.lane
    if a.steady:
        s = reader.steady(rec, a.steady, a.skip, scope)
        if s is None:
            print(f"warpline: too few `{a.steady}` regions in scope {scope!r}")
            return 1
        carrier, begin, end, windows = s
        lane = lane or carrier
    else:
        lanes = [ln for ln in rec.lanes if reader.scope_of(ln) == scope]
        times = [rec.lane_stamps(ln)[-1][2] for ln in lanes if rec.lane_stamps(ln)]
        begin = a.begin if a.begin is not None else min(rec.lane_stamps(ln)[0][2] for ln in lanes if rec.lane_stamps(ln))
        end = a.end if a.end is not None else max(times)
        lane = lane or max((ln for ln in lanes if rec.lane_stamps(ln)), key=lambda ln: rec.lane_stamps(ln)[-1][2])
    if a.begin is not None and a.steady:
        begin = a.begin
    if a.end is not None and a.steady:
        end = a.end
    if lane not in rec.lanes:
        raise SystemExit(f"warpline: no lane {lane!r}")
    p = reader.critical_path(rec, scope, begin, end, lane, a.blocking)
    per = f"cycles a window ({windows} windows)" if windows else "cycles"
    div = windows or 1
    print(f"scope {scope}: span {begin}..{end} = {end - begin:,} cycles, from lane {lane}; "
          f"blocking = a wait over {a.blocking} cycles; basis: measured unless marked upper_bound")
    total = sum(x.end - x.begin for x in p.pieces)
    print(f"pieces sum to {total:,} of {end - begin:,} cycles")

    by_role = defaultdict(float)
    members = defaultdict(int)
    for ln in rec.lanes:
        if reader.scope_of(ln) == scope:
            members[rec.lanes[ln].role] += 1
    for ln in (x for x in rec.lanes if reader.scope_of(x) == scope):
        role = rec.lanes[ln].role
        for s0, s1, st in reader._clip(rec, ln, begin, end):
            by_role[(role, _label(st))] += s1 - s0
    print(f"\nroles, {per} a warp, by state:")
    roles = defaultdict(list)
    for (role, st), v in by_role.items():
        roles[role].append((v / (members[role] * div), st))
    for role, rows in roles.items():
        print(f"  {role} ({members[role]} warps)")
        for v, st in sorted(rows, reverse=True)[:6]:
            print(f"    {v:10,.0f}  {st}")

    held = defaultdict(list)
    for w, src in reader.blocking_waits(rec, scope, begin, end, a.blocking):
        held[(rec.lanes[w.lane].role, reader.group(w.target), src)].append(w.end - w.begin)
    print(f"\nblocking waits, {per} a warp: held, by whom")
    for (role, tgt, src), v in sorted(held.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {sum(v) / (members[role] * div):10,.0f}  {role} at {tgt} <- {src}  [{len(v)} waits]")

    pieces = defaultdict(float)
    margins = defaultdict(list)
    bases = {}
    for x in p.pieces:
        label = f"{x.role}: {_label(x.state)}" if x.kind == "work" else f"{x.kind}: {x.state}"
        if x.kind != "work":
            label = x.state if x.kind == "flight" else f"{x.state} (waiter {x.role})"
        pieces[label] += x.end - x.begin
        bases[label] = x.basis
    for w, r in p.binding:
        key = (f"in flight: {reader.group(w.target)}" if r.kind == "unstamped"
               else f"release at {reader.group(w.target)} (waiter {rec.lanes[w.lane].role})")
        margins[key].append(w.end - w.begin)
    print(f"\nTHE PATH, {per} (a release: the mean margin it binds by = its waits' length):")
    for label, v in sorted(pieces.items(), key=lambda kv: -kv[1]):
        if v / div >= a.min:
            m = margins.get(label)
            extra = f"  (binding by {sum(m) / len(m):,.0f}, {len(m)} times)" if m else ""
            print(f"  {v / div:10,.0f}  {label}  [{bases[label]}]{extra}")

    known = sum(1 for r in p.slack if r.slack is not None)
    print(f"\nSLACK: {len(p.slack)} waits in the span are not on the path; slack = how much later the releaser could "
          f"have arrived before the lane would have joined the path (the path's time at the lane's next dependency "
          f"on it, minus the time the lane reached it); {known} known, {len(p.slack) - known} unknown")
    agg = defaultdict(list)
    unknown = defaultdict(int)
    why = {}
    for row in p.slack:
        key = (rec.lanes[row.wait.lane].role, reader.group(row.wait.target))
        if row.slack is None:
            unknown[key] += 1
            why.setdefault(key, row.reason)
        else:
            agg[key].append(row.slack)
    keys = sorted(set(agg) | set(unknown), key=lambda k: -(sum(agg.get(k, [])) + unknown.get(k, 0)))
    for key in keys[: a.rows]:
        role, tgt = key
        v = agg.get(key, [])
        line = f"  {role} at {tgt}: "
        line += (f"{len(v)} known, mean {sum(v) / len(v):8,.1f}, max {max(v):,}, {sum(1 for x in v if x == 0)} at 0"
                 if v else "none known")
        if unknown.get(key):
            line += f"; {unknown[key]} unknown (e.g. {why[key]})"
        print(line)
    return 0


def _label(state: str) -> str:
    if state.startswith("wait:"):
        return f"waits on {reader.group(state[5:])}"
    if state.startswith("poll:"):
        return f"polls {reader.group(state[5:])}"
    return state


def cmd_step(a) -> int:
    rec = load(a.record)
    r = reader.at(rec, a.at, _scope(rec, a.cta))
    print(f"scope {r['scope']!r} at t={r['time']}")
    for x in r["lanes"]:
        w = x["wait"]
        what = " > ".join(x["regions"]) or "(outside)"
        if w is not None:
            rel = w.release
            why = ""
            if rel is not None and rel.time is not None:
                why = f" (will be released by {rel.stamp} on {rel.lane} at {rel.time}; {rel.kind})"
            elif rel is not None and rel.note:
                why = f" ({rel.note})"
            what += f"  WAITING on {w.target} since {w.begin}{why}"
        last = f"last stamp `{x['last_stamp'][0]}` at {x['last_stamp'][1]}" if x["last_stamp"] else "no stamp yet"
        print(f"  {x['lane']:<14} {x['role']:<12} {what}   [{last}]")
    for resource, ops in r["resources"].items():
        if ops:
            for o in ops:
                done = f"learned done at {o.done_time} (upper bound)" if o.done_time is not None else "not known done"
                print(f"  resource {resource}: op {o.op_id} from {o.issue_lane} ({o.issue_region}) "
                      f"issued {o.issue_time}, {done}")
        else:
            print(f"  resource {resource}: idle")
    return 0


def cmd_perfetto(a) -> int:
    rec = load(a.record)
    doc = perfetto.build(rec)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    c = perfetto.counts(doc)
    print(f"{out}: {out.stat().st_size:,} bytes, {c['slices']:,} slices, {c['async_slices']:,} async slices, "
          f"{c['flows']:,} flows")
    return 0


def cmd_window(a) -> int:
    rec = load(a.record)
    try:
        av = window.average_window(rec, a.anchor, a.skip, [_scope(rec, a.cta)] if a.cta else None)
    except window.WindowError as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    if a.json or a.output:
        out = window.to_record(av)
        if a.output:
            dump(out, a.output)
            print(f"{a.output}: average window record, {len(out.stamps)} stamps", file=sys.stderr if a.json else sys.stdout)
        if a.json:
            print(json.dumps(out.to_dict(), separators=(",", ":")))
        return 0
    print(window.format_table(av), end="")
    return 0


def cmd_rates(a) -> int:
    rec = load(a.record)
    try:
        res = rates.analyze(rec, a.anchor, a.skip, [_scope(rec, a.cta)] if a.cta else None, a.tolerance / 100.0)
    except window.WindowError as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    print(rates.dumps(res) if a.json else rates.format_rates(res, a.rows), end="\n" if a.json else "")
    return 0


def cmd_merge(a) -> int:
    try:
        sources = [(p, load(p)) for p in a.records]
        m = merge.merge(sources, a.tolerance / 100.0, a.anchor, a.skip)
    except (merge.MergeError, window.WindowError) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    print(merge.format_report(m), end="")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    dump(m.record, a.output)
    print(f"wrote {a.output}")
    return 0


def _config(a) -> sampling.Config:
    cfg = sampling.Config(bins=a.bins, tolerance=a.tolerance / 100.0, skip=a.skip_windows, tail=a.tail,
                          min_score=a.min_score, live=a.live_threshold, length_tol=a.length_tolerance,
                          shift_slack=a.shift_slack)
    if a.range:
        cfg.lo, cfg.hi = a.range
    return cfg


def _fold(a):
    exports = [sampling.read_export(p) for p in a.csv]
    return exports, sampling.fold(exports, _config(a))


def cmd_fold(a) -> int:
    try:
        f = _fold(a)[1]
    except sampling.SamplingError as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    print(json.dumps(sampling.to_dict(f), indent=1) if a.json else sampling.format_fold(f), end="\n" if a.json else "")
    return 0


def cmd_overlay(a) -> int:
    try:
        rec = load(a.record)
        lc = overlay.launch_for(rec, a.cta)           # refused, naming what the kernel must stamp, before the samples
        view = overlay.view_of(rec, lc.scope, a.record)
        exports, f = _fold(a)
        pl = overlay.place(view, lc, exports, f)
    except (sampling.SamplingError, overlay.OverlayError) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(overlay.place_to_dict(pl), indent=1))
    else:
        print(overlay.format_overlay(pl, a.record, a.csv), end="")
    if a.perfetto:
        out = Path(a.perfetto)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(overlay.perfetto_doc(view, pl.values(), pl.bins, {"cross_check": pl.cross.note}),
                                  separators=(",", ":")))
        print(f"wrote {out}", file=sys.stderr if a.json else sys.stdout)
    return 0


def cmd_set(a) -> int:
    try:
        res = variants.run(a.manifest, _config(a), a.cost / 100.0, a.merge_unchecked,
                           ties_tolerance=a.ties_tolerance / 100.0)
    except (variants.SetError, sampling.SamplingError, overlay.OverlayError, merge.MergeError,
            window.WindowError) as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(variants.to_dict(res), indent=1))
    else:
        print(variants.format_set(res), end="")
    if a.output and res.merged is not None:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        dump(res.merged, a.output)
        print(f"wrote {a.output}", file=sys.stderr if a.json else sys.stdout)
    if a.perfetto and res.overlays:
        o = res.overlays[0]
        out = Path(a.perfetto)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(overlay.perfetto_doc(
            o.through.view, {x: ps.bins for x, ps in o.series.items() if ps.bins is not None}, o.through.placement.bins,
            {"plain_placed_through": o.through.name}), separators=(",", ":")))
        print(f"wrote {out}", file=sys.stderr if a.json else sys.stdout)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m warpline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="tier-0 checks")
    c.add_argument("record")
    c.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    c.add_argument("--show", type=int, default=5, help="findings shown per code")
    c.set_defaults(fn=cmd_check)
    p = sub.add_parser("path", help="critical path and slack")
    p.add_argument("record")
    p.add_argument("--cta", help="the scope (a lane id's part before /); default the first")
    p.add_argument("--lane", help="the lane the path starts from; default the one with the last stamp")
    p.add_argument("--steady", metavar="REGION", help="span by a repeating region (see --skip)")
    p.add_argument("--skip", type=int, default=2, help="regions skipped at the start of a --steady span")
    p.add_argument("--from", dest="begin", type=int, help="span start (cycles)")
    p.add_argument("--to", dest="end", type=int, help="span end (cycles)")
    p.add_argument("--blocking", type=int, default=reader.BLOCKING, help="a wait longer than this binds (cycles)")
    p.add_argument("--min", type=float, default=20, help="path pieces below this (a window) are not printed")
    p.add_argument("--rows", type=int, default=20)
    p.set_defaults(fn=cmd_path)
    s = sub.add_parser("step", help="what every lane and resource is doing at a time")
    s.add_argument("record")
    s.add_argument("--at", type=int, required=True)
    s.add_argument("--cta")
    s.set_defaults(fn=cmd_step)
    o = sub.add_parser("perfetto", help="write Chrome trace-event JSON")
    o.add_argument("record")
    o.add_argument("-o", "--output", required=True)
    o.set_defaults(fn=cmd_perfetto)
    w = sub.add_parser("window", help="the average window of a record: each stamp's offset in the steady windows")
    w.add_argument("record")
    w.add_argument("--anchor", help="the anchor region (default: the region whose begin recurs most regularly)")
    w.add_argument("--skip", type=int, default=window.SKIP, help="windows skipped at the start")
    w.add_argument("--cta", help="only this scope")
    w.add_argument("--json", action="store_true", help="print an average-window record instead of the table")
    w.add_argument("-o", "--output", help="write the average-window record here")
    w.set_defaults(fn=cmd_window)
    rt = sub.add_parser("rates", help="the rate between the cycle counter and time, read from a record's ties")
    rt.add_argument("record")
    rt.add_argument("--anchor", help="the window's anchor region (default: the region whose begin recurs most regularly)")
    rt.add_argument("--skip", type=int, default=window.SKIP, help="windows skipped at the start")
    rt.add_argument("--cta", help="only this scope")
    rt.add_argument("--tolerance", type=float, default=rates.TOLERANCE * 100,
                    help="percent: the lanes of a scope agree on the rate over one window within this")
    rt.add_argument("--rows", type=int, default=6, help="consecutive-tie pairs shown a lane")
    rt.add_argument("--json", action="store_true")
    rt.set_defaults(fn=cmd_rates)
    g = sub.add_parser("merge", help="merge runs on their anchors")
    g.add_argument("records", nargs="+", help="one record a run; the first is the reference when runs tie")
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--tolerance", type=float, default=merge.TOLERANCE * 100, help="percent of the window's period")
    g.add_argument("--anchor", help="the window's anchor region (default: the most regular one common to every run)")
    g.add_argument("--skip", type=int, default=window.SKIP)
    g.set_defaults(fn=cmd_merge)
    def sampling_args(p):
        p.add_argument("--bins", type=int, default=8, help="bins of the window (at least 10 samples a bin)")
        p.add_argument("--tolerance", type=float, default=1.0, help="percent: periods of series, halves and runs agree")
        p.add_argument("--skip-windows", type=float, default=4.0, help="windows skipped past a series' start")
        p.add_argument("--tail", type=int, default=3, help="samples dropped before a series' live end")
        p.add_argument("--min-score", type=float, default=0.25, help="a fold explaining less of the variance is flat")
        p.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"), help="period scan range, samples")
        p.add_argument("--live-threshold", type=float, default=0.0,
                       help="a sample is live when any series of its pass exceeds this (default: any nonzero value)")
        p.add_argument("--length-tolerance", type=float, default=0.25,
                       help="a fraction of the interval: the entry/exit assumption's own slack in the length check")
        p.add_argument("--shift-slack", type=float, default=0.5,
                       help="bins: the run alignment's resolution, carried into the intersection of the runs' intervals")

    f = sub.add_parser("fold", help="a profiler's sampled series: the window's period, the checks, the fold")
    f.add_argument("csv", nargs="+", help="one export a run")
    sampling_args(f)
    f.add_argument("--json", action="store_true")
    f.set_defaults(fn=cmd_fold)
    ov = sub.add_parser("overlay", help="lay the samples' fold under a stamp record's average window")
    ov.add_argument("record", help="a stamp record with the launch's two edges (`launch.begin`, `launch.end`, tied)")
    ov.add_argument("csv", nargs="+")
    sampling_args(ov)
    ov.add_argument("--cta", help="the scope of the record (default: the first)")
    ov.add_argument("--json", action="store_true")
    ov.add_argument("--perfetto", metavar="OUT", help="write the stamp lanes with a counter track a series")
    ov.set_defaults(fn=cmd_overlay)
    st = sub.add_parser("set", help="a set of variants the user promises are one kernel")
    st.add_argument("manifest")
    sampling_args(st)
    st.add_argument("--cost", type=float, default=6.0, help="percent: a variant's window against the plain one's")
    st.add_argument("--ties-tolerance", type=float, default=variants.TIES * 100,
                    help="percent: a variant's window in time from its ties against the samples'")
    st.add_argument("--merge-unchecked", action="store_true", help="merge variants whose promise could not be checked")
    st.add_argument("--json", action="store_true")
    st.add_argument("-o", "--output", help="write the merged record")
    st.add_argument("--perfetto", metavar="OUT")
    st.set_defaults(fn=cmd_set)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except RecordError as e:
        print(f"warpline: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
