"""`summary` and `diff`: a record, or two, in one screen (at most `LINES` lines of text).

`digest(rec)` reads a record once into plain data (the same data `--json` prints): its health, its launch, its window,
and per ROLE its lanes, its own period, its phases in order (cycles, time where the record has ties, share of the
window), its waits (a window and in all, by barrier) and its WORK, the period less the waits. The role whose work is the
largest share of the window is THE LONGEST CHAIN, named only where the record stamps waits to tell by.

`diff(before, after)` sets two digests of one kernel side by side: a row a role and phase, the change in cycles and
percent, and whether it is outside the spread. THE TEST: each record's own window-to-window standard deviation of a
quantity, over the square root of its windows, is the quantity's standard error; a change under TWO combined standard
errors (sqrt(se_before^2 + se_after^2)) is `~` (not resolved), over it `*`. It is the spread within one run: two runs of
one kernel can differ by more (clocks, placement, what else ran), so diff two runs of the same kernel first, and read a
change of a size they show between themselves as noise.
"""
from __future__ import annotations

import bisect
import math
import textwrap
from collections import defaultdict

from . import check, launch as launch_mod, rates, reader, window
from .record import Record, scope_of

LINES = 40
Z = 2.0               # a change under this many combined standard errors is not resolved
US = rates.US


class SummaryError(Exception):
    pass


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def label(name: str, occ: int, multi: bool) -> str:
    return f"{name}#{occ}" if multi else name


def digest(rec: Record, scope: str | None = None, anchor: str | None = None, skip: int = window.SKIP) -> dict:
    scopes = rec.scopes()
    if scope is None:
        scope = scopes[0]
    elif scope not in scopes:
        raise SummaryError(f"no scope {scope!r}; the record has {scopes}")
    lanes = [ln for ln in rec.lanes if scope_of(ln) == scope]
    findings = check.all_findings(rec)
    kinds: dict[str, int] = defaultdict(int)
    for f in findings:
        kinds[f.code] += 1
    open_end = {ln: v for ln, v in reader.open_at_end(rec).items() if scope_of(ln) == scope}
    out: dict = {
        "variant": rec.variant,
        "scope": scope,
        "scopes": len(scopes),
        "health": {"lanes": len(lanes), "roles": list(dict.fromkeys(rec.lanes[ln].role for ln in lanes)),
                   "stamps": sum(1 for s in rec.stamps if scope_of(s[0]) == scope),
                   "dropped": sum(n for ln, n in rec.dropped.items() if scope_of(ln) == scope),
                   "ties": sum(len(v) for ln, v in rec.ties.items() if scope_of(ln) == scope),
                   "findings": dict(kinds),
                   "open_at_end": {"lanes": len(open_end), "regions": sum(len(v) for v in open_end.values())}},
        "launch": None, "launch_note": None, "window": None, "window_note": None, "roles": [],
        "waits_stamped": False, "longest_chain": None,
    }
    try:
        lc = launch_mod.launch_of(rec, scope, anchor, skip, need_windows=False)
        out["launch"] = {"cycles": lc.cycles, "ns": lc.length_ns, "cycles_per_us": lc.cycles / lc.length_us,
                         "setup_cycles": lc.setup_cycles if lc.has_windows else None}
    except launch_mod.LaunchError as e:
        out["launch_note"] = "the record stamps no launch edges (`launch.begin`, `launch.end`) with ties: no launch length"
        if "REFUSED" not in str(e):
            out["launch_note"] = str(e)
    try:
        name, how, alane, starts = window.anchor_starts(rec, scope, anchor, skip)
    except window.WindowError as e:
        out["window_note"] = str(e)
        for role in dict.fromkeys(rec.lanes[ln].role for ln in lanes):
            out["roles"].append({"role": role, "lanes": sum(1 for ln in lanes if rec.lanes[ln].role == role),
                                 "period": None, "phases": [], "waits": None, "work": None, "work_share": None})
        return out
    spans = [b - a for a, b in zip(starts, starts[1:])]
    wp = _mean(spans)
    wns = None
    if rec.ties:
        try:
            wt = rates.window_in_time(rec, scope, name, skip)
        except window.WindowError:
            wt = None
        wns = None if wt is None else wt["period_ns"]
    out["window"] = {"anchor": name, "anchor_lane": alane, "windows": len(spans), "period": wp,
                     "period_std": _std(spans), "period_se": _std(spans) / math.sqrt(len(spans)),
                     "period_ns": wns, "skip": skip}
    # time of each phase, from the record's ties
    ns_of: dict[tuple[str, str, int], tuple[float, float]] = {}
    if rec.ties:
        try:
            res = rates.analyze(rec, name, skip, [scope], rates.TOLERANCE)
        except ValueError:
            res = None
        sr = None if res is None else res.scopes.get(scope)
        if sr is not None:
            for a in sr.by_role:
                ns_of[(a.role, a.name, a.occ)] = (a.ns, a.rate)
    # waits, by window
    ws = [w for w in reader.waits(rec) if scope_of(w.lane) == scope and w.end is not None]
    out["waits_stamped"] = bool(ws) or any(
        reader.classify(nm)[0] in ("wait_begin", "poll_begin") for ln in lanes for _i, nm, _t, _p in rec.lane_stamps(ln))
    nwin = len(spans)
    for rw in window.role_windows(rec, scope, name, skip):
        n_lanes = len(rw.lanes)
        phases = []
        for r in rw.rows:
            ns, rate = ns_of.get((rw.role, r.name, r.occ), (None, None))
            phases.append({"name": r.name, "occ": r.occ, "label": label(r.name, r.occ, r.multi),
                           "start": r.start, "cycles": r.dur, "ns": ns, "cycles_per_us": rate,
                           "share": r.dur / wp, "jitter": r.dur_jit, "se": r.dur_jit / math.sqrt(max(1, r.n)),
                           "held": r.share, "windows": r.n})
        per_total = [0.0] * nwin
        per_count = [0] * nwin
        by_barrier: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for w in ws:
            if w.lane not in rw.lanes or not (starts[0] <= w.begin < starts[-1]):
                continue
            k = bisect.bisect_right(starts, w.begin) - 1
            per_total[k] += (w.end - w.begin) / n_lanes
            per_count[k] += 1 / n_lanes
            b = by_barrier[reader.group(w.target)]
            b[0] += (w.end - w.begin) / n_lanes / nwin
            b[1] += 1 / n_lanes / nwin
        waits = None
        wait_mean = 0.0
        if out["waits_stamped"]:
            wait_mean = _mean(per_total)
            waits = {"count": _mean(per_count), "cycles": wait_mean, "share": wait_mean / wp,
                     "std": _std(per_total), "se": _std(per_total) / math.sqrt(nwin),
                     "by_barrier": [{"barrier": k, "count": v[1], "cycles": v[0], "share": v[0] / wp}
                                    for k, v in sorted(by_barrier.items(), key=lambda kv: -kv[1][0])]}
        work = rw.period - wait_mean
        out["roles"].append({"role": rw.role, "lanes": n_lanes, "period": rw.period, "period_std": rw.period_std,
                             "windows": rw.windows, "own_window": rw.own, "phases": phases, "waits": waits,
                             "work": work, "work_share": work / wp})
    if not out["waits_stamped"]:
        out["longest_chain"] = {"named": False, "role": None, "work_share": None,
                                "note": "the record stamps no waits (a coarse record): every role's work is its whole "
                                        "period, so no chain can be named; stamp the barrier waits to say"}
    else:
        ranked = sorted((r for r in out["roles"] if r["work_share"] is not None), key=lambda r: -r["work_share"])
        top = ranked[0]
        nxt = ranked[1] if len(ranked) > 1 else None
        out["longest_chain"] = {
            "named": True, "role": top["role"], "work_share": top["work_share"], "work": top["work"],
            "next_role": None if nxt is None else nxt["role"],
            "next_work_share": None if nxt is None else nxt["work_share"],
            "note": "work is the role's period less its stamped waits: a wait the record does not stamp counts as "
                    "work, and a role that waits on another's work is shown by its waits, not by what it waits for"}
    return out


# ------------------------------------------------------------------------------------------------ text

def _c(x: float) -> str:
    return f"{x:,.0f}"


def _t(ns) -> str:
    return "" if ns is None else (f"{ns / US:.3f} us" if ns >= 100 else f"{ns:.0f} ns")


def format_summary(d: dict, name: str = "", limit: int = LINES) -> str:
    h = d["health"]
    L: list[str] = []
    kinds = ", ".join(f"{k} x{v}" for k, v in h["findings"].items()) or "none"
    L.append(f"{name or 'record'}{' (variant ' + d['variant'] + ')' if d.get('variant') else ''}: scope {d['scope']}" + (f" (+{d['scopes'] - 1} more scopes: --cta)" if d["scopes"] > 1 else "")
             + f", {h['lanes']} lanes ({len(h['roles'])} roles), {h['stamps']:,} stamps, {h['dropped']:,} dropped, "
             f"{h['ties']:,} ties; findings: {kinds}" + ("  (`check` lists them)" if h["findings"] else "")
             + (f"; open at end: {h['open_at_end']['lanes']} lane(s), {h['open_at_end']['regions']} region(s) closed at "
                f"the last stamp (no launch.end; not findings)" if h["open_at_end"]["lanes"] else ""))
    lc = d["launch"]
    if lc is None:
        L.append(f"launch: {d['launch_note']}")
    else:
        setup = "" if lc["setup_cycles"] is None else f", {_c(lc['setup_cycles'])} cycles before the first window"
        L.append(f"launch: {_c(lc['cycles'])} cycles = {lc['ns'] / US:.3f} us ({lc['cycles_per_us']:,.0f} cycles an us "
                 f"over the whole launch){setup}")
    w = d["window"]
    if w is None:
        L.append(f"window: none: {d['window_note']}")
    else:
        pct = 100 * w["period_std"] / w["period"]
        L.append(f"window: `{w['anchor']}` on {w['anchor_lane']}: period {_c(w['period'])} cycles"
                 + (f" = {w['period_ns'] / US:.3f} us" if w["period_ns"] else "")
                 + f", spread {_c(w['period_std'])} cycles ({pct:.1f}%) over {w['windows']} steady windows (first "
                 f"{w['skip']} skipped)")
    roles = d["roles"]
    ch = d["longest_chain"]
    foot: list[str] = []
    if ch is not None:
        if ch["named"]:
            nx = "" if ch["next_role"] is None else f"; next {ch['next_role']} at {100 * ch['next_work_share']:.0f}%"
            foot.append(f"LONGEST CHAIN: role {ch['role']}: work {100 * ch['work_share']:.0f}% of the window{nx}.")
        else:
            foot.append("LONGEST CHAIN: not named.")
        foot += textwrap.wrap(f"the record cannot say: {ch['note']}.", 118, initial_indent="  ", subsequent_indent="    ")
    foot.append("phase time: cycles, and its own time from the ties where it has them (the cycle counter's rate changes "
                "between phases); +- the window-to-window spread")
    shown = [list(r["phases"]) for r in roles]
    fixed = len(L) + len(foot) + sum(1 + (1 if r["waits"] else 0) for r in roles)
    over = fixed + sum(len(p) for p in shown) - limit
    while over > 0 and any(shown):
        i = max(range(len(shown)), key=lambda k: len(shown[k]))
        j = min(range(len(shown[i])), key=lambda k: shown[i][k]["cycles"])
        shown[i].pop(j)
        over -= 1
    for r, ph in zip(roles, shown):
        if r["period"] is None:
            L.append(f"role {r['role']} ({r['lanes']} lanes): no window, no phases")
            continue
        cut = len(r["phases"]) - len(ph)
        wt = ""
        if r["waits"] is None and d["waits_stamped"] is False:
            wt = "waits: not stamped"
        worked = f"; work {_c(r['work'])} ({100 * r['work_share']:.0f}% of the window)" if d["waits_stamped"] else ""
        L.append(f"role {r['role']} ({r['lanes']} lanes): own period {_c(r['period'])} cycles{worked}" + (f"; {wt}" if wt else "")
                 + (f"; {cut} smaller phases not shown (--json has all)" if cut else ""))
        if r["waits"] is not None:
            x = r["waits"]
            top = ", ".join(f"{b['barrier']} {b['count']:.1f}x {_c(b['cycles'])} ({100 * b['share']:.0f}%)"
                            for b in x["by_barrier"][:3])
            more = len(x["by_barrier"]) - 3
            L.append(f"  waits a window: {x['count']:.1f} totalling {_c(x['cycles'])} cycles ({100 * x['share']:.0f}%)"
                     + (f"; by barrier: {top}" if top else "") + (f" +{more} more" if more > 0 else ""))
        for p in ph:
            held = "" if p["held"] >= 0.995 else f"  held {100 * p['held']:.0f}%"
            L.append(f"  {p['label']:<34} {_c(p['cycles']):>7} cycles {100 * p['share']:5.1f}%  "
                     f"{_t(p['ns']):>9}  +-{_c(p['jitter'])}{held}")
    L += foot
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ diff

class DiffError(Exception):
    pass


def comparable(a: Record, b: Record) -> str | None:
    """None when the two records are one kernel (the same lanes with the same roles), else what differs."""
    la = {ln: r.role for ln, r in a.lanes.items()}
    lb = {ln: r.role for ln, r in b.lanes.items()}
    if la == lb:
        return None
    parts = []
    only_a = [ln for ln in la if ln not in lb]
    only_b = [ln for ln in lb if ln not in la]
    if only_a:
        parts.append(f"lanes only in the first: {', '.join(only_a[:6])}" + (f" (+{len(only_a) - 6})" if len(only_a) > 6 else ""))
    if only_b:
        parts.append(f"lanes only in the second: {', '.join(only_b[:6])}" + (f" (+{len(only_b) - 6})" if len(only_b) > 6 else ""))
    diff_roles = [f"{ln}: {la[ln]!r} against {lb[ln]!r}" for ln in la if ln in lb and la[ln] != lb[ln]]
    if diff_roles:
        parts.append("roles differ on " + "; ".join(diff_roles[:4]))
    if a.variant is not None or b.variant is not None:
        parts.append(f"the records' variants: {a.variant or 'unnamed'!r} and {b.variant or 'unnamed'!r}")
    return "; ".join(parts)


def _row(kind: str, role: str, name: str, a: float, b: float, se_a: float, se_b: float) -> dict:
    ch = b - a
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    return {"kind": kind, "role": role, "name": name, "before": a, "after": b, "change": ch,
            "percent": (100 * ch / a if a else None), "se": se, "resolved": abs(ch) > Z * se}


def diff_digests(a: dict, b: dict) -> dict:
    rows = []
    out: dict = {"test": f"a change is `*` (outside the spread) when it exceeds {Z:g} combined standard errors, "
                         f"sqrt(se_before^2 + se_after^2), where a quantity's se is its window-to-window standard "
                         f"deviation over sqrt(windows); `~` when under (not resolved)",
                 "caveat": "this is the spread WITHIN each run, not between runs: two runs of one kernel can differ "
                           "by more; diff two runs of the same kernel first to see how much",
                 "launch": None, "window": None, "rows": rows, "only_before": [], "only_after": [],
                 "anchors": [(a["window"] or {}).get("anchor"), (b["window"] or {}).get("anchor")]}
    if a["launch"] and b["launch"]:
        x, y = a["launch"]["cycles"], b["launch"]["cycles"]
        out["launch"] = {"before": x, "after": y, "change": y - x, "percent": 100 * (y - x) / x,
                         "ns_before": a["launch"]["ns"], "ns_after": b["launch"]["ns"],
                         "note": "one value a record: no spread, so no mark"}
    if a["window"] and b["window"]:
        out["window"] = _row("window", "", "period", a["window"]["period"], b["window"]["period"],
                             a["window"]["period_se"], b["window"]["period_se"])
    ra = {r["role"]: r for r in a["roles"]}
    rb = {r["role"]: r for r in b["roles"]}
    for role, x in ra.items():
        y = rb[role]
        pa = {(p["name"], p["occ"]): p for p in x["phases"]}
        pb = {(p["name"], p["occ"]): p for p in y["phases"]}
        for k, p in pa.items():
            if k in pb:
                q = pb[k]
                rows.append(_row("phase", role, p["label"], p["cycles"], q["cycles"], p["se"], q["se"]))
            else:
                out["only_before"].append(f"{role}: {p['label']}")
        for k, q in pb.items():
            if k not in pa:
                out["only_after"].append(f"{role}: {q['label']}")
        if x["waits"] and y["waits"]:
            rows.append(_row("waits", role, "waits a window", x["waits"]["cycles"], y["waits"]["cycles"],
                             x["waits"]["se"], y["waits"]["se"]))
    out["waits_stamped"] = [a["waits_stamped"], b["waits_stamped"]]
    return out


def format_diff(d: dict, names=("before", "after"), limit: int = LINES) -> str:
    L = [f"diff {names[0]} -> {names[1]}"]
    L += textwrap.wrap("test: " + d["test"] + ".", 118, subsequent_indent="  ")
    L += textwrap.wrap("NOTE: " + d["caveat"] + ".", 118, subsequent_indent="  ")
    if None not in d["anchors"] and d["anchors"][0] != d["anchors"][1]:
        L.append(f"the two windows are anchored differently ({d['anchors'][0]} against {d['anchors'][1]}): phases are "
                 f"compared by their own durations, the window's period by each record's anchor")
    lc = d["launch"]
    if lc:
        L.append(f"launch: {_c(lc['before'])} -> {_c(lc['after'])} cycles ({lc['change']:+,.0f}, {lc['percent']:+.1f}%); "
                 f"{lc['ns_before'] / US:.3f} -> {lc['ns_after'] / US:.3f} us; {lc['note']}")
    else:
        L.append("launch: not in both records")
    w = d["window"]
    if w:
        L.append(f"window: {_c(w['before'])} -> {_c(w['after'])} cycles ({w['change']:+,.0f}, {w['percent']:+.1f}%)  "
                 f"{'*' if w['resolved'] else '~'}  (2 se = {_c(Z * w['se'])})")
    else:
        L.append("window: not in both records")
    if not all(d["waits_stamped"]):
        L.append("waits: " + ("not stamped in either record" if not any(d["waits_stamped"]) else
                              "stamped in only one record: not compared"))
    rows = d["rows"]
    tail = []
    if d["only_before"]:
        tail.append("only in before: " + ", ".join(d["only_before"][:6]) + (f" (+{len(d['only_before']) - 6})" if len(d["only_before"]) > 6 else ""))
    if d["only_after"]:
        tail.append("only in after: " + ", ".join(d["only_after"][:6]) + (f" (+{len(d['only_after']) - 6})" if len(d["only_after"]) > 6 else ""))
    room = limit - len(L) - len(tail) - 3
    ordered = sorted(rows, key=lambda r: (not r["resolved"], -abs(r["change"])))
    keep = ordered[: max(0, room)]
    L.append(f"  {'role':<12} {'phase / quantity':<30} {'before':>8} {'after':>8} {'change':>8} {'%':>7}   (2 se)")
    for r in keep:
        pct = "" if r["percent"] is None else f"{r['percent']:+.1f}%"
        L.append(f"  {r['role']:<12} {r['name']:<30} {_c(r['before']):>8} {_c(r['after']):>8} {r['change']:>+8,.0f} "
                 f"{pct:>7}  {'*' if r['resolved'] else '~'} ({_c(Z * r['se'])})")
    if len(keep) < len(rows):
        L.append(f"  ... {len(rows) - len(keep)} more rows, the smallest changes, not shown (--json has all)")
    L += tail
    nres = sum(1 for r in rows if r["resolved"])
    L.append(f"{nres} of {len(rows)} rows outside the spread (`*`); `~` rows are not resolved by this test")
    return "\n".join(L) + "\n"
