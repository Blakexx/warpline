"""`summary` and `diff`: a record, or two, in one screen (at most `LINES` lines of text).

`digest(rec)` reads a record once into plain data (the same data `--json` prints): its health, its launch, its window,
and per ROLE its lanes, its own period, its phases in order (cycles, time where the record has ties, share of the
window), its waits (a window and in all, by barrier) and its WORK, the period less the waits. The role whose work is the
largest share of the window is THE LONGEST CHAIN, named only where EVERY role's waits are stamped.

WHAT A RECORD CAN SAY ABOUT A ROLE'S WAITS (`wait_states`), from its stamps alone, one of three: STAMPED (a lane of the
role stamps a wait or a poll begin: its waits are measured), KNOWN_UNSTAMPED (no wait is stamped, but a lane arrives at a
barrier: the role takes part in barriers whose waits this record does not stamp), UNKNOWN (neither: the record cannot
say). A role not stamped has no waits figure, no work share and is never the longest chain: its period less nothing is
not its work. The format declares no role at a barrier (`facts.barriers` carries counts only), so a role that waits at a
barrier and stamps no arrival there cannot be seen from the record.

`digest_many` takes SEVERAL records of one kernel (one detail a run): the window and phases of the first, each other
record's period and phases checked against it, and each role's waits from the record(s) that stamp them.

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

from ..manifest import RunPart
from . import check, launch as launch_mod, merge as merge_mod, rates, reader, window
from .record import Record, scope_of

LINES = 40
Z = 2.0               # a change under this many combined standard errors is not resolved
US = rates.US


STAMPED, KNOWN_UNSTAMPED, UNKNOWN = "stamped", "known_unstamped", "unknown"
TOLERANCE = merge_mod.TOLERANCE   # runs agree within max(Z combined standard errors, this share of the window's period)


class SummaryError(Exception):
    pass


def wait_states(rec: Record, scope: str) -> dict[str, dict]:
    """role -> {"state": STAMPED | KNOWN_UNSTAMPED | UNKNOWN, "arrives": barrier groups the role's lanes arrive at}."""
    seen: dict[str, dict] = {}
    for ln, lane in rec.lanes.items():
        if scope_of(ln) != scope:
            continue
        o = seen.setdefault(lane.role, {"stamped": False, "arrives": set()})
        for _i, nm, _t, _p in rec.lane_stamps(ln):
            kind, key = reader.classify(nm)
            if kind in ("wait_begin", "poll_begin"):
                o["stamped"] = True
            elif kind == "arrive":
                o["arrives"].add(reader.group(key))
    return {role: {"state": STAMPED if o["stamped"] else (KNOWN_UNSTAMPED if o["arrives"] else UNKNOWN),
                   "arrives": sorted(o["arrives"])} for role, o in seen.items()}


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
    states = wait_states(rec, scope)
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
                                 "period": None, "phases": [], "waits": None, "work": None, "work_share": None,
                                 "waits_state": states[role]["state"], "arrives_at": states[role]["arrives"]})
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
    out["waits_stamped"] = any(v["state"] == STAMPED for v in states.values())
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
        st = states[rw.role]["state"]
        if st == STAMPED:
            wait_mean = _mean(per_total)
            waits = {"count": _mean(per_count), "cycles": wait_mean, "share": wait_mean / wp,
                     "std": _std(per_total), "se": _std(per_total) / math.sqrt(nwin),
                     "by_barrier": [{"barrier": k, "count": v[1], "cycles": v[0], "share": v[0] / wp}
                                    for k, v in sorted(by_barrier.items(), key=lambda kv: -kv[1][0])]}
        work = rw.period - wait_mean if st == STAMPED else None
        out["roles"].append({"role": rw.role, "lanes": n_lanes, "period": rw.period, "period_std": rw.period_std,
                             "windows": rw.windows, "own_window": rw.own, "phases": phases, "waits": waits,
                             "work": work, "work_share": None if work is None else work / wp,
                             "waits_state": st, "arrives_at": states[rw.role]["arrives"]})
    out["longest_chain"] = chain_of(out["roles"])
    return out


def chain_of(roles: list[dict]) -> dict:
    """The longest chain, named only among roles whose waits are stamped, and only when every role's are."""
    bad = [r["role"] for r in roles if r["waits_state"] != STAMPED]
    if not roles:
        return {"named": False, "role": None, "work_share": None, "unstamped_roles": [], "note": "the record has no roles"}
    if bad and len(bad) == len(roles):
        return {"named": False, "role": None, "work_share": None, "unstamped_roles": bad,
                "note": "the record stamps no waits (a coarse record): a role's period is not its work, so no chain can "
                        "be named; stamp the barrier waits to say"}
    if bad:
        return {"named": False, "role": None, "work_share": None, "unstamped_roles": bad,
                "note": f"the waits of {', '.join(bad)} are not stamped in this record (a role's period is not its "
                        f"work), so the chain is not named"}
    ranked = sorted(roles, key=lambda r: -r["work_share"])
    top = ranked[0]
    nxt = ranked[1] if len(ranked) > 1 else None
    return {"named": True, "role": top["role"], "work_share": top["work_share"], "work": top["work"],
            "unstamped_roles": [],
            "next_role": None if nxt is None else nxt["role"],
            "next_work_share": None if nxt is None else nxt["work_share"],
            "note": "work is the role's period less its stamped waits: a wait the record does not stamp counts as "
                    "work, and a role that waits on another's work is shown by its waits, not by what it waits for"}


# ------------------------------------------------------------------------------------------------ several runs

def _agreement(base: dict, d: dict, tol: float = TOLERANCE) -> dict:
    """How one run's window and phases agree with the base run's: the period, then each phase of each role, within the
    larger of `Z` combined standard errors and `tol` (a fraction) of the window's period. Names what does not."""
    out = {"agrees": True, "compared": 0, "disagree": [], "period_change": None}
    if d["window"] is None or base["window"] is None:
        out.update(agrees=False, disagree=["no window"])
        return out
    wp = base["window"]["period"]
    ch = d["window"]["period"] - wp
    se = math.sqrt(d["window"]["period_se"] ** 2 + base["window"]["period_se"] ** 2)
    out["period_change"] = 100 * ch / wp
    if abs(ch) > max(Z * se, tol * wp):
        out["disagree"].append(f"period {ch:+,.0f} cycles ({100 * ch / wp:+.1f}%)")
    ra = {r["role"]: r for r in base["roles"]}
    for r in d["roles"]:
        pa = {(p["name"], p["occ"]): p for p in ra[r["role"]]["phases"]}
        for p in r["phases"]:
            q = pa.get((p["name"], p["occ"]))
            if q is None:
                continue
            out["compared"] += 1
            c = p["cycles"] - q["cycles"]
            if abs(c) > max(Z * math.sqrt(p["se"] ** 2 + q["se"] ** 2), tol * wp):
                out["disagree"].append(f"{r['role']} {p['label']} {c:+,.0f} cycles ({100 * c / wp:+.1f}% of the window)")
    out["agrees"] = not out["disagree"]
    return out


def digest_many(sources: list[tuple[str, Record]], scope: str | None = None, anchor: str | None = None,
                skip: int = window.SKIP, tol: float = TOLERANCE) -> dict:
    """The digest of several records of ONE kernel (a detail a run): the first's launch, window and phases; each other
    record's period and phases checked against it (a record that disagrees is named and its waits are not taken); each
    role's waits from the record(s) that stamp them, a barrier from the first that stamps it; the longest chain named
    once every role's waits are stamped by some record. It reuses the merge's common anchor and tolerance; it does not
    use the merge itself, which overlays stamps and drops a run on its anchors' offsets: here the question is whether the
    runs' NUMBERS agree."""
    if len(sources) < 2:
        raise SummaryError("several records: name at least two")
    recs = [r for _n, r in sources]
    for (n, r) in sources[1:]:
        why = comparable(recs[0], r)
        if why:
            raise SummaryError(f"the records are not one kernel ({why}): {n} differs from {sources[0][0]}")
    sc = scope or recs[0].scopes()[0]
    try:
        name, _how = merge_mod._common_anchor(recs, sc, anchor)
    except (merge_mod.MergeError, window.WindowError) as e:
        raise SummaryError(f"no one window anchor for the records: {e}")
    ds = [digest(r, sc, name, skip) for r in recs]
    d = ds[0]
    if d["window"] is None:
        raise SummaryError(f"the first record has no window: {d['window_note']}")
    runs = []
    for i, ((n, r), x) in enumerate(zip(sources, ds)):
        ag = {"agrees": True, "compared": 0, "disagree": [], "period_change": 0.0} if i == 0 else _agreement(d, x, tol)
        tw = r.twin if r.twin and isinstance(r.twin.get("plain"), (int, float)) and r.twin["plain"] > 0 \
            and isinstance(r.twin.get("recorded"), (int, float)) else None
        runs.append({"name": n, "variant": r.variant, "base": i == 0, "period": None if x["window"] is None
                     else x["window"]["period"], "twin_ratio": None if tw is None else tw["recorded"] / tw["plain"],
                     "stamps_waits_of": [q["role"] for q in x["roles"] if q["waits_state"] == STAMPED],
                     "arrives_only": [q["role"] for q in x["roles"] if q["waits_state"] == KNOWN_UNSTAMPED], **ag})
    wp = d["window"]["period"]
    by_role = [{r["role"]: r for r in x["roles"]} for x in ds]
    for role in d["roles"]:
        taken: dict[str, tuple[int, dict]] = {}
        stamped_in = []
        arrives = set(role["arrives_at"])
        for i, x in enumerate(ds):
            q = by_role[i].get(role["role"])
            if q is None:
                continue
            arrives |= set(q["arrives_at"])
            if q["waits_state"] != STAMPED or not runs[i]["agrees"]:
                continue
            stamped_in.append(i)
            for b in q["waits"]["by_barrier"]:
                taken.setdefault(b["barrier"], (i, b))
        role["arrives_at"] = sorted(arrives)
        if not stamped_in:
            role.update(waits=None, work=None, work_share=None, waits_state=(KNOWN_UNSTAMPED if arrives else UNKNOWN),
                        waits_runs=[])
            continue
        by = [{"barrier": b, "count": e["count"], "cycles": e["cycles"], "share": e["cycles"] / wp,
               "run": sources[i][0]} for b, (i, e) in sorted(taken.items(), key=lambda kv: -kv[1][1]["cycles"])]
        cyc = sum(b["cycles"] for b in by)
        role.update(waits={"count": sum(b["count"] for b in by), "cycles": cyc, "share": cyc / wp, "std": None,
                           "se": None, "by_barrier": by},
                    waits_state=STAMPED, work=role["period"] - cyc, work_share=(role["period"] - cyc) / wp,
                    waits_runs=[sources[i][0] for i in stamped_in])
    d["waits_stamped"] = any(r["waits_state"] == STAMPED for r in d["roles"])
    d["longest_chain"] = chain_of(d["roles"])
    skipped = [r["name"] for r in runs if not r["agrees"]]
    if skipped and not d["longest_chain"]["named"]:
        d["longest_chain"]["note"] += f"; the waits of {', '.join(skipped)} were not taken (its window disagrees)"
    d["runs"] = runs
    d["anchor_common"] = name
    d["tolerance"] = tol
    return d


# ------------------------------------------------------------------------------------------------ a run manifest's figures

def run_figures(d: dict) -> RunPart:
    """What the recorder gives a run manifest (docs/run-manifest.md): a digest's figures (`digest`, `digest_many`) under names a
    bound can address, and the lines on what was not measured. Where the digest has null (no launch edges with ties, no
    window, a role's waits not stamped, a standard error several records do not give) the figure is absent and a line says
    why: never 0 in place of not measured. The records' names are not written; a record is named by its place among them."""
    said = []
    h = d["health"]
    out: dict = {"health": {"lanes": h["lanes"], "roles": len(h["roles"]), "stamps": h["stamps"], "dropped": h["dropped"],
                            "findings": dict(sorted(h["findings"].items())),
                            "open_at_end": {"lanes": h["open_at_end"]["lanes"], "regions": h["open_at_end"]["regions"]}}}
    if d["scopes"] > 1:
        said.append(f"note: the figures are of scope {d['scope']}, the first of {d['scopes']}: the others are not measured")
    lc = d["launch"]
    if lc is None:
        said.append(f"note: the launch is not measured: {d['launch_note']}")
    else:
        out["launch"] = {"cycles": lc["cycles"], **({"ns": lc["ns"]} if lc["ns"] is not None else {})}
        if lc["ns"] is None:
            said.append("note: the launch's time is not measured: its edges have no ties")
    w = d["window"]
    if w is None:
        said.append(f"note: no window, so no period and no role's phases: {d['window_note']}")
    else:
        out["window"] = {"anchor": w["anchor"], "period": w["period"], "period_std": w["period_std"],
                         "period_se": w["period_se"], "windows": w["windows"]}
    for i, run in enumerate(d.get("runs") or ()):
        if not run["agrees"]:
            said.append(f"note: record {i + 1} disagrees with the first, its waits are not taken: " + "; ".join(run["disagree"]))
    roles: dict = {}
    for r in d["roles"]:
        if r["period"] is None:
            continue
        role = {"period": r["period"]}
        x = r["waits"]
        if x is None:
            state = ("it arrives at " + ", ".join(r["arrives_at"]) if r["waits_state"] == KNOWN_UNSTAMPED
                     else "no barrier stamped on it")
            said.append(f"note: waits not stamped for role {r['role']} ({state}): no waits, no work")
        else:
            role["work"] = r["work"]
            role["work_share"] = r["work_share"]
            waits = {"count": x["count"], "cycles": x["cycles"]}
            if x["se"] is None:
                said.append(f"note: the waits of role {r['role']} come from several records: no standard error of them")
            else:
                waits["cycles_se"] = x["se"]
            waits["share"] = x["share"]
            waits["by_barrier"] = {b["barrier"]: {"count": b["count"], "cycles": b["cycles"], "share": b["share"]}
                                   for b in x["by_barrier"]}
            role["waits"] = waits
        role["phases"] = {p["label"]: {"cycles": p["cycles"], "cycles_se": p["se"], "share": p["share"], "jitter": p["jitter"],
                                       "held": p["held"]} for p in r["phases"]}
        roles[r["role"]] = role
    if roles:
        out["roles"] = roles
    return RunPart(figures=out, said=tuple(said))


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
    runs = d.get("runs")
    if runs:
        L += textwrap.wrap(
            f"runs: {len(runs)} records of one kernel; launch, window and phases are the first's; each run's period and "
            f"phases must agree with it within {Z:g} combined se or {100 * d['tolerance']:.0f}% of the window "
            f"(--tolerance). A role's waits come from the run that stamps them: different runs, each a few percent "
            f"from the plain kernel.", 118,
            subsequent_indent="  ")
        for r in runs:
            tw = "" if r["twin_ratio"] is None else f", {100 * (r['twin_ratio'] - 1):+.1f}% against its plain twin"
            who = ", ".join(r["stamps_waits_of"]) or "no role"
            if r["base"]:
                ag = "base"
            elif r["agrees"]:
                ag = f"agrees ({r['period_change']:+.1f}%; {r['compared']} phases)"
            else:
                ag = "DISAGREES, its waits are not taken: " + "; ".join(r["disagree"][:3]) \
                    + (f" (+{len(r['disagree']) - 3} more)" if len(r["disagree"]) > 3 else "")
            L += textwrap.wrap(f"  {r['name']}: {ag}{tw}; stamps the waits of: {who}", 118, subsequent_indent="      ")
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
        if r["waits_state"] == STAMPED:
            wt = ""
            worked = f"; work {_c(r['work'])} ({100 * r['work_share']:.0f}% of the window)"
        else:
            worked = ""
            wt = "waits: not stamped in this record" + (
                f" (arrives at {', '.join(r['arrives_at'][:2])}{' +' + str(len(r['arrives_at']) - 2) + ' more' if len(r['arrives_at']) > 2 else ''}; "
                f"its waits are not stamped)" if r["waits_state"] == KNOWN_UNSTAMPED else " (no barrier stamped on it)")
        L.append(f"role {r['role']} ({r['lanes']} lanes): own period {_c(r['period'])} cycles{worked}" + (f"; {wt}" if wt else "")
                 + (f"; {cut} smaller phases not shown (--json has all)" if cut else ""))
        if r["waits"] is not None:
            x = r["waits"]
            top = ", ".join(f"{b['barrier']} {b['count']:.1f}x {_c(b['cycles'])} ({100 * b['share']:.0f}%)"
                            for b in x["by_barrier"][:3])
            more = len(x["by_barrier"]) - 3
            src = f" from {', '.join(r['waits_runs'])}" if runs else ""
            L.append(f"  waits a window{src}: {x['count']:.1f} totalling {_c(x['cycles'])} cycles ({100 * x['share']:.0f}%)"
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
                 "launch": None, "window": None, "rows": rows, "waits_not_compared": [], "only_before": [], "only_after": [],
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
        if x["waits_state"] == STAMPED and y["waits_state"] == STAMPED:
            rows.append(_row("waits", role, "waits a window", x["waits"]["cycles"], y["waits"]["cycles"],
                             x["waits"]["se"], y["waits"]["se"]))
        else:
            # not stamped in one or both: not a quantity of that record, so no change is reported
            out["waits_not_compared"].append({"role": role, "before": x["waits_state"], "after": y["waits_state"],
                                              "before_cycles": None if x["waits"] is None else x["waits"]["cycles"],
                                              "after_cycles": None if y["waits"] is None else y["waits"]["cycles"]})
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
    if d["waits_not_compared"]:
        def _w(state, cyc):
            return f"{_c(cyc)} cycles" if state == STAMPED else "not stamped"
        L += textwrap.wrap(
            "waits not compared (a role's waits not stamped in a record is not 0, and a change from not stamped to a "
            "number is not a change): " + "; ".join(
                f"{x['role']} (before {_w(x['before'], x['before_cycles'])}, after {_w(x['after'], x['after_cycles'])})"
                for x in d["waits_not_compared"]) + ".", 118, subsequent_indent="  ")
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
