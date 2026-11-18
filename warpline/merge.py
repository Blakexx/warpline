"""The merge of several runs on their anchors.

A stamp is switched at compile time, so each run is its own binary and carries its own stamps; every run carries the
same few ANCHORS (stamp names present in every run) and, inside its stamp budget, one group of detail. Each run's
average window (see `window`) is taken on one anchor region common to all the runs; the runs are then overlaid.

  CONSISTENCY  for each anchor, scope and role, its mean offset must agree across runs within the tolerance (a
               fraction of the window's period), and so must the periods. The reference is the run that most others
               agree with (ties: the first run named). A run that disagrees is FLAGGED with the anchor and the sizes,
               and left out of the merged timeline; its detail is reported as untrusted.
  TWIN         a run whose record carries `twin` ({"plain", "recorded"} cycles) is flagged when recorded/plain
               exceeds 1 + 2 x tolerance; a run with no `twin` is reported as unchecked, never as passing. A twin flag
               alone does not leave the run out: its offsets already agree with the reference's.
  TIMELINE     one average window: the anchors at their mean over the accepted runs; every other stamp from the run
               that carried it (if two accepted runs carry it, their mean, and a disagreement beyond the tolerance is
               a finding, the first run's offset kept). Each stamp is tagged with its runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import record as rec_mod
from . import bound, window
from .record import Record, classify

TOLERANCE = 0.03   # a fraction of the window's period (the command takes percent)
HALF = 0.5         # an offset counts when its stamp is held by at least this share of the windows


class MergeError(Exception):
    pass


@dataclass
class Run:
    index: int
    source: str
    rec: Record
    av: window.Average
    twin_status: str = "unchecked"       # ok | over | unchecked
    twin_ratio: float | None = None
    flags: list[str] = field(default_factory=list)   # consistency findings, one line each
    accepted: bool = True
    roles: dict = field(default_factory=dict)        # (scope, role, name, occ) -> RoleRow


@dataclass
class Merged:
    tolerance: float
    runs: list[Run]
    reference: int
    reference_why: str
    anchor_region: dict[str, str]
    anchors: list[str]
    findings: list[str]
    untrusted: dict[int, list[str]]
    record: Record | None


def _names_in(av: window.Average) -> set[str]:
    return {r.name for sw in av.scopes.values() for r in sw.rows}


def _common_anchor(recs: list[Record], scope: str, requested: str | None) -> tuple[str, str]:
    if requested is not None:
        names = {window.resolve_anchor(r, requested) for r in recs}
        if len(names) != 1:
            raise MergeError(f"anchor {requested!r} is not a stamp name of every run")
        return names.pop(), "requested"
    cands = [window.candidates(r, scope) for r in recs]
    common = set(cands[0]).intersection(*[set(c) for c in cands[1:]])
    if not common:
        raise MergeError(f"scope {scope!r}: no region begin recurs in every run; name one with --anchor")
    ids = {name: nid for nid, name in recs[0].names.items()}
    best = min(common, key=lambda n: (sum(window._cv(c[n][1]) for c in cands), ids[n]))
    cvs = ", ".join(f"{window._cv(c[best][1]):.4f}" for c in cands)
    return best, f"the begin that recurs most regularly in every run (cv {cvs} by run)"


def _compare(run: Run, ref: Run, anchors: set[str], tol: float) -> list[tuple[float, str]]:
    """How `run` disagrees with `ref` on the periods and the anchors: [(size in periods, one line)]."""
    out = []
    for sc, sw in run.av.scopes.items():
        rp = ref.av.scopes[sc].period
        d = sw.period - rp
        if abs(d) > tol * rp:
            out.append((1e9 + abs(d) / rp, f"period of {sc}: {sw.period:,.1f} against {rp:,.1f} cycles "
                                     f"({d:+,.1f}, {100 * d / rp:+.1f}% of the window; bound {100 * tol:.1f}%)"))
    for key, row in run.roles.items():
        sc, role, name, occ = key
        if name not in anchors:
            continue
        other = ref.roles.get(key)
        rp = ref.av.scopes[sc].period
        used_a, used_b = run.av.scopes[sc].windows, ref.av.scopes[sc].windows
        if other is None:
            if row.n >= HALF * used_a:
                out.append((1.0, f"anchor `{name}`#{occ} role {role} scope {sc}: held {row.n}/{used_a} windows by "
                                  f"this run, not held by the reference (run {ref.index})"))
            continue
        d = row.mean - other.mean
        if abs(d) > tol * rp:
            out.append((abs(d) / rp, f"anchor `{name}`#{occ} role {role} scope {sc}: offset {row.mean:,.1f} against "
                                     f"{other.mean:,.1f} ({d:+,.1f} cycles, {100 * d / rp:+.1f}% of the period "
                                     f"{rp:,.0f}; bound {100 * tol:.1f}%)"))
    for key, other in ref.roles.items():
        sc, role, name, occ = key
        if name in anchors and key not in run.roles and other.n >= HALF * ref.av.scopes[sc].windows:
            out.append((1.0, f"anchor `{name}`#{occ} role {role} scope {sc}: held by the reference "
                             f"(run {ref.index}) in {other.n}/{ref.av.scopes[sc].windows} windows, not by this run"))
    out.sort(key=lambda x: -x[0])
    return out


def merge(sources: list[tuple[str, Record]], tolerance: float = TOLERANCE, anchor: str | None = None,
          skip: int = window.SKIP) -> Merged:
    if len(sources) < 2:
        raise MergeError("a merge needs at least two runs")
    recs = [r for _s, r in sources]
    findings: list[str] = []
    scopes = [sc for sc in recs[0].scopes() if all(sc in r.scopes() for r in recs[1:])]
    for i, r in enumerate(recs):
        lost = [sc for sc in r.scopes() if sc not in scopes]
        if lost:
            findings.append(f"run {i} has scope(s) {lost} that not every run has: left out of the merge")
    if not scopes:
        raise MergeError("the runs share no scope")
    region, how = {}, {}
    for sc in scopes:
        region[sc], how[sc] = _common_anchor(recs, sc, anchor)
    runs = []
    for i, (src, r) in enumerate(sources):
        av = window.average_window(r, dict(region), skip, scopes)
        run = Run(i, src, r, av)
        for sc, sw in av.scopes.items():
            sw.chosen = how[sc]
        for sc, sw in av.scopes.items():
            for role, _n, rows in window.by_role(r, sw):
                for rr in rows:
                    run.roles[(sc, role, rr.name, rr.occ)] = rr
        runs.append(run)
    anchors = sorted(set.intersection(*[_names_in(r.av) for r in runs]))
    aset = set(anchors)
    if not aset:
        raise MergeError("no stamp name is present in every run: there are no anchors")

    # the reference: the run most others agree with
    cmp = {(a.index, b.index): _compare(a, b, aset, tolerance) for a in runs for b in runs if a is not b}
    agrees = {a.index: sum(1 for b in runs if b is not a and not cmp[(b.index, a.index)]) for a in runs}
    ref = max(runs, key=lambda x: (agrees[x.index], -x.index))
    best = agrees[ref.index]
    tied = [x.index for x in runs if agrees[x.index] == best]
    why = (f"run {ref.index} agrees with {best} of the other {len(runs) - 1} runs"
           + (f"; tied with runs {[t for t in tied if t != ref.index]}: the first named is the reference"
              if len(tied) > 1 else ""))
    for run in runs:
        if run is ref:
            continue
        diffs = cmp[(run.index, ref.index)]
        run.flags = [line for _s, line in diffs]
        run.accepted = not diffs

    # twin
    for run in runs:
        tw = run.rec.twin
        if not tw or not isinstance(tw.get("plain"), (int, float)) or not isinstance(tw.get("recorded"), (int, float)) \
                or tw["plain"] <= 0:
            run.twin_status = "unchecked"
            continue
        run.twin_ratio = tw["recorded"] / tw["plain"]
        run.twin_status = "over" if run.twin_ratio > 1 + 2 * tolerance else "ok"

    acc = [r for r in runs if r.accepted]
    untrusted: dict[int, list[str]] = {}
    for run in runs:
        if not run.accepted:
            untrusted[run.index] = sorted(_names_in(run.av) - aset)
    rec = _build(acc, aset, region, how, tolerance, runs, ref, findings, untrusted)
    return Merged(tolerance, runs, ref.index, why, region, anchors, findings, untrusted, rec)


def _build(acc: list[Run], aset: set[str], region, how, tol, runs, ref, findings, untrusted) -> Record:
    lanes: dict[str, rec_mod.Lane] = {}
    by_key: dict[tuple[str, str, int], list[tuple[Run, window.Row]]] = {}
    for run in acc:
        for sw in run.av.scopes.values():
            for r in sw.rows:
                by_key.setdefault((r.lane, r.name, r.occ), []).append((run, r))
        for ln in run.rec.lanes.values():
            if ln.id not in lanes:
                lanes[ln.id] = rec_mod.Lane(ln.id, ln.role, ln.group)
            elif lanes[ln.id].role != ln.role:
                findings.append(f"lane {ln.id} has role {lanes[ln.id].role!r} in an earlier run and {ln.role!r} in run "
                                f"{run.index}: the first kept")
    period = {sc: sum(r.av.scopes[sc].period for r in acc) / len(acc) for sc in acc[0].av.scopes}
    stamps = []
    for (lane, name, occ), items in by_key.items():
        sc = rec_mod.scope_of(lane)
        means = [r.mean for _run, r in items]
        if name in aset:
            mean = sum(means) / len(means)
        elif len(items) > 1:
            if max(means) - min(means) > tol * period[sc]:
                findings.append(
                    f"stamp `{name}`#{occ} on {lane} is carried by runs {[run.index for run, _ in items]} at offsets "
                    f"{[round(m, 1) for m in means]}, apart by {max(means) - min(means):,.1f} cycles "
                    f"({100 * (max(means) - min(means)) / period[sc]:.1f}% of the period; bound {100 * tol:.1f}%): "
                    f"the first run's kept")
                mean = means[0]
            else:
                mean = sum(means) / len(means)
        else:
            mean = means[0]
        rows = [r for _run, r in items]
        stamps.append((lane, name, occ, mean, rows, [run.index for run, _ in items]))
    lane_order = list(lanes)
    stamps.sort(key=lambda x: (lane_order.index(x[0]), int(round(x[3])), x[2]))
    names: dict[str, int] = {}
    out_stamps, spread = [], []
    for lane, name, occ, mean, rows, run_ids in stamps:
        nid = names.setdefault(name, len(names))
        out_stamps.append([lane, nid, int(round(mean))])
        spread.append({"lane": lane, "name": name, "occ": occ, "mean": mean,
                       "std": sum(r.std for r in rows) / len(rows), "min": min(r.min for r in rows),
                       "max": max(r.max for r in rows), "n": min(r.n for r in rows), "runs": run_ids,
                       "anchor": name in aset})
    barriers: dict = {}
    operations: dict = {}
    for run in acc:
        for k, v in run.rec.facts.get("barriers", {}).items():
            if k in barriers and barriers[k] != v:
                findings.append(f"barrier `{k}` is declared differently in run {run.index}: the first kept")
            barriers.setdefault(k, v)
        for k, v in run.rec.facts.get("operations", {}).items():
            operations.setdefault(k, v)
    scope_facts = {}
    for sc in period:
        sws = [r.av.scopes[sc] for r in acc]
        scope_facts[sc] = {"period": period[sc], "period_std": sum(s.period_std for s in sws) / len(sws),
                           "period_min": min(s.period_min for s in sws), "period_max": max(s.period_max for s in sws),
                           "anchor": region[sc], "anchor_lane": sws[0].anchor_lane, "chosen": how[sc],
                           "windows": min(s.windows for s in sws), "skip": sws[0].skip}
    run_facts = []
    for run in runs:
        run_facts.append({
            "index": run.index, "source": run.source, "accepted": run.accepted, "reference": run is ref,
            "periods": {sc: sw.period for sc, sw in run.av.scopes.items()},
            "twin": run.rec.twin, "twin_status": run.twin_status,
            "twin_ratio": run.twin_ratio, "flags": run.flags})
    doc = {
        "format": rec_mod.FORMAT,
        "clock": acc[0].rec.clock,
        "names": {str(i): n for n, i in names.items()},
        "lanes": [{k: v for k, v in (("id", ln.id), ("role", ln.role), ("group", ln.group)) if v is not None}
                  for ln in lanes.values() if any(s[0] == ln.id for s in out_stamps)],
        "stamps": out_stamps,
        "dropped": {},
        "facts": {"barriers": barriers, "operations": operations, "runs": run_facts,
                  "window": {"scopes": scope_facts, "stamps": spread},
                  "merge": {"tolerance": tol, "anchors": sorted(aset), "findings": findings,
                            "untrusted": {str(k): v for k, v in untrusted.items()}}},
        "source": {"merged_from": [r.source for r in runs]},
    }
    return rec_mod.from_dict(doc)


def format_report(m: Merged) -> str:
    L = []
    tol = m.tolerance
    L.append(f"merge of {len(m.runs)} runs; tolerance {100 * tol:.1f}% of the window's period "
             f"(periods and anchors); twin bound recorded/plain <= {1 + 2 * tol:.3f}")
    for sc, name in m.anchor_region.items():
        L.append(f"window anchor, scope {sc}: `{name}`")
    for r in m.runs:
        per = ", ".join(f"{sc} {sw.period:,.1f}" for sc, sw in r.av.scopes.items())
        L.append(f"run {r.index}: {r.source}  period [{per}]  {_names_count(r)} stamp names in its window")
    L.append(f"anchors: {len(m.anchors)} stamp names present in every run: {', '.join(m.anchors)}")
    L.append(f"reference: run {m.reference} ({m.reference_why})")
    L.append("")
    for r in m.runs:
        if r.index == m.reference:
            verdict = "REFERENCE"
        elif r.accepted:
            verdict = "consistent with the reference: accepted"
        else:
            verdict = f"FLAGGED, anchors disagree with the reference ({len(r.flags)}): left out of the merged timeline"
        L.append(f"run {r.index} consistency: {verdict}")
        for line in r.flags[:12]:
            L.append(f"    {line}")
        if len(r.flags) > 12:
            L.append(f"    ... {len(r.flags) - 12} more")
        if r.twin_status == "unchecked":
            L.append(f"run {r.index} twin: UNCHECKED (the record carries no twin)")
        else:
            tw = r.rec.twin
            L.append(f"run {r.index} twin: {'FLAGGED, over the bound' if r.twin_status == 'over' else 'within the bound'}"
                     f" (recorded {tw['recorded']:,} / plain {tw['plain']:,} = {r.twin_ratio:.3f}; "
                     f"bound {1 + 2 * tol:.3f})")
    L.append("")
    for k, names in m.untrusted.items():
        L.append(f"run {k} detail reported separately as UNTRUSTED (not in the merged timeline): "
                 f"{len(names)} stamp names beyond the anchors"
                 + (f": {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}" if names else ""))
    L += bound.group_lines(m.findings, indent="finding: ")
    if m.record is not None:
        acc = [r.index for r in m.runs if r.accepted]
        n = len(m.record.stamps)
        L.append(f"merged timeline: runs {acc}; {n} stamps on {len(m.record.lanes)} lanes")
    return "\n".join(L) + "\n"


def _names_count(r: Run) -> int:
    return len(_names_in(r.av))
