"""A set of variants the user promises are one kernel.

warpline never switches a stamp: it is handed a MANIFEST naming the variants of one kernel, which one is PLAIN, and
where each one's recordings are (a stamp record, sampling exports, or both):

    {"kernel": "...", "variants": [{"name": "...", "plain": true, "stamps": null, "samples": ["a.csv", ...]},
                                   {"name": "...", "plain": false, "stamps": "rec.json", "samples": []}]}

Paths are relative to the manifest. Exactly one variant is plain.

  PERIOD     a variant's window in MICROSECONDS from its samples (the fold's period in samples times the interval),
             and, where it has stamps, in CYCLES from its stamp record (the average window). Their ratio, cycles an
             us, is the clock during that run: a MEASUREMENT, reported a variant.
  TIES       where a stamp record carries ties (`record.ties`), its window in TIME is read from them -- its own
             measurement, taken in the run that made the stamps -- and reported beside the period its samples give;
             the two must agree within a tolerance (default 2%), else the finding `time_bases_disagree` (the stamps'
             and the samples' time bases disagree for this variant). The variant's clock is then reported from the
             ties a region, not only as the one number above. A record without ties says so: time is the samples'.
  PROMISE    a variant's window over the plain one's is its COST. Beyond the tolerance (default 6%, in either
             direction: a variant that is faster than the plain one is not the same kernel either) the variant is
             named and left out of the merge. Compared in microseconds when both have samples; in cycles when both
             have stamps (the clock is then assumed the same: said in the report). A variant that cannot be compared
             with the plain one is UNCHECKED, never reported as passing, and is not merged unless asked.
  MERGE      the variants that pass and carry stamps are merged with the existing merge, each stamp tagged by its
             variant (`facts.variants`: run index -> variant name; `facts.window.stamps[i].runs`).
  LAUNCH     a stamped variant that has ties and its own samples: its launch length T from the ties (the two forced
             marks `launch.begin`, `launch.end`) against its samples' live span, per pass and run, shown even when
             it passes (the LENGTH CHECK of `overlay`); its samples are placed under its stamps by the launch's edges.
  OVERLAY    the PLAIN kernel has no stamps, so its samples are placed THROUGH a stamped variant: (a) the stamped
             variant's own samples are placed under its stamps by its launch edges; (b) the plain variant's fold of
             each varying series is rotated onto the stamped variant's placed fold of the same series by correlation
             (THE SHAPE CHECK: under 0.7, or a runner-up within 0.05, names the series and it is not overlaid); (c)
             the stamped variant's phases are shown with the plain variant's reasons placed by (a)+(b), beside the
             stamped variant's own. A set whose stamped variants carry no launch marks is REFUSED the overlay.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from . import merge as merge_mod
from . import launch as launch_mod
from . import overlay, placement, rates, sampling
from .record import Record, load

COST = 0.06
TIES = 0.02      # a variant's window in time from its ties against the one its samples give


class SetError(Exception):
    pass


@dataclass
class Variant:
    name: str
    plain: bool
    stamps: str | None
    samples: list[str]
    fold: sampling.Folded | None = None
    view: overlay.View | None = None
    record: Record | None = None
    period_us: float | None = None
    period_cycles: float | None = None
    status: str = "unchecked"       # plain | ok | over | unchecked
    ratio: float | None = None      # its window over the plain one's
    basis: str = ""                 # us | cycles
    note: str = ""
    ties_us: float | None = None    # its window in microseconds from its OWN ties (its stamps' measurement)
    ties_note: str = ""             # what the ties say beside the samples, or that there are none
    ties_findings: list[str] = field(default_factory=list)
    region_rates: list[rates.RoleRegion] = field(default_factory=list)   # the clock from the ties, per region
    exports: list[sampling.Export] = field(default_factory=list)
    launch: launch_mod.Launch | None = None        # from its ties, where its record carries the launch's two edges
    launch_note: str = ""                          # why it has none
    placement: overlay.Placement | None = None     # its own samples placed on cycles by its own launch's edges
    windows_from: str | None = None                # an EDGES variant: the stamped variant whose windows it takes
    edges_only: bool = False                       # the record has the launch's edges and no window anchors
    compare: list[placement.FoldCompare] = field(default_factory=list)   # an edges variant's fold against its source's

    @property
    def clock(self) -> float | None:
        """Cycles an us (MHz) during this run, where both periods are measured."""
        if self.period_us and self.period_cycles:
            return self.period_cycles / self.period_us
        return None


@dataclass
class PlainSeries:
    name: str
    short: str
    shift: sampling.Shift | None
    bins: list[sampling.Bin] | None       # the plain variant's fold rotated onto the stamped variant's placed fold
    half_windows: float | None = None     # the stamped series' placement half-width plus half a bin (the shift's step)
    finding: str = ""
    rotation: float | None = None         # the fractional rotation (bins) and the correlation as they are
    corr: float | None = None


@dataclass
class PlainOverlay:
    through: Variant
    series: dict[str, PlainSeries]
    findings: list[str]


@dataclass
class SetResult:
    kernel: str
    cost: float
    variants: list[Variant]
    merged: Record | None = None
    merge_report: str = ""
    merged_names: list[str] = field(default_factory=list)
    timeline_note: str = ""
    overlays: list[PlainOverlay] = field(default_factory=list)
    overlay_text: str = ""
    overlay_note: str = ""
    tags: dict[str, int] = field(default_factory=dict)


def read_manifest(path: str) -> tuple[str, list[Variant]]:
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise SetError(f"{path}: {e}")
    if not isinstance(doc, dict) or not isinstance(doc.get("variants"), list) or not doc["variants"]:
        raise SetError(f"{path}: a manifest is {{\"kernel\": ..., \"variants\": [{{name, plain, stamps, samples}}, ...]}}")
    base = os.path.dirname(os.path.abspath(path))

    def rel(p):
        return None if p is None else (p if os.path.isabs(p) else os.path.join(base, p))

    out: list[Variant] = []
    for i, v in enumerate(doc["variants"]):
        if not isinstance(v, dict) or not isinstance(v.get("name"), str):
            raise SetError(f"{path}: variant {i} has no `name`")
        samples = v.get("samples") or []
        if not isinstance(samples, list) or not all(isinstance(x, str) for x in samples):
            raise SetError(f"{path}: variant `{v['name']}`: `samples` is a list of paths")
        stamps = v.get("stamps")
        if stamps is not None and not isinstance(stamps, str):
            raise SetError(f"{path}: variant `{v['name']}`: `stamps` is a path or null")
        if stamps is None and not samples:
            raise SetError(f"{path}: variant `{v['name']}` has neither stamps nor samples")
        wf = v.get("windows_from")
        if wf is not None and not isinstance(wf, str):
            raise SetError(f"{path}: variant `{v['name']}`: `windows_from` is the name of a stamped variant")
        out.append(Variant(v["name"], bool(v.get("plain", False)), rel(stamps), [rel(x) for x in samples],
                           windows_from=wf))
    if len({v.name for v in out}) != len(out):
        raise SetError(f"{path}: variant names are not unique")
    n = sum(v.plain for v in out)
    if n != 1:
        raise SetError(f"{path}: exactly one variant is plain; {n} are")
    for v in out:
        if v.windows_from is not None and v.windows_from not in {x.name for x in out}:
            raise SetError(f"{path}: variant `{v.name}`: `windows_from` names no variant: {v.windows_from!r}")
    return str(doc.get("kernel", "")), out


def _read_ties(v: Variant, scope: str | None, tol: float) -> None:
    """A variant's own ties: its window in time, beside the samples', and its clock a region."""
    rec = v.record
    if not rec.ties:
        v.ties_note = "its stamp record carries no ties, so time comes from the samples alone"
        return
    w = rates.window_in_time(rec, scope, v.view.anchor)
    if w is None:
        v.ties_note = "its stamp record carries ties, but not on the stamps that start the window: no window in time"
        return
    v.ties_us = w["period_ns"] / rates.US
    res = rates.analyze(rec, v.view.anchor, scopes=[w["scope"]])
    v.ties_findings += [str(f) for f in res.findings]
    if w["scope"] in res.scopes:
        v.region_rates = res.scopes[w["scope"]].by_role
    if v.period_us:
        d = v.ties_us / v.period_us - 1
        if abs(d) > tol:
            v.ties_findings.append(
                f"time_bases_disagree: the ties give a window of {v.ties_us:.3f} us, the samples {v.period_us:.3f} us "
                f"({100 * d:+.2f}%; bound {100 * tol:g}%): the stamps' and the samples' time bases disagree for this "
                f"variant")
        v.ties_note = f"the ties' window and the samples' agree within {100 * tol:g}% ({100 * d:+.2f}%)" \
            if abs(d) <= tol else "the ties' window and the samples' DISAGREE"
    else:
        v.ties_note = "no samples of this variant to set against its ties"


def run(manifest: str, cfg: sampling.Config | None = None, cost: float = COST, merge_unchecked: bool = False,
        scope: str | None = None, ties_tolerance: float = TIES) -> SetResult:
    cfg = cfg or sampling.Config()
    kernel, vs = read_manifest(manifest)
    res = SetResult(kernel, cost, vs)
    for v in vs:
        if v.samples:
            v.exports = [sampling.read_export(p) for p in v.samples]
            v.fold = sampling.fold(v.exports, cfg)
            v.period_us = v.fold.period_us
        if v.stamps:
            v.record = load(v.stamps)
            try:
                v.view = overlay.view_of(v.record, scope, v.stamps)
            except overlay.OverlayError:
                # a record with the launch's two edges and no window anchors is the plain kernel with its edges
                try:
                    v.launch = overlay.launch_for(v.record, scope, need_windows=False)
                except overlay.OverlayError:
                    raise
                if v.launch.has_windows:
                    raise
                v.edges_only = True
            if v.view is not None:
                v.period_cycles = v.view.period
                _read_ties(v, scope, ties_tolerance)
                try:
                    v.launch = overlay.launch_for(v.record, v.view.scope)
                except overlay.OverlayError as e:
                    v.launch_note = str(e)
    for v in vs:
        if v.edges_only:
            src = next((x for x in vs if x.name == v.windows_from), None)
            if src is None:
                v.launch_note = (f"this record is the plain kernel with its two edges and no window anchors: its windows "
                                 f"are another variant's (the manifest's `windows_from`: the name of a stamped variant)")
            elif src.launch is None or src.view is None:
                v.launch_note = f"`windows_from` names `{src.name}`, which has no windows with the launch's edges"
    for v in vs:
        if v.launch is not None and v.exports and v.record is not None:
            if v.edges_only:
                src = next((x for x in vs if x.name == v.windows_from), None)
                if src is None or src.launch is None or src.view is None:
                    continue
                fr = src.launch.fractions_on(v.launch.cycles)
                v.placement = overlay.place(src.view, v.launch, v.exports, cfg, src.launch,
                                            f"variant `{src.name}`'s window anchors on this launch's cycles "
                                            f"(`windows_from`): the code before the first window "
                                            f"({src.launch.setup_cycles:,} cycles) is {src.name}'s, and the rest of this "
                                            f"launch ({v.launch.cycles - src.launch.setup_cycles:,} cycles) is divided by "
                                            f"{src.name}'s windows in its relative lengths (each "
                                            f"{(v.launch.cycles - src.launch.setup_cycles) / (src.launch.cycles - src.launch.setup_cycles):.4f} "
                                            f"of its)", fr)
            else:
                v.placement = overlay.place(v.view, v.launch, v.exports, cfg)
    for v in vs:
        if v.edges_only and v.placement is not None:
            src = next(x for x in vs if x.name == v.windows_from)
            if src.placement is not None:
                v.compare = _compare_placed(src, v)
    plain = next(v for v in vs if v.plain)
    plain.status = "plain"
    for v in vs:
        if v.plain:
            continue
        if v.period_us and plain.period_us:
            v.ratio, v.basis = v.period_us / plain.period_us, "us"
        elif v.period_cycles and plain.period_cycles:
            v.ratio, v.basis = v.period_cycles / plain.period_cycles, "cycles"
            v.note = ("compared in cycles: both have stamps and no samples; the clock is ASSUMED the same in both runs "
                      "(a sampled run of each would measure it)")
        else:
            v.status = "unchecked"
            pl = "samples" if plain.samples else "stamps"
            if v.stamps and not v.samples:
                v.note = (f"this variant has stamps (a window in cycles) and no samples; the plain variant has {pl} only "
                          f"(a window in microseconds): the two cannot be compared. To check it, record either a "
                          f"sampling export of THIS variant's own run (its window in microseconds against the plain "
                          f"one's) or a stamp record of the PLAIN variant (its window in cycles; then the clock is "
                          f"assumed equal)")
            else:
                v.note = (f"this variant has {'stamps' if v.stamps else 'samples'} only and the plain variant has "
                          f"{pl} only: no common measure")
            continue
        if abs(v.ratio - 1.0) > cost:
            v.status = "over"
            v.note = (v.note + "; " if v.note else "") + (
                f"window {100 * (v.ratio - 1):+.1f}% of the plain variant's ({'slower' if v.ratio > 1 else 'faster'}; "
                f"bound {100 * cost:.0f}%): left out of the merge")
        else:
            v.status = "ok"
    stamped = [v for v in vs if v.view is not None and (v.plain or v.status == "ok"
                                                        or (merge_unchecked and v.status == "unchecked"))]
    left = [v.name for v in vs if v.view is not None and v not in stamped]
    if len(stamped) >= 2:
        m = merge_mod.merge([(v.stamps, v.record) for v in stamped])
        res.merged = m.record
        res.merge_report = merge_mod.format_report(m)
        res.merged_names = [stamped[r.index].name for r in m.runs if r.accepted]
        m.record.facts["variants"] = {str(i): stamped[i].name for i in range(len(stamped))}
        for e in m.record.facts["window"]["stamps"]:
            for r in e.get("runs", []):
                res.tags[stamped[r].name] = res.tags.get(stamped[r].name, 0) + 1
        res.timeline_note = (f"merged timeline: variants {res.merged_names} of {[v.name for v in stamped]}; "
                             f"{len(m.record.stamps)} stamps, each tagged by its variant (facts.variants)")
    elif len(stamped) == 1:
        res.timeline_note = (f"one variant carries stamps that pass or are plain ({stamped[0].name}): nothing to merge; "
                             f"its own average window is the timeline")
    else:
        res.timeline_note = "no variant that passes carries stamps: there is no timeline"
    if left:
        res.timeline_note += f"; left out of the merge: {left}"
    if plain.fold is None:
        res.overlay_note = f"no overlay: the plain variant `{plain.name}` has no samples"
    else:
        through = [v for v in vs if v.record is not None and not v.plain and v.status == "ok"]
        if not through:
            res.overlay_note = "no overlay: no stamped variant passes the promise: there is no stamped timeline"
        else:
            why = []
            for v in through:
                if v.launch is None:
                    why.append(f"variant `{v.name}`: {v.launch_note}")
                elif v.placement is None:
                    why.append(f"variant `{v.name}` carries the launch's edges but has no samples of its own: its "
                               f"samples are what the launch's edges place, and the plain variant's samples are "
                               f"placed through them")
                else:
                    res.overlays.append(_through(plain, v))
            if res.overlays:
                res.overlay_text = "\n\n".join(format_through(plain, o) for o in res.overlays)
            else:
                res.overlay_note = ("OVERLAY REFUSED: the plain variant has no stamps, so its samples are placed through "
                                    "a stamped variant by that variant's launch edges, and none can: "
                                    + "; ".join(why))
    return res


def _compare_placed(src: Variant, e: Variant) -> list[placement.FoldCompare]:
    """THE CHECK of an edges variant: its fold, placed by its own edges and the stamped variant's windows, against the
    stamped variant's own placed fold, series by series (the two are one kernel up to the stamps' cost): a rotation over
    half a bin or a correlation under 0.7 is a named finding."""
    out = []
    a, b = src.placement.series, e.placement.series
    for name in src.placement.varying():
        short = overlay._short(name)
        if name not in b:
            out.append(placement.FoldCompare(name, short, None, None, None,
                                             f"`{short}` is placed in `{src.name}` and not in `{e.name}`"))
            continue
        c = placement.compare_folds(name, short, [x.mean for x in a[name].bins], [x.mean for x in b[name].bins])
        n = len(a[name].bins)
        c.bound = placement.SPREAD_Z * math.hypot(a[name].spread or 0.0, b[name].spread or 0.0) * n + 1.0
        c.within = c.rotation is not None and abs(c.rotation) <= c.bound
        out.append(c)
    return out


def _through(plain: Variant, st: Variant) -> PlainOverlay:
    """(b) the plain variant's fold of each varying series rotated onto the stamped variant's placed fold of the
    same series; THE SHAPE CHECK is the rotation's acceptance (correlation 0.7 or more, no runner-up within 0.05). The
    fractional rotation and the correlation as the folds are come with it."""
    pl = st.placement
    n = pl.bins
    series: dict[str, PlainSeries] = {}
    findings = []
    for name in plain.fold.varying():
        short = overlay._short(name)
        if name not in pl.series:
            msg = (f"SHAPE CHECK: {short}: `{st.name}` has no placed fold of this series (its pass was left out by "
                   f"the cycles-a-sample check, or the series is not in its samples): not overlaid")
            series[name] = PlainSeries(name, short, None, None, None, msg)
            findings.append(msg)
            continue
        target = [b.mean for b in pl.series[name].bins]
        fold = [b.mean for b in plain.fold.bins[name]]
        sh = sampling.best_shift(target, fold, sampling.MIN_CORR, sampling.MIN_GAP,
                                 sampling.guard_for(plain.fold.blur, n))
        frac = placement.fold_rotation(target, fold)
        if not sh.accepted:
            msg = (f"SHAPE CHECK: {short}: {sh.why}: the variants' sampled profiles differ, or cannot be told apart "
                   f"by rotation; the series is not overlaid")
            series[name] = PlainSeries(name, short, sh, None, None, msg, frac)
            findings.append(msg)
            continue
        bins = [plain.fold.bins[name][(b + sh.shift) % n] for b in range(n)]
        series[name] = PlainSeries(name, short, sh, bins, pl.series[name].half + 0.5 / n, "", frac, sh.corr)
    for name in pl.varying():
        if name not in series:       # varies in the stamped variant, and not in the plain one's fold
            short = overlay._short(name)
            why = ("flat in the plain variant's fold" if name in plain.fold.bins
                   else "not folded in the plain variant (no period of its own: flat, or left out by the period checks)")
            msg = (f"SHAPE CHECK: {short}: varies in `{st.name}` and is {why}: the variants' sampled profiles differ; "
                   f"the series is not overlaid")
            series[name] = PlainSeries(name, short, None, None, None, msg)
            findings.append(msg)
    return PlainOverlay(st, series, findings)


def format_through(plain: Variant, o: PlainOverlay) -> str:
    st = o.through
    pl = st.placement
    n = pl.bins
    L = [f"THE PLAIN VARIANT `{plain.name}` PLACED THROUGH `{st.name}`",
         f"(a) `{st.name}`'s own samples are placed on cycles by its own launch edges (its section above)",
         f"(b) the plain variant's fold of each varying series, rotated onto `{st.name}`'s placed fold of the same "
         f"series by the whole-bin rotation of largest correlation (THE SHAPE CHECK: correlation 0.7 or more and no "
         f"runner-up within 0.05); the rotation to a fraction of a bin (the phase of the first harmonics) and the "
         f"correlation of the two folds as they are come beside it. ASSUMED: the plain variant's profile is "
         f"`{st.name}`'s up to this rotation; checked by the shape alone:"]
    for ps in o.series.values():
        fr = "" if ps.rotation is None else f"; fractional rotation {ps.rotation:+.2f} bins"
        if ps.bins is None:
            L.append(f"  {ps.short:<22} {ps.finding}{fr}")
        else:
            sh = ps.shift
            ru = "" if sh.runner_corr is None else f"; runner-up shift {sh.runner_shift}, correlation {sh.runner_corr:.3f}"
            hw = ps.half_windows
            loose = hw * n > 0.5 + 1e-9
            L.append(f"  {ps.short:<22} rotation {sh.shift} bins ({sh.shift / n:.3f} of a window), correlation "
                     f"{sh.corr:.3f}{ru}{fr}; placement half-width {hw:.4f} of a window = {hw * n:.2f} bins (the "
                     f"stamped series' {pl.series[ps.name].half:.4f} plus half a bin, the rotation's step)"
                     + ("  LOOSE" if loose else ""))
    L.append("")
    own = pl.varying()
    placed = [x for x, ps in o.series.items() if ps.bins is not None]
    names = placed + [x for x in own if x not in placed]
    cols = []
    for x in names:
        sh = overlay._short(x)
        if x in placed:
            cols.append((f"{sh} PLAIN" + ("~" if o.series[x].half_windows * n > 0.5 + 1e-9 else ""), o.series[x].bins))
        if x in pl.series and x in own:
            cols.append((f"{sh} own" + ("~" if pl.series[x].half * n > 0.5 + 1e-9 else ""), pl.series[x].bins))
    L.append(f"(c) THE OVERLAY: `{st.name}`'s roles' phases (its own stamps) with the PLAIN variant's stall reasons placed by "
             f"(a)+(b) (columns `PLAIN`), and `{st.name}`'s own samples' placement printed beside each (columns `own`); "
             f"~ marks a placement over half a bin")
    L += overlay.table_lines(pl.view, n, cols, pl.profile, pl.roles)
    return "\n".join(L)


def format_set(r: SetResult) -> str:
    L = [f"variant set of kernel `{r.kernel}`: {len(r.variants)} variants; cost bound {100 * r.cost:.0f}% of the plain "
         f"variant's window"]
    for v in r.variants:
        L.append("")
        L.append(f"variant `{v.name}`{' (PLAIN)' if v.plain else ''}")
        if v.fold is not None:
            f = v.fold
            L.append(f"  samples: {len(v.samples)} run(s); window {f.period:.4f} samples x {f.interval:g} us = "
                     f"{f.period_us:.3f} us")
            for x in f.flags:
                if x.startswith(("RUNS", "RUN NOT", "the first")) or "NEAR" in x:
                    L.append(f"    flag: {x}")
        if v.view is not None:
            L.append(f"  stamps: {v.stamps}: window {v.period_cycles:,.1f} cycles (scope {v.view.scope}, "
                     f"{v.view.windows} steady windows)")
        if v.ties_us is not None:
            L.append(f"  ties: window {v.ties_us:.3f} us from its own ties (its stamps' measurement)"
                     + (f", the samples give {v.period_us:.3f} us" if v.period_us else "") + f": {v.ties_note}")
            for x in v.ties_findings:
                L.append(f"    FINDING {x}")
            if v.region_rates:
                L.append("  clock from the ties, per region (cycles an us; across lanes min..max):")
                for a in v.region_rates[:12]:
                    L.append(f"    {a.role} / {a.name}{'#%d' % a.occ if a.occ else ''}: {a.rate:,.1f} "
                             f"({a.lane_min:,.1f}..{a.lane_max:,.1f}), {a.cycles:,.0f} cycles in {a.ns / 1000:.3f} us")
                if len(v.region_rates) > 12:
                    L.append(f"    ... {len(v.region_rates) - 12} more (`rates` has all)")
        elif v.view is not None:
            L.append(f"  ties: {v.ties_note}")
        if v.record is not None:
            if v.launch is None:
                L.append(f"  launch: none: {v.launch_note.split(';')[0]}")
            else:
                L.append("  " + v.launch.describe().replace("\n", "\n  "))
                if v.edges_only:
                    L.append(f"  an EDGES variant: the plain kernel with its two edges; windows from "
                             f"{'`' + v.windows_from + '`' if v.windows_from else 'nobody'}"
                             + (f": {v.launch_note}" if v.launch_note else ""))
                if v.placement is None:
                    L.append("  no samples of this variant (or no windows): nothing is placed by its edges")
                else:
                    L.append("  its samples placed on cycles by its own launch's two edges:")
                    L += ["    " + x for x in overlay.placement_lines(v.placement)]
        if v.clock is not None:
            L.append(f"  clock during this run: {v.clock:,.0f} cycles an us ({v.clock / 1000:.3f} GHz) -- a MEASUREMENT "
                     f"of this run (stamped window over sampled window), not a setting")
        if v.plain:
            L.append("  promise: the plain variant: the reference")
        elif v.status == "unchecked":
            L.append(f"  promise: UNCHECKED -- {v.note}")
        else:
            word = "COST" if v.ratio >= 1 else "change"
            L.append(f"  promise: {'OVER, named' if v.status == 'over' else 'checked'}: window over the plain one's "
                     f"{v.ratio:.4f} in {v.basis} ({word} {100 * (v.ratio - 1):+.1f}%)" + (f"; {v.note}" if v.note else ""))
    L.append("")
    L.append(r.timeline_note)
    if r.merge_report:
        L.append("")
        L.append(r.merge_report.rstrip())
        L.append("stamps by variant: " + ", ".join(f"{k} {n}" for k, n in r.tags.items()))
    for v in r.variants:
        if v.placement is not None:
            L.append("")
            L.append(f"THE OVERLAY OF `{v.name}`'S OWN SAMPLES" + (f" (windows and roles from `{v.windows_from}`)"
                                                                  if v.edges_only else ""))
            L += overlay.overlay_table(v.placement)
    for v in r.variants:
        if v.edges_only and v.compare:
            src = next(x for x in r.variants if x.name == v.windows_from)
            L.append("")
            L.append(f"THE CHECK: `{v.name}`'s fold, placed by its own edges and `{src.name}`'s windows, against "
                     f"`{src.name}`'s own placed fold, series by series (rotation in bins: `{v.name}` lies that much "
                     f"later; over half a bin, or a correlation under {sampling.MIN_CORR:g}, is a finding). ASSUMED: "
                     f"the code before the first window costs the same cycles in both (the stamps add cycles to the windows "
                     f"and none to the setup), and the rest of `{v.name}`'s launch is `{src.name}`'s windows, in their "
                     f"relative lengths, each scaled alike")
            for c in v.compare:
                if c.rotation is None:
                    L.append(f"  {c.short:<22} {c.finding}")
                    continue
                L.append(f"  {c.short:<22} rotation {c.rotation:+.2f} bins, correlation {c.corr:.3f} (after the "
                         f"whole-bin rotation {c.corr_rot:.3f}); within the two placements' measured spreads"
                         f" ({placement.SPREAD_Z:g} each, in quadrature) and a bin, +-{c.bound:.2f} bins: "
                         f"{'yes' if c.within else 'NO'}" + (f"  FINDING: {c.finding}" if c.finding else ""))
    L.append("")
    if r.overlay_text:
        L.append(r.overlay_text.rstrip())
    else:
        L.append(r.overlay_note)
    return "\n".join(L) + "\n"


def to_dict(r: SetResult) -> dict:
    return {
        "kernel": r.kernel, "cost_bound": r.cost, "timeline": r.timeline_note, "merged_variants": r.merged_names,
        "stamps_by_variant": r.tags,
        "variants": [{"name": v.name, "plain": v.plain, "status": v.status, "period_us": v.period_us,
                      "period_cycles": v.period_cycles, "clock_cycles_per_us": v.clock, "ratio_to_plain": v.ratio,
                      "ratio_basis": v.basis, "note": v.note,
                      "ties": {"window_us": v.ties_us, "note": v.ties_note, "findings": v.ties_findings,
                               "regions": [{"role": a.role, "region": a.name, "occ": a.occ,
                                            "cycles_per_us": a.rate, "lane_min": a.lane_min,
                                            "lane_max": a.lane_max} for a in v.region_rates]},
                      "sample_flags": v.fold.flags if v.fold else [],
                      "windows_from": v.windows_from, "launch": v.launch.to_dict() if v.launch else None,
                      "placement": overlay.place_to_dict(v.placement) if v.placement else None,
                      "check_against_windows_from": [
                          {"series": c.short, "rotation_bins": c.rotation, "corr": c.corr, "bound_bins": c.bound,
                           "within_bound": c.within, "finding": c.finding or None}
                          for c in v.compare]} for v in r.variants],
        "overlay": {"note": r.overlay_note} if not r.overlays else [
            {"through": o.through.name, "placement": overlay.place_to_dict(o.through.placement),
             "plain": {x.short: {"finding": x.finding or None, "rotation_bins": x.shift.shift if x.shift else None,
                                 "corr": x.shift.corr if x.shift else None, "half_windows": x.half_windows,
                                 "fractional_rotation_bins": x.rotation,
                                 "bins": None if x.bins is None else [
                                     {"mean": b.mean, "se": b.se, "n": b.n} for b in x.bins]}
                       for x in o.series.values()}} for o in r.overlays],
    }
