"""A set of variants the user promises are one kernel.

warpline never switches a stamp: it is handed a MANIFEST naming the variants of one kernel, which one is PLAIN, and
where each one's recordings are (a stamp record, sampling exports, or both):

    {"kernel": "...", "variants": [{"name": "...", "plain": true, "stamps": null, "samples": ["a.csv", ...]},
                                   {"name": "...", "plain": false, "stamps": "rec.json", "samples": []}]}

Paths are relative to the manifest. Exactly one variant is plain.

  PERIOD     a variant's window in MICROSECONDS from its samples (the fold's period in samples times the interval),
             and, where it has stamps, in CYCLES from its stamp record (the average window). Their ratio, cycles an
             us, is the clock during that run: a MEASUREMENT, reported a variant.
  PROMISE    a variant's window over the plain one's is its COST. Beyond the tolerance (default 6%, in either
             direction: a variant that is faster than the plain one is not the same kernel either) the variant is
             named and left out of the merge. Compared in microseconds when both have samples; in cycles when both
             have stamps (the clock is then assumed the same: said in the report). A variant that cannot be compared
             with the plain one is UNCHECKED, never reported as passing, and is not merged unless asked.
  MERGE      the variants that pass and carry stamps are merged with the existing merge, each stamp tagged by its
             variant (`facts.variants`: run index -> variant name; `facts.window.stamps[i].runs`).
  OVERLAY    the plain variant's samples are laid on the merged timeline when a variant with waits allows the
             alignment (`overlay`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import merge as merge_mod
from . import overlay, sampling
from .record import Record, load

COST = 0.06


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

    @property
    def clock(self) -> float | None:
        """Cycles an us (MHz) during this run, where both periods are measured."""
        if self.period_us and self.period_cycles:
            return self.period_cycles / self.period_us
        return None


@dataclass
class SetResult:
    kernel: str
    cost: float
    variants: list[Variant]
    merged: Record | None = None
    merge_report: str = ""
    merged_names: list[str] = field(default_factory=list)
    timeline_note: str = ""
    alignment: overlay.Alignment | None = None
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
        out.append(Variant(v["name"], bool(v.get("plain", False)), rel(stamps), [rel(x) for x in samples]))
    if len({v.name for v in out}) != len(out):
        raise SetError(f"{path}: variant names are not unique")
    n = sum(v.plain for v in out)
    if n != 1:
        raise SetError(f"{path}: exactly one variant is plain; {n} are")
    return str(doc.get("kernel", "")), out


def run(manifest: str, cfg: sampling.Config | None = None, cost: float = COST, merge_unchecked: bool = False,
        scope: str | None = None) -> SetResult:
    cfg = cfg or sampling.Config()
    kernel, vs = read_manifest(manifest)
    res = SetResult(kernel, cost, vs)
    for v in vs:
        if v.samples:
            v.fold = sampling.fold([sampling.read_export(p) for p in v.samples], cfg)
            v.period_us = v.fold.period_us
        if v.stamps:
            v.record = load(v.stamps)
            v.view = overlay.view_of(v.record, scope, v.stamps)
            v.period_cycles = v.view.period
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
    stamped = [v for v in vs if v.record is not None and (v.plain or v.status == "ok"
                                                          or (merge_unchecked and v.status == "unchecked"))]
    left = [v.name for v in vs if v.record is not None and v not in stamped]
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
        timeline = overlay.view_of(m.record, scope, "the merged timeline")
    elif len(stamped) == 1:
        res.timeline_note = (f"one variant carries stamps that pass or are plain ({stamped[0].name}): nothing to merge; "
                             f"its own average window is the timeline")
        timeline = stamped[0].view
    else:
        res.timeline_note = "no variant that passes carries stamps: there is no timeline"
        timeline = None
    if left:
        res.timeline_note += f"; left out of the merge: {left}"
    if timeline is None:
        res.overlay_note = "no overlay: there is no stamped timeline"
    elif plain.fold is None:
        res.overlay_note = f"no overlay: the plain variant `{plain.name}` has no samples"
    else:
        try:
            res.alignment = overlay.align(timeline, plain.fold)
            res.overlay_text = overlay.format_overlay(res.alignment, timeline.source, plain.samples)
        except overlay.OverlayError as e:
            res.overlay_note = f"no overlay: {e}"
    return res


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
    L.append("")
    if r.overlay_text:
        L.append("plain variant's samples on the timeline:")
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
                      "sample_flags": v.fold.flags if v.fold else []} for v in r.variants],
        "overlay": overlay.to_dict(r.alignment, "the timeline") if r.alignment else {"note": r.overlay_note},
    }
