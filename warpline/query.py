"""`sql`: a record loaded into an in-memory SQLite database, for the question nobody anticipated.

Times are INTEGER CYCLES (the record's own clock); `ns` columns are the record's ties' readings. A column that is
REAL is derived (a share, a mean). The tables are `TABLES`; `--schema` prints them, and `NAMED` holds the queries for the
common asks, which print their SQL so they teach the schema. The database is read-only once loaded.
"""
from __future__ import annotations

import bisect
import csv
import io
import json
import sqlite3

from . import check, rates, reader, window
from .record import Record, classify, scope_of

DEFAULT_LIMIT = 50

# table -> (description, [(column, type, meaning)])
TABLES: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "lanes": ("one row a lane", [
        ("lane", "TEXT", "the lane's id"), ("scope", "TEXT", "the id's part before `/` (a CTA)"),
        ("role", "TEXT", "the lane's role"), ("grp", "TEXT", "its group, where the record gives one"),
        ("stamps", "INTEGER", "stamps written"), ("dropped", "INTEGER", "stamps dropped (counted)")]),
    "stamps": ("every stamp, in the order written (a record of a real kernel holds 8,000 to 31,000)", [
        ("idx", "INTEGER", "the stamp's index in the record"), ("lane", "TEXT", ""), ("t", "INTEGER", "cycles"),
        ("name", "TEXT", "the stamp's name"),
        ("kind", "TEXT", "region_begin | region_end | wait_begin | wait_end | arrive | op_issue | op_done | write | "
                          "poll_begin | poll_end | launch_begin | launch_end | mark"),
        ("key", "TEXT", "the name without its kind (the region, the barrier, the resource)"),
        ("payload", "INTEGER", "the stamp's payload, or NULL")]),
    "regions": ("one row a region (begin to end) of a lane, as the reader pairs them", [
        ("lane", "TEXT", ""), ("name", "TEXT", "the region's name, e.g. phase.adding"),
        ("depth", "INTEGER", "0 outermost"), ("t_begin", "INTEGER", "cycles"),
        ("t_end", "INTEGER", "cycles; NULL where the region never closes"),
        ("cycles", "INTEGER", "t_end - t_begin, or NULL"),
        ("window", "INTEGER", "the steady window (0 first, in the scope's anchor lane's windows) its begin falls in; "
                               "NULL outside every steady window")]),
    "windows": ("the steady windows: the scope's (role = '') and each role's own, where the role's lanes carry the anchor", [
        ("scope", "TEXT", ""), ("role", "TEXT", "'' for the scope's anchor lane; else the role whose own anchor it is"),
        ("idx", "INTEGER", "0 is the first steady window"), ("t_begin", "INTEGER", "cycles"),
        ("t_end", "INTEGER", "cycles (the next window's begin)"), ("cycles", "INTEGER", "t_end - t_begin")]),
    "ties": ("the record's ties: one moment read on both clocks", [
        ("lane", "TEXT", ""), ("cycles", "INTEGER", "the cycle counter"), ("ns", "INTEGER", "the time reading")]),
    "waits": ("one row a wait on a barrier or a poll, with what released it where the record says", [
        ("lane", "TEXT", ""), ("role", "TEXT", ""), ("kind", "TEXT", "barrier | poll"),
        ("barrier", "TEXT", "the barrier's name, or the polled field"),
        ("barrier_group", "TEXT", "the name with its numbers as `#`: the waits of a kind read together"),
        ("t_begin", "INTEGER", "cycles"), ("t_end", "INTEGER", "cycles; NULL where the wait never ends"),
        ("cycles", "INTEGER", "t_end - t_begin, or NULL"), ("window", "INTEGER", "the steady window its begin falls in, or NULL"),
        ("released_by", "TEXT", "the lane whose stamp released it, where the record says"),
        ("released_at", "INTEGER", "cycles of that stamp, or NULL"),
        ("release_kind", "TEXT", "arrival | write | unstamped | own | none")]),
    "phases": ("the per-role average window: one row a role's phase (a region), in the role's own window", [
        ("role", "TEXT", ""), ("name", "TEXT", "the region's name"),
        ("occurrence", "INTEGER", "the order it begins in within the window (0 first)"),
        ("start", "REAL", "cycles from the role's window start, mean"), ("cycles", "REAL", "duration, mean"),
        ("ns", "REAL", "its own time from the ties, where the record has them, else NULL"),
        ("share", "REAL", "cycles over the scope's window period"),
        ("jitter", "REAL", "standard deviation of the duration across windows (cycles)"),
        ("held", "REAL", "the fraction of windows that held it"), ("lanes", "INTEGER", "lanes of the role carrying it")]),
    "findings": ("what the checks found", [
        ("code", "TEXT", ""), ("lane", "TEXT", ""), ("t", "INTEGER", "cycles"), ("message", "TEXT", "")]),
}

NAMED: dict[str, tuple[str, str]] = {
    "phases-by-role": ("each role's phases in order, with duration, its time and share of the window",
                       "SELECT role, name, occurrence, ROUND(start) AS start, ROUND(cycles) AS cycles, "
                       "ROUND(ns) AS ns, ROUND(100 * share, 1) AS pct_of_window FROM phases ORDER BY role, start"),
    "longest-waits": ("the longest waits, with the lane that released each",
                      "SELECT lane, role, barrier, t_begin, cycles, released_by, release_kind FROM waits "
                      "WHERE cycles IS NOT NULL ORDER BY cycles DESC"),
    "waits-by-barrier": ("waits read together by barrier group: how many, how long in all, the longest",
                         "SELECT role, barrier_group, COUNT(*) AS n, SUM(cycles) AS total_cycles, "
                         "MAX(cycles) AS longest FROM waits WHERE cycles IS NOT NULL "
                         "GROUP BY role, barrier_group ORDER BY total_cycles DESC"),
    "window-jitter": ("the scope's windows: how long each is, from the shortest to the longest",
                      "SELECT idx, t_begin, cycles FROM windows WHERE role = '' ORDER BY cycles DESC"),
    "rate-by-phase": ("a phase's cycles an us from its ties (the counter's rate changes between phases)",
                      "SELECT role, name, occurrence, ROUND(cycles) AS cycles, ROUND(ns) AS ns, "
                      "ROUND(1000.0 * cycles / ns, 1) AS cycles_per_us FROM phases WHERE ns > 0 ORDER BY cycles_per_us"),
    "stamps-by-name": ("how many stamps of each name, over how many lanes",
                       "SELECT name, kind, COUNT(*) AS n, COUNT(DISTINCT lane) AS lanes FROM stamps "
                       "GROUP BY name ORDER BY n DESC"),
    "region-spread": ("a region's duration across its occurrences: mean, shortest, longest",
                      "SELECT name, COUNT(*) AS n, ROUND(AVG(cycles)) AS mean_cycles, MIN(cycles) AS shortest, "
                      "MAX(cycles) AS longest FROM regions WHERE cycles IS NOT NULL GROUP BY name ORDER BY mean_cycles DESC"),
}


def ddl() -> str:
    return "\n".join(f"CREATE TABLE {t} ({', '.join(f'{c} {ty}' for c, ty, _ in cols)});" for t, (_d, cols) in TABLES.items())


def load(rec: Record, scope: str | None = None, anchor: str | None = None, skip: int = window.SKIP) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(ddl())
    scope = scope or rec.scopes()[0]
    written: dict[str, int] = {}
    for lane, _n, _t, _p in rec.stamps:
        written[lane] = written.get(lane, 0) + 1
    con.executemany("INSERT INTO lanes VALUES (?,?,?,?,?,?)",
                    [(ln, scope_of(ln), r.role, r.group, written.get(ln, 0), rec.dropped.get(ln, 0))
                     for ln, r in rec.lanes.items()])
    rows = []
    for i, (lane, nid, t, p) in enumerate(rec.stamps):
        nm = rec.names[nid]
        kind, key = classify(nm)
        rows.append((i, lane, t, nm, kind, key, p))
    con.executemany("INSERT INTO stamps VALUES (?,?,?,?,?,?,?)", rows)
    starts: list[int] = []
    wp = None
    anchor_name = None
    try:
        anchor_name, _how, _lane, starts = window.anchor_starts(rec, scope, anchor, skip)
    except window.WindowError:
        starts = []

    def win_of(t: int):
        if len(starts) < 2 or not (starts[0] <= t < starts[-1]):
            return None
        return bisect.bisect_right(starts, t) - 1

    regs, _f = reader.regions(rec)
    con.executemany("INSERT INTO regions VALUES (?,?,?,?,?,?,?)",
                    [(ln, r.name, r.depth, r.begin, r.end, None if r.end is None else r.end - r.begin,
                      win_of(r.begin) if scope_of(ln) == scope else None)
                     for ln, rs in regs.items() for r in rs])
    wrows = []
    if len(starts) >= 2:
        wp = (starts[-1] - starts[0]) / (len(starts) - 1)
        wrows += [(scope, "", k, a, b, b - a) for k, (a, b) in enumerate(zip(starts, starts[1:]))]
        roles: dict[str, list[str]] = {}
        for ln in rec.lanes:
            if scope_of(ln) == scope:
                roles.setdefault(rec.lanes[ln].role, []).append(ln)
        for role, lanes in roles.items():
            for ln in lanes:
                own = window._own_begins(rec, ln, anchor_name, skip)
                if own is not None:
                    wrows += [(scope, role, k, a, b, b - a) for k, (a, b) in enumerate(zip(own, own[1:]))]
                    break
    con.executemany("INSERT INTO windows VALUES (?,?,?,?,?,?)", wrows)
    con.executemany("INSERT INTO ties VALUES (?,?,?)", [(ln, c, ns) for ln, lst in rec.ties.items() for c, ns in lst])
    con.executemany("INSERT INTO waits VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (w.lane, rec.lanes[w.lane].role, w.kind, w.target, reader.group(w.target), w.begin, w.end,
         None if w.end is None else w.end - w.begin, win_of(w.begin) if scope_of(w.lane) == scope else None,
         None if w.release is None else w.release.lane, None if w.release is None else w.release.time,
         None if w.release is None else w.release.kind) for w in reader.waits(rec)])
    if wp:
        ns_of = {}
        if rec.ties:
            try:
                sr = rates.analyze(rec, anchor_name, skip, [scope], rates.TOLERANCE).scopes.get(scope)
            except ValueError:
                sr = None
            for a in ([] if sr is None else sr.by_role):
                ns_of[(a.role, a.name, a.occ)] = a.ns
        prow = []
        for rw in window.role_windows(rec, scope, anchor_name, skip):
            for r in rw.rows:
                prow.append((rw.role, r.name, r.occ, r.start, r.dur, ns_of.get((rw.role, r.name, r.occ)), r.dur / wp,
                             r.dur_jit, r.share, r.lanes))
        con.executemany("INSERT INTO phases VALUES (?,?,?,?,?,?,?,?,?,?)", prow)
    con.executemany("INSERT INTO findings VALUES (?,?,?,?)",
                    [(f.code, f.lane, f.time, f.message) for f in check.all_findings(rec)])
    con.commit()
    con.execute("PRAGMA query_only = ON")
    return con


def schema_text() -> str:
    L = ["tables (times are INTEGER cycles unless a column says ns; REAL columns are derived):"]
    for t, (d, cols) in TABLES.items():
        L.append(f"  {t}({', '.join(c for c, _ty, _m in cols)})  -- {d}")
    L.append("columns:")
    for t, (_d, cols) in TABLES.items():
        for c, ty, m in cols:
            if m:
                L.append(f"  {t}.{c} {ty}: {m}")
    L.append("`sql <record> --named` lists named queries (with their SQL); `--named NAME` runs one.")
    return "\n".join(L) + "\n"


def named_text() -> str:
    L = ["named queries (`--named NAME`); the SQL is each one's, to adapt:"]
    for k, (d, q) in NAMED.items():
        L.append(f"  {k}: {d}")
        L.append(f"      {q}")
    return "\n".join(L) + "\n"


def run(con: sqlite3.Connection, query: str, limit: int = DEFAULT_LIMIT) -> tuple[list[str], list[tuple], bool]:
    cur = con.execute(query)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(limit + 1)
    return cols, rows[:limit], len(rows) > limit


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.4g}" if abs(v) >= 1000 or abs(v) < 0.01 else f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def format_table(cols, rows, cut: bool, limit: int) -> str:
    cells = [[_cell(v) if len(_cell(v)) <= 70 else _cell(v)[:69] + "~" for v in r] for r in rows]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(cols)]
    L = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip()]
    L += ["  ".join(v.rjust(w) if _num(rows[i][j]) else v.ljust(w) for j, (v, w) in enumerate(zip(r, widths))).rstrip()
          for i, r in enumerate(cells)]
    L.append(f"({len(rows)} rows" + (f"; CUT at --limit {limit}: more rows exist, narrow the query or raise --limit)" if cut else ")"))
    return "\n".join(L) + "\n"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def format_csv(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows(rows)
    return buf.getvalue()


def format_json(cols, rows, cut: bool, limit: int) -> str:
    return json.dumps({"columns": cols, "rows": [list(r) for r in rows], "truncated": cut, "limit": limit,
                       "row_count": len(rows)}, separators=(",", ":"))
