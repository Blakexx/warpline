"""THE FLOOR: a kernel's lower bound from the work it declares and a device's measured rates (docs/floor.md). The work is the
project's declaration, by phase; the rates are a table the probes wrote on a card (`probes/`). Nothing here predicts a time:
a floor is what the work cannot take less than, each stream's work over its rate, and the stream that binds it.

    {"format": "warpline-work/1", "kernel": "^stream_read\\\\(", "parameters": ["ctas"],
     "phases": [{"role": "reader", "phase": "read",
                 "work": {"hbm": {"read": "ctas * 8388608"}, "issue": {"global": "ctas * 16384"}}}],
     "overlap": [], "serial": {}}

    {"format": "warpline-rates/1", "device": "NVIDIA H100 80GB HBM3", "sms": 132, "clock_ghz": 1.98,
     "rows": [{"stream": "hbm.read", "rate": 13.1, "probe": "...", "check": "..."}, ...]}

- WORK, a phase's, the whole launch's: `hbm` (`read`, `write`: bytes), `l2` (a list of `{"pattern": "own" | "shared",
  "bytes": B}`, a shared one with its `footprint`, the bytes every CTA reads from), `shared` (`read`, `write`: bytes),
  `tensor` (a list of `{"shape": "m16n8k16", "dtype": "f16.f32", "count": N}`, warp-level MMA instructions), `issue` (warp
  instructions by the checker's instruction classes, spelled as a loop's `class/<name>` figures: `integer_address`,
  `float`, ...). Every value is a number or an expression over the declaration's `parameters` (`_parse`: numbers, names,
  + - * / // % **, min, max, ceil, floor), whose values are given to `floor`. The launch's SMs are `sms` (an expression;
  the table's by default) and its work spreads evenly over them.
- A RATE ROW prices one STREAM: `hbm.read`, `hbm.write`, `hbm` (a copy's: both), `l2.own`, `l2.shared` (at its
  `footprint`), `shared.read`, `shared.write`, `shared` (both), `tensor` (its `shape` and `dtype`), `issue` (every class),
  `issue.<class>`; `rate` is the stream's work a cycle an SM (bytes, MMA instructions, warp instructions), measured on `sms`
  SMs (the table's by default) at `clock_ghz` (the SMs' clock during its probe; the table's by default). A quantity of work
  is priced by every row whose stream covers it (`STREAMS`); a shared L2 quantity at footprint F by the shared row of the
  largest footprint at most F (a smaller footprint reads no slower). A row that covers two quantities (`hbm`, `shared`)
  prices their sum at the best of its own rate and its parts' (a copy streams slower than a read: a read priced at the
  copy's rate would be bounded too high). Work no row prices is refused: a stream with no probe is a gap, never a zero.
- TWO KINDS OF STREAM. An SM's OWN (`shared`, `tensor`, `issue`): its rate is the SM's, so its time is in cycles, and in
  nanoseconds at the table's `clock_ghz`, the highest clock the probes' SMs ran at. A DEVICE's (`hbm`, `l2`), every SM's at
  once: its rate times its row's own clock is the bytes a nanosecond the probe measured, and its nanoseconds are those.
  The nanoseconds hold for a run whose SMs' clock is at most the table's.
- A PHASE'S FLOOR is the largest, over the rows that price its work, of the work over the rate times the SMs: its BINDING
  row the largest in nanoseconds. Phases run IN SEQUENCE in the declaration's order on each SM, except an OVERLAP set
  (`overlap`, a list of lists of `role/phase` keys) that runs at once: its floor is the largest, over the rows, of its
  members' summed work on the row (the largest member's where the members' streams differ; more where two share one),
  placed at its first member. THE LAUNCH FLOOR is the larger of the SM's CHAIN, the groups' floors on the SMs' own streams
  summed, and each DEVICE stream's work over the whole launch at its rate: the CTAs need not pass a phase together, so a
  device stream one SM leaves idle another may fill (measured: the `serial` example, docs/floor.md#the-examples). Its
  binding row is the larger term's. A phase the design runs alone, proven serial, is `serial` with its reason: no
  overlapped form exists within the resources. It is priced like any other.
- THE BOUND (`declarations`): the floor times a project's factor, as the `bound` declarations the checkers read
  (`checkers.py`) on a run manifest's figure (`launch/grid_ns`, every scope's launch at once; or the first scope's
  `launch/ns` or `launch/cycles`): `at_most` floor x factor (the target) and `at_least` floor (a run under its floor
  refutes the rates or the work declared), the kernel `present`, and the `idle` check: every phase the run manifest
  measures is one the binding stream's family is busy in, directly or by a declared overlap, or is declared serial.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field

WORK_FORMAT = "warpline-work/1"
RATES_FORMAT = "warpline-rates/1"
#: the streams a rate row may price, each with the quantities of work it covers (a quantity's key, below)
STREAMS = {
    "hbm.read": ("hbm.read",), "hbm.write": ("hbm.write",), "hbm": ("hbm.read", "hbm.write"),
    "l2.own": ("l2.own",), "l2.shared": ("l2.shared",),
    "shared.read": ("shared.read",), "shared.write": ("shared.write",), "shared": ("shared.read", "shared.write"),
    "tensor": ("tensor",), "issue": ("issue",),
}
#: the DEVICE's streams, every SM's at once (their nanoseconds are their rows'); every other family is an SM's own
DEVICE = ("hbm", "l2")
#: the work a phase may declare, by kind
WORK_KINDS = ("hbm", "l2", "shared", "tensor", "issue")
#: an issue class: the checker's instruction class spelled as a manifest's `class/<name>` (`select/compare` -> `select_compare`)
_CLASS = re.compile(r"^[a-z][a-z_]*$")
#: what an EXPRESSION may hold: numbers, the declaration's parameters, + - * / // % **, parentheses and these functions
FUNCTIONS = {"min": min, "max": max, "ceil": math.ceil, "floor": math.floor}
_OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
              ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
              ast.Pow: lambda a, b: a ** b}
#: the run figures a bound may be written on: the grid's span in nanoseconds (every scope's launch edges: the work declared
#: is the launch's), the first scope's in nanoseconds, or in cycles
FIGURES = ("launch/grid_ns", "launch/ns", "launch/cycles")


class FloorError(ValueError):
    """A work declaration, a rates table or a parameter that is not usable, in words."""


# ------------------------------------------------------------------------------------------------ expressions

def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id not in FUNCTIONS}


def _parse(text: str, where: str) -> ast.Expression:
    """An expression's tree, refused unless every node is one `EXPRESSION` admits."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FloorError(f"{where}: {text!r} is not an expression ({e.msg})") from e
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Call,
                               *_OPERATORS))
        ok = ok or (isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool))
        if isinstance(node, ast.Call):
            ok = isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords
        if not ok:
            raise FloorError(f"{where}: {text!r} holds {type(node).__name__}: an expression is numbers, the declared "
                             f"parameters, + - * / // % **, parentheses and {', '.join(FUNCTIONS)}")
    return tree


def _value(node: ast.AST, values: dict[str, float]):
    if isinstance(node, ast.Expression):
        return _value(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _value(node.operand, values)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](_value(node.left, values), _value(node.right, values))
    return FUNCTIONS[node.func.id](*(_value(a, values) for a in node.args))


@dataclass(frozen=True)
class Amount:
    """An amount of work as declared: a number, or an expression's text and tree over the parameters."""
    text: str
    tree: ast.Expression | None = None
    number: float | None = None

    def of(self, values: dict[str, float], where: str) -> float:
        if self.tree is None:
            return self.number
        try:
            v = _value(self.tree, values)
        except (ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
            raise FloorError(f"{where}: {self.text!r} cannot be evaluated ({e})") from e
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise FloorError(f"{where}: {self.text!r} is {v}: work is a finite number, 0 or more")
        return float(v)


def _amount(value, where: str, parameters: tuple[str, ...]) -> Amount:
    if isinstance(value, bool):
        raise FloorError(f"{where}: work is a number or an expression, not {value!r}")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value < 0:
            raise FloorError(f"{where}: work is a finite number, 0 or more, not {value!r}")
        return Amount(repr(value), None, float(value))
    if not isinstance(value, str) or not value.strip():
        raise FloorError(f"{where}: work is a number or an expression, not {value!r}")
    tree = _parse(value, where)
    unknown = sorted(_names(tree) - set(parameters))
    if unknown:
        raise FloorError(f"{where}: {value!r} names {', '.join(unknown)}, which the declaration's parameters do not "
                         f"({', '.join(parameters) or 'none'})")
    return Amount(value, tree)


# ------------------------------------------------------------------------------------------------ the work

@dataclass(frozen=True)
class Quantity:
    """One quantity of a phase's work: `kind` (`hbm.read`, `l2.own`, `l2.shared`, `shared.read`, `tensor`, `issue`, ...),
    the key that tells two quantities of a kind apart (an L2 footprint, a tensor shape and dtype, an issue class) and the
    amount."""
    kind: str
    key: tuple = ()
    amount: Amount = Amount("0", None, 0.0)

    @property
    def label(self) -> str:
        if self.kind == "l2.shared":
            return f"l2.shared@{self.key[0]}"
        if self.kind == "tensor":
            return f"tensor {self.key[0]} {self.key[1]}"
        if self.kind == "issue":
            return f"issue.{self.key[0]}"
        return self.kind


@dataclass(frozen=True)
class Phase:
    role: str
    phase: str
    work: tuple[Quantity, ...]

    @property
    def key(self) -> str:
        return f"{self.role}/{self.phase}"


@dataclass(frozen=True)
class Work:
    """A kernel's declared work: its phases in order, the overlap sets, the serial phases with their reasons, the parameters
    its expressions read and, where it gives one, the kernel pattern its bound names."""
    phases: tuple[Phase, ...]
    overlap: tuple[tuple[str, ...], ...] = ()
    serial: dict[str, str] = field(default_factory=dict)
    parameters: tuple[str, ...] = ()
    kernel: str | None = None
    sms: Amount | None = None


def _mapping(node, where: str, keys: tuple[str, ...]) -> dict:
    if not isinstance(node, dict):
        raise FloorError(f"{where}: an object of {', '.join(keys)}")
    extra = sorted(set(node) - set(keys))
    if extra:
        raise FloorError(f"{where}: {', '.join(extra)} is not one of {', '.join(keys)}")
    return node


def _quantities(work, where: str, parameters) -> list[Quantity]:
    out = []
    work = _mapping(work, where, WORK_KINDS)
    for kind in ("hbm", "shared"):
        for way, value in _mapping(work.get(kind, {}), f"{where}/{kind}", ("read", "write")).items():
            out.append(Quantity(f"{kind}.{way}", (), _amount(value, f"{where}/{kind}/{way}", parameters)))
    for i, item in enumerate(work.get("l2", [])):
        at = f"{where}/l2/{i}"
        pattern = item.get("pattern") if isinstance(item, dict) else None
        if pattern == "own":
            _mapping(item, at, ("pattern", "bytes"))
            out.append(Quantity("l2.own", (), _amount(item.get("bytes"), f"{at}/bytes", parameters)))
        elif pattern == "shared":
            _mapping(item, at, ("pattern", "footprint", "bytes"))
            footprint = item.get("footprint")
            if isinstance(footprint, bool) or not isinstance(footprint, int) or footprint <= 0:
                raise FloorError(f"{at}: a shared pattern's footprint is the bytes every CTA reads from, a count above 0")
            out.append(Quantity("l2.shared", (footprint,), _amount(item.get("bytes"), f"{at}/bytes", parameters)))
        else:
            raise FloorError(f"{at}: an L2 quantity is {{\"pattern\": \"own\" | \"shared\", ...}}")
    for i, item in enumerate(work.get("tensor", [])):
        at = f"{where}/tensor/{i}"
        item = _mapping(item, at, ("shape", "dtype", "count"))
        if not all(isinstance(item.get(k), str) and item[k] for k in ("shape", "dtype")):
            raise FloorError(f"{at}: a tensor quantity names its shape and dtype (`m16n8k16`, `f16.f32`)")
        out.append(Quantity("tensor", (item["shape"], item["dtype"]), _amount(item.get("count"), f"{at}/count", parameters)))
    issue = work.get("issue", {})
    if not isinstance(issue, dict):
        raise FloorError(f"{where}/issue: an object of instruction classes to warp instructions")
    for klass, value in issue.items():
        if not _CLASS.match(klass):
            raise FloorError(f"{where}/issue: {klass!r} is not a class as a manifest spells it (`integer_address`)")
        out.append(Quantity("issue", (klass,), _amount(value, f"{where}/issue/{klass}", parameters)))
    return out


def load_work(text: str) -> Work:
    """The work declaration of a JSON text (`warpline-work/1`), validated: the format, each phase's role, phase and work, every
    expression over the declared parameters, the overlap sets (disjoint, of declared phases, two or more each) and the serial
    phases (declared ones, each with a reason)."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise FloorError(f"work: not JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("format") != WORK_FORMAT:
        raise FloorError(f"work: not a work declaration (format {WORK_FORMAT!r})")
    _mapping(doc, "work", ("format", "kernel", "parameters", "sms", "phases", "overlap", "serial", "about"))
    parameters = doc.get("parameters", [])
    if not isinstance(parameters, list) or not all(isinstance(p, str) and p.isidentifier() and p not in FUNCTIONS
                                                   for p in parameters):
        raise FloorError("work: parameters are a list of names")
    parameters = tuple(parameters)
    kernel = doc.get("kernel")
    if kernel is not None:
        try:
            re.compile(kernel)
        except (re.error, TypeError) as e:
            raise FloorError(f"work: kernel is not a regex ({e})") from e
    phases = []
    for i, p in enumerate(doc.get("phases") or []):
        where = f"work: phase {i}"
        p = _mapping(p, where, ("role", "phase", "work", "about"))
        if not all(isinstance(p.get(k), str) and p[k] and "/" not in p[k] for k in ("role", "phase")):
            raise FloorError(f"{where}: needs a role and a phase, names without `/` (the run manifest's `roles/<role>/phases/<phase>`)")
        phases.append(Phase(p["role"], p["phase"], tuple(_quantities(p.get("work", {}), f"work: {p['role']}/{p['phase']}",
                                                                      parameters))))
    if not phases:
        raise FloorError("work: no phases: a work declaration has at least one")
    keys = [p.key for p in phases]
    if len(set(keys)) != len(keys):
        raise FloorError(f"work: a phase is declared twice ({sorted(k for k in keys if keys.count(k) > 1)[0]})")
    overlap, seen = [], set()
    for i, group in enumerate(doc.get("overlap", [])):
        if not isinstance(group, list) or len(group) < 2 or len(set(group)) != len(group):
            raise FloorError(f"work: overlap set {i} is a list of two or more distinct phases (`role/phase`)")
        for k in group:
            if k not in keys:
                raise FloorError(f"work: overlap set {i} names {k!r}, which is no declared phase")
            if k in seen:
                raise FloorError(f"work: {k!r} is in two overlap sets: a set is the phases that run at once")
            seen.add(k)
        roles = [k.split("/")[0] for k in group]
        if len(set(roles)) != len(roles):
            raise FloorError(f"work: overlap set {i} holds two phases of one role: a role's phases run in sequence")
        overlap.append(tuple(group))
    serial = doc.get("serial", {})
    if not isinstance(serial, dict):
        raise FloorError("work: serial maps a phase to the reason it runs alone")
    for k, reason in serial.items():
        if k not in keys:
            raise FloorError(f"work: serial names {k!r}, which is no declared phase")
        if not isinstance(reason, str) or not reason.strip():
            raise FloorError(f"work: the serial phase {k!r} needs its reason (why no overlapped form exists)")
    sms = doc.get("sms")
    return Work(tuple(phases), tuple(overlap), dict(serial), parameters, kernel,
                None if sms is None else _amount(sms, "work: sms", parameters))


# ------------------------------------------------------------------------------------------------ the rates

@dataclass(frozen=True)
class Rate:
    """A row of the rates table: the stream it prices, its rate (work a cycle an SM), the SMs it was measured on, the SMs'
    clock it was measured at (its rate times its clock is the work a nanosecond an SM it measured), its key (an L2 footprint,
    a tensor shape and dtype, an issue class) and the probe's own words."""
    stream: str
    rate: float
    sms: int
    key: tuple = ()
    probe: str = ""
    check: str = ""
    counter: dict | None = None
    clock_ghz: float = 1.0

    @property
    def label(self) -> str:
        if self.stream == "l2.shared":
            return f"l2.shared@{self.key[0]}"
        if self.stream == "tensor":
            return f"tensor {self.key[0]} {self.key[1]}"
        if self.stream == "issue" and self.key:
            return f"issue.{self.key[0]}"
        return self.stream

    @property
    def family(self) -> str:
        """The stream's family (`hbm`, `l2`, `shared`, `tensor`, `issue`): what is busy or idle in a phase."""
        return self.stream.split(".")[0]

    def covers(self, q: Quantity) -> bool:
        """Whether this row prices the quantity (an L2 shared quantity's row is chosen by footprint: `Rates.priced`)."""
        if self.stream == "tensor":
            return q.kind == "tensor" and q.key == self.key
        if self.stream == "issue":
            return q.kind == "issue" and (not self.key or q.key == self.key)
        return q.kind in STREAMS[self.stream] and q.kind != "l2.shared"


@dataclass(frozen=True)
class Rates:
    device: str
    sms: int
    clock_ghz: float
    rows: tuple[Rate, ...]

    def priced(self, q: Quantity, sms: int) -> list[Rate]:
        """The rows that price a quantity on `sms` SMs: every row whose stream covers it, or, for an L2 shared quantity at
        footprint F, the shared row of the largest footprint at most F."""
        rows = [r for r in self.rows if r.sms == sms]
        if q.kind != "l2.shared":
            return [r for r in rows if r.covers(q)]
        shared = [r for r in rows if r.stream == "l2.shared" and r.key[0] <= q.key[0]]
        return [max(shared, key=lambda r: r.key[0])] if shared else []

    def speeds(self, r: Rate) -> tuple[float, float]:
        """A row's work a cycle an SM and a nanosecond an SM, as it prices: an SM's own stream at the table's clock, a device
        stream at its row's own; a row that covers two quantities at the best of its own and its parts' rows' (on its SMs)."""
        parts = [p for p in self.rows if p.sms == r.sms and len(STREAMS.get(r.stream, ())) > 1 and p.stream in
                 STREAMS[r.stream] and p.stream != r.stream] if r.key == () else []
        clock = (lambda x: x.clock_ghz) if r.family in DEVICE else (lambda x: self.clock_ghz)
        return max(x.rate for x in (r, *parts)), max(x.rate * clock(x) for x in (r, *parts))


def _positive(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise FloorError(f"rates: {what} must be a number above 0, not {v!r}")
    return float(v)


def load_rates(text: str) -> Rates:
    """The rates table of a JSON text (`warpline-rates/1`), validated: the device, its SMs and clock, each row's stream (one
    of `STREAMS` or `issue.<class>`), rate, SMs and key, no two rows pricing one stream with one key on one SM count."""
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise FloorError(f"rates: not JSON ({e})") from e
    if not isinstance(doc, dict) or doc.get("format") != RATES_FORMAT:
        raise FloorError(f"rates: not a rates table (format {RATES_FORMAT!r})")
    sms = doc.get("sms")
    if isinstance(sms, bool) or not isinstance(sms, int) or sms <= 0:
        raise FloorError("rates: sms is the device's SM count")
    clock = _positive(doc.get("clock_ghz"), "clock_ghz (the SMs' clock the rows were measured at)")
    rows, seen = [], set()
    for i, r in enumerate(doc.get("rows") or []):
        where = f"row {i}"
        if not isinstance(r, dict):
            raise FloorError(f"rates: {where} is an object")
        stream = r.get("stream")
        key: tuple = ()
        if isinstance(stream, str) and stream.startswith("issue."):
            klass = stream[len("issue."):]
            if not _CLASS.match(klass):
                raise FloorError(f"rates: {where}: {klass!r} is not a class as a manifest spells it")
            stream, key = "issue", (klass,)
        elif stream not in STREAMS:
            raise FloorError(f"rates: {where}: no stream {stream!r} (the streams: {', '.join(STREAMS)}, issue.<class>)")
        if stream == "l2.shared":
            footprint = r.get("footprint")
            if isinstance(footprint, bool) or not isinstance(footprint, int) or footprint <= 0:
                raise FloorError(f"rates: {where}: a shared L2 row names its footprint in bytes")
            key = (footprint,)
        if stream == "tensor":
            if not all(isinstance(r.get(k), str) and r[k] for k in ("shape", "dtype")):
                raise FloorError(f"rates: {where}: a tensor row names its shape and dtype")
            key = (r["shape"], r["dtype"])
        row_sms = r.get("sms", sms)
        if isinstance(row_sms, bool) or not isinstance(row_sms, int) or not 0 < row_sms <= sms:
            raise FloorError(f"rates: {where}: sms is the SMs it was measured on, 1 to {sms}")
        counter = r.get("counter")
        if counter is not None and not (isinstance(counter, dict) and isinstance(counter.get("metrics"), list)
                                        and all(isinstance(m, str) and m for m in counter["metrics"]) and counter["metrics"]):
            raise FloorError(f"rates: {where}: a counter is {{\"metrics\": [names], \"per\": work a count}}")
        if counter is not None:
            _positive(counter.get("per", 1), f"{where}'s counter per")
        ident = (stream, key, row_sms)
        if ident in seen:
            raise FloorError(f"rates: {where} prices {stream} {key or ''} on {row_sms} SMs a second time")
        seen.add(ident)
        rows.append(Rate(stream, _positive(r.get("rate"), f"{where}'s rate"), row_sms, key, str(r.get("probe", "")),
                         str(r.get("check", "")), counter, _positive(r.get("clock_ghz", clock), f"{where}'s clock_ghz")))
    if not rows:
        raise FloorError("rates: no rows")
    return Rates(str(doc.get("device", "")), sms, clock, tuple(rows))


# ------------------------------------------------------------------------------------------------ the floor

@dataclass(frozen=True)
class Floor:
    """A floor (a phase's, a group's, the SMs' chain, the device's): in nanoseconds and in cycles, the binding row (the
    largest in nanoseconds; None where there is no work), and each pricing row's cycles and nanoseconds, by its label."""
    keys: tuple[str, ...]
    cycles: float
    binding: Rate | None
    by_row: dict[str, float]
    ns: float = 0.0
    by_ns: dict[str, float] = field(default_factory=dict)
    rows: dict[str, Rate] = field(default_factory=dict)

    def only(self, device: bool) -> Floor:
        """This floor on the device's streams alone (`device`), or on the SM's own alone."""
        keep = {k: r for k, r in self.rows.items() if (r.family in DEVICE) == device}
        if not keep or not any(self.by_ns[k] for k in keep):
            return Floor(self.keys, 0.0, None, {}, 0.0, {}, {})
        best = max(keep, key=lambda k: self.by_ns[k])
        return Floor(self.keys, max(self.by_row[k] for k in keep), keep[best], {k: self.by_row[k] for k in keep},
                     self.by_ns[best], {k: self.by_ns[k] for k in keep}, keep)


@dataclass(frozen=True)
class Result:
    """The floor of a launch: each phase's, each group's (a phase alone or an overlap set, in the order they run), the SMs'
    CHAIN (the groups' floors on the SMs' own streams, summed) and the DEVICE's (the launch's work on the device's streams),
    the launch's (the larger, in nanoseconds and in cycles) and its binding row, the phases the binding stream's family is
    busy in (its own work, or a declared overlap with a phase that has some) and the serial phases."""
    phases: tuple[Floor, ...]
    groups: tuple[Floor, ...]
    cycles: float
    ns: float
    binding: Rate | None
    busy: tuple[str, ...]
    serial: dict[str, str]
    sms: int
    rates: Rates
    work: dict[str, dict[str, float]]
    chain: Floor | None = None
    device: Floor | None = None


def parameters_of(given: dict, work: Work) -> dict[str, float]:
    """The values of the declaration's parameters from those given (a cell's fields, `--param`): each declared one must be
    given as a finite number; the others given are not read."""
    out = {}
    for name in work.parameters:
        v = given.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise FloorError(f"parameter {name!r} is not given as a number (given: {', '.join(sorted(given)) or 'none'})")
        out[name] = v
    return out


def _floor(keys: tuple[str, ...], quantities: list[Quantity], values, rates: Rates, sms: int) -> tuple[Floor, dict]:
    by_row: dict[str, float] = {}
    by_ns: dict[str, float] = {}
    rows: dict[str, Rate] = {}
    amounts: dict[str, float] = {}
    for q, key in quantities:
        amount = q.amount.of(values, f"{key}: {q.label}")
        amounts[q.label] = amounts.get(q.label, 0.0) + amount
        priced = rates.priced(q, sms)
        if not priced:
            have = sorted({r.sms for r in rates.rows})
            raise FloorError(f"{key}: {q.label} is priced by no row of the rates on {sms} SMs (rows on {have} SMs): "
                             f"a stream with no probe is a gap, never a zero")
        for r in priced:
            per_cycle, per_ns = rates.speeds(r)
            by_row[r.label] = by_row.get(r.label, 0.0) + amount / (per_cycle * sms)
            by_ns[r.label] = by_ns.get(r.label, 0.0) + amount / (per_ns * sms)
            rows[r.label] = r
    if not any(by_ns.values()):
        return Floor(keys, 0.0, None, by_row, 0.0, by_ns, rows), amounts
    best = max(by_ns, key=lambda k: by_ns[k])
    return Floor(keys, max(by_row.values()), rows[best], by_row, by_ns[best], by_ns, rows), amounts


def floor(work: Work, rates: Rates, given: dict | None = None) -> Result:
    """The floor of the declared work at the rates, the parameters given (`parameters_of`): the larger of the SMs' chain and
    the device's (the module's text). The SMs the launch occupies are the declaration's `sms` (an expression) where it
    gives one, else the table's; every row that prices its work must have been measured on that many."""
    values = parameters_of(given or {}, work)
    sms = rates.sms if work.sms is None else work.sms.of(values, "work: sms")
    if sms != int(sms) or not 0 < sms <= rates.sms:
        raise FloorError(f"work: sms is {sms}: a count of SMs, 1 to the device's {rates.sms}")
    sms = int(sms)
    phases, amounts = {}, {}
    for p in work.phases:
        phases[p.key], amounts[p.key] = _floor((p.key,), [(q, p.key) for q in p.work], values, rates, sms)
    groups, placed = [], set()
    sets = {k: g for g in work.overlap for k in g}
    for p in work.phases:
        if p.key in placed:
            continue
        group = sets.get(p.key, (p.key,))
        placed.update(group)
        if len(group) == 1:
            groups.append(phases[p.key])
            continue
        members = [q for k in group for q in next(x for x in work.phases if x.key == k).work]
        groups.append(_floor(group, [(q, " + ".join(group)) for q in members], values, rates, sms)[0])
    own = [g.only(device=False) for g in groups]
    lead = max(own, key=lambda g: g.ns)
    chain = Floor(tuple(k for g in groups for k in g.keys), sum(g.cycles for g in own), lead.binding, {}, sum(g.ns for g in own))
    device = _floor(tuple(p.key for p in work.phases), [(q, p.key) for p in work.phases for q in p.work], values, rates,
                    sms)[0].only(device=True)
    top = max((chain, device), key=lambda f: f.ns)
    cycles, ns, binding = max(chain.cycles, device.cycles), top.ns, top.binding
    family = {r.label: r.family for r in rates.rows}
    busy = []
    for g in groups:
        if binding is not None and any(v > 0 and family[label] == binding.family
                                       for k in g.keys for label, v in phases[k].by_row.items()):
            busy += list(g.keys)
    return Result(tuple(phases[p.key] for p in work.phases), tuple(groups), cycles, ns, binding, tuple(busy),
                  dict(work.serial), sms, rates, amounts, chain, device)


# ------------------------------------------------------------------------------------------------ the bound

def declarations(result: Result, kernel: str, factor: float, figure: str = "launch/grid_ns", reason: str = "") -> dict:
    """The declarations the checkers read (`checkers.load`): the kernel present in the manifests judged (a bound whose pattern
    names no kernel would judge nothing), the floor on `figure` (`FIGURES`) at most times `factor`, at least the floor
    itself, and the `idle` check of the phases. `reason` is the project's words, after the floor's own."""
    if figure not in FIGURES:
        raise FloorError(f"the bound's figure is one of {', '.join(FIGURES)}, not {figure!r}")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1:
        raise FloorError(f"the factor is a number, 1 or more, not {factor!r}")
    try:
        re.compile(kernel)
    except (re.error, TypeError) as e:
        raise FloorError(f"the kernel is not a regex ({e})") from e
    value = result.cycles if figure == "launch/cycles" else result.ns
    unit = "cycles" if figure == "launch/cycles" else "ns"
    binds = result.binding.label if result.binding else "no stream"
    said = f"the launch floor {value:.6g} {unit} ({binds} binds, the rates of {result.rates.device or 'the device'} on " \
           f"{result.sms} SMs)" + (f"; {reason}" if reason else "")
    out = [{"kernel": kernel, "check": "present", "at_least": 1,
            "reason": "the bound's kernel is one the run manifests measure: a bound on no kernel judges nothing"},
           {"kernel": kernel, "check": "bound", "figure": figure, "at_most": _round(value * factor),
            "reason": f"{said} x {factor}: the target"},
           {"kernel": kernel, "check": "bound", "figure": figure, "at_least": _round(value, down=True),
            "reason": f"{said}: a run under its floor refutes the rates or the work declared"}]
    if result.binding is not None:
        out.append({"kernel": kernel, "check": "idle", "stream": result.binding.family, "busy": list(result.busy),
                    "serial": dict(result.serial),
                    "reason": f"{binds} binds the launch: a phase that idles {result.binding.family} is a design defect"})
    return {"declarations": out}


def _round(v: float, down: bool = False) -> float:
    """A bound's value at six significant digits, rounded away from refusing the floor's own value (up for an `at_most`,
    down for an `at_least`)."""
    if v == 0:
        return 0.0
    scale = 10 ** (math.floor(math.log10(abs(v))) - 5)
    return (math.floor(v / scale) if down else math.ceil(v / scale)) * scale


# ------------------------------------------------------------------------------------------------ the counters

def shown(result: Result, counters: dict[str, float], cycles: float) -> list[tuple[Rate, float]]:
    """What a profiled run's COUNTERS show: for each row that prices the declared work and names a counter (its `metrics`,
    summed, times `per`: the row's work a count), the share of the row's rate (as it prices, `Rates.speeds`) the run used
    over `cycles` (the run's elapsed SM cycles), largest first. The binding stream's family should lead (a counter tells a
    family's rows apart no better than its metrics do: every L2 row reads the L2's bytes): the probes' rates and the work
    declared then agree with what the hardware counted. A row whose metrics the counters lack is left out."""
    used = {r.label: r for f in result.phases for r in f.rows.values() if r.counter}
    out = []
    for r in used.values():
        metrics = r.counter["metrics"]
        if not all(m in counters for m in metrics):
            continue
        work = sum(counters[m] for m in metrics) * r.counter.get("per", 1)
        out.append((r, work / (result.rates.speeds(r)[0] * result.sms * cycles)))
    return sorted(out, key=lambda x: -x[1])


# ------------------------------------------------------------------------------------------------ text

def report(result: Result) -> str:
    """The floor in a screen: each phase's cycles and binding row, each overlap set's, the launch's."""
    rates = result.rates
    out = [f"floor on {rates.device or 'the device'} ({result.sms} SMs of {rates.sms}; an SM's streams at {rates.clock_ghz} GHz)"]
    for f in result.phases:
        binds = f"{f.binding.label} binds" if f.binding else "no work"
        rest = ", ".join(f"{k} {v:.6g}" for k, v in sorted(f.by_row.items(), key=lambda x: -x[1]) if f.binding and k != f.binding.label)
        out.append(f"  phase {f.keys[0]}: {f.ns:.6g} ns, {f.cycles:.6g} cycles, {binds}" + (f" (cycles: {rest})" if rest else "") +
                   (f"  [serial: {result.serial[f.keys[0]]}]" if f.keys[0] in result.serial else ""))
    for g in result.groups:
        if len(g.keys) > 1:
            out.append(f"  overlap {' + '.join(g.keys)}: {g.ns:.6g} ns, {g.cycles:.6g} cycles, " +
                       (f"{g.binding.label} binds" if g.binding else "no work"))
    for name, f, what in (("chain", result.chain, "the SMs' own streams, the groups in sequence"),
                          ("device", result.device, "the launch's work on HBM and L2")):
        out.append(f"  {name}: {f.ns:.6g} ns, {f.cycles:.6g} cycles ({what})" + (f", {f.binding.label} binds" if f.binding else ""))
    binds = result.binding.label if result.binding else "no stream"
    out.append(f"  launch: {result.ns:.6g} ns, {result.cycles:.6g} cycles, {binds} binds")
    if result.binding is not None:
        idle = [f.keys[0] for f in result.phases if f.keys[0] not in result.busy and f.keys[0] not in result.serial]
        out.append(f"  {result.binding.family} is busy in {', '.join(result.busy) or 'no phase'}" +
                   (f"; IDLE in {', '.join(idle)} (overlap it, or declare it serial with its reason)" if idle else ""))
    return "\n".join(out) + "\n"


def to_data(result: Result) -> dict:
    def one(f: Floor) -> dict:
        return {"keys": list(f.keys), "ns": f.ns, "cycles": f.cycles, "binding": f.binding.label if f.binding else None,
                "by_row": dict(sorted(f.by_row.items()))}
    return {"device": result.rates.device, "sms": result.sms, "clock_ghz": result.rates.clock_ghz,
            "phases": [{**one(f), "work": result.work[f.keys[0]]} for f in result.phases],
            "groups": [one(g) for g in result.groups], "chain": one(result.chain), "device": one(result.device),
            "cycles": result.cycles, "ns": result.ns,
            "binding": result.binding.label if result.binding else None, "busy": list(result.busy),
            "serial": result.serial}
