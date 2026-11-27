"""LAYER 1, THE READERS: one small function per vendor output, text in and plain data out. A reader never runs a tool
(`runner.py` does), so each is tested on a fixture with no toolkit installed.

    sass_functions(text)    `cuobjdump -sass`                 -> {function: [SassLine]}
    cfg_functions(text)     `nvdisasm -bbcfg -poff` (dot)     -> {function: Cfg}
    res_usage(text)         `cuobjdump -res-usage`            -> {function: Usage}
    demangled(text, names)  `c++filt` output, a line a name   -> {name: demangled}
    source_lines(text)      `nvdisasm --print-line-info`      -> {function: {address: (file, line)}}   (optional)
    inline_frames(text)     `nvdisasm --print-line-info-inline` -> {function: {address: ((file, line), ...)}}, innermost first
    life_ranges(text)       `nvdisasm --print-life-ranges`    -> {function: {address: Marks}}   (the registers read and written)

THE LINE GRAMMAR is here once, public, for a consumer that reads a listing these readers do not (a hash of a function's body):
`INSTRUCTION` (`/*0a10*/ @!P0 IMAD.WIDE R4, R6, 0x4 ;`: the address, the predicate, the opcode with its modifiers, the operands),
`ENCODING` (the line of an encoding word, `/* 0x000fe200078e0203 */`), `FUNCTION` and `ARCH` (`cuobjdump -sass`'s headers,
`Function : NAME` and `arch = sm_90a`) and `TEXT_SECTION` (`nvdisasm`'s `.section .text.NAME`, where a function's code starts: its
offsets restart there). An instruction's address has four hex digits or more (a function past 0xfffff bytes has six), its opcode
starts with a capital letter (nvdisasm's data lines, `/*0000*/ .byte 0x04, 0x2f`, are not instructions) and its line holds the `;`
that ends it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class ReaderError(ValueError):
    """A tool's text that is not the shape this reader knows."""


@dataclass(frozen=True)
class SassLine:
    """One instruction: its address, predicate (`@!P0`, or empty), opcode with its modifiers, and operands text."""
    address: int
    predicate: str
    opcode: str
    operands: str


@dataclass(frozen=True)
class CfgBlock:
    """One node of the graph: its id (`BB3`, `.L_x_2`, or the function's own name for the entry) and its instructions'
    offsets in order. A node that holds no instruction has no offsets."""
    id: str
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class Cfg:
    """A function's graph: the entry's id, the nodes in file order, and the edges (source id, target id)."""
    entry: str
    blocks: tuple[CfgBlock, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Usage:
    """A function's resource report: registers, stack bytes and static shared bytes a thread block."""
    registers: int
    stack: int
    shared: int


#: `/*0a10*/   @!P0 IMAD.WIDE R4, R6, 0x4 ;   /* 0x... */`: the address, the predicate, the opcode, the operands
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.$]*)\s*(.*?)\s*;")
#: the line of an encoding word, which follows an instruction in `cuobjdump -sass`
ENCODING = re.compile(r"^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")
#: `cuobjdump -sass`'s headers: a function, and the architecture of the functions after it
FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")
ARCH = re.compile(r"^\s*arch\s*=\s*(sm_\d+[a-z]?)\s*$")
#: `nvdisasm`'s section of a function's code: `.section .text.NAME,"ax",@progbits`
TEXT_SECTION = re.compile(r"^\s*\.section\s+\.text\.([^\s,]+)")


def sass_functions(text: str) -> dict[str, list[SassLine]]:
    """Every function of a `cuobjdump -sass` listing, its instructions in address order. A function named twice (one
    per architecture of a fat binary) is an error: select one architecture when the tool is run."""
    found: dict[str, list[SassLine]] = {}
    current = None
    for line in text.splitlines():
        if m := FUNCTION.match(line):
            current = m.group(1)
            if current in found:
                raise ReaderError(f"function {current} appears twice in the listing (one architecture only)")
            found[current] = []
        elif current is not None and (m := INSTRUCTION.match(line)):
            found[current].append(SassLine(int(m.group(1), 16), (m.group(2) or "").strip(), m.group(3), m.group(4)))
    return found


#: a dot file's text with its backslash escapes removed (`\ ` a space, `\[` a bracket, `\l` a line end kept as `\n`)
def _unescape(label: str) -> str:
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) == "l" else m.group(1), label)


_CLUSTER = re.compile(r'^subgraph "cluster_(.+)" \{$')
_NODE_ID = re.compile(r'^"([^"]+)"$')
_LABEL = re.compile(r'^\[label="\{(.*)\}"\]$')
_EDGE = re.compile(r'^"([^"]+)":\w+:\w+ -> "([^"]+)":\w+:\w+ \[style=\w+\];$')
_OFFSET = re.compile(r"^([0-9a-f]{4,}):\s")


def cfg_functions(text: str) -> dict[str, Cfg]:
    """Every function of `nvdisasm -bbcfg -poff` (a graphviz dot file): a cluster a function, a node a basic block
    whose label holds its instructions behind their offsets, an edge between two nodes. The function's own node is
    its entry. Edge styles are not read: the tool marks none."""
    found: dict[str, Cfg] = {}
    name = None
    blocks: list[CfgBlock] = []
    edges: list[tuple[str, str]] = []
    pending = None
    for line in text.splitlines():
        if m := _CLUSTER.match(line):
            name, blocks, edges, pending = m.group(1), [], [], None
        elif name is None:
            continue
        elif line == "}":
            found[name] = Cfg(name, tuple(blocks), tuple(edges))
            name = None
        elif m := _NODE_ID.match(line):
            pending = m.group(1)
        elif pending is not None and (m := _LABEL.match(line)):
            rows = [_unescape(part) for part in re.split(r"(?<!\\)\|", m.group(1))]
            rows = [row for part in rows for row in part.split("\n")]
            offsets = tuple(int(o.group(1), 16) for row in rows
                            if (o := _OFFSET.match(re.sub(r"^<\w+>", "", row.strip()))))
            blocks.append(CfgBlock(pending, offsets))
            pending = None
        elif m := _EDGE.match(line):
            edges.append((m.group(1), m.group(2)))
    return found


_USAGE_FUNCTION = re.compile(r"^\s*Function (\S+):")
_USAGE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)")


def res_usage(text: str) -> dict[str, Usage]:
    """Every function of a `cuobjdump -res-usage` report. `LOCAL` is not read: the tool reports 0 where local memory
    exists; the stack frame is `STACK`."""
    found: dict[str, Usage] = {}
    current = None
    for line in text.splitlines():
        if m := _USAGE_FUNCTION.match(line):
            current = m.group(1)
        elif current is not None and (m := _USAGE.search(line)):
            found[current] = Usage(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            current = None
    return found


def demangled(text: str, names: list[str]) -> dict[str, str]:
    """`c++filt` run over `names`, one per line, answers one line a name."""
    lines = text.splitlines()
    if len(lines) != len(names):
        raise ReaderError(f"{len(names)} names, {len(lines)} demangled lines")
    return dict(zip(names, lines))


_MARK = re.compile(r'^\s*//##\s+File\s+"([^"]+)",\s+line\s+(\d+)')
_EXECROOT = re.compile(r"^.*/execroot/[^/]+/")


def source_lines(text: str) -> dict[str, dict[int, tuple[str, int]]]:
    """Every function of a disassembly with source-line marks (`nvdisasm --print-line-info`, of a binary compiled with
    `--generate-line-info`): each instruction's address to its source `(file, line)`. A mark (`//## File "f.cu", line 12`)
    stands before the instructions it covers and is printed again only where the line changes, so an instruction with no mark
    of its own keeps the place of the one before it; an instruction's place is the FIRST mark since the instruction before it
    (marks that follow it, `... inlined at ...`, name the callers: the first is the innermost source line). The path is cut to
    what follows the build's execution root (`.../execroot/<workspace>/`) where one is present, so a listing built in a
    sandbox names the same file wherever it was built. A function is a `.text.<name>` section; an address before any mark of
    its function has no place."""
    found: dict[str, dict[int, tuple[str, int]]] = {}
    current = None
    place = None
    fresh = True
    for line in text.splitlines():
        if m := TEXT_SECTION.match(line):
            current = found.setdefault(m.group(1), {})
            place, fresh = None, True
        elif line.lstrip().startswith(".section"):
            current = None
        elif current is None:
            continue
        elif m := _MARK.match(line):
            if fresh:
                place, fresh = (_EXECROOT.sub("", m.group(1)), int(m.group(2))), False
        elif m := INSTRUCTION.match(line):
            if place is not None:
                current[int(m.group(1), 16)] = place
            fresh = True
    return found


def _place(m: re.Match) -> tuple[str, int]:
    return _EXECROOT.sub("", m.group(1)), int(m.group(2))


def inline_frames(text: str) -> dict[str, dict[int, tuple[tuple[str, int], ...]]]:
    """Every function of a disassembly with its inline chains (`nvdisasm --print-line-info-inline`, of a binary compiled with
    `--generate-line-info`): each instruction's address to the chain of source places it was inlined through, the innermost first,
    as `(file, line)`. The marks since the instruction before it are its chain, one a frame, the innermost first and each caller
    after it; an instruction with no mark of its own has the chain of the one before it. The path is cut as `source_lines` cuts
    it. A function is a `.text.<name>` section (its offsets restart there); an address before any mark of its function has no
    chain."""
    found: dict[str, dict[int, tuple[tuple[str, int], ...]]] = {}
    current = None
    chain: tuple[tuple[str, int], ...] = ()
    pending: list[tuple[str, int]] = []
    for line in text.splitlines():
        if m := TEXT_SECTION.match(line):
            current = found.setdefault(m.group(1), {})
            chain, pending = (), []
        elif line.lstrip().startswith(".section"):
            current = None
        elif current is None:
            continue
        elif m := _MARK.match(line):
            pending.append(_place(m))
        elif m := INSTRUCTION.match(line):
            if pending:
                chain, pending = tuple(pending), []
            if chain:
                current[int(m.group(1), 16)] = chain
    return found


@dataclass(frozen=True)
class Marks:
    """One instruction's registers by `nvdisasm --print-life-ranges`: those it reads (a `v` or an `x` under the register) and those
    it writes (`^` or `x`), named as the model names them (`R5`, `P0`, `UR4`, `UP1`)."""
    reads: frozenset[str]
    writes: frozenset[str]


#: a life-range listing's groups of columns, by the name the listing heads each with, and the prefix of a register of the group
LIFE_GROUPS = {"GPR": "R", "PRED": "P", "UGPR": "UR", "UPRED": "UP"}
_NUMBERS = re.compile(r"\S+")


def _columns(numbers: str, names: list[str], where: str) -> list[tuple[int, str]]:
    """The mark columns of a life-range header: `numbers` is its line of register numbers right of `// |` (a group a segment
    between bars, each `# 0 1 2 ...`, its live count's `#` then a register's number a column), `names` the groups' names in
    order. A column is the position of its number's last character and the register it stands for."""
    columns, at = [], 0
    for k, segment in enumerate(numbers.split("|")[:-1] if numbers.rstrip().endswith("|") else numbers.split("|")):
        tokens = [(m.start() + at, m.group()) for m in _NUMBERS.finditer(segment)]
        at += len(segment) + 1
        if not tokens:
            continue
        if k >= len(names):
            raise ReaderError(f"{where}: a life-range header with {k + 1} groups of columns and {len(names)} names")
        if tokens[0][1] != "#" or [t for _, t in tokens[1:]] != [str(n) for n in range(len(tokens) - 1)]:
            raise ReaderError(f"{where}: an unexpected register header in the life-range listing ({names[k]})")
        prefix = LIFE_GROUPS[names[k]]
        columns += [(pos + len(t) - 1, f"{prefix}{t}") for pos, t in tokens[1:]]
    return columns


def life_ranges(text: str) -> dict[str, dict[int, Marks]]:
    """Every function of `nvdisasm --print-life-ranges`: each instruction's address to the registers it reads and writes by the
    listing's marks. Right of `// |` the listing heads a group of columns a register kind, named on one line (`GPR`, `PRED`,
    `UGPR`, `UPRED`; a header of one unnamed group is `GPR`) with the registers' numbers on the next (`# 0 1 2 ...`, after the
    group's live count `#`), and marks each instruction under the registers it touches: `^` written, `v` read, `x` both, `:` live
    through. The marks are read; the tool's own live count is not (its control flow is not the function's: a call is marked as
    writing the calling convention's clobbered registers, and a call to a label of the function is taken for one that returns).
    A group this reader does not know, numbers that are not 0, 1, ... in order, and an instruction that names a vector register
    past the listing's columns are refused. A function is a `.text.<name>` section."""
    found: dict[str, dict[int, Marks]] = {}
    current, name, names, columns = None, None, [], None
    for line in text.splitlines():
        if m := TEXT_SECTION.match(line):
            name, current, names, columns = m.group(1), found.setdefault(m.group(1), {}), [], None
            continue
        if current is None or "// |" not in line:
            continue
        right = line[line.index("// |") + 4:]
        if (head := [t for t in re.split(r"\s*\|\s*", right.strip()) if t]) and all(t in LIFE_GROUPS for t in head):
            names = head
            continue
        if re.match(r"\s*#\s", right):
            columns = _columns(right, names or ["GPR"], name)
            continue
        if not (m := INSTRUCTION.match(line)):
            continue
        if columns is None:
            raise ReaderError(f"{name}: an instruction before the life-range listing's register header")
        vector = [int(r) for r in re.findall(r"\bR(\d+)\b", m.group(4))]
        listed = sum(1 for _, r in columns if r.startswith("R") and r[1:].isdigit())
        if vector and max(vector) >= listed:
            raise ReaderError(f"{name}: an instruction names R{max(vector)}, past the listing's {listed} columns")
        mark = lambda at: right[at] if at < len(right) else " "
        current[int(m.group(1), 16)] = Marks(frozenset(r for at, r in columns if mark(at) in "vx"),
                                             frozenset(r for at, r in columns if mark(at) in "^x"))
    return found
