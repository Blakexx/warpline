"""LAYER 1, THE READERS: one small function per vendor output, text in and plain data out. A reader never runs a tool
(`runner.py` does), so each is tested on a fixture with no toolkit installed.

    sass_functions(text)    `cuobjdump -sass`                 -> {function: [SassLine]}
    cfg_functions(text)     `nvdisasm -bbcfg -poff` (dot)     -> {function: Cfg}
    res_usage(text)         `cuobjdump -res-usage`            -> {function: Usage}
    demangled(text, names)  `c++filt` output, a line a name   -> {name: demangled}
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
_SASS = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.$]*)\s*(.*?)\s*;")
_FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")


def sass_functions(text: str) -> dict[str, list[SassLine]]:
    """Every function of a `cuobjdump -sass` listing, its instructions in address order. A function named twice (one
    per architecture of a fat binary) is an error: select one architecture when the tool is run."""
    found: dict[str, list[SassLine]] = {}
    current = None
    for line in text.splitlines():
        if m := _FUNCTION.match(line):
            current = m.group(1)
            if current in found:
                raise ReaderError(f"function {current} appears twice in the listing (one architecture only)")
            found[current] = []
        elif current is not None and (m := _SASS.match(line)):
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
