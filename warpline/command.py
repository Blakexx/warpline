"""THE ONE COMMAND, `warpline <verb>`, assembled from the modules present. Each module (`warpline.recorder`,
`warpline.checker`) has a `cli` with a `HEADING`, a `register(sub)` that adds its verbs and an `execute(args)` that runs one;
this file builds the parser, lists each module's verbs under its heading in `--help`, and refuses a verb two modules
share. The common verbs (`warpline/cli.py`: `compare`, `run-manifest`) are in every command, after the modules'. A verb is
told the modules the command was assembled from (`args.modules`): a common verb asks them what each can give. A consumer
that has one module (the other's directory is absent) gets the command of the one, with the common verbs."""
from __future__ import annotations

import argparse
import importlib
import os
import sys

MODULES = ("warpline.recorder.cli", "warpline.checker.cli")
#: the common verbs' `cli`, in every command
COMMON = "warpline.cli"


class _Section:
    """One module's verbs: a verb is added with its help, which `--help` prints under the module's heading."""

    def __init__(self, sub, module, taken, modules):
        self.sub, self.module, self.taken, self.modules, self.verbs = sub, module, taken, modules, []

    def add_parser(self, name, help):
        if name in self.taken:
            raise ValueError(f"warpline: the verb `{name}` is in two modules ({self.taken[name]}, {self.module.__name__})")
        self.taken[name] = self.module.__name__
        self.verbs.append((name, help))
        parser = self.sub.add_parser(name)    # no `help`: the choices list stays empty, the epilog prints the verbs by module
        parser.set_defaults(execute=self.module.execute, modules=self.modules)
        return parser


def available():
    """The modules' `cli` that this installation has."""
    found = []
    for name in MODULES:
        try:
            found.append(importlib.import_module(name))
        except ModuleNotFoundError as e:
            if not (e.name or "").startswith("warpline"):
                raise
    return found


def build(modules, prog="warpline") -> argparse.ArgumentParser:
    """The parser of the command of `modules` (the modules' `cli`), the common verbs added after them."""
    common = importlib.import_module(COMMON)
    modules = [*[m for m in modules if m is not common], common]
    ap = argparse.ArgumentParser(prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="verb")
    taken, sections = {}, []
    for module in modules:
        section = _Section(sub, module, taken, tuple(modules))
        module.register(section)
        sections.append((module.HEADING, section.verbs))
    lines = []
    for heading, verbs in sections:
        lines.append(heading)
        for name, text in verbs:
            lines.append(f"  {name:<12} {text}")
        lines.append("")
    ap.epilog = "\n".join(lines).rstrip()
    return ap


def main(argv=None, modules=None, prog="warpline") -> int:
    ap = build(modules if modules is not None else available(), prog)
    a = ap.parse_args(argv)
    return a.execute(a)


def run(modules=None, prog="warpline") -> None:
    """The entry of a `python -m` or of `bazel run`: from the directory it was invoked in."""
    os.chdir(os.environ.get("BUILD_WORKING_DIRECTORY", "."))
    sys.exit(main(None, modules, prog))
