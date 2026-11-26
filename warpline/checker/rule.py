"""LAYER 3, THE RULE INTERFACE. A rule is one small unit: a name, a one-line statement of what it flags,
`check(kernel) -> findings` (each a `warpline.manifest.Finding`: a site of the manifest, where, how many and what), the
refusal's message and its fix, and at least two fixtures (one where it fires, one where it must not). Write it as one module in
`rules/` that defines `RULE`; the registry (`rules/__init__.py`) finds it.
A rule reads the model (`model.Kernel`) and never text. A BOUND rule (`bound=True`) reports a figure (bytes a step,
instructions a step): a project's declaration is its bound, and with none declared the figure is reported (`manifest.figures`)
and never refused. A PROFILE rule (`ProfileRule`) reads a profile (`ncu.Profile`)
instead, has fixtures of the profiler's exports, and writes into a run manifest (`manifest.run_rules`), never the static one."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .model import Kernel

if TYPE_CHECKING:
    from ..manifest import Finding


@dataclass(frozen=True)
class Fixture:
    """A directory of tool texts under `tests/checker/fixtures/rules/<rule>/<name>/` (`kernel.sass`,
    `kernel.cfg.dot`, `kernel.res`) and the count the rule must report on it: 0 where it must not fire."""
    name: str
    count: int


@dataclass(frozen=True)
class Rule:
    name: str
    statement: str
    check: Callable[[Kernel], list[Finding]]
    refusal: str
    fix: str
    fixtures: tuple[Fixture, ...]
    bound: bool = False

    def __post_init__(self):
        counts = {f.count for f in self.fixtures}
        if len(self.fixtures) < 2 or not any(counts) or (len(counts) < 2 if self.bound else all(counts)):
            raise ValueError(f"rule {self.name}: needs at least two fixtures, one where it fires and one where it does not "
                             f"(a bound rule that measures every kernel: two fixtures of different figures)")

    def count(self, kernel: Kernel) -> int:
        """The rule's findings on a kernel, summed."""
        return sum(f.count for f in self.check(kernel))


@dataclass(frozen=True)
class ProfileRule(Rule):
    """A rule over a profiler's exports: `check(profile) -> findings`. `needs` is the export it reads, `details` (the
    built-in rules' results) or `source` (the per-instruction counters); a profile without it says the rule was not run.
    Its fixtures are directories under `tests/checker/fixtures/profile_rules/<rule>/<name>/` holding `details.csv` and/or
    `source.csv`. Findings and declarations are as a model rule's: what a declaration allows is the sum of the counts."""
    needs: str = "source"

    def __post_init__(self):
        super().__post_init__()
        if self.needs not in ("details", "source"):
            raise ValueError(f"rule {self.name}: needs is `details` or `source`")
