"""LAYER 3, THE RULE INTERFACE. A rule is one small unit: a name, a one-line statement of what it flags,
`check(kernel) -> findings`, the refusal's message and its fix, and at least two fixtures (one where it fires, one
where it must not). Write it as one module in `rules/` that defines `RULE`; the registry (`rules/__init__.py`) finds it.
A rule reads the model (`model.Kernel`) and never text. A PROFILE rule (`ProfileRule`) reads a profile (`ncu.Profile`)
instead, and has fixtures of the profiler's exports."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model import Kernel


@dataclass(frozen=True)
class Finding:
    """`where`: an address or the kernel; `count`: how many sites or units this finding stands for (what a declaration
    allows is the sum over a rule's findings); `what`: the instance in a line."""
    where: str
    count: int
    what: str


@dataclass(frozen=True)
class Fixture:
    """A directory of tool texts under `tests/checker_fixtures/rules/<rule>/<name>/` (`kernel.sass`,
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

    def __post_init__(self):
        if len(self.fixtures) < 2 or not any(f.count for f in self.fixtures) or all(f.count for f in self.fixtures):
            raise ValueError(f"rule {self.name}: needs at least two fixtures, one where it fires and one where it does not")

    def count(self, kernel: Kernel) -> int:
        """The rule's findings on a kernel, summed."""
        return sum(f.count for f in self.check(kernel))


@dataclass(frozen=True)
class ProfileRule(Rule):
    """A rule over a profiler's exports: `check(profile) -> findings`. `needs` is the export it reads, `details` (the
    built-in rules' results) or `source` (the per-instruction counters); a profile without it says the rule was not run.
    Its fixtures are directories under `tests/checker_fixtures/profile_rules/<rule>/<name>/` holding `details.csv` and/or
    `source.csv`. Findings and declarations are as a model rule's: what a declaration allows is the sum of the counts."""
    needs: str = "source"

    def __post_init__(self):
        super().__post_init__()
        if self.needs not in ("details", "source"):
            raise ValueError(f"rule {self.name}: needs is `details` or `source`")
