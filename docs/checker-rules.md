# Writing a rule

A rule is one module in `warpline/checker/rules/` that defines `RULE`, plus its fixtures. The registry
(`rules/__init__.py`) finds every module that defines `RULE`; nothing else is edited.

```python
from ..model import Kernel
from ..rule import Finding, Fixture, Rule

def check(kernel: Kernel) -> list[Finding]:
    ...                                   # read the model, never text

RULE = Rule(
    name="my_rule",                       # what a declaration names
    statement="one line: what it flags",
    check=check,
    refusal="what is wrong, in a sentence",
    fix="what to change",
    fixtures=(Fixture("fires", 2), Fixture("quiet", 0)),
)
```

A `Finding` is `(where, count, what)`: an address (or `kernel`), how many sites or units it stands for, and the
instance in a line. A rule's count on a kernel is the sum of its findings' counts; that is what a declaration allows.

## Fixtures

Each fixture is a directory `tests/checker_fixtures/rules/<rule>/<fixture>/` of a small kernel's tool texts, named as
the readers' fixtures are: `kernel.sass` (`cuobjdump -sass`), `kernel.cfg.dot` (`nvdisasm -bbcfg -poff`), `kernel.res`
(`cuobjdump -res-usage`) and `kernel.names` (`c++filt`). `Fixture(name, count)` is the count the rule must report: a
rule needs at least two fixtures, one where it fires and one where it must not (`Rule` refuses otherwise).
`tests/test_checker_rules.py` runs every registered rule on every fixture it declares and checks that no fixture
directory is undeclared, so writing the module and its fixtures adds the rule's tests.

## The stage-1 rules

| rule | flags | count | fixtures |
|---|---|---|---|
| `step_rolled` | a tensor step in a natural loop (the innermost around it) with no tensor wait in the loop's blocks: such steps run at about half a straight block's rate | the loop's steps, a finding a loop | `rolled` 4, `nested` 2; quiet: `straight`, `outer_wait` |
| `wait_after_issue` | a `WARPGROUP.DEPBAR.LE gsb0, 0x0` straight after its group's last step with no work of substance (memory access, copy, barrier arrival) between, within one basic block | 1 a wait | `bare_wait` 1; quiet: `work_between`, `partial_wait` |
| `stack_memory` | any stack frame in the resource report | the stack bytes; the detail names the `LDL`/`STL` instructions | `spill` 128; quiet: `no_stack` |

## Rules over a profile and over a build's notices

A rule that needs a profiler's counters is a `ProfileRule` in `PROFILE_RULE`, with fixtures of the profiler's CSV
exports: [rules backed by the profiler](checker-profile.md). The toolkit's build messages are not rules but rows of a
ledger: [the notice ledger](checker-notices.md). Both are judged by the same declarations.
