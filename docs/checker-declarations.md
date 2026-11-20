# How a project declares

The module holds no threshold. A project's JSON file says, per kernel and rule, how many findings it allows and why:

```json
{"declarations": [
  {"kernel": "^sample::tile\\(\\)$", "rule": "stack_memory", "allow": 128,
   "reason": "a per-thread staging array indexed by a runtime value; measured once and accepted"}
]}
```

- `kernel`: a regex, matched with `re.search` against the DEMANGLED name.
- `rule`: a registered rule's name (a rule of the model, a [profile rule](checker-profile.md), or a rule a `declare` row of the [notice ledger](checker-notices.md) counts under). `allow`: the count of its findings the kernel may have (what `Finding.count` sums to). For a notice rule the kernel is the function the notice names (demangled).
- `reason`: required, a sentence. An empty reason, an unknown rule or a bad regex is an error (exit 2).
- Undeclared means zero. When several declarations match a kernel and a rule, the first in file order applies.
- A declaration whose regex names no kernel is listed as a note, not a refusal: the declaration has gone stale.

`declare.check(kernels, rules, declarations)` returns the refusals: each rule whose count on a kernel is above its
allowance. `python -m warpline check <binary> --declare <file>` prints them and exits 1; `warpline_check_test` in
`checker.bzl` runs it as a Bazel test (see [the checker](checker.md)).

## A note a rule

The file (or a file beside it, `--notes <file>`, `notes =` in the Bazel rule, the same shape) may give, per rule name, a
note of the project's own:

```json
{"notes": {"stack_memory": "the project's own words: where the measurement is, who decides, what to do instead"}}
```

A refusal prints the rule's finding, the rule's generic fix, then the project's note, last:

    sample::tile(): stack_memory: 128 found, 0 declared
      the kernel uses stack memory (local memory): its accesses are far slower than registers
      at 0x0020 x128: a 128-byte stack frame; ...
      fix: remove the spill (...)
      note: the project's own words ...

The module attaches no meaning to a note. A name must be a rule (an unknown name is an error, exit 2) and a note must not
be empty. A note on `notice_without_row` is printed after an unknown notice's refusal.

`declare.rule_names(ledger=None)` returns every rule name a declaration or a note may name, sorted: the registry's, the
profile rules', and the `declare` rows' of the base ledger (or of `ledger`, a merged ledger, to include a project's own).
A consumer lints that each has a note:

```python
from warpline.checker import declare
missing = set(declare.rule_names()) - set(json.load(open("declarations.json")).get("notes", {}))
```

Today the list is `mma_serialized`, `profiler_verdicts`, `shared_wavefronts_excess`, `stack_memory`, `step_rolled`,
`wait_after_issue`, `wait_injected`.
