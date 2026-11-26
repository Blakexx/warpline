# How a project declares

The module holds no threshold. A project's JSON file says, per kernel and rule, how many findings it allows and why:

```json
{"declarations": [
  {"kernel": "^sample::tile\\(\\)$", "rule": "stack_memory", "allow": 128,
   "reason": "a per-thread staging array indexed by a runtime value; measured once and accepted"}
]}
```

- `kernel`: a regex, matched with `re.search` against the DEMANGLED name.
- `rule`: a registered rule's name (a rule of the model, a [profile rule](checker-profile.md), or a rule a row of the [notice ledger](checker-notices.md) counts its notices under). `allow`: the count of its findings the kernel may have (what `Finding.count` sums to). For a notice rule the kernel is the function the notice names (demangled). A declaration carries no architecture: a project attaches its declarations per build target, which is one architecture.
- `reason`: required, a sentence. An empty reason, an unknown rule or a bad regex is an error (exit 2).
- Undeclared means the rule's default: zero for a rule of the model, and the notice row's own allowance for a notice (0, a count, or none: accepted). When several declarations match a kernel and a rule, the first in file order applies.
- A declaration may TIGHTEN as well as allow: `allow: 0` on a notice the ledger accepts by default (`wgmma_arrive_injected`, `unreachable_loop`) refuses it at the kernels the regex names.
- A declaration whose regex names no kernel is listed as a note, not a refusal: the declaration has gone stale.

The declarations are compared with a [manifest](checker-manifest.md), a binary's or a kernel's [run manifest](run-manifest.md),
or several judged as one: `warpline.declare.compare(manifest, declarations)` (the common module) returns the refusals, each
naming the kernel, the rule, the count found, the count allowed and the declaration's reason (or `undeclared`);
`warpline.checker.manifest.check(kernels, rules, declarations)` is `collect` and `compare` over the model's rules.
`python -m warpline compare <manifest> [<manifest> ...] --declare <file>` (or `check <binary> --declare <file> [--run <run
manifest>]`, which writes the binary's manifest first) prints them and exits 1; `warpline_check_test` and `warpline_check` in
`checker.bzl` run it as a Bazel test and as two build actions, and `warpline_compare_test` on manifest files (see
[the checker](checker.md)). A rule a declaration names is one of the modules present (the checker's rules, profile rules and
notice rows) or one a manifest being judged holds.

## Beyond counts

The entries above are the `violations` checker (an entry with a `rule` and no `check`). The same list may hold entries with
`"check": "bound"` (a figure of the kernel's manifest, or of its run's, at most, at least or exactly a value), `"check": "hash"` (the values of
the fields it covers against an expected hash) and `"check": "waive"`, and a top-level `"coverage": true` asks that every numeric
manifest field be covered by one of them or waived with a reason; all are in [the manifest](checker-manifest.md). A project
with none of them is judged by its counts alone.

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

`warpline.checker.manifest.rule_names(ledger=None)` returns every rule name of the checker a declaration or a note may name,
sorted: the registry's, the profile rules', and the rows' rules of the base ledger (or of `ledger`, a merged ledger, to include
a project's own). A consumer lints that each has a note:

```python
from warpline.checker import manifest
missing = set(manifest.rule_names()) - set(json.load(open("declarations.json")).get("notes", {}))
```

Today the list is `accumulator_moved`, `build_info`, `clock_uniform`, `convergence_call`, `function_size`, `load_in_trip`, `local_memory_used`,
`loop_convergence`, `mma_serialized`, `profiler_verdicts`, `register_spill`, `setmaxnreg_ignored`,
`shared_wavefronts_excess`, `stack_memory`, `step_branches`, `step_conversion`, `step_density`, `step_gap`, `step_global`,
`step_port`, `step_predicated`, `step_rolled`, `step_serialized`, `step_work`, `store_reload`, `store_unfenced`, `unreachable_loop`,
`unused_variable`, `variable_unset`, `wait_after_issue`, `wait_all_with_work`, `wait_injected`, `wgmma_arrive_injected`.

## A bound

Five rules report a figure, not a count: `function_size` (instructions a function, padding excluded), `step_port` (bytes a
step puts through the shared-memory port), `step_work` (instructions a step), `step_density` (static instructions a step) and
`step_gap` (the longest run without a step in a step loop). Their `allow` is the project's bound, and with none declared the figure is printed and nothing is refused:

```json
{"kernel": "^example::main\\(", "rule": "step_port", "allow": 4096, "reason": "two steps of the 64x16x16 form: the port's budget a step"}
```

    1 figure(s) of a bound rule, with no bound declared:
      example::main(...): step_port: 3584 (no bound declared): 3,584 B a step (28 wavefronts) at unrolled at 0x0680 ...
