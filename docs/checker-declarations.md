# How a project declares

The module holds no threshold. A project's JSON file says, per kernel and rule, how many findings it allows and why:

```json
{"declarations": [
  {"kernel": "^sample::tile\\(\\)$", "rule": "stack_memory", "allow": 128,
   "reason": "a per-thread staging array indexed by a runtime value; measured once and accepted"}
]}
```

- `kernel`: a regex, matched with `re.search` against the DEMANGLED name.
- `rule`: a registered rule's name. `allow`: the count of its findings the kernel may have (what `Finding.count` sums to).
- `reason`: required, a sentence. An empty reason, an unknown rule or a bad regex is an error (exit 2).
- Undeclared means zero. When several declarations match a kernel and a rule, the first in file order applies.
- A declaration whose regex names no kernel is listed as a note, not a refusal: the declaration has gone stale.

`declare.check(kernels, rules, declarations)` returns the refusals: each rule whose count on a kernel is above its
allowance. `python -m warpline check <binary> --declare <file>` prints them and exits 1; `warpline_check_test` in
`checker.bzl` runs it as a Bazel test (see [the checker](checker.md)).
