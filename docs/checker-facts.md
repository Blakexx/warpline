# The facts, and their comparison

A run of the checker writes FACTS: what each rule found on each kernel of a binary, as a stable JSON file. The facts are data
about a build, not a verdict on it. A second step, the COMPARISON, takes a facts file and a project's declarations
([declaring](checker-declarations.md)) and refuses. The two are separate so that a consumer can keep the facts (to diff two
builds, to print a table, to feed a gate of its own) and judge them as often as it likes.

    python -m warpline facts <binary> -o build.facts.json [--arch sm_90a] [--toolkit DIR]
        [--notices <build log>] [--ledger <ledger.json>] [--profile <csv>]...
    python -m warpline compare build.facts.json --declare declarations.json [--notes notes.json] [--ledger ledger.json]

`facts` exits 0 (2 when a tool or a text is not usable); `compare` exits 0 when nothing is refused, 1 on a refusal, 2 when a
file is not usable. `check <binary> --declare FILE` is the two in one command, and `--facts FILE` also writes the file.
`facts` reads a saved set of tool outputs with `--fixture <stem>`, as `check` does.

## The file

    {"format": "warpline-facts/1",
     "notices": 6,
     "said": ["note: profile rules not run (no --profile): profiler_verdicts, shared_wavefronts_excess"],
     "places": [
      {"place": "sample::tile()", "kernel": true, "name": "_ZN6sample4tileEv",
       "rules": {
        "stack_memory": {"count": 128, "bound": false, "default": 0, "declarable": true,
                         "message": "the kernel uses stack memory ...", "fix": "remove the spill ...",
                         "sites": [{"where": "0x0020", "count": 128, "what": "a 128-byte stack frame; ..."}]}}}]}

- A PLACE is a kernel (`kernel: true`, `name` the mangled name, `place` the demangled one), or the function, `file(line)` or
  `the build` a notice or a profile names where it is not a kernel of the binary.
- A kernel lists every rule of the model that ran, counts of 0 included, in the registry's order. A notice rule is listed
  where a notice was counted under it, a profile rule where a profile was read. The counts of a rule are over its SITES
  (where, how many, what), and `count` is their sum: what a declaration's `allow` is compared with.
- `message` and `fix` are the rule's own words (a notice rule's are its rows' reasons and fixes), so a facts file is
  enough to print a refusal.
- `bound`: the rule reports a figure (a byte count, an instruction count): with no declaration it is shown, never refused.
- `default`: what is allowed where no declaration names the rule at the place: 0, a count a notice row gives, or null for a
  notice accepted by default (counted, refused only once a declaration gives it a number).
- `declarable`: false for `notice_without_row`, the one count no declaration can lift.
- `notices` is the number of notices read from the build's messages (null where none were given); `said` are the lines about
  what was not run.

The file has no time, path or run identity and is written in a fixed order: the same binary and the same inputs write the
same bytes. `facts.loads(text)` reads and validates it (`FactsError`: the format, the shape, a count that is not its sites'
sum); `facts.dumps(facts)` writes it; `facts.collect(kernels, rules, notices=, ledger=, profiles=, profile_rules=)` makes it
from a model and the readers' output (`checker/facts.py`).

## The comparison

`declare.compare(facts, declarations, notes=None)` returns the refusals, the figures of bound rules with no bound declared,
the declarations that name no place (stale: a note, never a refusal) and the lines the facts say. For each place and rule:

- a bound rule with no declaration at the place is a figure;
- any other rule is refused when its count is above the first declaration (file order) whose regex matches the place and
  names the rule; with none, above the rule's `default`. A declaration may tighten as well as allow: `allow: 0` on a rule a
  notice row accepts by default refuses the notice; `allow: 2` on a refused one accepts two.

A refusal names the kernel, the rule, the count found, the count allowed and the declaration's reason, or `undeclared`:

    sample::tile(): stack_memory: 128 found, 64 declared (the staging array)
      the kernel uses stack memory (local memory): its accesses are far slower than registers
      at 0x0020 x128: a 128-byte stack frame; 0 LDL and 0 STL instruction(s) in the listing
      fix: ...

## As build actions

`checker.bzl`'s `warpline_check(name, binary, declarations)` is the two steps as two actions: the facts, then the comparison.
A refusal fails the build and prints itself. The facts file is a side output, the output group `facts`:

    load("@warpline//:checker.bzl", "warpline_check")
    warpline_check(name = "tile_check", binary = ":tile.cubin", declarations = "tile.declarations.json",
                   toolkit = "@cuda_tools//:toolkit", notices = ":tile.ptxas")
    filegroup(name = "tile_facts", srcs = [":tile_check"], output_group = "facts")     # tile_check.facts.json

A declarations file belongs to a build target, which is one architecture: a declaration carries none. `toolkit` is a label
whose files hold `bin/cuobjdump`, `bin/nvdisasm` and `bin/cu++filt` (the actions run them; without it they must be on PATH,
which a sandboxed action does not have). `examples/checker/` has three: `steps_check`, `mma_check` and `collectives_check`,
over the example kernels the graph compiles, and `//:graph_facts_test` reads their facts files.

The toolkit's `cu++filt` writes `float *` where GNU `c++filt` writes `float*`: a declaration's regex over a parameter list
should not depend on the spacing (`^ns::kernel\(` and not the whole signature).
