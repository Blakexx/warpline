# The manifest, and the checkers that judge it

The checker writes a MANIFEST of a binary: one stable JSON file describing each kernel. It holds what each rule found on the
kernel (the detectors' findings) and DESCRIPTIVE FIGURES: the kernel's totals and, for each loop, what it holds and does. It is a
build output and data about a build, not a verdict on it: it can be read and judged on its own. It is STATIC: everything in it is
read from the compiled code; what exists only after a run (a record's timeline, a profiler's counters) is a
[run manifest](run-manifest.md), the same family of file, judged by the same checkers. A second step, the CHECKERS, takes a
manifest and a project's [declarations](checker-declarations.md), which hold the checkers' arguments, and refuses. The two are
separate so that a consumer can keep the manifest (to diff two builds, to print a table, to feed a gate of its own) and judge it
as often as it likes.

| checker | judges | arguments (in the declarations) |
|---|---|---|
| `violations` | a rule's count at a kernel | `rule`, `allow` (the declarations as ever; with none, the rule's default) |
| `bound` | a figure of the manifest | `figure`, one of `at_most` / `at_least` (a number) / `exactly` (a count) |
| `hash` | the values of a set of fields | `fields`, `hash` |
| `present` | how many kernels a pattern names | one of `at_most` / `at_least` / `exactly` (a count) |
| (coverage) | that every numeric field is judged by something | `"coverage": true`; `waive` entries |
| (strict) | that every `violations` declaration that allows findings names a place that has one | `"strict": true` |

    python -m warpline manifest <binary> -o build.manifest.json [--arch sm_90a] [--toolkit DIR] [--lines <listing>]
        [--notices <build log>]... [--ledger <ledger.json>]
    python -m warpline manifest <binary> --text [--kernel REGEX] [--lines <listing>]    # every field, a line each, for a person
    python -m warpline compare build.manifest.json [run.manifest.json ...] --declare declarations.json [--notes notes.json]
        [--ledger ledger.json]

`manifest` exits 0 (2 when a tool or a text is not usable); `compare` exits 0 when nothing is refused, 1 on a refusal, 2 when a
file is not usable. `check <binary> --declare FILE` is the two in one command, `--manifest FILE` also writes the file, and
`--run FILE` judges a kernel's run manifest with it ([one declarations file](run-manifest.md#with-the-static-manifest-one-declarations-file)).
`manifest` reads a saved set of tool outputs with `--fixture <stem>`, as `check` does. `manifest` and `check` are the checker's;
`compare` is a common verb, in every command ([the two modules](modules.md)).

## The file

    {"format": "warpline-manifest/1",
     "notices": 6,
     "said": [],
     "places": [
      {"place": "sample::tile()", "kernel": true, "name": "_ZN6sample4tileEv",
       "figures": {"kernel": {...}, "loops_keyed_by": "source file and position", "loops": {...}},
       "rules": {
        "stack_memory": {"count": 128, "bound": false, "default": 0, "declarable": true,
                         "message": "the kernel uses stack memory ...", "fix": "remove the spill ...",
                         "sites": [{"where": "0x0020", "count": 128, "what": "a 128-byte stack frame; ..."}]}}}]}

- A PLACE is a kernel (`kernel: true`, `name` the mangled name, `place` the demangled one, `figures` its figures), or the
  function, `file(line)` or `the build` a notice names where it is not a kernel of the binary (no `figures`).
- A kernel lists every rule of the model that ran, counts of 0 included, in the registry's order. A notice rule is listed
  where a notice was counted under it. No profile rule is ever in it: those are a run's ([the run manifest](run-manifest.md)).
  The counts of a rule are over its SITES (where, how many, what), and `count` is their sum: what a declaration's `allow` is
  compared with.
- `message` and `fix` are the rule's own words (a notice rule's are its rows' reasons and fixes), so the manifest is enough to
  print a refusal.
- `bound`: the rule reports a figure (a byte count, an instruction count): with no declaration it is shown, never refused.
- `default`: what is allowed where no declaration names the rule at the place: 0, a count a notice row gives, or null for a
  notice accepted by default (counted, refused only once a declaration gives it a number).
- `declarable`: false for `notice_without_row`, the one count no declaration can lift.
- `notices` is the number of notices read from the build's messages (null where none were given); `said` are the lines about
  what was not run (given by the caller of `collect`; the command writes none).

The file has no time, path or run identity and is written in a fixed order: the same binary and the same inputs write the same
bytes. Its format is `warpline-manifest/1`; the reader takes it and `warpline-run-manifest/1` and refuses any other. The form
of both, their reading, writing and validation are the common module's (`warpline/manifest.py`): `warpline.manifest.loads(text)`
reads and validates (`ManifestError`: the format, the shape, a count that is not its sites' sum, a kernel with no figures);
`warpline.manifest.dumps(m)` writes. The checker makes it: `warpline.checker.manifest.collect(kernels, rules, notices=, ledger=)`,
from a model and the readers' output.

## A kernel's figures

`--text` prints every field of each kernel, `path = value`, with each loop's note above its fields:

    held_many(const float *, const float *, float *, int)  (loops keyed by source file and position)
      kernel/instructions = 102
      kernel/registers = 40
      ...
      # held_many.cu/1 at held_many.cu:15
      loops/held_many.cu/1/copies = 1
      loops/held_many.cu/1/live = 30
      loops/held_many.cu/1/read = 4
      loops/held_many.cu/1/instructions = 8
      loops/held_many.cu/1/class/integer_address = 4
      loops/held_many.cu/1/global_loads = 1
      ...
      rules/loop_held = 30

The same figures are in the file, in each kernel's `figures`:

    "figures": {"kernel": {"instructions": 102, "registers": 40, "stack_bytes": 0, "shared_bytes": 0, "barriers": 0, "tensor_steps": 0,
                           "local_loads": 0, "local_stores": 0, "local_in_loops": 0},
                "loops_keyed_by": "source file and position",
                "loops": {"held_many.cu/1": {"at": "held_many.cu:15", "copies": 1, "live": 30, "read": 4, "instructions": 8,
                                             "class": {"tensor": 0, ..., "integer_address": 4, "other": 0},
                                             "global_loads": 1, "global_stores": 0, "shared_loads": 0, "shared_stores": 0}}}

- `kernel/...`: the kernel's totals: `instructions`, `registers`, `stack_bytes`, `shared_bytes`, `barriers`, `tensor_steps`,
  `local_loads` and `local_stores` (its local-memory loads and stores, `LDL` and `STL`, a dead one not counted) and
  `local_in_loops` (how many of those a natural loop holds); and, where the build's notices were read (`--notices`) and the
  assembler's report names the kernel, `spill_stores` and `spill_loads`, the bytes it spilled (`ptxas -v`'s `Function
  properties` of the entry compiled for the binary's architecture, `--arch`; with none given, the one entry of the name the
  report holds).
- A LOOP is a natural loop ([the model](checker-model.md)). Its key holds no source line and no instruction address, because a
  line moves when code above it is edited and an address moves with every edit, and a hash over loop figures must not fire on
  an edit that changed nothing. The listing with source marks (`--lines`) gives, for an instruction, a file and a line (and for
  inlined code the file and line it was inlined at) and NO function name, so the key is built from what it does give: the
  innermost source FILE of the loop's header (its base name) and the loop's POSITION, its rank by line among the loops of that file at
  the same nesting: `held_many.cu/2.1` is the first loop inside the second loop of that file's loops. `loops_keyed_by` is
  `source file and position`. A line number moves the rank only if a loop is added or removed above it in its file (that is a
  change); a blank line or a comment above moves nothing (`//:graph_shifted_test` builds `examples/checker/shifted/held_many.cu`, the
  kernel with lines inserted above its loops, and compares the two manifests: the same paths and values, the same hash). Several
  compiled copies of one source loop (an unrolled or inlined one: the same file and line) are ONE entry: `copies` is how many and
  every other figure is the most of any copy. Without a listing, or if a loop of the kernel has no line, every loop of the kernel
  is keyed by its nesting path, `loop 2.1` (the first loop inside the second loop of the function), and `loops_keyed_by` is
  `nesting path`.
- `at`: a note for a person, `held_many.cu:15`, where the loop is. It is text, written beside the key; it is never a field, never
  in a path, never hashed or bound. It moves with every edit above the loop: it is the one place a line number is written.
- `live`: the vector registers live at the loop's top; `read`: how many of those an instruction of the loop reads
  ([what a loop holds](checker.md#what-a-loop-holds)); `instructions`: the loop's own instructions (a dead one under `@!PT` is
  not counted, the code of inner loops is); `class/<name>`: those by the model's unit classes ([the model](checker-model.md)),
  spelled with `_` for a space or a slash (`shared_load`, `select_compare`, `integer_address`) and every class present, zeros
  included, with `other` for none; `global_loads`, `global_stores`, `shared_loads`, `shared_stores`: the loop's instructions of
  those kinds (atomics and asynchronous copies are not counted).
- `rules/<rule>`: each rule's count at the kernel: a finding is a field.

A FIELD is the number at a path (the names above joined by `/`). Notices at places that are not kernels have no figures.

## The checkers, in the declarations

Each entry of `declarations` has a `kernel` regex (over the demangled name, as ever) and a `reason`. An entry with a `rule` and no
`check` is `violations`. A figure or fields argument is a regular expression that must match the WHOLE path.

    {"declarations": [
      {"kernel": "^held_many\\(", "rule": "loop_held", "allow": 30, "reason": "..."},
      {"kernel": "^held_many\\(", "check": "bound", "figure": "loops/.*/live", "at_most": 32, "reason": "the register budget"},
      {"kernel": "^held_many\\(", "check": "hash", "fields": "loops/.*/(live|read)", "hash": "37977f91bc7650c2",
       "reason": "the loop's shape is the design"},
      {"kernel": "^held_many\\(", "check": "waive", "fields": "loops/.*/class/.*", "reason": "too noisy"}],
     "coverage": true}

**`violations`**: [as declared](checker-declarations.md): the first entry of the rule whose kernel matches gives the allowance,
else the rule's default. A refusal names the kernel, the rule, the count found, the count allowed and the declaration's reason, or
`undeclared`:

    sample::tile(): stack_memory: 128 found, 64 declared (the staging array)
      the kernel uses stack memory (local memory): its accesses are far slower than registers
      at 0x0020 x128: a 128-byte stack frame; 0 LDL and 0 STL instruction(s) in the listing
      fix: ...

**`bound`**: each field `figure` matches, at each kernel `kernel` matches, is compared with `at_most`, `at_least` (one, a number
0 or more) or `exactly` (a count). Every bound that applies is enforced; they are not alternatives. A figure that matches no field
of a matched kernel is refused (a renamed loop does not pass quietly). On a run manifest `exactly` holds the structural figures
only, and a hash none ([judging values that vary](run-manifest.md#judging-values-that-vary)):

    held_many(...): bound: loops/held_many.cu/1/live is 30, at most 8 declared (the register budget)

**`hash`**: the values of the fields `fields` matches, in the matched kernels, hashed: the lines `path=value` (each kernel's paths
sorted, kernels in the manifest's order), SHA-256, its first sixteen hex digits. The kernel's name is not hashed. It is refused while
the hash differs, and the refusal prints the covered fields' current values and the hash to write: look at the manifest as it stands,
then update the hash. Nothing is stored of a past manifest, and there is no tolerance: any change of a covered value is a refusal.
To start, write any text as the hash; the first refusal prints the real one.

    held_many(...): hash: the 2 covered field(s) hash to 496c062fc6b6e774, stale declared (the loop's shape is the design)
      loops/held_many.cu/1/live = 30
      loops/held_many.cu/1/read = 4
      fix: look at the values above; if they are intended, write "hash": "496c062fc6b6e774" in the declaration

A hash that covers no field is refused.

**`waive`**: the fields `fields` matches, at the kernels `kernel` matches, are not asked about by coverage. The reason is the
project's ("too noisy" is one).

**`present`**: the number of kernels `kernel` matches (over the manifests judged as one) is compared with `at_most`, `at_least` or
`exactly` (one, a count). A project says what its binary carries: an instance it built (`exactly: 1` on the instance's exact
name), every kernel its source defines (`at_least: 1` on each). It reads no field and covers none. The refusal names the pattern,
then the kernels it names, a line each:

    ^void k<\(int\)3>\(: present: names 0 kernel(s), exactly 1 declared (the build carries the arm k<3>)

**`"strict": true`** (a top-level key; absent or false: off): a `violations` entry that allows findings (`allow` above 0, of a rule
that is no bound rule) where no place it names has a finding of the rule is refused: a declaration says only what is so, and one
the code has outgrown goes with it.

    ^void k<\(int\)2>\(: strict: declares step_conversion (allow 2), which no place it names has (a kept conversion)

An entry that names a `check` that is not one of these, a bound or a presence with none or two comparisons, a missing argument or
reason, or a regex that does not compile is an error (exit 2). A bound, hash or waiver whose `kernel` regex names no kernel is
listed as a note, as a stale declaration is; a presence is never stale, a pattern that names no kernel being what it counts.

The comparison, `warpline.declare.compare(manifest, declarations, notes=None, checks=None)` (`manifest` one manifest or several
judged as one: a kernel's fields are those of every manifest that holds its place), returns the refusals of both kinds, the
figures of bound rules with no bound declared, the declarations that name no place (stale: a note, never a refusal) and the
lines the manifests say. For each place and rule: a bound rule with no declaration at the place is a figure; any other rule is
refused when its count is above the first declaration (file order) whose regex matches the place and names the rule, else above
the rule's `default`. A declaration may tighten as well as allow: `allow: 0` on a rule a notice row accepts by default refuses the
notice; `allow: 2` on a refused one accepts two.

## Coverage

`"coverage": true` at the top of the declarations (absent or false: off, so a project's existing declarations never break) asks
that every numeric field of every kernel, the rule counts included, be COVERED or WAIVED. A field is covered by a `bound` whose
figure matches it, a `hash` whose fields match it, or, for `rules/<rule>`, a `violations` entry for the rule at the kernel or the
rule's default. The default covers a rule that is refused above it: not a notice accepted by default (a null default) and not a
bound rule (`function_size`, `step_work`, `step_port`, `step_density`, `step_gap`, `loop_held`: a figure shown and never refused),
which need an entry. An uncovered, unwaived field is refused by name:

    held_many(...): coverage: 33 field(s) neither covered by a checker nor waived
      kernel/instructions
      kernel/registers
      loops/held_many.cu/1/live
      ...
      rules/function_size
      fix: cover each with a bound or a hash, or waive it with a reason ({"check": "waive", "fields": ..., "reason": ...})

Coverage does not judge whether a covering checker is strict enough: `at_most: 999` covers. `examples/checker/manifest.declarations.json`
is a declarations file with all of it, run by `//examples/checker:manifest_check` (and, on the shifted source, `shifted_check`).

## As build actions

`checker.bzl`'s `warpline_check(name, binary, declarations)` is the two steps as two actions: the manifest, then the comparison
(with the kernels' run manifests where `runs` lists them, [in the build graph](run-manifest.md#in-the-build-graph)). A refusal
fails the build and prints itself. The manifest is a side output, the output group `manifest`:

    load("@warpline//:checker.bzl", "warpline_check")
    warpline_check(name = "tile_check", binary = ":tile.cubin", declarations = "tile.declarations.json",
                   toolkit = "@cuda_tools//:toolkit", notices = [":tile.ptxas"], lines = ":tile.lines")
    filegroup(name = "tile_manifest", srcs = [":tile_check"], output_group = "manifest")     # tile_check.manifest.json

A rule of the project's own makes the same two actions with `warpline_check_actions(ctx, name, binary, declarations, arch=,
toolkit=, notices=, ledger=, notes=, lines=, runs=)` (Files, not labels), its attributes extended by `WARPLINE_CHECK_ATTRS` (the
checker's command): the rule that links a library calls it on the library, once an architecture, so the library's own build
fails on a refusal, and returns the manifest's and the comparison's files (`struct(manifest, report)`) as its outputs.

A declarations file belongs to a build target, which is one architecture: a declaration carries none. `toolkit` is a label
whose files hold `bin/cuobjdump`, `bin/nvdisasm` and `bin/cu++filt` (the actions run them; without it they must be on PATH,
which a sandboxed action does not have). `examples/checker/` has several: `steps_check`, `mma_check` and `collectives_check`,
over the example kernels the graph compiles, and `//:graph_manifest_test` reads their manifests.

The toolkit's `cu++filt` writes `float *` where GNU `c++filt` writes `float*`: a declaration's regex over a parameter list
should not depend on the spacing (`^ns::kernel\(` and not the whole signature).

## In code

The checker's: `warpline.checker.manifest.of_kernel(kernel)` makes a kernel's figures and `collect` the manifest. The common
module's: `warpline.manifest.fields(place)` gives a place's `{path: number}` with the rule counts, `warpline.checkers.load(text)`
reads the entries above (and `coverage` and `strict`), and `warpline.checkers.judge(manifests, checks, declarations)` returns the
refusals of the bounds, hashes, presences, coverage and strictness over one manifest or several.
