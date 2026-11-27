# The checker: what is wrong with a compiled kernel

warpline has two modules and a common part ([the two modules](modules.md)). The RECORDER (`warpline/recorder/`, the rest of
these docs) reads a kernel's timeline out of stamps. The CHECKER (`warpline/checker/`) takes a compiled CUDA kernel and says
what is wrong with it, from the code the toolkit made, with no GPU and no stamps. It needs the CUDA toolkit's `cuobjdump` and
`nvdisasm` to read a binary, and nothing at all to read their saved output (which is what its tests do). The manifests' form,
the checkers, the declarations and the comparison are the common part's (`warpline/manifest.py`, `checkers.py`, `declare.py`):
the checker writes the static manifest into it, and gives a kernel's [run manifest](run-manifest.md) its profile rules.

## Four layers, each depending only on the one before

1. READERS (`readers.py`): one small function a vendor output, text in and plain data out.
   `sass_functions` (`cuobjdump -sass`: each function's instructions, address, predicate, opcode, operands),
   `cfg_functions` (`nvdisasm -bbcfg -poff`: a function's basic blocks and edges), `res_usage` (`cuobjdump
   -res-usage`: registers, stack bytes, shared bytes), `demangled` (`c++filt`), and, optionally, `source_lines` (`nvdisasm
   --print-line-info`: each instruction's source file and line). A reader takes TEXT, so each is tested
   on a fixture; `runner.py` is the only module that starts a process (the toolkit's tools on a binary, and the profiler for
   `warpline run --profile`, [the run layer](run-layer.md)). Two more readers: `notices.read_notices` (the
   build's messages) and `ncu` (Nsight Compute's CSV exports).
2. THE KERNEL MODEL (`model.py`, `loops.py`): one plain structure a kernel. [The model](checker-model.md).
3. RULES (`rule.py`, `rules/`): a small unit each; the registry lists them. [Writing a rule](checker-rules.md).
4. THE MANIFEST'S COLLECTION AND THE COMMAND (`manifest.py`, `cli.py`, `checker.bzl`): the checker writes the MANIFEST of a
   binary, what each rule found on a kernel and the figures of the kernel and of each loop (keyed by file and position, never by
   line), as stable JSON ([the manifest](checker-manifest.md)); a project says, in its declarations, how many findings of a rule a
   kernel may have and why, and may bound a figure, hash a set of fields or ask that every field be judged; the CHECKERS (the
   common `warpline/checkers.py` and `warpline/declare.py`) judge the manifest and refuse the rest.
   [Declaring](checker-declarations.md), with a note of the project's own per rule.

Beside the rules of the model, two more sources of findings, both judged by the same declarations: the build's notices, each
counted under a rule by a row of a ledger ([the notice ledger](checker-notices.md)), and a profile's counters and verdicts
([rules backed by the profiler](checker-profile.md)), which are a run's: they are written into a kernel's
[run manifest](run-manifest.md), not the binary's.

Rules read the model, never text. Thresholds and reasons are the project's data, never the module's.

## The command

    python -m warpline check <binary> --declare <declarations.json> [--arch sm_90a] [--toolkit DIR]
    python -m warpline check <stem> --fixture --declare <declarations.json>     # saved tool outputs, no toolkit
        [--notices <build log>]... [--ledger <ledger.json>] [--run <run.manifest.json>]... [--notes <notes.json>]

`--notices` reads the build's saved messages (repeated for a log a compile) against the module's base ledger of the toolkit's notices, and `--ledger`
gives the project's rows (override or add by tool, code and text); `--run` a kernel's run manifest (`warpline run-manifest`,
[the run manifest](run-manifest.md)), judged with the binary's against the one declarations file, repeated for several;
`--notes` a file of the project's notes per rule (the declarations file may carry them: `"notes"`). Each is optional.
`--lines <listing>` (on `check`, `manifest` and `held`) gives the disassembly with source-line marks, so a finding can name
source lines; it is optional everywhere, and without it a finding names addresses.

`<binary>` is an executable, a shared library, a fatbin or a cubin; give `--arch` for a fat binary of several
architectures. The toolkit is `--toolkit`, then `CUDA_HOME`, then `PATH`. `--fixture` reads `<stem>.sass`,
`<stem>.cfg.dot`, `<stem>.res` and, optionally, `<stem>.names` (`c++filt`'s answer, a line a function) and `<stem>.lines` (the
disassembly with source-line marks). The verb is the checker's alone: the
recorder's own checks of a record are `warpline verify` ([the two modules](modules.md)).

`manifest` writes the binary's manifest (`--text` prints its fields) and `compare` (a common verb) judges manifests
([the manifest](checker-manifest.md)); `check` is both:

    python -m warpline manifest <binary> -o build.manifest.json
    python -m warpline compare build.manifest.json [run.manifest.json ...] --declare <declarations.json>

A refusal prints the kernel, the rule, the count found and declared (and the declared reason, or `undeclared`), what the rule says, each
finding with where it is, the fix, and the project's note for the rule, if it gave one. Exit 0: nothing refused; 1: a refusal; 2: a tool, text or declarations file that
is not usable.

## What a loop holds

    python -m warpline held <binary> [--kernel REGEX] [--lines <listing>] [--arch sm_90a] [--toolkit DIR]
    python -m warpline held <stem> --fixture [--kernel REGEX]

For each kernel (those whose demangled name matches `--kernel`, all by default), each natural loop, outermost first, nested
loops indented under the loop around them: the vector registers live at the loop's top (the header's entry,
[the model's liveness](checker-model.md#register-liveness)), grouped by the place that last made each before the loop, most first. A
place is a source `file:line` where the listing with source-line marks is given (`--lines`, or `<stem>.lines` with
`--fixture`), the address of the instruction otherwise. Per group: how many registers, how many of them an instruction in the
loop reads at all (a held value the loop never touches is the one a spill costs most to keep), how many the loop also
rewrites, and the first registers. Plain text, kernels in the binary's order, loops by address: the same text on every run, no
times, and paths cut to what follows the build's execution root.

    held_many(const float *, const float *, float *, int)  (40 registers, 1 loop(s))
      loop at 0x02a0: 30 registers live at its top, 2 uniform
          25     0 read in the loop  examples/checker/held_many.cu:12  [R0 R4 R5 R6 R7 R8 ...]
           4     4 read in the loop, 3 rewritten in it  examples/checker/held_many.cu:15  [R3 R17 R19 R21]
           1     0 read in the loop  examples/checker/held_many.cu:9  [R23]

Under a small register count the values that do not fit are spilled: the stack frame of rule `stack_memory`. `held` says which
source lines made the values a loop holds; the rule [`loop_held`](checker-rules.md#loop_held) reports the largest count as a
figure a project can bound. Exit 0; 2 when a tool, a text or the pattern is not usable.

## As a Bazel test

`checker.bzl` exports `warpline_check_test`; `examples/checker/` is a complete target:

    load("@warpline//:checker.bzl", "warpline_check_test")
    warpline_check_test(name = "tile_check", binary = ":my_kernel", declare = "declarations.json", arch = "sm_90a",
                        notices = ["build.log"], ledger = "ledger.json", runs = ["tile.run.manifest.json"])

`notices` (a list of labels of the saved build messages, a file a compile), `ledger`, `runs` (a list of labels of run manifests, `--run`) and `notes` are
optional and are the command's options of the same names. `warpline_compare_test(name, manifests, declare)` judges manifest
files alone, a run manifest the project keeps ([in the build graph](run-manifest.md#in-the-build-graph)).

`binary` is a label of one compiled kernel (a cc target's library, a fatbin, a cubin) and the test needs the toolkit's
tools. `fixture = "tile"` instead checks saved tool outputs (`tile.sass`, `tile.cfg.dot`, `tile.res`): the example, and
what `bazel test //...` runs with no GPU and no toolkit installed.

`warpline_check(name, binary, declarations)` is the same check as two BUILD actions, the manifest and then the comparison, with the
manifest a side output (`manifest` output group); a refusal fails the build ([the manifest](checker-manifest.md#as-build-actions)).
`warpline_check_actions(ctx, name, binary, declarations, ...)` makes the same two actions inside a rule of the project's own (the
rule that links the binary, so that its own build fails on a refusal). The example kernels themselves are compiled by that run with
the repository's pinned toolkit, and their outputs are checked against the committed fixtures
([the model](checker-model.md#the-example-kernel-and-its-saved-outputs)).

## What the readers read from the real tools (facts the code relies on)

- `cuobjdump -res-usage` states the stack frame as `STACK`; its `LOCAL` is 0 even where local memory exists.
- `nvdisasm -bbcfg -poff` marks no edge kind (every edge is `solid`); a function's own node is its entry. The listing
  from `cuobjdump -sass` also holds padding after the last instruction (a self-branch and `NOP`s) that the graph does
  not; the model drops instructions after the graph's last offset, and refuses any other disagreement.
- A tensor step's commit is the `gsb0` flag on a group's last `HGMMA`/`IGMMA`; the wait is `WARPGROUP.DEPBAR.LE gsb0, N`. A synchronous step (`mma.sync`, `HMMA`/`IMMA`) has no commit and no wait: it is a step of kind `sync` ([the model](checker-model.md#step-kinds)).

## Adopting the module

Built: the readers, the model, the rules (stage 1), the notice ledger, notes and the profile rules (stage 2), and the static
rules of [the rules page](checker-rules.md#the-stage-3-rules) (stage 3). A project adopting it does this, and only this:

- writes its own declarations (`declarations.json`: per kernel and rule, the count allowed and why; for a bound rule, the bound),
  its own ledger rows and its notes, and has a note for each rule name (`warpline.checker.manifest.rule_names()` lists them)
  checked by a lint;
- puts a `warpline_check_test` (or a `warpline_check`, in the graph) on each kernel it ships (a cc target, a fatbin or a cubin, with the toolkit's tools on PATH);
- keeps what is its own convention out of the module: a comment marker, a registry of kernels, a build policy such as
  "cubins only", a named helper, a clock scheme. A rule here is true of any CUDA kernel on this toolchain; a check that is
  true of one project's conventions is that project's lint, written over the same model (`runner.kernels_of`).
