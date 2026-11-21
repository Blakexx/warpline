# The checker: what is wrong with a compiled kernel

warpline has two modules. The RECORDER (`warpline/`, the rest of these docs) reads a kernel's timeline out of stamps.
The CHECKER (`warpline/checker/`) takes a compiled CUDA kernel and says what is wrong with it, from the code the
toolkit made, with no GPU and no stamps. It needs the CUDA toolkit's `cuobjdump` and `nvdisasm` to read a binary, and
nothing at all to read their saved output (which is what its tests do).

## Four layers, each depending only on the one before

1. READERS (`readers.py`): one small function a vendor output, text in and plain data out.
   `sass_functions` (`cuobjdump -sass`: each function's instructions, address, predicate, opcode, operands),
   `cfg_functions` (`nvdisasm -bbcfg -poff`: a function's basic blocks and edges), `res_usage` (`cuobjdump
   -res-usage`: registers, stack bytes, shared bytes), `demangled` (`c++filt`). A reader takes TEXT, so each is tested
   on a fixture; `runner.py` is the only module that starts a process. Two more readers: `notices.read_notices` (the
   build's messages) and `ncu` (Nsight Compute's CSV exports).
2. THE KERNEL MODEL (`model.py`, `loops.py`): one plain structure a kernel. [The model](checker-model.md).
3. RULES (`rule.py`, `rules/`): a small unit each; the registry lists them. [Writing a rule](checker-rules.md).
4. DECLARATIONS AND THE COMMAND (`declare.py`, `cli.py`, `checker.bzl`): a project says how many findings of a rule a
   kernel may have and why; the rest are refusals. [Declaring](checker-declarations.md), with a note of the project's own
   per rule.

Beside the rules of the model, two more sources of findings, both judged by the same declarations: the build's notices,
read against a ledger ([the notice ledger](checker-notices.md)), and a profile's counters and verdicts
([rules backed by the profiler](checker-profile.md)).

Rules read the model, never text. Thresholds and reasons are the project's data, never the module's.

## The command

    python -m warpline check <binary> --declare <declarations.json> [--arch sm_90a] [--toolkit DIR]
    python -m warpline check <stem> --fixture --declare <declarations.json>     # saved tool outputs, no toolkit
        [--notices <build log>] [--ledger <ledger.json>] [--profile <details.csv>] [--profile <source.csv>] [--notes <notes.json>]

`--notices` reads the build's saved messages against the module's base ledger of the toolkit's notices, and `--ledger`
gives the project's rows (override or add by tool, code and text); `--profile` gives Nsight Compute's exports, any number;
`--notes` a file of the project's notes per rule (the declarations file may carry them: `"notes"`). Each is optional;
without `--profile` the profile rules are reported as not run.

`<binary>` is an executable, a shared library, a fatbin or a cubin; give `--arch` for a fat binary of several
architectures. The toolkit is `--toolkit`, then `CUDA_HOME`, then `PATH`. `--fixture` reads `<stem>.sass`,
`<stem>.cfg.dot`, `<stem>.res` and, optionally, `<stem>.names` (`c++filt`'s answer, a line a function). The verb is the
recorder's `check`: with `--declare` or `--fixture` it is the checker, without it the argument is a record.

A refusal prints the kernel, the rule, the count found and declared (and the declared reason), what the rule says, each
finding with where it is, the fix, and the project's note for the rule, if it gave one. Exit 0: nothing refused; 1: a refusal; 2: a tool, text or declarations file that
is not usable.

## As a Bazel test

`checker.bzl` exports `warpline_check_test`; `examples/checker/` is a complete target:

    load("@warpline//:checker.bzl", "warpline_check_test")
    warpline_check_test(name = "tile_check", binary = ":my_kernel", declare = "declarations.json", arch = "sm_90a",
                        notices = "build.log", ledger = "ledger.json", profiles = ["details.csv", "source.csv"])

`notices` (a label of the saved build messages), `ledger`, `profiles` (a list of labels) and `notes` are optional and
are the command's options of the same names.

`binary` is a label of one compiled kernel (a cc target's library, a fatbin, a cubin) and the test needs the toolkit's
tools. `fixture = "tile"` instead checks saved tool outputs (`tile.sass`, `tile.cfg.dot`, `tile.res`): the example, and
what `bazel test //...` runs with no GPU and no toolkit installed. The example kernels themselves are compiled by that run with
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
  its own ledger rows and its notes, and has a note for each rule name (`declare.rule_names()` lists them) checked by a lint;
- puts a `warpline_check_test` on each kernel it ships (a cc target, a fatbin or a cubin, with the toolkit's tools on PATH);
- keeps what is its own convention out of the module: a comment marker, a registry of kernels, a build policy such as
  "cubins only", a named helper, a clock scheme. A rule here is true of any CUDA kernel on this toolchain; a check that is
  true of one project's conventions is that project's lint, written over the same model (`runner.kernels_of`).
