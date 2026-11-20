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
   on a fixture; `runner.py` is the only module that starts a process.
2. THE KERNEL MODEL (`model.py`, `loops.py`): one plain structure a kernel. [The model](checker-model.md).
3. RULES (`rule.py`, `rules/`): a small unit each; the registry lists them. [Writing a rule](checker-rules.md).
4. DECLARATIONS AND THE COMMAND (`declare.py`, `cli.py`, `checker.bzl`): a project says how many findings of a rule a
   kernel may have and why; the rest are refusals. [Declaring](checker-declarations.md).

Rules read the model, never text. Thresholds and reasons are the project's data, never the module's.

## The command

    python -m warpline check <binary> --declare <declarations.json> [--arch sm_90a] [--toolkit DIR]
    python -m warpline check <stem> --fixture --declare <declarations.json>     # saved tool outputs, no toolkit

`<binary>` is an executable, a shared library, a fatbin or a cubin; give `--arch` for a fat binary of several
architectures. The toolkit is `--toolkit`, then `CUDA_HOME`, then `PATH`. `--fixture` reads `<stem>.sass`,
`<stem>.cfg.dot`, `<stem>.res` and, optionally, `<stem>.names` (`c++filt`'s answer, a line a function). The verb is the
recorder's `check`: with `--declare` or `--fixture` it is the checker, without it the argument is a record.

A refusal prints the kernel, the rule, the count found and declared (and the declared reason), what the rule says, each
finding with where it is, and the fix. Exit 0: nothing refused; 1: a refusal; 2: a tool, text or declarations file that
is not usable.

## As a Bazel test

`checker.bzl` exports `warpline_check_test`; `examples/checker/` is a complete target:

    load("@warpline//:checker.bzl", "warpline_check_test")
    warpline_check_test(name = "tile_check", binary = ":my_kernel", declare = "declarations.json", arch = "sm_90a")

`binary` is a label of one compiled kernel (a cc target's library, a fatbin, a cubin) and the test needs the toolkit's
tools. `fixture = "tile"` instead checks saved tool outputs (`tile.sass`, `tile.cfg.dot`, `tile.res`): the example, and
what `bazel test //...` runs with no GPU and no toolkit.

## What the readers read from the real tools (facts the code relies on)

- `cuobjdump -res-usage` states the stack frame as `STACK`; its `LOCAL` is 0 even where local memory exists.
- `nvdisasm -bbcfg -poff` marks no edge kind (every edge is `solid`); a function's own node is its entry. The listing
  from `cuobjdump -sass` also holds padding after the last instruction (a self-branch and `NOP`s) that the graph does
  not; the model drops instructions after the graph's last offset, and refuses any other disagreement.
- A tensor step's commit is the `gsb0` flag on a group's last `HGMMA`/`IGMMA`; the wait is `WARPGROUP.DEPBAR.LE gsb0, N`.

## The module's next stages (not built)

- the toolkit's build messages (ptxas notices) as a reader and a ledger of them, so a notice a project accepted is
  declared once and a new one is a refusal;
- rules backed by a profiler's counters (shared-memory bank conflicts, stall shares);
- the other hazards of a hand-tuned Hopper kernel, as rules over the model.
