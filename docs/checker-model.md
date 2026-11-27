# The kernel model

`model.Kernel` is what every rule reads: one plain, immutable structure built by `model.kernels_from_text` (or
`runner.kernels_of` from a binary) out of the readers' data. The graph's instruction offsets must be the listing's
addresses.

| field | holds |
|---|---|
| `name`, `demangled` | the mangled name, and `c++filt`'s (a declaration's regex is over this one) |
| `instructions` | every `Instruction(address, predicate, opcode, operands, block)`; `.base` is the opcode without modifiers, `.dead` is `@!PT`, `.klass` the unit class ([the opcode table](#the-opcode-table): tensor, shared load, global, conversion unit, ...), `.writes` and `.reads` the registers (`R5`, uniform `UR5`; a `.64` or `.128` result is a run), `.address_operand` and `.address_registers`, `.target` the address a branch or call names |
| `blocks` | `Block(id, start, end, instructions, successors, loop, cold)`: a basic block of the control-flow graph; `loop` is the header of the innermost natural loop holding it, or None; `cold` is true where no path from the entry reaches it but through a fallback a converged warp does not take (the target of a `BRA.DIV`, the fall-through of a `BRA.CONV`, a branch under `@!PT`) |
| `loops` | `Loop(header, address, blocks, tails, spin)`: the natural loops |
| `steps` | `TensorStep(instruction, commit, shape, operand_bytes, null, kind)`: each tensor instruction, of one of two KINDS ([below](#step-kinds)): `async` (`HGMMA`/`IGMMA`/`QGMMA`/`BGMMA`) or `sync` (`HMMA`/`IMMA`/`DMMA`, `mma.sync`); `commit` is the `gsb0` flag of an async group's last (false for a sync step); `shape` is (m, n, k) from the opcode (`HGMMA.64x16x16...`; for a sync step its digits, `HMMA.16816` is 16, 8, 16); `operand_bytes` what the step reads through its descriptors (A is m x k and B is k x n elements at the input type's size, or B alone where A is in registers; None for a form this does not read; 0 for a sync step, which reads registers); `null` the empty group (`RZ, gdesc[URZ]`) the toolkit issues to hold a wait |
| `waits` | `TensorWait(instruction, left)`: each `WARPGROUP.DEPBAR.LE gsb0, N`; `left` groups may stay in flight (only async steps are waited on) |
| `arrives` | each `WARPGROUP.ARRIVE` |
| `accesses` | `MemoryAccess(instruction, space, kind, width)`: space shared, global, local or generic; kind load, store, atomic or async (`LDGSTS`, the bulk and tensor-map copies); width in bits; `.warp_bytes` the bytes a warp moves (32 lanes of the width; 128 a matrix of `LDSM`/`STSM`) |
| `barriers` | `BAR`, `SYNCS` (the mbarrier instructions) and `MEMBAR` instructions |
| `fences` | `FENCE` instructions (`FENCE.VIEW.ASYNC.S`, the async-proxy fence) |
| `convergence` | `BSSY`, `BSYNC`, `BREAK`, `WARPSYNC` and `BRA.DIV` |
| `calls` | `Call(instruction, target, subroutine, hot)`: each `CALL`; `subroutine` whether its target is a convergence subroutine (a body of 8 instructions or fewer, up to its return, with a `WARPSYNC` and a collective); `hot` whether a converged warp executes it (its block is not cold) |
| `regions` | `StepRegion(kind, address, steps, instructions, accesses, tensor)`: what a step's cost is read over (a step of either kind), an innermost natural loop that holds steps (`loop`) or one step outside such a loop with the instructions before it in its basic block (`unrolled`); the dead instructions and the cold blocks left out, the null step not counted |
| `registers`, `stack_bytes`, `shared_bytes` | from the resource report |
| `lines` | instruction address -> `(source file, line)`, from the disassembly with source-line marks (`--lines`, [below](#source-lines)); empty where none was given, and not part of what makes two kernels equal |
| `marks` | instruction address -> `Marks(reads, writes)`, the registers it reads and writes by the life-range listing ([below](#register-pressure)); empty where none was given (then `kernel.registers_of(instruction)` is its operands'), and not part of what makes two kernels equal |

`kernel.async_steps` is the steps of kind `async`.

`kernel.loop_of(instruction)` is the innermost natural loop around an instruction; `kernel.innermost()` the loops with
no loop inside them (with a predicate over an instruction, the loops that hold one and no inner loop that does).
`flow.entry_states(kernel, step)` and `flow.walk(kernel, step)` are a forward flow over the graph (a set that joins by union)
for a rule that asks what may hold at a point; `flow.exit_states(kernel, step)` and `flow.walk_backward(kernel, step)` the same
run backward, for one that asks what may still come. `kernel.live_in()` is [register liveness](#register-liveness).

## Register liveness

`kernel.live_in()` (`flow.live_in`) is, for each block id, the registers live at the block's entry: a register is live where
some path from there reads it before writing it. It is `flow.exit_states` (the backward flow, a set that joins by union, to a
fixed point over the graph's edges) with the step `flow.liveness(kernel)` (`flow.live_step` over `kernel.registers_of`), then
each block's own instructions from its end: it fits `walk_backward`'s shape without change. The step reads the instruction's
reads and kills its writes (its life-range marks where the kernel has them, else `Instruction.reads` and `.writes`); an instruction
under a predicate (`@P0 IADD3 R4, ...`) keeps the old value where the predicate is false, so it kills nothing and also
reads what it writes; code under `@!PT` changes nothing. Registers are named as in the model, vector `R5` and uniform `UR5`
(`RZ` and `URZ` are none).

What this needs of the operands (checked on the toolkit's output): the destination comes first, behind the predicates some
opcodes write (`SHFL.IDX PT, R3, R2, ...`, `LOP3.LUT P0, R6, ...`); a comparison (`ISETP`, `FSETP`, `PLOP3`) writes
predicates alone; a `.64`, `.WIDE` or double-precision result is two registers, `.128` four; the data of a `.64` or
`.128` store or atomic, an `IMAD.WIDE`'s addend and every operand of a double-precision operation are runs too; an atomic
returns a value (a discarded one is `RZ`). Not modelled by the operands: a tensor instruction's fragment registers beyond the
first of a run, a conversion's operand width (`F2F.F64.F32`), and what a `CALL`'s callee reads and writes (the callee is not in
the graph, so a register only a convergence subroutine reads is not live at the call); the life-range marks
([register pressure](#register-pressure)) model all three. Cold blocks (a `BRA.DIV` fallback) are paths like any
other, so what only they read is live.

`held.holds(kernel)` (`held.py`) uses it: for each natural loop, the vector registers live at the header's entry, each with
the definition that last made it before the loop (a forward flow over the graph keeps every definition that reaches the
header; of those outside the loop the latest address is taken) and whether the loop reads or rewrites it. The rule
[`loop_held`](checker-rules.md#loop_held) and the verb `held` read it.

## Register pressure

`flow.pressure(kernel)` is, for each instruction, the registers HELD across it: those live into it or written by it that some
write reaches (a forward flow of the writes from the entry, or its own). A register no write has made is not held, so one
written and read under one predicate does not count from the kernel's entry; the count of its vector registers is the program
point's register pressure, what a budget of registers is read against (the allocation is not). The instructions of the outlined
block are not in it.

It is exact only over the LIFE-RANGE MARKS: `readers.life_ranges` reads `nvdisasm --print-life-ranges`, which marks under each
register's column what an instruction does to it (`^` written, `v` read, `x` both), for the vector, predicate and uniform
registers (`GPR`, `PRED`, `UGPR`, `UPRED`), widths included (`CS2R R12, SRZ` writes R12 and R13, which the operands do not say).
`kernels_from_text(..., life_text=)`, `kernels_of(binary, life=True)` and a fixture's `<stem>.life` give the kernel its `marks`.
The listing's own live count is not read, because its control flow is not the function's, and two marks are not what the code
does:

- every call is marked as writing the calling convention's clobbered registers, which a call never touches: a call to a
  SUBROUTINE (its target in the outlined code) reads and writes what its body does, from the target to its `RET`;
- a `CALL.REL.NOINC` to a LABEL of the function (how the toolkit leaves a loop through a convergence barrier) is a jump that
  never returns, and reads and writes nothing. The graph takes it for a call that returns, so the model makes it the jump it
  is (`model._label_calls`): the block ends at the call, its successors the label's block and, where the call is predicated,
  the instruction after it; the instructions after it in its block are a block of their own (`<id>.after`).

Both drop a value live across the call from the listing's count: a value held across a division's slow-path call
(`examples/checker/ranges.cu`) counts before the call (`tests/checker/test_ranges.py`), and one held across a loop's exit
through a call counts in the loop. A function at 255 registers is one nvdisasm prints no life ranges for; `runner.life_listing`
reads a copy whose `EIATTR_REGCOUNT` says 254 (`runner.listable`): the attribute only sizes the listing.

## The opcode table

`model.INSTRUCTION_CLASSES` is the one opcode table: each class one unit or one kind of overhead (tensor, shared load, shared
store, shared atomic, control, shuffle, global, pack, conversion unit, float, select/compare, integer/address), matched by opcode
prefix, the first match winning. `model.instruction_class(opcode)` is an opcode's class (`other` for none; a uniform-datapath
opcode, `UIADD3`, has its vector opcode's). It is public: a consumer that budgets instructions by class reads this table, and a
loop's `class/<name>` figures are its classes ([the manifest](checker-manifest.md#a-kernels-figures)).

## Source lines

`readers.source_lines` reads `nvdisasm --print-line-info` of a cubin built with `--generate-line-info` (the pinned toolkit's
spellings; `--print-line-info-inline` adds the `inlined at` marks, which the reader accepts): a function is a `.text.<name>`
section, a mark `//## File "f.cu", line 12` stands before the instructions it covers and is printed again only where the line
changes, and an instruction's place is the FIRST mark since the instruction before it, the innermost source line (an instruction
with no mark of its own keeps the place before it). The path is cut to what follows `.../execroot/<workspace>/`, so a listing
built in a Bazel sandbox names `examples/checker/held_many.cu`, the same on every machine; a path with no execution root is kept
whole. `readers.inline_frames` reads the same listing with every mark (`--print-line-info-inline`): each instruction's whole chain, the
innermost place first and each caller after it, what attributes an instruction to the function it was inlined from. The
listing is an optional input: `runner.kernels_of_files` reads `<stem>.lines` when it exists, `kernels_of(binary,
lines=path)` and `--lines` give another, and `examples/cuda.bzl`'s `kernel_outputs(lines = True)` compiles with
`--generate-line-info` and writes `<name>.lines` from the cubin. A binary is not asked for its listing by the checker itself:
the listing is the size of the whole disassembly (hundreds of megabytes for a large kernel), so it is given, not made.

The graph from `nvdisasm -bbcfg` does not hold the bodies the toolkit outlines past the function's last instruction (a
convergence subroutine a `CALL` reaches, laid out before the final self-branch and its padding). The model keeps them in a
block of its own (`model.OUTLINED`, cold, no edge into it) and drops the padding.

## Step kinds

A tensor step is ASYNC or SYNC, and `TensorStep.kind` says which.

- `async`: Hopper's warpgroup steps (`HGMMA`, `IGMMA`, `QGMMA`, `BGMMA`). They are issued and return, committed in groups
  (`commit`, the `gsb0` flag) and waited on (`WARPGROUP.DEPBAR`); they read shared memory through descriptors, so the model
  reads their `operand_bytes`.
- `sync`: `mma.sync` (`HMMA`, `IMMA`, `DMMA`, as on `sm_80` and `sm_86`). A warp issues it from registers; there is no commit
  group, no wait and no descriptor, so `commit` is false, `operand_bytes` is 0 and the step is never the empty group.

A rule that reads steps states which kinds it reads ([the rules](checker-rules.md#step-kinds-and-the-rules-that-read-them)):
the rules about commit groups, waits and descriptors read `async` only; those about what surrounds a step (its branches,
predicate, conversions, global accesses, the gap between steps, the work and the density) read both. `regions` cover both
kinds, a region's `steps` counting either.

## Natural loops, not backward branches

A loop is a BACK EDGE of the graph: an edge whose target dominates its source (`loops.py`, dominators by Cooper,
Harvey and Kennedy). It is not "a branch to a lower address". On the example kernel (24 blocks under toolkit 13.5, 19 under 13.0, `examples/checker/steps.cu`,
compiled for `sm_90a`) 8 edges jump back in address; 6 are back edges, and the other 2 are the returns from the two
out-of-line try-wait stubs at the end of the function into its body, which close no loop. Of the 6 loops, 4 do work
(the counted loop, the loop around steps and their wait, the rolled loop around steps without their wait, and the long
loop the waits sit in) and 2 are SPINS: a loop whose body is the address move, a barrier try-wait (`SYNCS.PHASECHK.TRANS64.TRYWAIT`, in a block of its own under 13.5, one block with the move under 13.0),
the toolkit's sleep and the retry branch (`Loop.spin`). A count of "backward branches" would have found 8 loops.

## The example kernel and its saved outputs

(Stage 1's example, `examples/checker/steps.cu`; the stage-3 examples are [below](#the-example-kernels-and-their-saved-outputs).)

The source is `examples/checker/steps.cu`: one `__global__` function holding, on purpose, a counted loop, a loop around
steps that holds their wait, a rolled loop around steps without it, two suspended `mbarrier.try_wait` waits (which the
toolkit emits as out-of-line stubs that jump back into the body), a tensor wait directly after its steps and one with
loads and stores between. Its toolkit outputs are saved in `tests/checker/fixtures/example/` (`kernel.sass`,
`kernel.cfg.dot`, `kernel.res`, and `kernel.names`, the `c++filt` of the function's name), made by

    nvcc -arch=sm_90a -cubin steps.cu -o steps.cubin
    cuobjdump -sass steps.cubin > kernel.sass ; nvdisasm -bbcfg -poff steps.cubin > kernel.cfg.dot
    cuobjdump -res-usage steps.cubin > kernel.res

and the build's own messages, `kernel.ptxas` (the same compile with `-Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage`,
standard error saved): what [the notice ledger](checker-notices.md) reads.

The four example sources (`steps.cu` here, `stages.cu`, `collectives.cu`, `mma.cu`) are Bazel targets, built with the repository's
pinned toolkit (CUDA 13.5, fsla-devtools' `pins/cuda_toolkit_pin.json`, a dev dependency: a workspace that depends on warpline never
fetches it): `bazel build //examples/checker:steps` writes `steps.sass`, `.cfg.dot`, `.res`, `.names` and `.ptxas`, the recipes
below. `bazel test //:fixtures_regenerated` compares them with the committed fixtures; the committed files are the contract, and a
regenerated one that differs fails (ptxas's compile times and the demangler's spacing aside; see the test). The commands by hand
are the same.

The tests read these files and need no toolkit. The stubs appear because the waits are inline PTX `try_wait`s with a
time hint (`0x989680`) in a loop of rounds: the toolkit moves the retry out of the body. The loops are compiled with
`unroll 1` so that each shape stays one loop.

Back edges to one header make one loop. The model's loops are found on the graph alone; `spin` is the only field that
reads an instruction.

## The example kernels and their saved outputs

Stage 3 adds two sources under `examples/checker/`, compiled with the same tools and saved as fixtures:

- `stages.cu`, `sm_90a`: one small kernel a shape (a divergent inner loop in a loop of steps, a global load read in its own
  trip, a shared store loaded back, a barrier arrival with and without the async-proxy fence, a conversion and global accesses in
  a step loop, a clock read, a guarded step, a divergent region after a suspended wait). Saved in `tests/checker/fixtures/stages/`:

      nvcc -arch=sm_90a -cubin stages.cu -o stages.cubin
      cuobjdump -sass stages.cubin > kernel.sass ; nvdisasm -bbcfg -poff stages.cubin > kernel.cfg.dot
      cuobjdump -res-usage stages.cubin > kernel.res
      cuobjdump -sass stages.cubin | grep "Function :" | awk '{print $3}' | c++filt > kernel.names
      nvcc -arch=sm_90a -cubin stages.cu -o stages.cubin -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage 2> kernel.ptxas

- `collectives.cu`, `sm_80`: warp collectives under role-dependent control flow; the toolkit outlines each into a call to a
  convergence subroutine and guards it. Saved in `tests/checker/fixtures/collectives/` the same way.

The fixtures hold several kernels in one text, the readers' functions; a test picks one by its demangled name.
`tests/checker/test_stage3.py` asserts each rule's count on each kernel. Facts the model relies on from them: the toolkit
emits the clock as `CS2R` into a vector register, except `S2UR` into a uniform register where the value feeds a warp
reduction; it branches around a guarded tensor step (no predicated step) and issues the empty group; every outlined
collective it emitted for `sm_80` was behind a `BRA.CONV` or in a `BRA.DIV` fallback; the graph does not hold the
subroutines' bodies.

A fourth source, `mma.cu` (`sm_80`), is the synchronous-step example: `mma.sync` in a loop of operands read from shared memory
(`mma_loop`), the same loop with a data-dependent branch, a conversion and global accesses (`mma_branch`), and a step the
source guards by lane (`mma_predicated`, which the toolkit if-converts to `@!P0 HMMA`; it also if-converts a small guarded body in a
loop, so the loop's branch is held by a `__nanosleep` the toolkit cannot predicate). Saved in `tests/checker/fixtures/mma/` the same
way (`nvcc -arch=sm_80 -cubin mma.cu`, then the outputs as for `stages.cu`); `tests/checker/test_mma.py` asserts each step rule on
it.

Five more single-kernel sources (`sm_90a`) are the fixtures of the step-pattern rules ([the rules page](checker-rules.md#the-step-pattern-rules)):
`serialized.cu`, `fenced.cu`, `mixed_shapes.cu`, `wait_all_work.cu`, `wait_all_reads.cu`. Each is built by `//examples/checker:<name>`
and its four outputs are saved under `tests/checker/fixtures/rules/<rule>/<name>/` for each rule that shows it; `//:fixtures_regenerated`
compares every saved copy with the rebuild.
