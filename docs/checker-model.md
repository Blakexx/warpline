# The kernel model

`model.Kernel` is what every rule reads: one plain, immutable structure built by `model.kernels_from_text` (or
`runner.kernels_of` from a binary) out of the readers' data. The graph's instruction offsets must be the listing's
addresses.

| field | holds |
|---|---|
| `name`, `demangled` | the mangled name, and `c++filt`'s (a declaration's regex is over this one) |
| `instructions` | every `Instruction(address, predicate, opcode, operands, block)`; `.base` is the opcode without modifiers, `.dead` is `@!PT`, `.klass` the unit class (`INSTRUCTION_CLASSES`: tensor, shared load, global, conversion unit, ...), `.writes` and `.reads` the registers (`R5`, uniform `UR5`; a `.64` or `.128` result is a run), `.address_operand` and `.address_registers`, `.target` the address a branch or call names |
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

`kernel.async_steps` is the steps of kind `async`.

`kernel.loop_of(instruction)` is the innermost natural loop around an instruction; `kernel.innermost()` the loops with
no loop inside them (with a predicate over an instruction, the loops that hold one and no inner loop that does).
`flow.entry_states(kernel, step)` and `flow.walk(kernel, step)` are a forward flow over the graph (a set that joins by union)
for a rule that asks what may hold at a point.

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
pinned toolkit (CUDA 13.5, `toolchain/cuda_toolkit_pin.json`, a dev dependency: a workspace that depends on warpline never
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
