# The kernel model

`model.Kernel` is what every rule reads: one plain, immutable structure built by `model.kernels_from_text` (or
`runner.kernels_of` from a binary) out of the readers' data. The graph's instruction offsets must be the listing's
addresses.

| field | holds |
|---|---|
| `name`, `demangled` | the mangled name, and `c++filt`'s (a declaration's regex is over this one) |
| `instructions` | every `Instruction(address, predicate, opcode, operands, block)`; `.base` is the opcode without modifiers |
| `blocks` | `Block(id, start, end, instructions, successors, loop)`: a basic block of the control-flow graph; `loop` is the header of the innermost natural loop holding it, or None |
| `loops` | `Loop(header, address, blocks, tails, spin)`: the natural loops |
| `steps` | `TensorStep(instruction, commit)`: each `HGMMA`/`IGMMA`/`QGMMA`/`BGMMA`; `commit` is the `gsb0` flag of a group's last |
| `waits` | `TensorWait(instruction, left)`: each `WARPGROUP.DEPBAR.LE gsb0, N`; `left` groups may stay in flight |
| `arrives` | each `WARPGROUP.ARRIVE` |
| `accesses` | `MemoryAccess(instruction, space, kind, width)`: space shared, global, local or generic; kind load, store, atomic or async (`LDGSTS`, the bulk and tensor-map copies); width in bits |
| `barriers` | `BAR`, `SYNCS` (the mbarrier instructions) and `MEMBAR` instructions |
| `convergence` | `BSSY`, `BSYNC`, `BREAK`, `WARPSYNC` and `BRA.DIV` |
| `registers`, `stack_bytes`, `shared_bytes` | from the resource report |

`kernel.loop_of(instruction)` is the innermost natural loop around an instruction.

## Natural loops, not backward branches

A loop is a BACK EDGE of the graph: an edge whose target dominates its source (`loops.py`, dominators by Cooper,
Harvey and Kennedy). It is not "a branch to a lower address". On the example kernel (19 blocks, `examples/checker/steps.cu`,
compiled for `sm_90a`) 8 edges jump back in address; 6 are back edges, and the other 2 are the returns from the two
out-of-line try-wait stubs at the end of the function into its body, which close no loop. Of the 6 loops, 4 do work
(the counted loop, the loop around steps and their wait, the rolled loop around steps without their wait, and the long
loop the waits sit in) and 2 are SPINS: one block whose body is a barrier try-wait (`SYNCS.PHASECHK.TRANS64.TRYWAIT`),
the toolkit's sleep and the retry branch (`Loop.spin`). A count of "backward branches" would have found 8 loops.

## The example kernel and its saved outputs

The source is `examples/checker/steps.cu`: one `__global__` function holding, on purpose, a counted loop, a loop around
steps that holds their wait, a rolled loop around steps without it, two suspended `mbarrier.try_wait` waits (which the
toolkit emits as out-of-line stubs that jump back into the body), a tensor wait directly after its steps and one with
loads and stores between. Its toolkit outputs are saved in `tests/checker_fixtures/example/` (`kernel.sass`,
`kernel.cfg.dot`, `kernel.res`, and `kernel.names`, the `c++filt` of the function's name), made by

    nvcc -arch=sm_90a -cubin steps.cu -o steps.cubin
    cuobjdump -sass steps.cubin > kernel.sass ; nvdisasm -bbcfg -poff steps.cubin > kernel.cfg.dot
    cuobjdump -res-usage steps.cubin > kernel.res

and the build's own messages, `kernel.ptxas` (the same compile with `-Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage`,
standard error saved): what [the notice ledger](checker-notices.md) reads.

The tests read these files and need no toolkit. The stubs appear because the waits are inline PTX `try_wait`s with a
time hint (`0x989680`) in a loop of rounds: the toolkit moves the retry out of the body. The loops are compiled with
`unroll 1` so that each shape stays one loop.

Back edges to one header make one loop. The model's loops are found on the graph alone; `spin` is the only field that
reads an instruction.
