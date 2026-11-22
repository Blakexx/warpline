# Writing a rule

A rule is one module in `warpline/checker/rules/` that defines `RULE`, plus its fixtures. The registry
(`rules/__init__.py`) finds every module that defines `RULE`; nothing else is edited.

```python
from ..model import Kernel
from ..rule import Finding, Fixture, Rule

def check(kernel: Kernel) -> list[Finding]:
    ...                                   # read the model, never text

RULE = Rule(
    name="my_rule",                       # what a declaration names
    statement="one line: what it flags",
    check=check,
    refusal="what is wrong, in a sentence",
    fix="what to change",
    fixtures=(Fixture("fires", 2), Fixture("quiet", 0)),
)
```

A `Finding` is `(where, count, what)`: an address (or `kernel`), how many sites or units it stands for, and the
instance in a line. A rule's count on a kernel is the sum of its findings' counts; that is what a declaration allows.

## Fixtures

Each fixture is a directory `tests/checker/fixtures/rules/<rule>/<fixture>/` of a small kernel's tool texts, named as
the readers' fixtures are: `kernel.sass` (`cuobjdump -sass`), `kernel.cfg.dot` (`nvdisasm -bbcfg -poff`), `kernel.res`
(`cuobjdump -res-usage`) and `kernel.names` (`c++filt`). `Fixture(name, count)` is the count the rule must report: a
rule needs at least two fixtures, one where it fires and one where it must not (`Rule` refuses otherwise; a bound rule that measures every kernel, such as `function_size`, has two of different figures).
`tests/checker/test_rules.py` runs every registered rule on every fixture it declares and checks that no fixture
directory is undeclared, so writing the module and its fixtures adds the rule's tests.

## Step kinds and the rules that read them

A tensor step is `async` (Hopper's `HGMMA`, `IGMMA`, `QGMMA`, `BGMMA`: committed in groups and waited on) or `sync`
(`mma.sync`, SASS `HMMA`, `IMMA`, `DMMA`: issued by a warp from registers, with no commit group and no wait); see
[the model](checker-model.md#step-kinds). Each rule that reads steps reads the kinds its statement names:

| reads | rules | why |
|---|---|---|
| `async` only | `step_rolled`, `wait_after_issue`, `store_unfenced`, `step_port`, `step_serialized`, `accumulator_moved`, `wait_all_with_work` | they are about commit groups and waits (`step_rolled`, `wait_after_issue`, `step_serialized`, `wait_all_with_work`), the async proxy's view of shared memory (`store_unfenced`), the descriptors a warpgroup reads through the shared port, 4 warps (`step_port`), and the accumulator registers an async step names as its destination (`accumulator_moved`): a sync step has none of these |
| both | `step_branches`, `step_predicated`, `step_conversion`, `step_global`, `step_gap`, `step_work`, `step_density` | they are about what surrounds a step in the thread's instruction stream (a branch, a predicate, the conversion unit, a global access, the run between steps, the work and the instructions per step), which an `HMMA` suffers as an `HGMMA` does |

A rule's fixtures under `tests/checker/fixtures/rules/` are async kernels; the sync kinds are shown on the toolkit's own output,
`examples/checker/mma.cu` (`sm_80`, `tests/checker/fixtures/mma/`, read by `tests/checker/test_mma.py`).

## The stage-1 rules

| rule | flags | count | fixtures |
|---|---|---|---|
| `step_rolled` | a tensor step in a natural loop (the innermost around it) with no tensor wait in the loop's blocks: such steps run at about half a straight block's rate | the loop's steps, a finding a loop | `rolled` 4, `nested` 2; quiet: `straight`, `outer_wait` |
| `wait_after_issue` | a `WARPGROUP.DEPBAR.LE gsb0, 0x0` straight after its group's last step with no work of substance (memory access, copy, barrier arrival) between, within one basic block | 1 a wait | `bare_wait` 1; quiet: `work_between`, `partial_wait` |
| `stack_memory` | any stack frame in the resource report | the stack bytes; the detail names the `LDL`/`STL` instructions | `spill` 128; quiet: `no_stack` |

## The stage-3 rules

Static rules over the model, one module each, each with fixtures where it fires and where it must not. Two kinds: a COUNT
rule (undeclared means zero, as in the table above) and a BOUND rule (`Rule(bound=True)`), whose count is a figure the rule
reports (bytes, instructions): with no declaration the figure is printed (`figures of a bound rule, with no bound
declared`) and never refused; a declaration's `allow` is the project's bound, and a figure above it is refused.

| rule | flags | count | fixtures |
|---|---|---|---|
| `loop_convergence` | `BSSY`, `BSYNC` or `BRA.DIV` inside a natural loop | 1 an instruction | `in_loop` 2; quiet: `outside_loop`, `stub_return` |
| `load_in_trip` | a global load whose value an instruction reads that the same trip of an innermost loop reaches from it: forward over the loop's blocks a converged warp executes, never across the back edge (a `.64`/`.128` load writes a run of registers; a write of one ends that path). A read that precedes the load (the next trip's) and a read only in a cold fallback block (`BRA.DIV`'s, past a `BRA.CONV`, under `@!PT`) are not counted, and are not reported either: a finding carries no notes | 1 a load | `read_in_trip` 1, `wide_pair` 1, `cold_and_hot_reader` 1; quiet: `a_stage_ahead`, `outside_loop`, `cold_reader` |
| `store_reload` | a shared load of the operand a store of the same trip wrote, with no write of the address registers between; "the same trip" is reachability from the store inside the loop without the back edge, over blocks a converged warp executes (as `load_in_trip`) | 1 a store | `reload` 1; quiet: `other_address`, `base_rewritten`, `outside_loop`, `cold_reload` |
| `store_unfenced` | in a kernel with tensor steps, a barrier arrival (`SYNCS.ARRIVE`) that a shared store reaches with no `FENCE.VIEW.ASYNC` on the path (a forward flow over the graph: the next trip of a loop counts) | 1 an arrival | `arrive_after_store` 1, `loop_carried` 1; quiet: `fenced`, `no_steps` |
| `clock_uniform` | a cycle-counter reading moved into a uniform register (`S2UR`) and summed there (`UIADD3`), until the register is next written | 1 a sum | `summed` 1; quiet: `overwritten`, `vector_sum` |
| `convergence_call` | a `CALL` into a convergence subroutine (a body of 8 instructions or fewer that syncs the warp, runs one collective and returns) in a block a converged warp reaches; a call behind a `BRA.CONV`, in a `BRA.DIV` fallback or in dead code is cold, and a call into a longer slow path is no subroutine | 1 a call | `hot_call` 1; quiet: `guarded_conv`, `guarded_div`, `slow_path` |
| `step_predicated` | a tensor step under a predicate (`@P0 HGMMA`), not `@!PT` dead code | 1 a step | `predicated` 1; quiet: `plain`, `dead` |
| `step_branches` | a `BSSY` or a forward conditional `BRA` inside an innermost loop that holds steps; not the conditional back edge, the `BRA.DIV`/`BRA.CONV` guards, or the exit laid out as the loop's latch (a forward exit after the last step followed only by register copies and the unconditional back edge) | 1 a branch | `in_step` 2; quiet: `latch_exit`, `straight_body` |
| `step_conversion` | `F2I`, `I2F`, `F2F` or `MUFU` inside an innermost loop that holds steps | 1 an instruction | `in_loop` 2; quiet: `outside_loop`, `fma_form` |
| `step_global` | a global access (`LDG`, `STG`, `RED`, `ATOMG`, a bulk copy out) inside an innermost loop that holds steps, other than a staging copy in (`LDGSTS`, `UBLKCP.S.G`, `UTMALDG`) | 1 an instruction | `in_loop` 2; quiet: `staged`, `outside_loop` |
| `function_size` (bound) | the instructions of a function as the model holds them: the listing up to the graph's last instruction and the bodies the graph does not hold, padding (the self-branch and `NOP`s the toolkit appends) excluded; `@!PT` code stays, it is code the function has | instructions | `small` 4, `padded` 4, `large` 11 (a bound rule that measures every kernel has no quiet fixture: quiet is a kernel under its declared bound) |
| `step_port` (bound) | the bytes a step puts through the shared-memory port: its operands (A is m x k and B is k x n at the input type's size; B alone where A is in registers) and its region's shared loads and stores (4 warps), over the largest region | bytes a step | `operands_only` 2,560, `loop_with_loads` 5,120, `a_in_registers` 512; quiet: `no_steps` |
| `step_work` (bound) | the instructions a thread executes for a step in its region, each class (`Instruction.klass`) rounded up a step and summed; the finding names the classes | instructions a step | `loop` 5, `unrolled_pair` 4; quiet: `no_steps` |
| `step_density` (bound) | the kernel's static instructions (dead ones left out) over its tensor steps (the empty group is no step), rounded up | instructions a step | `sparse` 12, `dense` 4; quiet: `no_steps` |
| `step_gap` (bound) | the longest run of instructions without a step in an innermost loop that holds steps: between two steps of a trip, or around the back edge | instructions | `gapped` 5, `dense_steps` 4; quiet: `no_steps` |

A STEP REGION (`kernel.regions`, steps of either kind) is what a step's cost is read over: an innermost natural loop that holds steps (all the
instructions of its blocks that a converged warp reaches; a loop inside it that holds no step is part of its body), or, for
a step outside such a loop, the instructions of its own basic block after the previous step through it. The loops of
the rules above are natural loops, never the span a backward branch closes: a `try_wait` stub that returns into the body
closes no loop, and a block laid out inside a loop's address range but outside the loop is not in it.

Each of the stage-3 rules is also shown on the toolkit's own output: `examples/checker/stages.cu` (one small kernel a shape,
`sm_90a`) and `examples/checker/collectives.cu` (`sm_80`), saved in `tests/checker/fixtures/stages/` and
`tests/checker/fixtures/collectives/`, and read by `tests/checker/test_stage3.py` ([the model](checker-model.md#the-example-kernels-and-their-saved-outputs)).
Where the toolkit (13.5) did not produce a shape that fires a rule, the example shows the rule quiet on the real output and the
fixtures under `tests/checker/fixtures/rules/` show it firing: a hot `convergence_call` (every outlined collective the
toolkit emitted was guarded), a predicated step (the toolkit branches around a guarded step) and a `clock_uniform` sum (it
reads the counter into a uniform register for a warp reduction, and sums it in the vector datapath).

"The same trip" and "between" in the rules over a loop's trip: `load_in_trip` and `store_reload` read control flow (reachable from
the instruction inside the loop, not across the back edge, cold blocks not entered). `store_unfenced` is a forward flow over the
whole graph, a correctness property over every path, a cold one included. `wait_after_issue` scans one basic block backwards, no
address order across blocks. The `step_*` rules read `kernel.regions`, which leave cold blocks out. `step_rolled` asks only whether
the loop's blocks hold a wait (membership, no order). On the toolkit's output for a stage-ahead load in a loop with a guarded
collective (13.5, `sm_80`/`sm_90a`, small kernels) the fallbacks the toolkit emitted held no read of the loaded register, so the
cold-reader shape is shown by the synthetic fixtures `cold_reader`, `cold_and_hot_reader` and `cold_reload` only.

## The step-pattern rules

Three rules over the async steps (`HGMMA`, `IGMMA`, `QGMMA`, `BGMMA`) of an `sm_90a` kernel, one module each, read from the
listing's `WARPGROUP.ARRIVE`, steps and `WARPGROUP.DEPBAR.LE gsb0, N` waits. They share `rules/_accumulators.py`: the registers a
step accumulates into (the first operand, N/2 consecutive registers for an `m64nNk16` step with a 32-bit accumulator, N/4 with a
16-bit one; none for the null group or a step whose opcode names no shape or accumulator type) and the registers an instruction
touches (the model's reads and writes, every operand widened by the instruction's width: 4 for `.128`, 2 for `.64`, `.WIDE`
and the double-precision operations, so a wide store or a double is not read as touching one register).

| rule | flags | count | fixtures |
|---|---|---|---|
| `step_serialized` | a run of 4 steps or more (same innermost loop, in address order, no unconditional branch, `EXIT` or `RET` between) in which at most one step in eight lacks its own `DEPBAR` 0 before the next step: the toolkit serialized the run, with or without its `C7511`/`C7514`/`C7520` notice. The finding names the run's length and its waits' addresses | 1 a run | `serialized` 2, `mixed_shapes` 1; quiet: `fenced` |
| `accumulator_moved` | an instruction that is not a step that writes a register of a step's destination between two steps of the chain (a step wrote it before, and a step reads it as its accumulator input after, with no other write between: a forward and a backward flow over the graph, the back edge included; a step that starts a fresh accumulation, `RZ, !UPT` as its input, reads none, so registers reused for an epilogue's packing are no move) or between a step and its wait (any wait ends that), and that reads a register of some step's destination: a copy or an update of an accumulator. A write that reads none (a zeroing) or whose registers are no step's destination is not counted | 1 an instruction; one finding a basic block that holds them | `mixed_shapes` 64; quiet: `serialized`, `fenced` |
| `wait_all_with_work` | a `DEPBAR` 0 after which no instruction up to the next step reads or writes a register of the newest group (the steps back to the previous commit or wait), on any path, while a path holds work of substance (as `wait_after_issue`: loads, stores, atomics, copies, barriers, arrivals): a wait for fewer groups would let that work overlap | 1 a wait | `wait_all_work` 1; quiet: `wait_all_reads`, `serialized` |

The fixtures are the toolkit's own output (`examples/checker/serialized.cu`, `fenced.cu`, `mixed_shapes.cu`, `wait_all_work.cu`,
`wait_all_reads.cu`, 13.5, `sm_90a`), saved under `tests/checker/fixtures/rules/<rule>/<kernel>/` and compared with a rebuild by
`//:fixtures_regenerated`:

- `serialized.cu`: eight groups of eight loop-carried accumulator registers flow into two blocks of eight steps behind a
  branch each thread decides for itself, the `wgmma.fence` before the branch. ptxas reports `C7519` (`C7520` under 13.0) and issues a wait for every
  group after each step (16 steps, 16 waits). `fenced.cu` puts the `wgmma.fence` inside each block and an operand fence (an empty
  `asm volatile("" : "+f"(d[j][i]) ...)` over every accumulator register) before the branch and after each block's wait: one wait a
  block. On 13.0 the operand fence alone does not stop the serialization (tried: it still serializes; not tried again under 13.5); the `wgmma.fence` in the
  block does. A branch every thread of the warpgroup takes alike (`flags[i]`) did not serialize in any variant tried.
- `mixed_shapes.cu`: thirty-two accumulator registers; one block runs four `m64n16k16` steps into groups of eight, the other one
  `m64n64k16` into all thirty-two, the first block naming each pair of its group's registers swapped. The toolkit copies the
  accumulators at the switch (64 instructions in the block that issues the `m64n64k16`, `C7511`). With the groups named in order
  it emits no copy, so the rule stays quiet there.
- `wait_all_work.cu`, `wait_all_reads.cu`: four steps, a wait for every group, a barrier and shared loads and stores, four more
  steps. In the first the work is of unrelated words; in the second it adds the first accumulator. Global loads and stores after
  the wait are hoisted above it by ptxas, even `volatile` ones, so the fixtures use shared-memory work behind a barrier, which the
  toolkit leaves where it is.

What these rules cannot decide, and so say nothing about:

- `step_serialized` reads the listing in address order. It does not know which waits the source wrote: a kernel that waits
  after every step on purpose (a reduction that needs each result) is a run too, and is declared with its reason.
- `accumulator_moved` cannot tell the toolkit's copy from the program's own update (`d[0] += f` between steps is `FADD R24, R24, R3`:
  `stages.cu`'s `loop_join` shows it, 1), and sees a register only by name: a step whose destination it cannot read (no shape
  or accumulator type) starts no chain. The "between a step and its wait" view ends at any wait, a partial one included, so an
  update of registers a partial wait does not yet cover is missed there (it is still reported when a later step reaches the
  register).
- `wait_all_with_work` does not decide a wait whose group it cannot read (no step behind it before a join of several predecessors, or a
  step with no shape or accumulator type), a path that reaches a `CALL`, `JMP`, `BRX` or `RET` (the callee or target is unseen), and
  any wait on whose path an instruction touches the group, including a wide access (conservatively widened) and the path out of a
  loop that stores the accumulators: that wait is needed, so it is quiet. It does not know whether the work could have been
  moved by the toolkit; it asks only whether a wait for fewer groups would have let it overlap.

## Rules over a profile and over a build's notices

A rule that needs a profiler's counters is a `ProfileRule` in `PROFILE_RULE`, with fixtures of the profiler's CSV
exports: [rules backed by the profiler](checker-profile.md). The toolkit's build messages are not rules but rows of a
ledger: [the notice ledger](checker-notices.md). Both are judged by the same declarations.
