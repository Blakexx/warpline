"""THE BASE LEDGER: the toolkit's notice templates this module knows, each counted under a rule with a default allowance
(0, or null for a notice accepted by default) a project may override in its own ledger file (`notices.py`), and a
declaration may override for a kernel. The reasons and fixes are about the hardware and the toolchain; a
project's own notes on a declared rule belong in its declarations file (`checker-declarations.md`).

The rules counted here, so a project declares them like a rule of the model, are the rows' `rule` fields; the two a
step's serialization is counted under are:
- `mma_serialized`: the assembler serialized asynchronous warpgroup matrix multiplies (the tensor pipe's steps wait
  for one another instead of overlapping);
- `wait_injected`: the assembler injected a warpgroup wait of its own into the step stream.
"""
SERIALIZED = "Potential Performance Loss: wgmma.mma_async instructions are serialized due to "

#: (code, the cause as the toolkit words it, reason, fix) of each serialization notice
_SERIALIZED = (
    ("C7511", "insufficient register resources for the function '{}'",
     "The function's register count cannot hold the operand sets of its in-flight steps, so the assembler made each "
     "asynchronous step wait for the one before it: the tensor pipe idles between steps.",
     "raise the register count the function may use (fewer live values, `__launch_bounds__`, a register reallocation "
     "at entry), or issue fewer steps per group"),
    ("C7512", "insufficient register resources for the wgmma pipeline in the function '{}'",
     "The registers available to the code between a step's issue and its wait cannot hold what the pipeline needs, so "
     "the assembler made the steps wait for one another.",
     "write a step's operand registers before its commit group, and give the issuing warps the register count the "
     "pipeline needs"),
    ("C7514", "non wgmma instructions reading accumulator registers of a wgmma between start and end of the pipeline "
              "stage in the function '{}'",
     "Other instructions read a step's accumulators between its issue and its wait; the hardware cannot let them "
     "overtake the step, so the assembler serialized the steps.",
     "read accumulators only after the group's wait, and keep non-step code that touches them out of the stage"),
    ("C7520", "program dependence on compiler-inserted WG.AR in divergent path in the function '{}'",
     "A warpgroup arrive the assembler inserted sits on a divergent path, which the steps depend on, so the assembler "
     "serialized them.",
     "issue the arrive and the steps on a path every thread of the warpgroup takes together (no divergence between "
     "the arrive and the commit)"),
    (None, "program dependence on compiler-inserted WG.DP in divergent path in the function '{}'",
     "A warpgroup dependence wait the assembler inserted sits on a divergent path the steps depend on, so the "
     "assembler serialized them.",
     "place the wait where the whole warpgroup converges, outside any divergent branch"),
    (None, "non wgmma instructions defining accumulator registers of a wgmma between start and end of the pipeline "
           "stage in the function '{}'",
     "Other instructions write a step's accumulators inside its stage, so the assembler serialized the steps.",
     "initialize accumulators before the stage opens (or use the step's own zeroing), never inside it"),
    (None, "non wgmma instructions defining input registers of a wgmma between start and end of the pipeline stage in "
           "the function '{}'",
     "Other instructions write a step's register operands inside its stage, so the assembler serialized the steps.",
     "finish writing a step's register operands before the stage opens"),
    (None, "ill formed pipeline stage in the function '{}'",
     "The assembler could not find a fence, steps, a commit and a wait that form one stage, so it serialized the steps.",
     "keep each stage in one straight piece: fence, steps, commit, then the wait, with no branch out of it"),
    (None, "the presence of Extern calls in the function '{}'",
     "A call to a function the assembler cannot see sits in the pipeline, so it serialized the steps.",
     "inline the callee or move the call out of the stage"),
    (None, "wgmma pipeline crossing function boundary at a function call in the function '{}'",
     "A stage opens before a call and closes after it, so the assembler serialized the steps.",
     "close the stage before the call, or inline the callee"),
)

ROWS = [
    *({"tool": "ptxas", "code": code, "text": SERIALIZED + cause, "rule": "mma_serialized",
       "reason": reason, "fix": fix} for code, cause, reason, fix in _SERIALIZED),
    {"tool": "ptxas", "code": "C7517", "rule": "wait_injected",
     "text": "warpgroup.wait is injected in around line {} by compiler to allow use of registers defined by GMMA in "
             "function '{}'",
     "reason": "Code reads registers a step defines before the step's group has been waited for, so the assembler put a "
               "wait into the step stream: the issuing warp stalls until the group completes. The line is a line of "
               "the PTX, not of the source.",
     "fix": "read a step's result registers only after the wait you wrote for its group"},
    {"tool": "ptxas", "code": "C7519", "rule": "wgmma_arrive_injected", "allow": None,
     "text": "warpgroup.arrive is injected in around line {} by compiler to allow use of registers in GMMA in "
             "function '{}'",
     "reason": "The assembler added a warpgroup arrive (the fence before a step that reads registers); it costs a few "
               "cycles at the step and changes no result.",
     "fix": "write the fence yourself where the registers settle, if the cycles matter"},
    {"tool": "ptxas", "code": "C7505", "rule": "setmaxnreg_ignored",
     "text": "Potential Performance Loss: 'setmaxnreg' ignored to allow debugging.",
     "reason": "A debug build ignores the register reallocation between warp roles, so every role runs at the launch's "
               "register count and the measured code is not the code that ships.",
     "fix": "build without device debugging (no `-G`)"},
    {"tool": "ptxas", "code": None, "rule": "setmaxnreg_ignored",
     "text": "Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility across compilation units.",
     "reason": "The register reallocation between warp roles was dropped because another compilation unit may call "
               "this code, so every role runs at the launch's register count.",
     "fix": "compile the kernel and everything it calls as one unit (whole-program, no separate compilation)"},
    {"tool": "ptxas", "code": None, "rule": "setmaxnreg_ignored",
     "text": "Potential Performance Loss: 'setmaxnreg' ignored; unable to determine register count at entry.",
     "reason": "The assembler could not tell how many registers the function starts with, so it dropped the register "
               "reallocation between warp roles.",
     "fix": "state the register count at entry (`__launch_bounds__` with a minimum block count, or `maxnreg`)"},
    {"tool": "ptxas", "code": None, "rule": "setmaxnreg_ignored",
     "text": "Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility into 'extern' call.",
     "reason": "A call to an external function made the assembler drop the register reallocation between warp roles.",
     "fix": "inline the callee or remove the external call from the kernel"},
    {"tool": "ptxas", "code": "C7507", "rule": "setmaxnreg_ignored",
     "text": "Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum register requirements.",
     "reason": "The register count a role asked for is below what its code needs, so the assembler dropped the request "
               "and the role runs at the launch's register count.",
     "fix": "ask for at least the registers the role's code needs, and give the other roles what is left"},
    {"tool": "nvcc", "code": "#549-D", "rule": "variable_unset", "text": 'variable "{}" is used before its value is set',
     "reason": "A read of a variable no write reaches: the value is whatever the register held.",
     "fix": "initialize the variable on every path that reaches the read"},
    {"tool": "nvcc", "code": "#128-D", "rule": "unreachable_loop", "allow": None, "text": "loop is not reachable",
     "reason": "The front end proved a loop cannot run; its code is dropped, so a count or a time taken over it "
               "measures nothing.",
     "fix": "remove the loop, or fix the condition that makes it unreachable"},
    {"tool": "nvcc", "code": "#177-D", "rule": "unused_variable", "allow": None, "text": 'variable "{}" was declared but never referenced',
     "reason": "An unused variable emits no instruction.", "fix": ""},
    {"tool": "nvcc", "code": None, "rule": "build_info", "allow": None,
     "text": 'The warnings can be suppressed with "-diag-suppress <warning-number>"',
     "reason": "The front end's own hint that follows a warning; the warning is judged by its own row.", "fix": ""},
    {"tool": "ptxas", "code": None, "rule": "register_spill",
     "text": "Registers are spilled to local memory in function '{}', {} bytes spill stores, {} bytes spill loads",
     "reason": "Live values did not fit the registers and were stored to local memory (a stack frame in device "
               "memory behind the L1 cache): every reload is a memory access in the middle of the compute.",
     "fix": "reduce live values (a smaller tile, fewer accumulators in flight), or raise the register limit"},
    {"tool": "ptxas", "code": None, "rule": "local_memory_used",
     "text": "Local memory used for function '{}', size of stack frame: {} bytes",
     "reason": "The function has a stack frame in local memory: a spill, or an array indexed by a runtime value; "
               "its accesses are far slower than registers.",
     "fix": "index arrays with compile-time constants so they live in registers, or accept the frame by declaring "
            "the `stack_memory` rule for the kernel"},
    {"tool": "ptxas", "code": None, "rule": "build_info", "allow": None, "text": "Compiling entry function '{}' for '{}'",
     "reason": "Names the function and architecture the lines after it describe.", "fix": ""},
    {"tool": "ptxas", "code": None, "rule": "build_info", "allow": None, "text": "Function properties for {}",
     "reason": "Heads the stack, spill-store and spill-load counts of a function (indented lines the notice owns); "
               "a nonzero count is refused by the spill and local-memory rows.", "fix": ""},
    {"tool": "ptxas", "code": None, "rule": "build_info", "allow": None, "text": "Used {} registers{}",
     "reason": "The function's register, barrier and shared-memory counts; the resource report carries them.", "fix": ""},
    {"tool": "ptxas", "code": None, "rule": "build_info", "allow": None, "text": "Compile time = {} ms",
     "reason": "The assembler's own run time for the function.", "fix": ""},
    {"tool": "ptxas", "code": None, "rule": "build_info", "allow": None, "text": "{} bytes gmem",
     "reason": "The module's static global memory; a size, not a hazard.", "fix": ""},
]
