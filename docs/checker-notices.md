# The notice ledger

A build prints notices: the assembler's `info` and `warning` lines, the front end's warnings, bare performance lines.
Each is a fact about the code the toolkit made (a step it serialized, a wait it injected, a register request it
ignored). `warpline check <binary> --notices <build log>` reads the build's messages and judges each notice by a ROW of
a ledger. A notice with no row is REFUSED, quoted whole: a new notice is a decision, never a silence.

Save the build's messages to a file (`nvcc ... -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage 2> build.log`;
`-v` prints the per-function summary, the other two the spill and local-memory warnings). Pass the compiler's own output
and nothing else: a line no reader knows is a notice of tool `unreadable`, and has no row.

## The reader

`notices.read_notices(text)` (text in, data out) returns `Notice(tool, code, text, function, where, raw, message)` in
order. It reads ptxas's `info`/`warning` lines with or without a `(Cnnnn)` code and a function name, nvcc's
`file(line): warning #N-D: ...`, a bare `Potential Performance Loss: ...` line and nvcc's `Remark:` lines. What follows a
notice belongs to it (`raw`): the indented lines of ptxas's per-function summary, an nvcc warning's source excerpt and
instantiations up to the blank line. `text` is what the ledger matches: whitespace folded and the function's name
replaced by `'{}'`; `function` is the name as printed (mangled), `where` is `file(line)` with the directory dropped.

## The schema

A ledger file is `{"rows": [...]}`; a row is ONE kind of row: a notice code and a rule.

| field | |
|---|---|
| `tool` | `ptxas`, `nvcc`, `nvlink`, `cicc`, `fatbinary` |
| `code` | the notice's own (`C7512`, `#549-D`), or `null`: the row then matches the text under any code, and none |
| `text` | the template's fixed words; `{}` stands for a number or a name, a function's own name is `'{}'` |
| `rule` | required: the rule name the notice is counted under |
| `allow` | the row's default allowance: 0 when absent (a refusal), a count, or `null` (accepted by default) |
| `reason` | required, one sentence: what the notice means for the kernel |
| `fix` | what to change (may be empty) |

A notice is a rule input: it is counted under its row's `rule` at the notice's function, and the project's declarations
([declaring](checker-declarations.md)) govern the count exactly as they govern a rule of the model: a declaration of the rule
on a kernel (a regex over the demangled function name) says how many that kernel may have. A declaration may allow more than
the row's default, or tighten it: `allow: 0` on a rule whose row says `allow: null` refuses a notice the project otherwise
accepts. The notices of all of a rule's rows at one function are summed (the rule's default is the strictest of its rows'). The
counts go to the run's [facts](checker-facts.md); the refusal prints the notices, the rows' reasons and fixes, and the
project's note for the rule.

There is no other kind of row: what was a `refuse`, a `declare`, a `report` and an `ignore` row is a row with a rule and an
allowance (0, 0, null, null).

A notice matches a row by tool, then code (where the row names one), then text. A notice no row matches is refused
(rule `notice_without_row`, which no declaration can lift: an unknown notice is a decision, never a count): the toolkit's lines, quoted
whole, and the fix `add a row to the project's ledger`. No two
rows may match one notice: the ledger refuses two rows with one key, and refuses a row whose template matches what
another of the same tool (with a code that agrees) matches, when it is loaded (exit 2); a notice that two rows match is
the same error.

## The base ledger and a project's rows

`base_ledger.py` holds the 28 templates of the vendor's the module knows, each with a rule and a default allowance:

| rows | rule | allow |
|---|---|---|
| ten ptxas notices that asynchronous warpgroup MMAs are serialized (`C7511`, `C7512`, `C7514`, `C7520` and six that print no code) | `mma_serialized` | 0 |
| the injected warpgroup wait (`C7517`) | `wait_injected` | 0 |
| five `setmaxnreg ignored` notices (`C7505`, `C7507` and three with no code) | `setmaxnreg_ignored` | 0 |
| a variable used before it is set (`#549-D`) | `variable_unset` | 0 |
| registers spilled to local memory | `register_spill` | 0 |
| local memory used | `local_memory_used` | 0 |
| the injected warpgroup arrive (`C7519`) | `wgmma_arrive_injected` | accepted |
| an unreachable loop (`#128-D`) | `unreachable_loop` | accepted |
| a declared-but-unreferenced variable (`#177-D`) | `unused_variable` | accepted |
| the front end's suppression hint; the assembler's per-function summary lines (compiling, properties, registers, compile time, global bytes) | `build_info` | accepted |

A project passes `--ledger <file>` (or `ledger =` to the Bazel rule). Its rows OVERRIDE the base row of the same
(tool, code, text), in place, and any other row is ADDED. So a project that accepts local-memory use because it counts
the frame under the `stack_memory` rule gives the row's own key with another allowance:

```json
{"rows": [{"tool": "ptxas", "code": null, "text": "Local memory used for function '{}', size of stack frame: {} bytes",
           "rule": "local_memory_used", "allow": null,
           "reason": "the stack_memory rule counts the frame, and the declarations govern it"}]}
```

A project's row may name a new rule; the name is then valid in the project's declarations and notes. The
merged ledger is checked as a whole: a project row whose template would match what a different base row matches is an
error, not an override. `examples/checker/ledger.json` is a complete project ledger.

## Where a notice is, and stale declarations

A notice's place is the function it names, demangled through the binary's own names (the kernels read from the binary),
or `file(line)`, or `the build`. A declaration whose regex names no kernel and no place a notice or a profile was read
from is listed as stale, as in stage 1.
