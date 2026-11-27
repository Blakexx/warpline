# Rules backed by the profiler

Static rules read the compiled kernel. Some facts exist only when the kernel has run: how many wavefronts a shared access
took, what the profiler's own rules concluded. The checker reads Nsight Compute's CSV exports and runs PROFILE rules on them, and
their findings are a run's: they are written into the kernel's [run manifest](run-manifest.md), never into the binary's static
manifest.

    python -m warpline run --profile SourceCounters,SpeedOfLight --place 'sample::tile()' -o run -- ./task   # the run, profiled
    python -m warpline run-manifest --profile details.csv --profile source.csv --place 'sample::tile()' -o run.manifest.json
    python -m warpline compare run.manifest.json --declare declarations.json
    python -m warpline check <binary> --declare declarations.json --run run.manifest.json     # with the binary's own rules

`warpline run --profile` runs the task under Nsight Compute, limited to its marked regions, exports the pages below and writes
the run manifest ([the run layer](run-layer.md)); `run-manifest` reads exports made by hand. `manifest` and `check` take no
profile. A run manifest made without `--profile` says `note: profile rules not run (no --profile)`.

## The exports (`ncu.py`)

Text in, data out; a reader never runs the profiler. The flags each page is printed with are declared once, `ncu.PAGES`, and
the runner exports with them:

    ncu --import report.ncu-rep --page details --csv > details.csv
    ncu --import report.ncu-rep --page source --print-source sass --csv --launch-skip I --launch-count 1 > source.I.csv

and pass any of them to `run-manifest`, `--profile` repeated. The two are told apart by their header (the source page starts
with a `Kernel Name` line; the details page has a `Rule Name` column). A SOURCE PAGE IS ONE RESULT'S: a report of several
results prints a block a result, so result I is exported alone (`--launch-skip I --launch-count 1`); Nsight Compute may print
a result's block more than once, and identical copies are read once, while blocks that differ are refused.

A COLUMN A READER READS IS REQUIRED: a page without one is refused, naming it, never read as 0 or as empty. A profiler version
that stops exporting a column is a refusal, not a silent zero.

- `details_rules(text)` returns `RuleResult(kernel, launch, section, name, type, description, speedup_type, speedup)` for
  every row that carries a rule: the built-in rules' Rule Name, Rule Type (`OPT`, `WRN`, `INF`), Rule Description and
  Estimated Speedup (percent; `None` where the rule gives none). The page has every column it reads (`DETAILS_COLUMNS`: ID,
  Kernel Name, Section Name, Rule Name, Rule Type, Rule Description, Estimated Speedup Type, Estimated Speedup).
  `details_results(text)` lists its results, `(ID, kernel name)` in order.
- `source_rows(text)` returns a `SourceProfile(kernel, rows, reasons)`, a `SourceRow(address, source, executed, space,
  wavefronts, ideal, excess, stalls)` for each instruction: Instructions Executed, its Address Space, the shared-memory
  wavefronts (L1 Wavefronts Shared, Ideal, Excessive) and the stall samples by reason (`stall_long_sb` as `long_sb`, zero
  counts left out; `reasons` lists the reasons the page exports, so a reason it does not export is not read as one that
  sampled nothing). The source page's columns are the metrics of the sections collected: Address Space and the instructions'
  counts are required (`SOURCE_COLUMNS`; a profile without `SourceCounters` has none, and the refusal names the section), and
  the three wavefront columns where an instruction's address space is shared. Nsight Compute exports no wavefront column for a
  kernel with no shared-memory instruction (seen with 2025.3 and the full set): there each instruction's wavefronts are 0,
  since none takes one.
- `profiles_of(texts)` groups both into a `Profile(kernel, rules, source, has_details, has_source, reasons)` for each kernel
  name: the source pages of several results of one kernel are its rows together (a count over the profiled window).

A profile is matched to the run manifest's place (`--place`, the binary's demangled kernel name) by the name before the
parameter list, with the profiler's `void ` prefix, its `(bool)` casts and spaces removed and a bool template argument read
as `1`/`0` (`warpline.checker.ncu.kernel_key`, which also pairs a profile's two pages: Nsight Compute 2026 prints
`k<1>` on the details page and `k<(bool)1>` on the source page, `c++filt` `k<true>`; `warpline.checker.manifest.same_kernel`;
`match_profile(profile, kernels)` is the same rule over a binary's kernels); a lone profile is paired with the place whatever
it is called, and a profile that matches it not is named in a note and judged under the profiler's own kernel name, a place of
its own. A declaration's regex is searched over that place. A rule needs one of the pages (`needs`): where the profile has not
got it, the run manifest says `profile rule X not run on 'K': the profile has no source page`, never a silent zero.
`warpline.checker.manifest.run_rules(profiles, place, profile_rules)` gives a run manifest its rules.

## The two rules

A profile rule is a `ProfileRule` (a `Rule` whose `check` takes a profile), in a module of `rules/` that defines
`PROFILE_RULE`; `rules.profile_registry()` lists them, and `registry()` does not: a binary's rules are the static ones, and a
profile's are a run's. Fixtures are directories under `tests/checker/fixtures/profile_rules/<rule>/<name>/`
holding `details.csv` and/or `source.csv`: small neutral cuts of a real export, with the kernels renamed.

| rule | flags | count | fixtures |
|---|---|---|---|
| `shared_wavefronts_excess` | shared-memory wavefronts in excess of the conflict-free count, by instruction; `where` is the instruction's offset in the kernel (its address less the first row's) | the excess wavefronts EXECUTED, summed: a count over the profiled window | `conflicted` 98304 (three 16-bit loads at 2 wavefronts for 1); quiet: `conflict_free` |
| `profiler_verdicts` | each `OPT` or `WRN` result of the profiler's built-in rules; `where` is the profiler's rule name, the text its description and estimated speedup; `INF` results are information | 1 a verdict | `fired` 3 (the bank-conflict rule on two kernels); quiet: `informational` |

A declaration allows a count of the profiled window (`shared_wavefronts_excess`), or a number of verdicts
(`profiler_verdicts`, which does not say which). The profiled window is whatever the profiler captured: a project that
declares a count states it for its own window.
