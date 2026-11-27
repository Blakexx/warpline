"""`warpline_check_test`: the checker (docs/checker.md) as a Bazel test on a compiled kernel; `warpline_check`: the same as two
build actions, the manifest and then the comparison, with the manifest file a side output (docs/checker-manifest.md), and
`warpline_check_actions` the same two actions for a rule of the consumer's own (the rule that links the binary, so its build
fails on a refusal). Each may judge the kernels' RUN MANIFESTS with the binary's (`runs`, docs/run-manifest.md) against the one
declarations file; `warpline_compare_test` judges manifest files alone (a run manifest the project keeps)."""

load("@rules_python//python:defs.bzl", "py_test")

def _data_args(args, data, flag, label):
    args.extend([flag, "$(rootpath %s)" % label])
    if label not in data:
        data.append(label)

def warpline_check_test(name, declare, binary = None, fixture = None, arch = None, toolkit = None, notices = None, ledger = None, runs = None, notes = None, lines = None, **kwargs):
    """Runs `warpline check` on a kernel and fails on a refusal.

    Args:
      name: the test's name.
      declare: the project's declarations file (JSON): per kernel, the findings a rule may report, and why; it may
        also carry `notes`, the project's own text per rule, printed after a refusal's fix.
      binary: a label whose single file is the compiled kernel (an executable, a library, a fatbin or a cubin); the
        test needs the CUDA toolkit's `cuobjdump`, `nvdisasm` and `c++filt` (see `toolkit`).
      fixture: instead of `binary`, the stem, in this package, of saved tool outputs (`<stem>.sass`, `<stem>.cfg.dot`,
        `<stem>.res`, optionally `<stem>.names`): no toolkit, no GPU.
      arch: the architecture to read from a fat binary (`sm_90a`).
      toolkit: the toolkit's root, when it is not CUDA_HOME or on PATH.
      notices: a list of labels of the build's saved messages (ptxas and nvcc output, a file a compile): each notice is
        read against the notice ledger, and one with no row is a refusal.
      ledger: a label of the project's ledger file (JSON), whose rows override or add to the module's base ledger.
      runs: a list of labels of run manifests of the binary's kernels (`warpline run-manifest`, docs/run-manifest.md):
        judged with the binary's manifest against `declare`.
      notes: a label of a file of notes per rule, when they are not in `declare`.
      lines: a label of the binary's disassembly with source-line marks (`nvdisasm --print-line-info` of its cubin, built with
        `--generate-line-info`, e.g. `kernel_outputs(lines = True)`'s `<name>.lines`): a finding then names source lines.
        With a `fixture`, `<stem>.lines` is read when it exists.
      **kwargs: passed to the test (`tags`, `size`).
    """
    if (binary == None) == (fixture == None):
        fail("warpline_check_test %s: give exactly one of binary and fixture" % name)
    args = ["check"]
    data = [declare]
    if binary:
        args.append("$(rootpath %s)" % binary)
        data.append(binary)
        if arch:
            args += ["--arch", arch]
        if toolkit:
            args += ["--toolkit", toolkit]
    else:
        args += ["$(rootpath :%s.sass)" % fixture, "--fixture"]
        data += native.glob([fixture + ".*"])
    args += ["--declare", "$(rootpath %s)" % declare]
    for flag, label in (("--ledger", ledger), ("--notes", notes), ("--lines", lines)):
        if label:
            _data_args(args, data, flag, label)
    for flag, labels in (("--notices", notices), ("--run", runs)):
        for label in labels or []:
            _data_args(args, data, flag, label)
    py_test(
        name = name,
        srcs = [Label("//:warpline/checker/__main__.py")],
        main = Label("//:warpline/checker/__main__.py"),
        args = args,
        data = data,
        deps = [Label("//:checker")],
        **kwargs
    )

def warpline_compare_test(name, manifests, declare, notes = None, ledger = None, **kwargs):
    """Runs `warpline compare` on manifest files and fails on a refusal: a run manifest the project keeps (a file a run of
    the kernel wrote, `warpline run-manifest`), alone or with others, judged against its declarations. No GPU, no toolkit.

    Args:
      name: the test's name.
      manifests: a list of labels of manifest files (`warpline-run-manifest/1` or `warpline-manifest/1`), judged as one.
      declare: the project's declarations file (JSON): its `bound`s on the run's figures, its `violations` of the profile
        rules, `waive`s and `coverage` (docs/run-manifest.md).
      notes: a label of a file of notes per rule, when they are not in `declare`.
      ledger: a label of the project's ledger file, for the rule names its rows count under.
      **kwargs: passed to the test (`tags`, `size`).
    """
    args = ["compare"]
    data = [declare]
    for label in manifests:
        args.append("$(rootpath %s)" % label)
        if label not in data:
            data.append(label)
    args += ["--declare", "$(rootpath %s)" % declare]
    for flag, label in (("--notes", notes), ("--ledger", ledger)):
        if label:
            _data_args(args, data, flag, label)
    py_test(
        name = name,
        srcs = [Label("//:warpline/checker/__main__.py")],
        main = Label("//:warpline/checker/__main__.py"),
        args = args,
        data = data,
        deps = [Label("//:checker")],
        **kwargs
    )

def _toolkit_root(files):
    """The root of the toolkit whose files these are: the directory above its `bin/cuobjdump`."""
    for file in files:
        if file.path.endswith("/bin/cuobjdump"):
            return file.path[:-len("/bin/cuobjdump")]
    fail("warpline_check: the toolkit has no bin/cuobjdump")

#: what a rule that calls `warpline_check_actions` adds to its attributes: the checker's command, built for the host the actions
#: run on (`ctx.attr._warpline_cli`)
WARPLINE_CHECK_ATTRS = {
    "_warpline_cli": attr.label(default = Label("//:checker_cli"), executable = True, cfg = "exec"),
}

def warpline_check_actions(ctx, name, binary, declarations, arch = None, toolkit = [], notices = [], ledger = None, notes = None, lines = None, runs = []):
    """THE CHECKER AS TWO BUILD ACTIONS of the calling rule: the MANIFEST of `binary`, then its COMPARISON with the declarations
    (and the kernels' run manifests). The comparison's action fails on a refusal, printing it, so whatever depends on its output
    (the rule's own default outputs, say) is not built while one stands. The rule carries `WARPLINE_CHECK_ATTRS`.

    Args:
      ctx: the calling rule's context.
      name: the outputs' stem in the rule's package: `<name>.manifest.json` (the manifest, docs/checker-manifest.md) and
        `<name>.check.txt` (the comparison's text).
      binary: the File of the compiled kernel (an executable, a library, a fatbin or a cubin).
      declarations: the File of the declarations (JSON, docs/checker-declarations.md), for the one architecture read.
      arch: the image to read from a fat binary (`sm_90a`).
      toolkit: the CUDA toolkit's Files, whose `bin/` holds `cuobjdump`, `nvdisasm` and `cu++filt` (none: the tools on PATH).
      notices: the Files of the build's saved messages, a file a compile.
      ledger: the File of the project's ledger.
      notes: the File of the project's notes per rule, when they are not in the declarations.
      lines: the File of the disassembly with source-line marks.
      runs: the Files of run manifests of the binary's kernels.

    Returns:
      struct(manifest = the manifest's File, report = the comparison's File).
    """
    cli = ctx.attr._warpline_cli[DefaultInfo].files_to_run
    manifest = ctx.actions.declare_file(name + ".manifest.json")
    report = ctx.actions.declare_file(name + ".check.txt")

    # ACTION 1, THE MANIFEST: what each rule found on each kernel of the binary, written whole whatever it found.
    run = ctx.actions.args()
    run.add("manifest")
    run.add(binary)
    run.add("-o", manifest)
    if arch:
        run.add("--arch", arch)
    if toolkit:
        run.add("--toolkit", _toolkit_root(toolkit))
    run.add_all(notices, before_each = "--notices")
    for flag, file in (("--ledger", ledger), ("--lines", lines)):
        if file:
            run.add(flag, file)
    ctx.actions.run(
        executable = cli,
        arguments = [run],
        inputs = depset([binary] + notices + [f for f in (ledger, lines) if f], transitive = [depset(toolkit)]),
        outputs = [manifest],
        mnemonic = "WarplineManifest",
        progress_message = "warpline manifest %{label} (" + name + ")",
    )

    # ACTION 2, THE COMPARISON: the manifest (and the kernels' run manifests) against the declarations; a refusal fails the
    # build and prints itself.
    compare = ctx.actions.args()
    compare.add("compare")
    compare.add(manifest)
    compare.add_all(runs)
    compare.add("--declare", declarations)
    for flag, file in (("--ledger", ledger), ("--notes", notes)):
        if file:
            compare.add(flag, file)
    ctx.actions.run_shell(
        tools = [cli],
        arguments = [compare],
        inputs = [manifest, declarations] + runs + [f for f in (ledger, notes) if f],
        outputs = [report],
        command = '"%s" "$@" > "%s"; status=$?; if [ $status -ne 0 ]; then cat "%s"; fi; exit $status' % (
            cli.executable.path,
            report.path,
            report.path,
        ),
        mnemonic = "WarplineCompare",
        progress_message = "warpline compare %{label} (" + name + ")",
    )
    return struct(manifest = manifest, report = report)

def _warpline_check_impl(ctx):
    checked = warpline_check_actions(
        ctx,
        name = ctx.label.name,
        binary = ctx.file.binary,
        declarations = ctx.file.declarations,
        arch = ctx.attr.arch,
        toolkit = ctx.files.toolkit,
        notices = ctx.files.notices,
        ledger = ctx.file.ledger,
        notes = ctx.file.notes,
        lines = ctx.file.lines,
        runs = ctx.files.runs,
    )
    return [
        DefaultInfo(files = depset([checked.report])),
        OutputGroupInfo(manifest = depset([checked.manifest])),
    ]

_warpline_check = rule(
    implementation = _warpline_check_impl,
    attrs = WARPLINE_CHECK_ATTRS | {
        "binary": attr.label(allow_single_file = True, mandatory = True),
        "declarations": attr.label(allow_single_file = True, mandatory = True),
        "arch": attr.string(),
        "toolkit": attr.label(allow_files = True),
        "notices": attr.label_list(allow_files = True),
        "ledger": attr.label(allow_single_file = True),
        "runs": attr.label_list(allow_files = True),
        "notes": attr.label(allow_single_file = True),
        "lines": attr.label(allow_single_file = True),
    },
)

def warpline_check(name, binary, declarations, **kwargs):
    """The checker as two build actions on a compiled kernel: the MANIFEST of the binary, then its COMPARISON (with the
    kernels' run manifests, `runs`) with the declarations. A refusal fails the build, printing itself; `bazel build :name`
    succeeds when nothing is refused.

    The manifest file (`<name>.manifest.json`, docs/checker-manifest.md) is a side output, the output group `manifest`: a consumer
    reads it for the counts and sites the rules found (`filegroup(srcs = [":name"], output_group = "manifest")`).

    Args:
      name: the target's name; its default output is the comparison's text, `<name>.check.txt`.
      binary: a label whose single file is the compiled kernel (an executable, a library, a fatbin or a cubin).
      declarations: a label of the project's declarations file (JSON, docs/checker-declarations.md), for the binary's
        build target: one target is one architecture, so a declaration carries none. It may hold the checkers over the
        kernels' manifests (`bound`, `hash`, `waive`, `present`, `"coverage": true`, `"strict": true`,
        docs/checker-manifest.md): they run in the comparison.
      **kwargs: `arch` (the image to read from a fat binary), `toolkit` (a label of the CUDA toolkit's files, whose `bin/`
        holds `cuobjdump`, `nvdisasm` and `cu++filt`; none: the tools on PATH, which Bazel does not provide), `notices`
        (labels of the build's saved messages, a file a compile), `ledger`, `runs` (labels of run manifests of the binary's kernels,
        docs/run-manifest.md), `notes`, `lines` (a label of the disassembly with source-line marks, so a finding names source
        lines), and the rule's common attributes (`tags`, `visibility`).
    """
    _warpline_check(name = name, binary = binary, declarations = declarations, **kwargs)
