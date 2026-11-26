"""`warpline_check_test`: the checker (docs/checker.md) as a Bazel test on a compiled kernel; `warpline_check`: the same as two
build actions, the manifest and then the comparison, with the manifest file a side output (docs/checker-manifest.md). Each may
judge the kernels' RUN MANIFESTS with the binary's (`runs`, docs/run-manifest.md) against the one declarations file;
`warpline_compare_test` judges manifest files alone (a run manifest the project keeps)."""

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
      notices: a label of the build's saved messages (ptxas and nvcc output): each notice is read against the notice
        ledger, and one with no row is a refusal.
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
    for flag, label in (("--notices", notices), ("--ledger", ledger), ("--notes", notes), ("--lines", lines)):
        if label:
            _data_args(args, data, flag, label)
    for label in runs or []:
        _data_args(args, data, "--run", label)
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

def _warpline_check_impl(ctx):
    cli = ctx.attr._cli[DefaultInfo].files_to_run
    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest.json")
    report = ctx.actions.declare_file(ctx.label.name + ".check.txt")
    binary = ctx.file.binary
    optional = []
    for attr, flag in ((ctx.file.notices, "--notices"), (ctx.file.ledger, "--ledger"), (ctx.file.lines, "--lines")):
        if attr:
            optional.append((flag, attr))
    toolkit = ctx.files.toolkit

    # ACTION 1, THE MANIFEST: what each rule found on each kernel of the binary, written whole whatever it found.
    run = ctx.actions.args()
    run.add("manifest")
    run.add(binary)
    run.add("-o", manifest)
    if ctx.attr.arch:
        run.add("--arch", ctx.attr.arch)
    if toolkit:
        run.add("--toolkit", _toolkit_root(toolkit))
    for flag, file in optional:
        run.add(flag, file)
    ctx.actions.run(
        executable = cli,
        arguments = [run],
        inputs = depset([binary] + [f for _, f in optional], transitive = [depset(toolkit)]),
        outputs = [manifest],
        mnemonic = "WarplineManifest",
        progress_message = "warpline manifest %{label}",
    )

    # ACTION 2, THE COMPARISON: the manifest (and the kernels' run manifests) against the declarations; a refusal fails the
    # build and prints itself.
    compare = ctx.actions.args()
    compare.add("compare")
    compare.add(manifest)
    compare.add_all(ctx.files.runs)
    compare.add("--declare", ctx.file.declarations)
    if ctx.file.ledger:
        compare.add("--ledger", ctx.file.ledger)
    if ctx.file.notes:
        compare.add("--notes", ctx.file.notes)
    ctx.actions.run_shell(
        tools = [cli],
        arguments = [compare],
        inputs = [manifest, ctx.file.declarations] + ctx.files.runs + ([ctx.file.ledger] if ctx.file.ledger else []) + ([ctx.file.notes] if ctx.file.notes else []),
        outputs = [report],
        command = '"%s" "$@" > "%s"; status=$?; if [ $status -ne 0 ]; then cat "%s"; fi; exit $status' % (
            cli.executable.path,
            report.path,
            report.path,
        ),
        mnemonic = "WarplineCompare",
        progress_message = "warpline compare %{label}",
    )
    return [
        DefaultInfo(files = depset([report])),
        OutputGroupInfo(manifest = depset([manifest])),
    ]

_warpline_check = rule(
    implementation = _warpline_check_impl,
    attrs = {
        "binary": attr.label(allow_single_file = True, mandatory = True),
        "declarations": attr.label(allow_single_file = True, mandatory = True),
        "arch": attr.string(),
        "toolkit": attr.label(allow_files = True),
        "notices": attr.label(allow_single_file = True),
        "ledger": attr.label(allow_single_file = True),
        "runs": attr.label_list(allow_files = True),
        "notes": attr.label(allow_single_file = True),
        "lines": attr.label(allow_single_file = True),
        "_cli": attr.label(default = Label("//:checker_cli"), executable = True, cfg = "exec"),
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
        kernels' manifests (`bound`, `hash`, `waive`, `"coverage": true`, docs/checker-manifest.md): they run in the comparison.
      **kwargs: `arch` (the image to read from a fat binary), `toolkit` (a label of the CUDA toolkit's files, whose `bin/`
        holds `cuobjdump`, `nvdisasm` and `cu++filt`; none: the tools on PATH, which Bazel does not provide), `notices`
        (a label of the build's saved messages), `ledger`, `runs` (labels of run manifests of the binary's kernels,
        docs/run-manifest.md), `notes`, `lines` (a label of the disassembly with source-line marks, so a finding names source
        lines), and the rule's common attributes (`tags`, `visibility`).
    """
    _warpline_check(name = name, binary = binary, declarations = declarations, **kwargs)
