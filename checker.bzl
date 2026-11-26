"""`warpline_check_test`: the checker (docs/checker.md) as a Bazel test on a compiled kernel; `warpline_check`: the same as two
build actions, the facts and then the comparison, with the facts file a side output (docs/checker-facts.md)."""

load("@rules_python//python:defs.bzl", "py_test")

def warpline_check_test(name, declare, binary = None, fixture = None, arch = None, toolkit = None, notices = None, ledger = None, profiles = None, notes = None, lines = None, **kwargs):
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
      profiles: a list of labels of Nsight Compute exports (`--page details --csv`, `--page source --print-source sass
        --csv`): the profile rules run on them; without any they are reported as not run.
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
            args += [flag, "$(rootpath %s)" % label]
            if label not in data:
                data.append(label)
    for label in profiles or []:
        args += ["--profile", "$(rootpath %s)" % label]
        if label not in data:
            data.append(label)
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
    facts = ctx.actions.declare_file(ctx.label.name + ".facts.json")
    report = ctx.actions.declare_file(ctx.label.name + ".check.txt")
    binary = ctx.file.binary
    optional = []
    for attr, flag in ((ctx.file.notices, "--notices"), (ctx.file.ledger, "--ledger"), (ctx.file.lines, "--lines")):
        if attr:
            optional.append((flag, attr))
    toolkit = ctx.files.toolkit

    # ACTION 1, THE FACTS: what each rule found on each kernel of the binary, written whole whatever it found.
    run = ctx.actions.args()
    run.add("facts")
    run.add(binary)
    run.add("-o", facts)
    if ctx.attr.arch:
        run.add("--arch", ctx.attr.arch)
    if toolkit:
        run.add("--toolkit", _toolkit_root(toolkit))
    for flag, file in optional:
        run.add(flag, file)
    for file in ctx.files.profiles:
        run.add("--profile", file)
    ctx.actions.run(
        executable = cli,
        arguments = [run],
        inputs = depset([binary] + ctx.files.profiles + [f for _, f in optional], transitive = [depset(toolkit)]),
        outputs = [facts],
        mnemonic = "WarplineFacts",
        progress_message = "warpline facts %{label}",
    )

    # ACTION 2, THE COMPARISON: the facts against the declarations; a refusal fails the build and prints itself.
    compare = ctx.actions.args()
    compare.add("compare")
    compare.add(facts)
    compare.add("--declare", ctx.file.declarations)
    if ctx.file.ledger:
        compare.add("--ledger", ctx.file.ledger)
    if ctx.file.notes:
        compare.add("--notes", ctx.file.notes)
    ctx.actions.run_shell(
        tools = [cli],
        arguments = [compare],
        inputs = [facts, ctx.file.declarations] + ([ctx.file.ledger] if ctx.file.ledger else []) + ([ctx.file.notes] if ctx.file.notes else []),
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
        OutputGroupInfo(facts = depset([facts])),
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
        "profiles": attr.label_list(allow_files = True),
        "notes": attr.label(allow_single_file = True),
        "lines": attr.label(allow_single_file = True),
        "_cli": attr.label(default = Label("//:checker_cli"), executable = True, cfg = "exec"),
    },
)

def warpline_check(name, binary, declarations, **kwargs):
    """The checker as two build actions on a compiled kernel: the FACTS of the binary, then their COMPARISON with the
    declarations. A refusal fails the build, printing itself; `bazel build :name` succeeds when nothing is refused.

    The facts file (`<name>.facts.json`, docs/checker-facts.md) is a side output, the output group `facts`: a consumer
    reads it for the counts and sites the rules found (`filegroup(srcs = [":name"], output_group = "facts")`).

    Args:
      name: the target's name; its default output is the comparison's text, `<name>.check.txt`.
      binary: a label whose single file is the compiled kernel (an executable, a library, a fatbin or a cubin).
      declarations: a label of the project's declarations file (JSON, docs/checker-declarations.md), for the binary's
        build target: one target is one architecture, so a declaration carries none.
      **kwargs: `arch` (the image to read from a fat binary), `toolkit` (a label of the CUDA toolkit's files, whose `bin/`
        holds `cuobjdump`, `nvdisasm` and `cu++filt`; none: the tools on PATH, which Bazel does not provide), `notices`
        (a label of the build's saved messages), `ledger`, `profiles` (labels of Nsight Compute exports), `notes`, `lines` (a label
        of the disassembly with source-line marks, so a finding names source lines), and the
        rule's common attributes (`tags`, `visibility`).
    """
    _warpline_check(name = name, binary = binary, declarations = declarations, **kwargs)
