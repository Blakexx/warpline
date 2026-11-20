"""`warpline_check_test`: the checker (docs/checker.md) as a Bazel test on a compiled kernel."""

load("@rules_python//python:defs.bzl", "py_test")

def warpline_check_test(name, declare, binary = None, fixture = None, arch = None, toolkit = None, notices = None, ledger = None, profiles = None, notes = None, **kwargs):
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
    for flag, label in (("--notices", notices), ("--ledger", ledger), ("--notes", notes)):
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
        srcs = [Label("//:warpline_cli.py")],
        main = "warpline_cli.py",
        args = args,
        data = data,
        deps = [Label("//:warpline")],
        **kwargs
    )
