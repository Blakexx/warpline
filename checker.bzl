"""`warpline_check_test`: the checker (docs/checker.md) as a Bazel test on a compiled kernel."""

load("@rules_python//python:defs.bzl", "py_test")

def warpline_check_test(name, declare, binary = None, fixture = None, arch = None, toolkit = None, **kwargs):
    """Runs `warpline check` on a kernel and fails on a refusal.

    Args:
      name: the test's name.
      declare: the project's declarations file (JSON): per kernel, the findings a rule may report, and why.
      binary: a label whose single file is the compiled kernel (an executable, a library, a fatbin or a cubin); the
        test needs the CUDA toolkit's `cuobjdump`, `nvdisasm` and `c++filt` (see `toolkit`).
      fixture: instead of `binary`, the stem, in this package, of saved tool outputs (`<stem>.sass`, `<stem>.cfg.dot`,
        `<stem>.res`, optionally `<stem>.names`): no toolkit, no GPU.
      arch: the architecture to read from a fat binary (`sm_90a`).
      toolkit: the toolkit's root, when it is not CUDA_HOME or on PATH.
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
    py_test(
        name = name,
        srcs = [Label("//:warpline_cli.py")],
        main = "warpline_cli.py",
        args = args,
        data = data,
        deps = [Label("//:warpline")],
        **kwargs
    )
