"""`warpline_floor_bound`: a kernel's FLOOR as a build action (docs/floor.md): its work (a declaration's file, or a work function
called with the cell) at a card's rates, written as the bound declarations its run manifests are judged by
(`warpline_compare_test`, `checker.bzl`): the time within the project's factor of the floor, and each counter the work
expects within its band. A project's gate is floor, then bound, then check, with no number copied by hand. Its default output
is the declarations file (`<name>.json`); the output group `floor` holds the floor's text (`<name>.floor.txt`)."""

load("@rules_python//python:defs.bzl", "py_binary")

def _floor_bound_impl(ctx):
    bound = ctx.actions.declare_file(ctx.label.name + ".json")
    text = ctx.actions.declare_file(ctx.label.name + ".floor.txt")
    if (ctx.file.work == None) == (ctx.attr.work_function == ""):
        fail("%s: give the work as one of `work` (a declaration's file) or `work_function` (`module.path:function`)" % ctx.label)
    args = ctx.actions.args()
    args.add("floor")
    args.add(ctx.file.work if ctx.file.work else ctx.attr.work_function)
    args.add("--rates", ctx.file.rates)
    if ctx.file.parameters:
        args.add("--parameters", ctx.file.parameters)
    for name, value in sorted(ctx.attr.params.items()):
        args.add("--param", "%s=%s" % (name, value))
    args.add("--bound", bound)
    args.add("--factor", ctx.attr.factor)
    args.add("--tolerance", ctx.attr.tolerance)
    args.add("--figure", ctx.attr.figure)
    if ctx.attr.kernel:
        args.add("--kernel", ctx.attr.kernel)
    if ctx.attr.reason:
        args.add("--reason", ctx.attr.reason)
    args.add("-o", text)
    ctx.actions.run(
        executable = ctx.attr.cli[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = [f for f in (ctx.file.work, ctx.file.rates, ctx.file.parameters) if f],
        outputs = [bound, text],
        mnemonic = "WarplineFloor",
        progress_message = "warpline floor %{label}",
    )
    return [DefaultInfo(files = depset([bound])), OutputGroupInfo(floor = depset([text]))]

_floor_bound = rule(
    implementation = _floor_bound_impl,
    attrs = {
        "work": attr.label(allow_single_file = [".json"], doc = "the kernel's work declaration (`warpline-work/1`)"),
        "work_function": attr.string(doc = "the kernel's work function, `module.path:function`, called with the cell"),
        "rates": attr.label(mandatory = True, allow_single_file = [".json"],
                            doc = "the card's rates (`warpline-rates/1`: the probes' table, or a shipped descriptor, " +
                                  "`@warpline//:warpline/rates/<device>.json`)"),
        "parameters": attr.label(allow_single_file = [".json"], doc = "the cell, a JSON object (`params` after it)"),
        "params": attr.string_dict(doc = "fields of the cell: a name to a number's text (or a text)"),
        "factor": attr.string(mandatory = True, doc = "the bound's factor over the floor, the project's (`1.15`)"),
        "tolerance": attr.string(default = "0.05", doc = "the counters' band, a share of the floor at each stream's rate"),
        "figure": attr.string(default = "launch/grid_ns", values = ["launch/grid_ns", "launch/ns", "launch/cycles"],
                              doc = "the run manifest's figure the time's bound is on"),
        "kernel": attr.string(doc = "the kernel's pattern (default: the work's `kernel`)"),
        "reason": attr.string(doc = "the project's words, written after the floor's in each bound's reason"),
        "cli": attr.label(default = Label("//:checker_cli"), executable = True, cfg = "exec"),
    },
)

def warpline_floor_bound(name, work = None, work_function = None, deps = [], **kwargs):
    """A kernel's floor at a card's rates, written as the bound declarations its run manifests are judged by.

    Args:
      name: the target; its default output is `<name>.json`, the declarations.
      work: the work declaration's file (`warpline-work/1`); or
      work_function: the work function, `module.path:function`, imported from `deps` and called with the cell (`parameters`
        and `params`): its object is the work.
      deps: the py_library targets the work function's module is in.
      **kwargs: `rates` (mandatory), `parameters`, `params`, `factor` (mandatory), `tolerance`, `figure`, `kernel`,
        `reason`, and the common attributes.
    """
    cli = None
    if work_function:
        cli = name + ".floor_cli"
        py_binary(
            name = cli,
            srcs = [Label("//:warpline/checker/__main__.py")],
            main = Label("//:warpline/checker/__main__.py"),
            deps = [Label("//:checker")] + deps,
            visibility = ["//visibility:private"],
            tags = kwargs.get("tags", []),
        )
    _floor_bound(
        name = name,
        work = work,
        work_function = work_function or "",
        cli = (":" + cli) if cli else Label("//:checker_cli"),
        **kwargs
    )
