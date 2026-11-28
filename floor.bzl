"""`warpline_floor_bound`: a kernel's FLOOR as a build action (docs/floor.md): its work declaration at a card's rates and the
parameters given, written as the bound declarations a run manifest is judged by (`warpline_compare_test`, `checker.bzl`), so a
project's gate is floor, then bound, then check, with no number copied by hand. Its default output is the declarations file
(`<name>.json`); the output group `floor` holds the floor's text (`<name>.floor.txt`)."""

def _warpline_floor_bound_impl(ctx):
    bound = ctx.actions.declare_file(ctx.label.name + ".json")
    text = ctx.actions.declare_file(ctx.label.name + ".floor.txt")
    args = ctx.actions.args()
    args.add("floor")
    args.add(ctx.file.work)
    args.add("--rates", ctx.file.rates)
    for name, value in sorted(ctx.attr.params.items()):
        args.add("--param", "%s=%s" % (name, value))
    args.add("--bound", bound)
    args.add("--factor", ctx.attr.factor)
    args.add("--figure", ctx.attr.figure)
    if ctx.attr.kernel:
        args.add("--kernel", ctx.attr.kernel)
    if ctx.attr.reason:
        args.add("--reason", ctx.attr.reason)
    args.add("-o", text)
    ctx.actions.run(
        executable = ctx.attr._warpline_cli[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = [ctx.file.work, ctx.file.rates],
        outputs = [bound, text],
        mnemonic = "WarplineFloor",
        progress_message = "warpline floor %{label}",
    )
    return [DefaultInfo(files = depset([bound])), OutputGroupInfo(floor = depset([text]))]

warpline_floor_bound = rule(
    implementation = _warpline_floor_bound_impl,
    attrs = {
        "work": attr.label(mandatory = True, allow_single_file = [".json"], doc = "the kernel's work declaration (`warpline-work/1`)"),
        "rates": attr.label(mandatory = True, allow_single_file = [".json"], doc = "the card's rates table (`warpline-rates/1`, the probes')"),
        "params": attr.string_dict(doc = "the parameters the declaration's expressions read: name to a number's text"),
        "factor": attr.string(mandatory = True, doc = "the bound's factor over the floor, the project's (`1.15`)"),
        "figure": attr.string(default = "launch/grid_ns", values = ["launch/grid_ns", "launch/ns", "launch/cycles"],
                              doc = "the run manifest's figure the bound is on"),
        "kernel": attr.string(doc = "the kernel's pattern (default: the declaration's `kernel`)"),
        "reason": attr.string(doc = "the project's words, written after the floor's in each bound's reason"),
        "_warpline_cli": attr.label(default = Label("//:checker_cli"), executable = True, cfg = "exec"),
    },
    doc = "A kernel's floor at a card's rates, written as the bound declarations a run manifest is judged by.",
)
