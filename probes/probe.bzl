"""THE RATE PROBE AS ANY WORKSPACE BUILDS IT (docs/floor.md#the-probes): `warpline_rates_probe` compiles `rates.cu` with the
CUDA toolchain the building workspace registers for `@warpline//probes:cuda_toolchain_type` (`warpline_cuda_toolchain`: its
`nvcc` and the toolkit's files), nvcc's host compiler the workspace's own C++ toolchain (`-ccbin`). warpline's own build
registers its pinned toolkit (MODULE.bazel, `//toolchain:cuda_probe_toolchain`); a consumer registers its own:

    load("@warpline//probes:probe.bzl", "warpline_cuda_toolchain")
    warpline_cuda_toolchain(name = "cuda", nvcc = "@my_cuda//:bin/nvcc", files = ["@my_cuda//:toolkit"])
    toolchain(name = "cuda_toolchain", toolchain = ":cuda", toolchain_type = "@warpline//probes:cuda_toolchain_type")
    # MODULE.bazel: register_toolchains("//:cuda_toolchain")

and then `bazel run @warpline//probes:rates -- rates.json COMMIT` on a card writes the card's table."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

_TYPE = Label("//probes:cuda_toolchain_type")

def _toolchain_impl(ctx):
    files = depset([ctx.file.nvcc], transitive = [f[DefaultInfo].files for f in ctx.attr.files])
    return [platform_common.ToolchainInfo(nvcc = ctx.file.nvcc, files = files)]

warpline_cuda_toolchain = rule(
    implementation = _toolchain_impl,
    attrs = {
        "nvcc": attr.label(mandatory = True, allow_single_file = True, cfg = "exec", doc = "the toolkit's `bin/nvcc`"),
        "files": attr.label_list(allow_files = True, cfg = "exec", doc = "every file of the toolkit nvcc compiles and links with"),
    },
    doc = "A CUDA toolkit for `warpline_rates_probe`: its nvcc and its files (headers, compilers, the static runtime).",
)

def _probe_impl(ctx):
    cuda = ctx.toolchains[_TYPE]
    cc = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    driver = cc_common.get_tool_for_action(feature_configuration = features, action_name = ACTION_NAMES.cpp_compile)
    env = cc_common.get_environment_variables(
        feature_configuration = features,
        action_name = ACTION_NAMES.cpp_compile,
        variables = cc_common.empty_variables(),
    )
    out = ctx.actions.declare_file(ctx.label.name + ".bin")
    lib = cuda.nvcc.dirname + "/.."
    args = ctx.actions.args()
    args.add("-ccbin", driver)
    args.add("-std=c++17")
    for arch in ctx.attr.archs:
        args.add("-gencode", "arch=compute_%s,code=sm_%s" % (arch[3:], arch[3:]))
    args.add("-L" + lib + "/lib")
    args.add("-L" + lib + "/lib64")
    args.add(ctx.file.src)
    args.add("-o", out)
    ctx.actions.run(
        executable = cuda.nvcc,
        arguments = [args],
        inputs = depset([ctx.file.src], transitive = [cuda.files, cc.all_files]),
        outputs = [out],
        env = env,
        #: the shell's tools (a toolchain's wrapper script reads its own path with them), as a genrule's command has them
        use_default_shell_env = True,
        mnemonic = "WarplineProbe",
        progress_message = "compiling the rate probe %{label}",
    )
    return [DefaultInfo(executable = out, files = depset([out]))]

warpline_rates_probe = rule(
    implementation = _probe_impl,
    executable = True,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".cu"], doc = "the probe's program (`rates.cu`)"),
        "archs": attr.string_list(default = ["sm_80", "sm_90a"], doc = "the SASS images it carries (a card runs its own)"),
    },
    toolchains = use_cc_toolchain() + [_TYPE],
    fragments = ["cpp"],
    doc = "The rate probe, compiled with the building workspace's CUDA toolchain and C++ toolchain: `<name>.bin OUT COMMIT`.",
)
