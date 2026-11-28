"""The example kernels, compiled with the pinned toolkit (`@cuda_tools`, MODULE.bazel), nvcc's host compiler the build's
C++ toolchain (warpline's own build registers the pinned one, MODULE.bazel's `host_toolchain`): no CUDA and no compiler of
the machine's."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

#: the build's C++ toolchain, which nvcc drives as its host compiler (`-ccbin $(HOST_CC)`): a target of its own, so every
#: compile names it and takes its files
_HOST_CC = Label("//examples:host_cc")
_TOOLKIT = ["@cuda_tools//:toolkit"]
_NVCC = "$(execpath @cuda_tools//:bin/nvcc) -ccbin $(HOST_CC)"
_TOOLS = ["@cuda_tools//:bin/nvcc", "@cuda_tools//:bin/cuobjdump", "@cuda_tools//:bin/nvdisasm", "@cuda_tools//:bin/cu++filt", _HOST_CC]

def _host_compiler_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    features = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    driver = cc_common.get_tool_for_action(feature_configuration = features, action_name = ACTION_NAMES.cpp_compile)
    return [
        platform_common.TemplateVariableInfo({"HOST_CC": driver}),
        DefaultInfo(files = cc_toolchain.all_files),
    ]

host_compiler = rule(
    implementation = _host_compiler_impl,
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    doc = "The build's C++ compiler driver as `$(HOST_CC)`, the toolchain's files the target's: a genrule that runs nvcc " +
          "names it in `toolchains` and `tools`, so nvcc's `-ccbin` is the registered toolchain's, never the machine's.",
)

def kernel_outputs(name, src, arch, lines = False, **kwargs):
    """The checker's saved tool outputs of one example kernel, regenerated: `<name>.cubin` (the binary `warpline_check`
    reads), `<name>.sass`, `.cfg.dot`, `.res`, `.names` and `.ptxas` (the same recipe docs/checker-model.md gives by hand).
    With `lines`, the kernel is compiled with `--generate-line-info` and `<name>.lines` is `nvdisasm --print-line-info` of
    the cubin: the disassembly with source-line marks (`--lines`, docs/checker-model.md)."""
    cuobjdump = "$(execpath @cuda_tools//:bin/cuobjdump)"
    info = " --generate-line-info" if lines else ""
    native.genrule(
        name = name,
        srcs = [src],
        outs = [name + ext for ext in (".cubin", ".sass", ".cfg.dot", ".res", ".names", ".ptxas") + ((".lines",) if lines else ())],
        tools = _TOOLS + _TOOLKIT,
        toolchains = [_HOST_CC],
        cmd = " && ".join([
            "%s -arch=%s%s -cubin $(location %s) -o $(@D)/%s.cubin -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage 2> $(location %s.ptxas)" % (_NVCC, arch, info, src, name, name),
            "%s -sass $(@D)/%s.cubin > $(location %s.sass)" % (cuobjdump, name, name),
            "$(execpath @cuda_tools//:bin/nvdisasm) -bbcfg -poff $(@D)/%s.cubin > $(location %s.cfg.dot)" % (name, name),
            "%s -res-usage $(@D)/%s.cubin > $(location %s.res)" % (cuobjdump, name, name),
            "%s -sass $(@D)/%s.cubin | grep 'Function :' | awk '{print $$3}' | $(execpath @cuda_tools//:bin/cu++filt) > $(location %s.names)" % (cuobjdump, name, name),
        ] + (["$(execpath @cuda_tools//:bin/nvdisasm) --print-line-info $(@D)/%s.cubin > $(location %s.lines)" % (name, name)] if lines else [])),
        **kwargs
    )

def host_and_device_binary(name, src, archs, includes, hdrs, **kwargs):
    """An example program (host and device code in one `.cu`), compiled by nvcc with one SASS image for each of `archs`
    (`sm_80`, `sm_90a`, ...), so each of those GPUs runs it with no JIT; `<name>.sass` is what `cuobjdump -sass` prints
    of it, every image."""
    native.genrule(
        name = name,
        srcs = [src] + hdrs,
        outs = [name + ".bin", name + ".sass"],
        tools = _TOOLS + _TOOLKIT,
        toolchains = [_HOST_CC],
        cmd = "%s -std=c++17 %s -L$$(dirname $(execpath @cuda_tools//:bin/nvcc))/../lib %s $(location %s) -o $(location %s.bin) && $(execpath @cuda_tools//:bin/cuobjdump) -sass $(location %s.bin) > $(location %s.sass)" % (
            _NVCC,
            " ".join(["-gencode arch=compute_%s,code=sm_%s" % (a[3:], a[3:]) for a in archs]),
            " ".join(["-I" + i for i in includes]),
            src,
            name,
            name,
            name,
        ),
        **kwargs
    )
