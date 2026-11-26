"""The example kernels, compiled with the pinned toolkit (`@cuda_tools`, MODULE.bazel): no CUDA on the machine."""

_TOOLKIT = ["@cuda_tools//:toolkit"]
_NVCC = "$(execpath @cuda_tools//:bin/nvcc)"
_TOOLS = ["@cuda_tools//:bin/nvcc", "@cuda_tools//:bin/cuobjdump", "@cuda_tools//:bin/nvdisasm", "@cuda_tools//:bin/cu++filt"]

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
        cmd = " && ".join([
            "%s -arch=%s%s -cubin $(location %s) -o $(@D)/%s.cubin -Xptxas -v,--warn-on-spills,--warn-on-local-memory-usage 2> $(location %s.ptxas)" % (_NVCC, arch, info, src, name, name),
            "%s -sass $(@D)/%s.cubin > $(location %s.sass)" % (cuobjdump, name, name),
            "$(execpath @cuda_tools//:bin/nvdisasm) -bbcfg -poff $(@D)/%s.cubin > $(location %s.cfg.dot)" % (name, name),
            "%s -res-usage $(@D)/%s.cubin > $(location %s.res)" % (cuobjdump, name, name),
            "%s -sass $(@D)/%s.cubin | grep 'Function :' | awk '{print $$3}' | $(execpath @cuda_tools//:bin/cu++filt) > $(location %s.names)" % (cuobjdump, name, name),
        ] + (["$(execpath @cuda_tools//:bin/nvdisasm) --print-line-info $(@D)/%s.cubin > $(location %s.lines)" % (name, name)] if lines else [])),
        **kwargs
    )

def host_and_device_binary(name, src, arch, includes, hdrs, **kwargs):
    """An example program (host and device code in one `.cu`), compiled by nvcc for `arch`; `<name>.sass` is what
    `cuobjdump -sass` prints of it."""
    native.genrule(
        name = name,
        srcs = [src] + hdrs,
        outs = [name + ".bin", name + ".sass"],
        tools = _TOOLS + _TOOLKIT,
        cmd = "%s -std=c++17 -arch=%s -L$$(dirname %s)/../lib %s $(location %s) -o $(location %s.bin) && $(execpath @cuda_tools//:bin/cuobjdump) -sass $(location %s.bin) > $(location %s.sass)" % (
            _NVCC,
            arch,
            _NVCC,
            " ".join(["-I" + i for i in includes]),
            src,
            name,
            name,
            name,
        ),
        **kwargs
    )
