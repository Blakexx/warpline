package(default_visibility = ["//visibility:public"])

exports_files(["bin/cuobjdump", "bin/nvdisasm", "bin/nvcc", "bin/cu++filt"])

#: the toolkit whole: nvcc reads nvcc.profile, include/, lib64/ and nvvm/ beside it (lib64 is a link to lib, which the
#: glob would list twice)
filegroup(name = "toolkit", srcs = glob(["**"], exclude = ["BUILD.bazel", "REPO.bazel", "redistrib.json", "lib64/**"]))
