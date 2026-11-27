package(default_visibility = ["//visibility:public"])

#: `ncu` at the root as the archive lays it out: the profiler `warpline run --profile` runs (`--toolkit` this root)
exports_files(["ncu"])

#: Nsight Compute whole: `ncu` reads the sections, the target libraries and the report tools beside it
filegroup(name = "profiler", srcs = glob(["**"], exclude = ["BUILD.bazel", "REPO.bazel"]))
