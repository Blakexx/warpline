"""THE THIN RUNNER: invokes the toolkit's tools on a binary and hands their text to the readers, and puts a task under the
profiler for `warpline run --profile` and exports what it captured. Nothing here parses but the details page's list of results;
it is the only module of the checker that starts a process, and the one a test without a toolkit does not use.

The toolkit is found in this order: the `--toolkit` argument (a toolkit's root or its `bin`), `CUDA_HOME`, `PATH`. The
demangler is the toolkit's `cu++filt`, else `c++filt`. The profiler is Nsight Compute's `ncu`, found the same way.

THE PROFILED RUN. `profile_command` is the prefix the task runs under: `ncu --profile-from-start off`, so the profiler captures
the kernels between the task's `cudaProfilerStart` and `cudaProfilerStop` (`warpline::run::Region`, include/warpline/run.hpp)
and nothing else, the sections and metrics asked for, and the report written as `DIR/profile.ncu-rep` (`.ncu-repz` from Nsight Compute 2026). `export` prints the
report's pages with the flags its readers read (`ncu.PAGES`): the details page once, and the source page a result
(`--launch-skip I --launch-count 1`) where the source page's section (`ncu.SOURCE_SECTION`) was collected; and the profiler's
version (`ncu --version`), written beside them."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import model, ncu, readers


class ToolError(RuntimeError):
    """A tool that is missing or that failed."""


def tool(name: str, toolkit: str | None = None) -> str:
    for root in (toolkit, os.environ.get("CUDA_HOME")):
        if root:
            for candidate in (Path(root) / "bin" / name, Path(root) / name):
                if candidate.is_file():
                    return os.path.abspath(candidate)
    if found := shutil.which(name):
        return found
    raise ToolError(f"{name} not found: pass --toolkit DIR, set CUDA_HOME, or put the CUDA toolkit's bin on PATH")


def demangler(toolkit: str | None = None) -> str:
    """The demangler: the toolkit's own `cu++filt` (beside its `cuobjdump`), else `c++filt` on PATH."""
    try:
        return tool("cu++filt", toolkit)
    except ToolError:
        return tool("c++filt", toolkit)


def _run(argv: list[str], cwd: str | None = None) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    if done.returncode:
        raise ToolError(f"{' '.join(argv)}: exit {done.returncode}: {done.stderr.strip()[:400]}")
    return done.stdout


def kernels_of(binary: str, arch: str | None = None, toolkit: str | None = None,
               lines: str | None = None) -> dict[str, model.Kernel]:
    """Every kernel of a binary (an executable, a shared library, a fatbin or a cubin), for one architecture when
    `arch` (`sm_90a`) is given. The binary must hold one image per function name: select an architecture otherwise.
    `lines`: the path of the disassembly with source-line marks (`nvdisasm --print-line-info` of the binary's cubin, built
    with `--generate-line-info`); none: the kernels know no source lines."""
    cuobjdump, nvdisasm, filt = tool("cuobjdump", toolkit), tool("nvdisasm", toolkit), demangler(toolkit)
    select = ["-arch", arch] if arch else []
    sass = _run([cuobjdump, "-sass", *select, binary])
    res = _run([cuobjdump, "-res-usage", *select, binary])
    with tempfile.TemporaryDirectory(prefix="warpline_cubin_") as scratch:
        _run([cuobjdump, "-xelf", "all", *select, str(Path(binary).resolve())], cwd=scratch)
        cubins = sorted(Path(scratch).glob("*.cubin"))
        if not cubins:
            raise ToolError(f"{binary}: no cubin to read the control-flow graph from")
        cfg = "\n".join(_run([nvdisasm, "-bbcfg", "-poff", str(c)]) for c in cubins)
    names = list(readers.sass_functions(sass))
    pretty = subprocess.run([filt], input="\n".join(names) + "\n", capture_output=True, text=True, check=True).stdout
    return model.kernels_from_text(sass, cfg, res, pretty, Path(lines).read_text() if lines else None)


_SUFFIXES = (".sass", ".cfg.dot", ".res", ".names", ".lines")


def fixture_stem(path: str) -> str:
    """A fixture's stem: `dir/kernel` for `dir/kernel`, `dir/kernel.sass`, `dir/kernel.cfg.dot` and the others."""
    for suffix in _SUFFIXES:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return path


def kernels_of_files(stem: str, lines: str | None = None) -> dict[str, model.Kernel]:
    """The kernels of tool outputs saved as `<stem>.sass` (`cuobjdump -sass`), `<stem>.cfg.dot` (`nvdisasm -bbcfg
    -poff`), `<stem>.res` (`cuobjdump -res-usage`) and, optionally, `<stem>.names` (`c++filt` over the functions in
    the listing's order) and `<stem>.lines` (`nvdisasm --print-line-info`: the source lines; `lines` names another file for
    them): no toolkit needed."""
    stem = fixture_stem(stem)

    def read(suffix: str) -> str:
        return Path(stem + suffix).read_text()

    names, marked = Path(stem + ".names"), Path(lines) if lines else Path(stem + ".lines")
    return model.kernels_from_text(read(".sass"), read(".cfg.dot"), read(".res"), names.read_text() if names.exists() else None,
                                   marked.read_text() if marked.exists() or lines else None)


#: the profiler's report in a run's directory, its version beside the exports; `ncu --export` adds the extension its version
#: writes -- `.ncu-rep` up to 2025, `.ncu-repz` (compressed) from 2026 -- and `--import` reads either
REPORT, VERSION = "profile", "ncu-version.txt"
REPORT_SUFFIXES = (".ncu-rep", ".ncu-repz")


def report_of(out: Path) -> Path | None:
    """The report the profiler wrote into the run's directory `out`, or None when it wrote none."""
    found = [out / (REPORT + s) for s in REPORT_SUFFIXES if (out / (REPORT + s)).is_file()]
    if len(found) > 1:
        raise ToolError(f"the run's directory holds two reports ({', '.join(p.name for p in found)}); a run writes one")
    return found[0] if found else None


def profile_items(specs: list[str]) -> tuple[list[str], list[str]]:
    """The sections and the metrics of `--profile SPEC`s: each a comma list; an item with `__` is a metric (Nsight Compute's
    names are `unit__counter...`), any other a section identifier (`SourceCounters`, `MemoryWorkloadAnalysis_Tables`)."""
    items = [item.strip() for spec in specs for item in spec.split(",")]
    if not all(items):
        raise ToolError(f"{','.join(specs)!r} holds an empty item (a section identifier or a metric is expected)")
    return [i for i in items if "__" not in i], [i for i in items if "__" in i]


def profile_command(out: Path, specs: list[str], toolkit: str | None = None) -> list[str]:
    """The prefix a task runs under to be profiled into the run's directory `out`: only its marked regions, the sections and
    metrics of `specs`."""
    sections, metrics = profile_items(specs)
    return [tool("ncu", toolkit), "--profile-from-start", "off", "--export", str(out / REPORT),
            *[arg for s in sections for arg in ("--section", s)], *(["--metrics", ",".join(metrics)] if metrics else [])]


def profiler_version(toolkit: str | None = None) -> str:
    """The profiler's version, as `ncu --version` states it (`2025.3.0.0 (build 36273991) (public-release)`)."""
    lines = [line for line in _run([tool("ncu", toolkit), "--version"]).splitlines() if line.startswith("Version ")]
    if len(lines) != 1:
        raise ToolError("ncu --version states no `Version ...` line")
    return lines[0][len("Version "):].strip()


def export(out: Path, specs: list[str], toolkit: str | None = None) -> tuple[list[Path], str]:
    """The pages of the report in `out` as files beside it, `details.csv` and, where the source page's section was collected,
    `source.<I>.csv` a result I; and the profiler's version, written as `ncu-version.txt`. A run that captured nothing (no
    kernel between the task's profiler start and stop) has no report, and is refused."""
    report = report_of(out)
    if report is None:
        raise ToolError(f"the profiler captured nothing (no {REPORT}{'/'.join(REPORT_SUFFIXES)} was written): no kernel ran "
                        f"inside a marked region of the task (`warpline::run::Region`, or cudaProfilerStart/cudaProfilerStop)")
    ncu_bin = tool("ncu", toolkit)
    details = _run([ncu_bin, "--import", str(report), *ncu.PAGES["details"]])
    written = [out / "details.csv"]
    written[0].write_text(details)
    if ncu.SOURCE_SECTION in profile_items(specs)[0]:
        for i, _ in enumerate(ncu.details_results(details)):
            page = out / f"source.{i}.csv"
            page.write_text(_run([ncu_bin, "--import", str(report), *ncu.PAGES["source"], "--launch-skip", str(i),
                                  "--launch-count", "1"]))
            written.append(page)
    version = profiler_version(toolkit)
    (out / VERSION).write_text(version + "\n")
    return written, version
