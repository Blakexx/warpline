"""THE THIN RUNNER: invokes the toolkit's tools on a binary and hands their text to the readers, and puts a task under the
profiler for `warpline run --profile` and exports what it captured, and keeps the cubins it extracts. Nothing here parses text but
the details page's list of results (`listable` edits one attribute of a cubin's ELF); it is the only module of the checker that
starts a process, and the one a test without a toolkit does not use.

The toolkit is found in this order: the `--toolkit` argument (a toolkit's root or its `bin`), `CUDA_HOME`, `PATH`. The
demangler is the toolkit's `cu++filt`, else `c++filt`. The profiler is Nsight Compute's `ncu`, found the same way.

THE PROFILED RUN. `profile_command` is the prefix the task runs under: `ncu --profile-from-start off`, so the profiler captures
the kernels between the task's `cudaProfilerStart` and `cudaProfilerStop` (`warpline::run::Region`, include/warpline/run.hpp)
and nothing else, the sections and metrics asked for, and the report written as `DIR/profile.ncu-rep` (`.ncu-repz` from Nsight Compute 2026). `export` prints the
report's pages with the flags its readers read (`ncu.PAGES`): the details page once, and the source page a result
(`--launch-skip I --launch-count 1`) where the source page's section (`ncu.SOURCE_SECTION`) was collected; and the profiler's
version (`ncu --version`), written beside them.

THE CUBIN CACHE. A tool that reads one image of a built binary more than once a build (a frame map, then the life ranges)
extracts its images once a build: `cubins(binary, cache)` writes every ELF image `cuobjdump -xelf` gives into `cache`, under the
binary's content hash, so the directory names the binary it came from; the cache keeps `KEEP_BUILDS` builds and prunes a build
(or a crashed extraction's staging) unused for `PRUNE_AFTER_S`. `cubin(binary, cache, member, arch)` is the one image whose name
holds a member and an architecture (a fat binary holds an image an architecture: a launch is the image the device ran).

THE LISTINGS a reader of a cubin's registers and source reads: `inline_listing(cubin)` is `nvdisasm --print-line-info-inline`
(`readers.inline_frames`), `life_listing(cubin, scratch)` is `nvdisasm --print-life-ranges` (`readers.life_ranges`), of a copy whose
255-register functions say 254 where the cubin has one (`listable`: nvdisasm refuses to print a function's life ranges at 255
registers); `kernels_of(binary, ..., life=True)` reads the life ranges into the kernels' marks (`Kernel.marks`)."""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
import time
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
               lines: str | None = None, life: bool = False) -> dict[str, model.Kernel]:
    """Every kernel of a binary (an executable, a shared library, a fatbin or a cubin), for one architecture when
    `arch` (`sm_90a`) is given. The binary must hold one image per function name: select an architecture otherwise.
    `lines`: the path of the disassembly with source-line marks (`nvdisasm --print-line-info` of the binary's cubin, built
    with `--generate-line-info`); none: the kernels know no source lines. `life`: each kernel's marks are its images' life
    ranges (`life_listing`), so its registers are the listing's, not its operands'."""
    cuobjdump, nvdisasm, filt = tool("cuobjdump", toolkit), tool("nvdisasm", toolkit), demangler(toolkit)
    select = ["-arch", arch] if arch else []
    sass = _run([cuobjdump, "-sass", *select, binary])
    res = _run([cuobjdump, "-res-usage", *select, binary])
    with tempfile.TemporaryDirectory(prefix="warpline_cubin_") as scratch:
        if Path(binary).suffix == ".cubin":
            cubins = [Path(binary)]
        else:
            _run([cuobjdump, "-xelf", "all", *select, str(Path(binary).resolve())], cwd=scratch)
            cubins = sorted(Path(scratch).glob("*.cubin"))
        if not cubins:
            raise ToolError(f"{binary}: no cubin to read the control-flow graph from")
        cfg = "\n".join(_run([nvdisasm, "-bbcfg", "-poff", str(c)]) for c in cubins)
        ranges = "\n".join(life_listing(c, Path(scratch), toolkit) for c in cubins) if life else None
    names = list(readers.sass_functions(sass))
    pretty = subprocess.run([filt], input="\n".join(names) + "\n", capture_output=True, text=True, check=True).stdout
    return model.kernels_from_text(sass, cfg, res, pretty, Path(lines).read_text() if lines else None, ranges)


#: the extracted builds the cache keeps, the one just used among them, by last use
KEEP_BUILDS = 4
#: a build (or a crashed extraction's staging) unused this long may be pruned: a reader of it has finished
PRUNE_AFTER_S = 3600


def cubins(binary: Path, cache: Path, toolkit: str | None = None) -> list[Path]:
    """The cubins of `binary`: itself when it is one, else every ELF image `cuobjdump -xelf all` writes out of it, in `cache`
    under the binary's content hash -- extracted once a build (concurrent readers race to one directory, and the loser's copy
    is dropped), so the directory names the binary it came from and the cache holds a few builds, not one copy a run."""
    binary = Path(binary)
    if binary.suffix == ".cubin":
        return [binary]
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with binary.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    out = cache / digest.hexdigest()[:16]
    if not out.is_dir():
        staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=cache))
        _run([tool("cuobjdump", toolkit), "-xelf", "all", str(binary.resolve())], cwd=str(staging))
        try:
            staging.rename(out)
        except OSError:
            if not out.is_dir():
                raise
            shutil.rmtree(staging)
    os.utime(out)
    _prune(cache, out)
    return sorted(out.iterdir())


def _prune(cache: Path, used: Path) -> None:
    """Every build past the newest `KEEP_BUILDS` and every staging directory, once unused for `PRUNE_AFTER_S`."""
    idle = time.time() - PRUNE_AFTER_S
    seen = []
    for d in cache.iterdir():
        try:
            if d.is_dir() and d != used:
                seen.append((d.stat().st_mtime, d))
        except FileNotFoundError:
            continue
    kept = 0
    for mtime, d in sorted(seen, reverse=True):
        if not d.name.startswith(".") and kept < KEEP_BUILDS - 1:
            kept += 1
        elif mtime < idle:
            shutil.rmtree(d, ignore_errors=True)


def cubin(binary: Path, cache: Path, member: str, arch: str, toolkit: str | None = None) -> Path:
    """The one cubin of `binary` whose name holds `member` and `arch` (`sm_86`); refused where that is not exactly one image. A
    fat binary holds one image an architecture, and a reader attributes a launch to the image the device ran: the first match
    by name alone is another architecture's."""
    found = [image for image in cubins(binary, cache, toolkit) if member in image.name and arch in image.name]
    if len(found) != 1:
        raise ToolError(f"{member} for {arch} names {len(found)} ELF images in {Path(binary).name}: one is read")
    return found[0]


def inline_listing(cubin: Path, toolkit: str | None = None) -> str:
    """`nvdisasm --print-line-info-inline` of a cubin: its instructions with their inline chains (`readers.inline_frames`)."""
    return _run([tool("nvdisasm", toolkit), "--print-line-info-inline", str(cubin)])


#: nvdisasm prints no life ranges for a function at 255 registers (`Invalid register count`): such a function is read from a
#: copy whose EIATTR_REGCOUNT says 254 -- the listing's columns still reach R253, past the R252 a 255-register function's code
#: reaches (two of the count are reserved)
EIATTR_REGCOUNT, EIFMT_SVAL, MAX_LISTED = 0x2F, 0x04, 254


def listable(cubin: Path, scratch: Path) -> Path:
    """`cubin`, or a copy in `scratch` whose 255-register functions' EIATTR_REGCOUNT says `MAX_LISTED` (`.nv.info`'s attributes:
    a format byte, an id byte, a u16 size, then for EIFMT_SVAL the function's symbol and the value). The attribute only sizes
    the listing."""
    data = bytearray(Path(cubin).read_bytes())
    shoff, = struct.unpack_from("<Q", data, 0x28)
    entsize, count, names = struct.unpack_from("<HHH", data, 0x3A)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * entsize) for i in range(count)]
    strtab = headers[names][4]
    patched = False
    for h in headers:
        if data[strtab + h[0]: data.index(b"\0", strtab + h[0])] != b".nv.info":
            continue
        p, end = h[4], h[4] + h[5]
        while p < end:
            fmt, attr, size = data[p], data[p + 1], struct.unpack_from("<H", data, p + 2)[0]
            if fmt == EIFMT_SVAL and attr == EIATTR_REGCOUNT and struct.unpack_from("<I", data, p + 8)[0] > MAX_LISTED:
                struct.pack_into("<I", data, p + 8, MAX_LISTED)
                patched = True
            p += 4 + (size if fmt == EIFMT_SVAL else 0)
    if not patched:
        return Path(cubin)
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f"{Path(cubin).stem}.listable.cubin"
    out.write_bytes(bytes(data))
    return out


def life_listing(cubin: Path, scratch: Path, toolkit: str | None = None) -> str:
    """`nvdisasm --print-life-ranges` of a cubin (`readers.life_ranges`), of its `listable` copy in `scratch`."""
    return _run([tool("nvdisasm", toolkit), "--print-life-ranges", str(listable(cubin, scratch))])


_SUFFIXES = (".sass", ".cfg.dot", ".res", ".names", ".lines", ".life")


def fixture_stem(path: str) -> str:
    """A fixture's stem: `dir/kernel` for `dir/kernel`, `dir/kernel.sass`, `dir/kernel.cfg.dot` and the others."""
    for suffix in _SUFFIXES:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return path


def kernels_of_files(stem: str, lines: str | None = None) -> dict[str, model.Kernel]:
    """The kernels of tool outputs saved as `<stem>.sass` (`cuobjdump -sass`), `<stem>.cfg.dot` (`nvdisasm -bbcfg
    -poff`), `<stem>.res` (`cuobjdump -res-usage`) and, optionally, `<stem>.names` (`c++filt` over the functions in
    the listing's order), `<stem>.lines` (`nvdisasm --print-line-info`: the source lines; `lines` names another file for
    them) and `<stem>.life` (`nvdisasm --print-life-ranges`: the registers' marks): no toolkit needed."""
    stem = fixture_stem(stem)

    def read(suffix: str) -> str:
        return Path(stem + suffix).read_text()

    names, marked, life = Path(stem + ".names"), Path(lines) if lines else Path(stem + ".lines"), Path(stem + ".life")
    return model.kernels_from_text(read(".sass"), read(".cfg.dot"), read(".res"), names.read_text() if names.exists() else None,
                                   marked.read_text() if marked.exists() or lines else None,
                                   life.read_text() if life.exists() else None)


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
