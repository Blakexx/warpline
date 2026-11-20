"""THE THIN RUNNER: invokes the toolkit's tools on a binary and hands their text to the readers. Nothing here parses;
it is the only module that starts a process, and the one a test without a toolkit does not use.

The toolkit is found in this order: the `--toolkit` argument (a toolkit's root or its `bin`), `CUDA_HOME`, `PATH`."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import model, readers


class ToolError(RuntimeError):
    """A tool that is missing or that failed."""


def tool(name: str, toolkit: str | None = None) -> str:
    for root in (toolkit, os.environ.get("CUDA_HOME")):
        if root:
            for candidate in (Path(root) / "bin" / name, Path(root) / name):
                if candidate.is_file():
                    return str(candidate)
    if found := shutil.which(name):
        return found
    raise ToolError(f"{name} not found: pass --toolkit DIR, set CUDA_HOME, or put the CUDA toolkit's bin on PATH")


def _run(argv: list[str], cwd: str | None = None) -> str:
    done = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    if done.returncode:
        raise ToolError(f"{' '.join(argv)}: exit {done.returncode}: {done.stderr.strip()[:400]}")
    return done.stdout


def kernels_of(binary: str, arch: str | None = None, toolkit: str | None = None) -> dict[str, model.Kernel]:
    """Every kernel of a binary (an executable, a shared library, a fatbin or a cubin), for one architecture when
    `arch` (`sm_90a`) is given. The binary must hold one image per function name: select an architecture otherwise."""
    cuobjdump, nvdisasm, filt = tool("cuobjdump", toolkit), tool("nvdisasm", toolkit), tool("c++filt")
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
    return model.kernels_from_text(sass, cfg, res, pretty)


_SUFFIXES = (".sass", ".cfg.dot", ".res", ".names")


def fixture_stem(path: str) -> str:
    """A fixture's stem: `dir/kernel` for `dir/kernel`, `dir/kernel.sass`, `dir/kernel.cfg.dot` and the others."""
    for suffix in _SUFFIXES:
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return path


def kernels_of_files(stem: str) -> dict[str, model.Kernel]:
    """The kernels of tool outputs saved as `<stem>.sass` (`cuobjdump -sass`), `<stem>.cfg.dot` (`nvdisasm -bbcfg
    -poff`), `<stem>.res` (`cuobjdump -res-usage`) and, optionally, `<stem>.names` (`c++filt` over the functions in
    the listing's order): no toolkit needed."""
    stem = fixture_stem(stem)

    def read(suffix: str) -> str:
        return Path(stem + suffix).read_text()

    names = Path(stem + ".names")
    return model.kernels_from_text(read(".sass"), read(".cfg.dot"), read(".res"), names.read_text() if names.exists() else None)
