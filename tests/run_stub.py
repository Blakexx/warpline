"""Two stand-ins for `tests/test_run.py`, so the runner is tested with no GPU and no profiler.

    run_stub.py task [--record SRC NAME]... [--note FILE] [--exit N] [ARG...]
        A user's task: copies each SRC into the run's record directory (WARPLINE_RECORD_DIR) as NAME.record.json, writes
        its own arguments and that variable to FILE, and exits N (default 0).
    run_stub.py ncu ARG...
        Nsight Compute as the runner calls it, every call appended to STUB_NCU_LOG as a JSON line:
        `--version`; a profiled run (`--profile-from-start off --export P ... <command>`: runs the command, then writes
        P.ncu-rep unless STUB_NCU_CAPTURE is 0); an import (`--import R --page details|source ...`): the page from
        STUB_NCU_DETAILS, or from STUB_NCU_SOURCE<I> for `--launch-skip I`.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def task(argv):
    note, code, rest = None, 0, []
    while argv:
        arg = argv.pop(0)
        if arg == "--record":
            src, name = argv.pop(0), argv.pop(0)
            shutil.copy(src, Path(os.environ["WARPLINE_RECORD_DIR"]) / f"{name}.record.json")
        elif arg == "--note":
            note = argv.pop(0)
        elif arg == "--exit":
            code = int(argv.pop(0))
        else:
            rest.append(arg)
    if note:
        Path(note).write_text(json.dumps({"args": rest, "record_dir": os.environ.get("WARPLINE_RECORD_DIR")}))
    return code


def ncu(argv):
    with open(os.environ["STUB_NCU_LOG"], "a") as log:
        log.write(json.dumps(argv) + "\n")
    if argv == ["--version"]:
        print("NVIDIA (R) Nsight Compute Command Line Profiler\nCopyright (c) 2018-2025 NVIDIA Corporation\n"
              "Version 2025.3.0.0 (build 36273991) (public-release)")
        return 0
    if argv[:1] == ["--import"]:
        if "details" in argv:
            sys.stdout.write(Path(os.environ["STUB_NCU_DETAILS"]).read_text())
        else:
            sys.stdout.write(Path(os.environ["STUB_NCU_SOURCE" + argv[argv.index("--launch-skip") + 1]]).read_text())
        return 0
    assert argv[:2] == ["--profile-from-start", "off"], argv
    report = argv[argv.index("--export") + 1]
    i = 4
    while argv[i] in ("--section", "--metrics"):
        i += 2
    done = subprocess.run(argv[i:])
    if os.environ.get("STUB_NCU_CAPTURE", "1") != "0":
        Path(report + ".ncu-rep").write_text("a report\n")
    return done.returncode


if __name__ == "__main__":
    sys.exit({"task": task, "ncu": ncu}[sys.argv[1]](sys.argv[2:]))
