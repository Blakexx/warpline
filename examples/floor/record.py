"""RECORD THE FLOOR'S FIXTURES on a card (docs/floor.md#the-fixtures): the rates table the probes write, and for each example
of `floor_examples.cu` a stamped run's manifest and a profiled run's counters, into `tests/floor/fixtures/<device>/` of the
checkout. `//:floor_fixtures_<device>` then holds the floor against them, with no GPU.

    bazel run //:record_floor_fixtures                 # on a machine with a GPU, the profiler's counters granted
    bazel run //:record_floor_fixtures -- --out DIR    # elsewhere than the checkout's fixtures

It runs `probes/rates.bin` once, then each example twice, both by the run layer (`warpline run`): stamped (`--stamps`: its
run manifest, `launch/ns` and the phases) and profiled (`--profile` with every metric a row of the table names, beside the
elapsed cycles: its counters), the profiler the pinned Nsight Compute. Files: `rates.json`; `<example>/run.manifest.json`,
`<example>/counters.csv` (the details page) and `<example>/parameters.json` (the launch's CTAs, one an SM)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from warpline import command, floor

#: the examples: their argument to the program, and the place a run manifest names (the kernel's demangled name)
EXAMPLES = {
    "stream": "floor_examples::stream<true>(warpline::Entry*, uint4 const*, unsigned int*)",
    "shared": "floor_examples::shared<true>(warpline::Entry*, unsigned int*)",
    "mma": "floor_examples::mma<true>(warpline::Entry*, float*)",
    "serial": "floor_examples::serial<true>(warpline::Entry*, uint4 const*, unsigned int*)",
}
#: the counters every profiled run collects beside the rows' own: the SMs' elapsed cycles (a row's share is over them) and
#: the launch's time
ELAPSED = ("sm__cycles_elapsed.max", "gpu__time_duration.sum")
#: the section collected beside them: its rules give the details page the rule columns the run's reader requires (a page of
#: metrics alone has none)
SECTION = "SpeedOfLight"


def slug(device: str) -> str:
    """A device's directory name: its name past the vendor's word, lower case, `-` between words (`h100-80gb-hbm3`)."""
    return re.sub(r"[^a-z0-9]+", "-", device.lower().replace("nvidia", "")).strip("-")


def warpline(*argv: str) -> None:
    code = command.main(list(argv))
    if code:
        raise SystemExit(f"record_floor_fixtures: `warpline {' '.join(argv)}` exited {code}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="record_floor_fixtures", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rates-bin", required=True, help="the probes' program (the target's own argument)")
    ap.add_argument("--examples-bin", required=True, help="the examples' program (the target's own argument)")
    ap.add_argument("--ncu", required=True, help="the pinned Nsight Compute's `ncu` (the target's own argument)")
    ap.add_argument("--out", help="the device's fixture directory (default: the checkout's tests/floor/fixtures/<device>)")
    a = ap.parse_args(argv)
    rates_bin, examples_bin, ncu = (str(Path(p).resolve()) for p in (a.rates_bin, a.examples_bin, a.ncu))
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "rates.json"
        subprocess.run([rates_bin, str(table)], check=True)
        rates = floor.load_rates(table.read_text())
        checkout = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
        if not a.out and not checkout:
            raise SystemExit("record_floor_fixtures: give --out, or run it with `bazel run`")
        out = Path(a.out) if a.out else Path(checkout) / "tests" / "floor" / "fixtures" / slug(rates.device)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(table, out / "rates.json")
        metrics = sorted({m for r in rates.rows if r.counter for m in r.counter["metrics"]} | set(ELAPSED))
        os.environ["CUDA_HOME"] = str(Path(ncu).parent)
        for name, place in EXAMPLES.items():
            here = out / name
            here.mkdir(exist_ok=True)
            stamped, profiled = Path(tmp) / f"{name}.stamps", Path(tmp) / f"{name}.profile"
            warpline("run", "--stamps", "--place", place, "-o", str(stamped), "--", examples_bin, name)
            warpline("run", "--profile", ",".join([SECTION, *metrics]), "--place", place, "-o", str(profiled), "--",
                     examples_bin, name)
            shutil.copy(stamped / "run.manifest.json", here / "run.manifest.json")
            shutil.copy(profiled / "details.csv", here / "counters.csv")
            (here / "parameters.json").write_text(json.dumps({"ctas": rates.sms}) + "\n")
        print(f"wrote {out}: rates.json and {', '.join(EXAMPLES)}; `bazel test //:floor_fixtures_{slug(rates.device)}` holds "
              f"the floor against them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
