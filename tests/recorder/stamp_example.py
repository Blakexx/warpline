"""The stamp example's claim, on the pinned toolkit's own build of it: with both switches off, the kernel holds no clock
read, and the switched-on ones do. Argument: the example's `cuobjdump -sass` text."""

import re
import sys
from pathlib import Path

CLOCK = re.compile(r"CLOCK|GLOBALTIMER|CS2R")


def main(path):
    functions = re.split(r"(?=^\s*Function : )", Path(path).read_text(), flags=re.M)
    clocks = {}
    for f in functions:
        m = re.match(r"\s*Function : (\S+)", f)
        inst = re.search(r"ILb(\d)ELb(\d)", m.group(1)) if m else None
        if inst:
            clocks[inst.groups()] = len(CLOCK.findall(f))
    if clocks.get(("0", "0")) != 0:
        print(f"the plain kernel holds clock reads: {clocks}", file=sys.stderr)
        return 1
    if not all(n > 0 for k, n in clocks.items() if k != ("0", "0")) or len(clocks) < 2:
        print(f"a stamped kernel holds no clock read: {clocks}", file=sys.stderr)
        return 1
    print(f"clock reads by <Coarse, Detail>: {clocks}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
