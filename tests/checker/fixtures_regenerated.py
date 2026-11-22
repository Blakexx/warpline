"""A kernel built with the pinned toolkit reproduces the checker's committed fixtures.

Arguments: `<name>=<regenerated dir>/<name>` pairs are not used; each argument is `kind:regenerated:committed`. The
committed files are the contract: a regenerated one that differs fails, and the fix is the kernel or the toolkit pin, never
a quiet update of the fixture. Two spellings are not the toolkit's compile: ptxas's `Compile time` lines vary run to run
(dropped), and `cu++filt` writes `float *` and `unsigned int const*` where GNU c++filt (which made the committed
`kernel.names`) writes `float*` and `const unsigned int*` (compared without spaces and `const`)."""

import re
import sys
from pathlib import Path


def canon(kind, text):
    if kind == "ptxas":
        return [l for l in text.splitlines() if "Compile time" not in l]
    if kind == "names":
        return [re.sub(r"\s|const", "", l) for l in text.splitlines()]
    return text.splitlines()


def main(argv):
    bad = []
    for arg in argv:
        kind, got, want = arg.split(":")
        if canon(kind, Path(got).read_text()) != canon(kind, Path(want).read_text()):
            bad.append(f"{want}: differs from the pinned toolkit's output {got}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"{len(argv)} fixtures reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
