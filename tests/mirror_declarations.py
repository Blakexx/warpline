"""Every tracked path of this checkout is declared in `.github/mirror/declarations.json`: the mirror tool's `--check`
(rola_devtools.mirror, at the commit the mirror workflow installs), run in the checkout over its git index so a path
`git add`ed and undeclared fails here, before the push the workflow would refuse."""
import os
import sys
from pathlib import Path

from rola_devtools import mirror


def main() -> int:
    """The tool's exit status; the checkout is where this test's `MODULE.bazel` input links to."""
    anchor = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "MODULE.bazel"
    os.chdir(anchor.resolve().parent)
    return mirror.main(["--check"])


if __name__ == "__main__":
    sys.exit(main())
