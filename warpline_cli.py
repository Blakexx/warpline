"""`bazel run //:cli -- <command> ...`: the command, run from the directory it was invoked in."""
import os
import sys

from warpline.__main__ import main

if __name__ == "__main__":
    os.chdir(os.environ.get("BUILD_WORKING_DIRECTORY", "."))
    sys.exit(main())
