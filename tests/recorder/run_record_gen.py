"""A stamped run of one kernel (`helpers.stamped_run`: a producer and a consumer, every wait stamped, the launch's two edges with
ties) written as a record to the path given: what the graph's run manifest (`//:run_manifest_example`) is made of."""
import sys

from tests.recorder.helpers import stamped_run
from warpline.recorder import record

if __name__ == "__main__":
    record.dump(stamped_run(), sys.argv[1])
