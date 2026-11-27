"""The pinned Nsight Compute (`@nsight_compute`, fsla-devtools' `pins/nsight_compute_pin.json`) is the release the pin names: its
`ncu --version` states the pin's version string. No GPU."""
import json
import subprocess
import sys
import unittest

NCU, PIN = sys.argv[1:3]


class PinnedProfiler(unittest.TestCase):
    def test_ncu_is_the_pins_release(self):
        want = json.load(open(PIN))["version_string"]
        lines = subprocess.run([NCU, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertIn(want, [line.strip() for line in lines])


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
