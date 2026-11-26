"""The manifest the build's first action (`warpline_check`) wrote for each example kernel are the manifest of the committed fixtures
(the same kernels, read from the toolkit's output rather than from the saved text), and the declarations that pass in the
build refuse when each is made one short: kernel, rule, count, allowed and the declaration's reason.

Arguments, one a kernel: `NAME:MANIFEST:DECLARATIONS:FIXTURE_STEM` (runfiles-relative paths)."""
import json
import sys
import unittest
from pathlib import Path

from warpline import declare, manifest
from warpline.checker import manifest as collection, notices, runner
from warpline.checker.rules import registry

CASES = [a.split(":") for a in sys.argv[1:]]
sys.argv = sys.argv[:1]


def load(path):
    return manifest.loads(Path(path).read_text())


class GraphManifest(unittest.TestCase):
    def test_the_graphs_manifest_are_the_fixtures_manifest(self):
        for name, manifest_path, _, stem in CASES:
            with self.subTest(kernel=name):
                made = load(manifest_path)
                log = Path(stem + ".ptxas")                  # `steps` has the build's messages, read in the graph as well
                saved = collection.collect(list(runner.kernels_of_files(stem).values()), registry(),
                                        notices=notices.read_notices(log.read_text()) if log.exists() else None)
                # the graph demangles with the toolkit's `cu++filt` (`float *`), the fixtures were made with GNU `c++filt`
                # (`float*`): the kernels are matched by the name before the parameter list
                counts = lambda fs: {p.place.split("(")[0]: {n: r.count for n, r in p.rules.items()} for p in fs.places if p.kernel}
                self.assertEqual(counts(made), counts(saved))
                self.assertTrue(counts(made))

    def test_the_declarations_hold_the_build_and_each_refuses_one_short(self):
        names = collection.rule_names()
        for name, manifest_path, decl_path, _ in CASES:
            doc = json.loads(Path(decl_path).read_text())
            made = load(manifest_path)
            self.assertEqual(declare.compare(made, declare.load_declarations(json.dumps(doc), registry(), names)).refusals, [], name)
            for i, entry in enumerate(doc["declarations"]):
                if not entry["allow"]:
                    continue
                short = {"declarations": [{**e, "allow": e["allow"] - 1} if j == i else e for j, e in enumerate(doc["declarations"])]}
                refusals = declare.compare(made, declare.load_declarations(json.dumps(short), registry(), names)).refusals
                with self.subTest(kernel=name, declaration=i):
                    self.assertEqual(len(refusals), 1, [r.lines()[0] for r in refusals])
                    (r,) = refusals
                    self.assertEqual((r.rule, r.allowed, r.count, r.reason),
                                     (entry["rule"], entry["allow"] - 1, entry["allow"],
                                      entry["reason"]))

    def test_undeclared_is_refused_naming_the_kernel(self):
        for name, manifest_path, _, _ in CASES:
            got = declare.compare(load(manifest_path), []).refusals
            self.assertEqual({r.reason for r in got}, {None} if got else set(), name)


if __name__ == "__main__":
    unittest.main()
