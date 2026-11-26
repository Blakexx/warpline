"""A line inserted above a loop moves no field. The build wrote the manifest of `held_many` and of the same kernel with lines
inserted above its loops (`examples/checker/shifted/held_many.cu`); both are read from the graph, and their field paths and values
are the same (a hash over them is the same), while the loops' `at` notes, the one place a line number is written, differ.

Arguments: the manifest of `held_many` and the manifest of the shifted one (runfiles-relative paths)."""
import sys
import unittest
from pathlib import Path

from warpline import checkers, manifest

ORIGINAL, SHIFTED = sys.argv[1:3]
sys.argv = sys.argv[:1]


def load(path):
    return manifest.loads(Path(path).read_text())


class Shifted(unittest.TestCase):
    def test_no_field_path_and_no_value_moves(self):
        a, b = load(ORIGINAL), load(SHIFTED)
        (pa,), (pb,) = [[p for p in m.places if p.kernel] for m in (a, b)]
        self.assertEqual(manifest.fields(pa), manifest.fields(pb))
        self.assertTrue([p for p in manifest.fields(pa) if p.startswith("loops/")])
        self.assertFalse([p for p in manifest.fields(pa) if any(c.isdigit() and ":" in p for c in p)])

    def test_a_hash_over_the_loops_is_the_same(self):
        hashed = lambda m: checkers.digest([(p.place, path, v) for p in m.places if p.kernel
                                            for path, v in sorted(manifest.fields(p).items()) if path.startswith("loops/")])
        self.assertEqual(hashed(load(ORIGINAL)), hashed(load(SHIFTED)))

    def test_only_the_note_of_the_line_differs(self):
        notes = lambda m: [loop["at"] for p in m.places if p.kernel for loop in p.figures["loops"].values()]
        self.assertNotEqual(notes(load(ORIGINAL)), notes(load(SHIFTED)))
        self.assertEqual([n.split(":")[0] for n in notes(load(ORIGINAL))], [n.split(":")[0] for n in notes(load(SHIFTED))])


if __name__ == "__main__":
    unittest.main()
