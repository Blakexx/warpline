import json
import os
import tempfile
import unittest

from warpline import record, variants
from tests.helpers import hump, regular, synth, write_csv

P = 2.6
WAITS = [("k/b", "barrier.Q.wait.begin", 100), ("k/b", "barrier.Q.wait.end", 400)]


def samples(path, period, n=200, seed=1):
    s = {"barrier": synth(hump(0.3, 0.12, 8.0, 1.0), period, n, seed=seed, noise=0.2),
         "wait": synth(hump(0.6, 0.12, 5.0, 1.0), period, n, seed=seed + 1, noise=0.2),
         "mio": synth(hump(0.8, 0.12, 4.0, 1.0), period, n, seed=seed + 2, noise=0.2)}
    return write_csv(path, s)


class Set(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        p = lambda n: os.path.join(self.d.name, n)
        samples(p("plain.csv"), P, seed=1)
        samples(p("a.csv"), P * 1.01, seed=10)
        samples(p("d.csv"), P * 0.98, seed=20)
        samples(p("b.csv"), P * 1.12, seed=30)
        for name, stretch in (("a", 1.0), ("d", 1.02), ("c", 1.0)):
            record.dump(regular(stretch=stretch, extra=WAITS), p(name + ".json"))
        self.manifest = {"kernel": "k", "variants": [
            {"name": "plain", "plain": True, "stamps": None, "samples": ["plain.csv"]},
            {"name": "a", "plain": False, "stamps": "a.json", "samples": ["a.csv"]},
            {"name": "d", "plain": False, "stamps": "d.json", "samples": ["d.csv"]},
            {"name": "b", "plain": False, "stamps": None, "samples": ["b.csv"]},
            {"name": "c", "plain": False, "stamps": "c.json", "samples": []}]}

    def write(self):
        path = os.path.join(self.d.name, "m.json")
        with open(path, "w") as f:
            json.dump(self.manifest, f)
        return path

    def test_cost_over_the_bound_unchecked_and_the_merge(self):
        res = variants.run(self.write())
        by = {v.name: v for v in res.variants}
        self.assertEqual({k: v.status for k, v in by.items()},
                         {"plain": "plain", "a": "ok", "d": "ok", "b": "over", "c": "unchecked"})
        self.assertAlmostEqual(by["a"].ratio, 1.01, delta=0.01)
        self.assertAlmostEqual(by["b"].ratio, 1.12, delta=0.02)
        self.assertIn("slower", by["b"].note)
        self.assertIn("sampling export of THIS variant", by["c"].note)
        # the clock is a measurement of the run: stamped cycles over sampled microseconds
        self.assertAlmostEqual(by["a"].clock, by["a"].period_cycles / by["a"].period_us)
        # only variants that pass and carry stamps are merged: not b (over, no stamps), not c (unchecked)
        self.assertEqual(res.merged_names, ["a", "d"])
        self.assertEqual(res.merged.facts["variants"], {"0": "a", "1": "d"})
        self.assertEqual(set(res.tags), {"a", "d"})
        text = variants.format_set(res)
        for needle in ("UNCHECKED", "OVER, named", "a MEASUREMENT", "left out of the merge", "stamps by variant"):
            self.assertIn(needle, text)

    def test_the_plain_samples_are_laid_on_the_merged_timeline(self):
        res = variants.run(self.write())
        self.assertIsNotNone(res.alignment)          # a variant with waits allows the alignment (accepted or refused)
        self.assertIn("SHIFT", res.overlay_text)
        json.dumps(variants.to_dict(res))

    def test_a_stamped_variant_without_samples_is_never_reported_as_passing(self):
        self.manifest["variants"] = [self.manifest["variants"][0], self.manifest["variants"][4]]
        res = variants.run(self.write())
        c = res.variants[1]
        self.assertEqual(c.status, "unchecked")
        self.assertIsNone(c.ratio)
        self.assertIsNone(res.merged)
        self.assertIn("no variant that passes carries stamps", res.timeline_note)
        text = variants.format_set(res)
        self.assertIn("UNCHECKED", text)
        self.assertNotIn("checked:", text)
        # asked for, an unchecked variant is merged -- and still reported unchecked
        res = variants.run(self.write(), merge_unchecked=True)
        self.assertEqual(res.variants[1].status, "unchecked")
        self.assertIn("nothing to merge", res.timeline_note)

    def test_a_variant_faster_than_the_plain_one_is_not_the_same_kernel_either(self):
        samples(os.path.join(self.d.name, "d.csv"), P * 0.85, seed=21)
        res = variants.run(self.write())
        d = next(v for v in res.variants if v.name == "d")
        self.assertEqual(d.status, "over")
        self.assertIn("faster", d.note)

    def test_stamps_against_stamps_are_compared_in_cycles_with_the_clock_assumed(self):
        self.manifest["variants"] = [
            {"name": "plain", "plain": True, "stamps": "a.json", "samples": []},
            {"name": "d", "plain": False, "stamps": "d.json", "samples": []}]
        res = variants.run(self.write())
        d = res.variants[1]
        self.assertEqual((d.status, d.basis), ("ok", "cycles"))
        self.assertIn("ASSUMED", d.note)
        self.assertAlmostEqual(d.ratio, 1.02, delta=0.005)

    def test_a_manifest_needs_exactly_one_plain_variant(self):
        self.manifest["variants"][1]["plain"] = True
        with self.assertRaises(variants.SetError):
            variants.run(self.write())
        for v in self.manifest["variants"]:
            v["plain"] = False
        with self.assertRaises(variants.SetError):
            variants.run(self.write())

    def test_a_variant_with_nothing_recorded_is_refused(self):
        self.manifest["variants"][3]["samples"] = []
        with self.assertRaises(variants.SetError):
            variants.run(self.write())


if __name__ == "__main__":
    unittest.main()
