import unittest

from warpline.recorder import check, reader
from warpline.recorder.record import from_dict
from warpline.recorder.importers.stamped_clocks import convert_row

PHASE, WAITS, WAITED, ARRIVES, ISSUES = 1, 2, 3, 4, 5


def ev(t, kind, site):
    return [t, kind << 16 | site]


class Importer(unittest.TestCase):
    def row(self):
        legend = {"100": "slot landed", "101": "tile ready", "32769": "pair meet", "49152": "work",
                  "49153": "own steps"}
        w0 = [ev(0, PHASE, 49152), ev(5, ISSUES, 100), ev(6, WAITS, 100), ev(30, WAITED, 100), ev(31, ARRIVES, 101),
              ev(32, WAITS, 32769), ev(40, WAITED, 32769), ev(41, WAITS, 49153), ev(50, WAITED, 49153)]
        w1 = [ev(0, PHASE, 49152), ev(35, WAITS, 32769), ev(40, WAITED, 32769), ev(41, WAITS, 101),
              ev(60, WAITED, 101)]
        return {"variant": "v", "roles": ["a", "b"], "legend": legend,
                "ctas": [{"cta": 0, "warps": [{"count": len(w0), "stamps": sum(w0, [])},
                                              {"count": len(w1) + 2, "stamps": sum(w1, [])}]}]}

    def test_maps_the_kinds_onto_the_convention(self):
        doc, unmapped = convert_row(self.row())
        rec = from_dict(doc)
        self.assertEqual(unmapped, [])
        names = set(rec.names.values())
        self.assertIn("phase.work.begin", names)
        self.assertIn("op.slot_landed_copy.issue", names)
        self.assertIn("barrier.pair_meet.arrive", names)  # synthesized at the wait's begin
        self.assertIn("barrier.tile_ready.arrive", names)
        self.assertEqual(rec.facts["barriers"]["own_steps"]["count"], 0)
        self.assertEqual(rec.facts["barriers"]["slot_landed"]["stamped"], 0)
        self.assertEqual(rec.facts["barriers"]["pair_meet"]["count"], 2)
        self.assertEqual(rec.dropped, {"cta0/warp1": 2})

    def test_the_imported_record_reads(self):
        rec = from_dict(convert_row(self.row())[0])
        ws = {(w.lane, w.target): w for w in reader.waits(rec)}
        self.assertEqual(ws[("cta0/warp0", "slot_landed")].release.kind, "unstamped")
        self.assertEqual(ws[("cta0/warp0", "slot_landed")].release.issue.time, 5)
        self.assertEqual(ws[("cta0/warp1", "tile_ready")].release.time, 31)
        self.assertEqual(ws[("cta0/warp0", "pair_meet")].release.time, 35)  # the last to reach it
        self.assertEqual(check.stamps_accounted(rec), [])

    def test_unknown_kinds_are_reported(self):
        row = self.row()
        row["ctas"][0]["warps"][0]["stamps"] += ev(60, 9, 1)
        row["ctas"][0]["warps"][0]["count"] += 1
        _doc, unmapped = convert_row(row)
        self.assertEqual(unmapped, ["kind 9: 1 event(s)"])


if __name__ == "__main__":
    unittest.main()
