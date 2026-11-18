import json
import unittest

from warpline import record
from warpline.record import RecordError, StrictError, classify
from tests.helpers import make, three_lane


def doc():
    return {"format": "warpline/0", "clock": {"unit": "cycles", "hz": None}, "names": {"0": "r.begin", "1": "r.end"},
            "lanes": [{"id": "k/w0", "role": "x"}], "stamps": [["k/w0", 0, 1], ["k/w0", 1, 5, 7]],
            "dropped": {}, "facts": {}}


class Loading(unittest.TestCase):
    def test_valid_round_trip(self):
        rec = record.from_dict(doc())
        self.assertEqual(len(rec.stamps), 2)
        self.assertEqual(rec.stamps[1], ("k/w0", 1, 5, 7))
        self.assertEqual(record.loads(json.dumps(rec.to_dict())).stamps, rec.stamps)

    def test_named_errors(self):
        def code(mutate):
            d = doc()
            mutate(d)
            with self.assertRaises(RecordError) as cm:
                record.from_dict(d)
            return cm.exception.code
        self.assertEqual(code(lambda d: d.update(format="x/1")), "format_unsupported")
        self.assertEqual(code(lambda d: d.pop("facts")), "missing_field")
        self.assertEqual(code(lambda d: d["stamps"].append(["nope", 0, 1])), "unknown_lane")
        self.assertEqual(code(lambda d: d["stamps"].append(["k/w0", 9, 1])), "unknown_name_id")
        self.assertEqual(code(lambda d: d["stamps"].append(["k/w0", 0, 1.5])), "bad_time")
        self.assertEqual(code(lambda d: d["stamps"].append(["k/w0", 0, 1, "p"])), "bad_payload")
        self.assertEqual(code(lambda d: d.update(clock={"unit": "ns"})), "bad_clock")
        self.assertEqual(code(lambda d: d.update(dropped={"zz": 1})), "unknown_lane")
        self.assertEqual(code(lambda d: d.update(facts={"barriers": {"B": {}}})), "bad_facts")
        self.assertEqual(code(lambda d: d["names"].update({"2": "r.end"})), "duplicate_name")
        with self.assertRaises(RecordError) as cm:
            record.loads("{not json")
        self.assertEqual(cm.exception.code, "bad_json")

    def test_the_variant_is_an_optional_validated_field_that_round_trips(self):
        self.assertIsNone(record.from_dict(doc()).variant)
        d = doc()
        d["variant"] = "fine"
        rec = record.from_dict(d)
        self.assertEqual(rec.variant, "fine")
        self.assertEqual(record.loads(json.dumps(rec.to_dict())).variant, "fine")
        self.assertNotIn("variant", record.from_dict(doc()).to_dict())
        for bad in ("", "  ", 3, ["fine"]):
            d["variant"] = bad
            with self.assertRaises(RecordError) as cm:
                record.from_dict(d)
            self.assertEqual(cm.exception.code, "bad_variant")

    def test_a_rendezvous_fact_is_validated(self):
        def code(facts):
            d = doc()
            d["facts"] = {"barriers": {"B": facts}}
            with self.assertRaises(RecordError) as cm:
                record.from_dict(d)
            return cm.exception.code
        d = doc()
        d["facts"] = {"barriers": {"B": {"count": 2, "rendezvous": True, "skew": 8}}}
        record.from_dict(d)
        self.assertEqual(code({"count": 2, "rendezvous": 1}), "bad_facts")
        self.assertEqual(code({"count": 2, "skew": 8}), "bad_facts")                         # a skew of a non-rendezvous
        self.assertEqual(code({"count": 2, "rendezvous": True, "skew": -1}), "bad_facts")
        self.assertEqual(code({"count": 2, "rendezvous": True, "stamped": 0}), "bad_facts")  # its arrivals are stamped
        self.assertEqual(code({"count": 2, "rendezvous": True, "issue": "R"}), "bad_facts")
        self.assertEqual(code({"count": 0, "rendezvous": True}), "bad_facts")

    def test_strict_loading_raises_on_findings(self):
        bad = json.dumps(doc())  # r.begin never has an end before a second begin... r.end follows: fine
        record.loads(bad, strict=True)
        d = doc()
        d["names"]["2"] = "launch.end"
        d["stamps"] = [["k/w0", 0, 1], ["k/w0", 2, 5]]  # an unclosed region in a lane that stamps its exit
        with self.assertRaises(StrictError) as cm:
            record.loads(json.dumps(d), strict=True)
        self.assertEqual(cm.exception.findings[0].code, "region_unclosed")

    def test_naming_convention(self):
        self.assertEqual(classify("barrier.full.arrive"), ("arrive", "full"))
        self.assertEqual(classify("barrier.full.wait.begin"), ("wait_begin", "full"))
        self.assertEqual(classify("barrier.full.wait.end"), ("wait_end", "full"))
        self.assertEqual(classify("op.tensor.issue"), ("op_issue", "tensor"))
        self.assertEqual(classify("op.tensor.done"), ("op_done", "tensor"))
        self.assertEqual(classify("write.flag"), ("write", "flag"))
        self.assertEqual(classify("poll.flag.begin"), ("poll_begin", "flag"))
        self.assertEqual(classify("fold.step.begin"), ("region_begin", "fold.step"))
        self.assertEqual(classify("fold.step.end"), ("region_end", "fold.step"))
        self.assertEqual(classify("anything"), ("mark", "anything"))


if __name__ == "__main__":
    unittest.main()
