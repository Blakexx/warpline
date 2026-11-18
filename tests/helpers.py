"""Build small records from a compact description."""
from __future__ import annotations

from warpline import record


def make(lanes, stamps, facts=None, dropped=None, issued=None):
    """lanes: {id: role}; stamps: (lane, name, time[, payload]); facts: the declared facts."""
    names: dict[str, int] = {}
    rows = []
    for s in stamps:
        nid = names.setdefault(s[1], len(names))
        rows.append([s[0], nid, s[2]] + ([s[3]] if len(s) > 3 else []))
    doc = {
        "format": "warpline/0",
        "clock": {"unit": "cycles", "hz": None},
        "names": {str(i): n for n, i in names.items()},
        "lanes": [{"id": k, "role": v, **({"issued": issued[k]} if issued and k in issued else {})}
                  for k, v in lanes.items()],
        "stamps": rows,
        "dropped": dropped or {},
        "facts": facts or {},
    }
    return record.from_dict(doc)


def three_lane():
    """Three lanes in one scope `k`; the answer of its critical path is stated in test_path.

    c: region load 0..50; arrives at Y and X at 50; waits on X 50..90.
    b: region work 0..80; waits on Y 20..70 (released by c's arrival at 50); arrives at X at 80.
    a: region w 0..100; waits on X 10..90 (X has count 2: c's arrival at 50 and b's at 80 -- b's, the last, releases).
    """
    lanes = {"k/a": "consumer", "k/b": "middle", "k/c": "producer"}
    stamps = [
        ("k/a", "w.begin", 0), ("k/a", "barrier.X.wait.begin", 10), ("k/a", "barrier.X.wait.end", 90),
        ("k/a", "w.end", 100),
        ("k/b", "work.begin", 0), ("k/b", "barrier.Y.wait.begin", 20), ("k/b", "barrier.Y.wait.end", 70),
        ("k/b", "barrier.X.arrive", 80), ("k/b", "work.end", 80),
        ("k/c", "load.begin", 0), ("k/c", "load.end", 50), ("k/c", "barrier.Y.arrive", 50),
        ("k/c", "barrier.X.arrive", 50), ("k/c", "barrier.X.wait.begin", 50), ("k/c", "barrier.X.wait.end", 90),
    ]
    facts = {"barriers": {"X": {"count": 2}, "Y": {"count": 1}}}
    return make(lanes, stamps, facts)
