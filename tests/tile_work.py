"""A WORK FUNCTION (docs/floor.md#the-work-function): the graph's `sample::tile()` as a function of its cell, `{"n": windows}`,
the same work `tests/floor/tile.work.json` declares with expressions, here computed: `//:tile_floor_function_bound` and the
floor's tests call it by its name, `tile_work:work`."""


def work(cell: dict) -> dict:
    n = int(cell["n"])
    return {"format": "warpline-work/1", "kernel": "^sample::tile\\(",
            "phases": [
                {"role": "producer", "phase": "tick", "work": {"issue": {"control": n}}},
                {"role": "producer", "phase": "load", "work": {"hbm": {"read": n * 8000}, "issue": {"global": n * 16}}},
                {"role": "producer", "phase": "compute", "work": {"issue": {"float": n * 200}}},
                {"role": "producer", "phase": "drain", "work": {}},
                {"role": "consumer", "phase": "tick", "work": {"issue": {"control": n}}},
                {"role": "consumer", "phase": "serve", "work": {"hbm": {"read": n * 2000}, "issue": {"float": n * 400}}},
                {"role": "consumer", "phase": "idle", "work": {"hbm": {"read": n * 1000}}}],
            "overlap": [["producer/load", "consumer/tick"], ["producer/compute", "consumer/serve"],
                        ["producer/drain", "consumer/idle"]],
            "serial": {"producer/tick": "the window's anchor: ten cycles a window, before the load is issued"}}
