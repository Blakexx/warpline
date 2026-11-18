# warpline

A kernel's timeline, read out of stamps.

For any GPU kernel: at each cycle of a window, what is each warp doing, what is each resource carrying and for whom,
what is each warp waiting on, and what ended the wait. An agent or a person reads it to choose the next change.

## The one primitive

A **stamp**: a name and the clock's value, written by a warp at a point of its code. Nothing else is recorded.
Regions, waits, who released whom, what an asynchronous resource carried and the critical path are all computed by
the reader afterwards. See [docs/format.md](docs/format.md).

## The naming convention (the reader's only knowledge of meaning)

| name | meaning |
|---|---|
| `X.begin` / `X.end` | region `X` |
| `barrier.B.arrive`, `barrier.B.wait.begin` / `.end` | arrivals at, and waits on, barrier `B` |
| `op.R.issue` / `op.R.done` | an operation on resource `R`; the payload is its id |
| `write.F`, `poll.F.begin` / `.end` | the stamped write a polled wait watches, and the poll |

Facts declared once about a kernel (a barrier's arrival count, any pairing the names do not give) travel in the record.

## Commands (Python 3 standard library only)

    python -m warpline check    <record> [--strict]     tier-0 logical checks
    python -m warpline path     <record> [--cta C] [--lane L] [--steady REGION --skip N | --from T --to T]
                                                         critical path, the blocking waits with their releasers, slack
    python -m warpline step     <record> --at T [--cta C]  what every lane and resource is doing at time T
    python -m warpline perfetto <record> -o out.json     Chrome trace-event JSON for Perfetto's UI

Perfetto output: a track a lane (regions as nested slices, waits as slices named by their barrier), a track a resource
(its operations as async slices), and flow arrows from the stamp that released a wait to the wait it ended.

`python -m warpline.importers.stamped_clocks <input.json> -o <dir>` reads a record of per-lane (clock, event) stamps
with named kinds and writes warpline records.

## Tests

    python3 -m unittest discover -s tests -v

## Not built yet

The in-kernel stamp (a pinned clock read and a floating store, conditional at compile time) and its service (a
shared-memory ring drained in the background, dropping and counting when full) do not exist yet. Today the reader
works on records written by other recorders through an importer. Also not built: the twin comparison, a card's
capacity descriptor, the sampled and inferred bases for the work between stamps, a simulator.

## License

Apache-2.0, Copyright 2026 Blake Bottum.
