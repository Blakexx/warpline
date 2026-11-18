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
    python -m warpline window   <record> [--anchor NAME] [--skip N] [--cta C] [--json] [-o out]
                                                         the average window: each stamp's offset in the steady windows
    python -m warpline merge    <record> <record> ... -o out [--tolerance PCT] [--anchor NAME]
                                                         several runs overlaid on their anchors
    python -m warpline perfetto <record> -o out.json     Chrome trace-event JSON for Perfetto's UI

`path` prints the slack of every wait off the critical path: how much later its releaser could have arrived before
the wait's lane would have joined the path (the path's time at the lane's next dependency on the path, minus the time
the lane reached it). Where the record cannot say, the slack is "unknown", with the reason; it is never the wait's length.

### The average window

A stamp costs, so a kernel traced with every stamp runs slower than the kernel. The method: every run carries the same
few ANCHORS (a phase's start, say) and, inside a small stamp budget, one group of detail; the runs are overlaid on the
anchors. `window` is the first half. A window is the span between consecutive `begin` stamps of the anchor region on a
scope's first lane carrying it (default: the region whose begin recurs most regularly there; the command says which).
Over the steady windows (the first `--skip`, default 2, and the last partial one dropped) it gives each lane's each
stamp: its offset from the window's start (mean), its spread (min, max, standard deviation) and how many windows held
it. A lane's stamps are reported as they fall in the anchor lane's window, so a lane whose own window is shifted shows
it. The table is by role (lanes of a role averaged, their spread shown); `--json` prints an average-window record.

### The merge

`merge` is the second half. Each input is one run (its own binary: a stamp is switched at compile time). The ANCHORS
are the stamp names present in every run. Per anchor and role, the mean offset must agree across runs within
`--tolerance` (default 3, a percent of the window's period), and so must the periods; a run that disagrees is flagged
with the anchor and the sizes and left out of the merged timeline, its detail reported as untrusted. A run whose record
carries `twin` is flagged when recorded/plain exceeds 1 + 2 x tolerance; one without is reported as unchecked. The
merged record holds the anchors (mean over the accepted runs) and every other stamp from the run that carried it, each
tagged with its runs (`facts.runs`, `facts.window.stamps[i].runs`). The reader runs on it; a wait and its arrivals come
from one run, and the join refuses (`cross_run_join_refused`) to match a wait from one run with an arrival of another.

Perfetto output: a track a lane (regions as nested slices, waits as slices named by their barrier), a track a resource
(its operations as async slices), and flow arrows from the stamp that released a wait to the wait it ended.

`python -m warpline.importers.stamped_clocks <input.json> -o <dir>` reads a record of per-lane (clock, event) stamps
with named kinds and writes warpline records.

## Tests

    python3 -m unittest discover -s tests -v

## Not built yet

The in-kernel stamp (a pinned clock read and a floating store, conditional at compile time) and its service (a
shared-memory ring drained in the background, dropping and counting when full) do not exist yet. Today the reader
works on records written by other recorders through an importer. Also not built: a run's twin measured (the merge reads a `twin` it is given), a card's
capacity descriptor, the sampled and inferred bases for the work between stamps, a simulator.

## License

Apache-2.0, Copyright 2026 Blake Bottum.

## Building and installing

Everything is built by Bazel; there is no other packaging file.

    bazel test //...                       # the tests
    bazel run //:cli -- check <record>     # the command, from a checkout
    bazel build //:wheel                   # the wheel: pip install bazel-bin/warpline-0.1.0-py3-none-any.whl

Another Bazel workspace depends on it as a module (`bazel_dep(name = "warpline")` at a commit).
