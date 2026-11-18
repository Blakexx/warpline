# The guide: how to stamp a kernel

warpline's in-kernel half is a few pieces, nothing owning (`include/warpline/stamp.cuh`). Where stamps go, how many,
in which register or memory, and when they drain are YOUR kernel's design. This guide is the rest: what each piece
costs, and the rules a budget of stamps follows. Every number below was measured on one data-centre GPU (an Ampere
card) unless it says otherwise; read them as orders of magnitude on your own, and re-measure before you rely on one.

## 1. The primitive, and what it costs

A STAMP is the cycle counter's low word and an event id, stored at an address you give. In SASS: one clock read, one
move of the id, one store (and, if the address is a cursor, a compare against the capacity and an address add).

- About 4 instructions. A lone warp pays about 2 cycles an instruction; a stamp on a step that waits on other work
  costs 0-1 cycles, on an integer-bound step about 8. (Built here for the example: a stamp at a caller-held address is 3
  instructions, with a one-lane predicate 4, through the cursor 5; a mark 8. See `examples/README.md`.)
- A MARK is a stamp and a second entry, the nanosecond timer's low word with the reserved kind `kTime`: one moment on
  both clocks. It costs a second entry. The launch's two edges, taken once each outside every loop, cost a window nothing.
- An entry is 8 bytes: the clock word and `kind << 16 | site`. The reserved kinds are in `entry.hpp`; kinds from
  `kUser` up are yours.
- One predicated store by the warp's first lane, no branch. An entry past the row's capacity is counted, not stored
  (`dropped` in the record says how many).

## 2. Switch at compile time, never at run time

The library has no on/off switch and no levels. A stamp you do not want is a stamp that is not written: put your own
`if constexpr`, macro or template parameter around the call (see `Line<Coarse, Detail>` in `examples/two_roles.cu`).

- A stamp behind a RUN-TIME branch costs 22-25 cycles a site, several times a stamp: the toolkit ends the block with the
  stalls its pending instructions are owed. So each variant of the kernel is its own binary.
- Compiled out means compiled out: the example with both switches off holds no clock read in its SASS
  (`cuobjdump -sass`, zero `CLOCK`/`GLOBALTIMER`). The compiler also drops a clock read whose value nothing stores, so a
  clock read written for some other use vanishes: stamp it, or use it, but do not count on it.

## 3. Never use a stamp's value

A clock read's VALUE arrives about 53 cycles after it is issued (either clock). The stamp pays for that by not waiting:
the read is issued, the store floats to where an issue slot is free. Feed the value into the work (a loop bound, a
branch) and the warp waits those cycles at that point. Stamp, and let the host read the stamps.

A clock read issued right after a named-barrier wait (`bar.sync`) does NOT wait for the barrier: the warp keeps issuing
until an instruction needs the barrier's completion. A wait-end stamp placed directly after `bar.sync` read the wait's
start (measured on a consumer card, a 13-cycle "wait" that was 1700+ cycles). `warpline::after_wait()` (a `membar.cta`)
between the barrier and the stamp makes it honest; a wait on an mbarrier's `try_wait` loop needs nothing, the loop's exit
is the release. Put it under the same compile-time condition as the stamp.

## 4. Stamps outside step bodies

A step is the unit of work whose time you are measuring (a tile's loads, an MMA group). A stamp inside it adds its
instructions to the thing measured and breaks up the compiler's schedule; stamp BETWEEN steps, at the point a phase
starts, a wait begins or ends, an arrival happens.

## 5. Coarse first, then one detail, merged on anchors

- COARSE: the launch's two edges and each window's and phase's start as marks, 3-8 stamps a warp a window. That costs
  3-4% of a window.
- DETAIL: every wait and every arrival costs 15-35% of a window: too much to leave on, and a kernel that slow is a
  different kernel. So take ONE detail a run (one barrier's waits and its arrivals, which interact and belong in the
  same run), recorded at steady state, in a run whose coarse stamps are the same in every run.
- CHECK EACH RUN AGAINST THE PLAIN KERNEL: the variant set (`python -m warpline set`), with the stamps compiled out, is the
  yardstick. A run whose window is more than the tolerance over the plain one is flagged, not trusted. Put the pair in
  the record's `twin` (`Record::twin(plain, recorded)`).
- MERGE the runs on their anchors (`python -m warpline merge`): the coarse marks are in every run, so the runs overlay.

## 6. Where entries go

- GLOBAL where entries are sparse and beside waits: a row a warp, written directly. A wait is long against a store.
- SHARED where they are dense: a ring in shared memory, and one `warpline::dump(global_row, shared_ring, count)` run by a
  warp of your choosing at a quiet point (a phase edge, the window's end, the exit). The batch drain is yours to place.
- A row is `kHead` entries of head and then the capacity; `Cursor<CAP>` binds one for you and `close()` writes the
  count. Ignore the cursor and address entries yourself when a fixed slot or a loop counter you already hold does it with
  no register; the host takes such a row's entries and count issued apart (section 11).

## 7. Which lane writes

One lane, the warp's first. The cursor's `mine` is 1 only there, and every store is predicated on it: no branch, and
no 32 lanes storing the same entry.

## 8. Capacity, and what "dropped" means

The cursor counts every entry and stores those that fit. `close()` writes the count and the capacity; the host's
record says `dropped` (entries past capacity) and `issued` (stamps written plus dropped), so a reader accounts for every
stamp. A dropped entry is the NEWEST: a full row has the beginning of the run and loses its end, and a lost launch end is
a finding (`launch_edge_missing`). Size the row for the run you take, or stamp less.

## 9. Both clocks on anchors only, and why

An anchor (a window's start, a phase's start, the launch's edges) is a MARK: both clocks. Everything else is a stamp.
- The cycle counter is not a steady clock: its rate moved 9% between phases of one kernel, and a profiler's samples are
  on time, not on cycles. Time comes from the marks (`python -m warpline rates`).
- A mark costs a second entry, so marks are the coarse set, not the detail.
- The nanosecond timer steps (1 us here): a mark's two readings are exact only to that step, so windows should be long
  against it (the example's are 40 us).
- The cycle counter's 32-bit word wraps in about a second, the timer's in about 4.3 s: `host.hpp` unwraps each lane's
  words in order, so two consecutive entries of a lane must be less than 2^32 apart.

## 10. With a sampling profiler

A time-sampling profiler stretches the kernel's time 14-18% and leaves its cycles unchanged: stamps and samples cannot
be joined on time as taken.
- The kernel carries the launch's TWO EDGES, entry and exit, each a mark on both clocks (`kLaunchBegin`, `kLaunchEnd`,
  forced: always on, outside every loop). A record without them cannot be overlaid.
- JOIN ON CYCLES, through the edges: the samples are placed under the stamps by the launch's entry and exit, not by
  correlating stalls with waits.
- The profiler must sample THE LAUNCH WHOSE STAMPS ARE RECORDED (the same launch, not a twin), or the edges are not
  the samples'.
- Expect the stretched time: take the plain kernel's time from the variant set, the stamps' cycles from the record,
  and let `overlay` reconcile them ([sampling.md](sampling.md)).

## 11. What the host writer declares beyond the row

`include/warpline/host.hpp` has four more calls, each optional, for a kernel that is not shaped like the example.

- A ROW WITH A HEAD OF YOUR OWN. `add_row(scope, lane, role, entries, issued, cap)` takes the entries themselves and how
  many the warp issued; `add_row(scope, lane, role, row, cap)` (a head of `kHead` entries, the count in its first) calls it.
  A row whose head is yours (a count and a base address in 32 words, say) is given as it lies, at `entries`.
- OPERATIONS. `operation(resource, issue_site, done_barrier)`: the `kIssue` entries at `issue_site` are issues of
  `resource` (`op.<resource>.issue`, the running count as the id), and `done_barrier`'s wait end is where a warp learns
  they are done (`facts.operations`). Unless you declare that barrier yourself, its one arrival is the work's landing,
  not stamped, with the issue as its cause (`barrier(done_barrier, 1, 0, resource)`). Without it a barrier released by a
  copy leaves every issue `operation_unfinished`. If the issue site's legend name already is the resource,
  `operation(resource, done_barrier)` does it.
- A RENDEZVOUS. `rendezvous(name, count, skew = 0)` for a barrier that completes by the count of its own waiters (a
  `bar.sync` where every warp waits is also an arrival). Stamp the wait's begin and end as for any wait and do not stamp
  an arrival (the writer refuses one): it writes an arrival at each wait begin, and the phase's release is the last
  waiter's wait begin. A clock read after `bar.sync` is not ordered with it (section 3), so the last waiter's begin can be
  stamped after another waiter's end by a few cycles; declare `skew`, the most you accept, and `check` allows exactly
  that. Without the call the barrier is read as before, and a barrier with no stamped arrival is a finding.
- THE VARIANT. `variant("fine")` writes `variant`. `summary` shows it, `diff` names the two when it refuses, and a
  `set` manifest may omit a variant's `name` where its stamp record carries one.

A lane that never stamps `launch.end` is read by one rule: its open regions close at its last stamp, and `check` and
`summary` count them as `open_at_end` instead of finding `region_unclosed` a lane ([format](format.md)). A kernel that
wants the finding stamps its exit.

## What you still write yourself

The kernel's own conditions and their switch; the site ids and the legend (id to name); where each warp's row is and its
capacity; clearing the rows before a launch and copying them back; the barrier counts (`Record::barrier`, or
`Record::rendezvous`) and operations (`Record::operation`); a
`dump` call where a shared ring drains; and deciding which variants make a set.
