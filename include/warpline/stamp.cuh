// warpline: the stamp, in a kernel. Header-only, CUDA C++17, no dependency; plain CUDA can use it.
//
// The pieces, each usable alone, nothing owning:
//   stamp(at, id) / stamp_if(at, id, on)   the cycle counter's low word and an event id, stored at `at`
//   mark(at, id)  / mark_if(at, id, on)    a stamp, then a second entry: the nanosecond timer, kind kTime (two entries)
//   Cursor<CAP>                            an optional cursor over a row; a user may address entries themselves
//   dump(to, from, count)                  copy a shared-memory ring's entries to a global row
// There is NO on/off switch here: a stamp you do not want is a stamp you do not write. Put your own `if constexpr` or
// macro around the call (examples/two_roles.cu) and nothing is compiled. See docs/guide.md.
#pragma once
#include <cstdint>

#include "entry.hpp"

namespace warpline {

// The cycle counter's low word. A clock read depends on nothing, so it is volatile: it stays where it is written.
__device__ __forceinline__ uint32_t clock_lo() {
  uint32_t now;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(now));
  return now;
}

// The nanosecond timer's low word (it wraps in about 4.3 s; the host half unwraps).
__device__ __forceinline__ uint32_t time_lo() {
  uint32_t ns;
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ns));
  return ns;
}

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t l;
  asm("mov.u32 %0, %%laneid;" : "=r"(l));
  return l;
}

// One store of {word, event} at `at` (global or shared: a generic address).
__device__ __forceinline__ void put(Entry* at, uint32_t word, uint32_t event) {
  asm volatile("st.v2.u32 [%0], {%1, %2};" ::"l"(at), "r"(word), "r"(event) : "memory");
}

// The same store, predicated: one predicated store, no branch.
__device__ __forceinline__ void put_if(Entry* at, uint32_t word, uint32_t event, uint32_t on) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n @p st.v2.u32 [%0], {%1, %2};\n}" ::"l"(at), "r"(word), "r"(event),
      "r"(on)
      : "memory");
}

// Put this between a `bar.sync` and the stamp that ends the wait. A clock read issued right after a named-barrier
// wait does NOT wait for the barrier: the warp keeps issuing until an instruction needs the barrier's completion, so the
// stamp would read the clock at the wait's start (measured: a 13-cycle wait that really lasted 1700+). A `membar.cta`
// needs it. Waits on an mbarrier's try_wait loop need nothing: the loop's exit is the release. Compile it out with the
// stamps (it is part of the stamp's cost, not of the kernel's).
__device__ __forceinline__ void after_wait() { asm volatile("membar.cta;" ::: "memory"); }

// A stamp: the cycle counter's low word and `id` (warpline::id(kind, site)) stored at `at`. Every lane that calls it
// stores; have one lane call it (or use stamp_if with a one-lane predicate).
__device__ __forceinline__ void stamp(Entry* at, uint32_t id) { put(at, clock_lo(), id); }

// A stamp whose store happens only where `on` is nonzero.
__device__ __forceinline__ void stamp_if(Entry* at, uint32_t id, uint32_t on) { put_if(at, clock_lo(), id, on); }

// A MARK: two entries, at[0] the stamp, at[1] the nanosecond timer's low word with kind kTime at the same site: one
// moment on both clocks. Use it on anchors only.
__device__ __forceinline__ void mark(Entry* at, uint32_t id) {
  const uint32_t now = clock_lo();
  const uint32_t ns = time_lo();
  put(at, now, id);
  put(at + 1, ns, ::warpline::id(kTime, site_of(id)));
}

// Both predicated stores sit in one asm block. ptxas may still choose to branch around the pair (seen on a mark outside
// any loop, in the micro-build; inside the example's window loop it predicated them): see docs/guide.md, "what a stamp is".
__device__ __forceinline__ void mark_if(Entry* at, uint32_t id, uint32_t on) {
  const uint32_t now = clock_lo();
  const uint32_t ns = time_lo();
  const uint32_t tid = ::warpline::id(kTime, site_of(id));
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n @p st.v2.u32 [%0], {%1, %2};\n @p st.v2.u32 [%0+8], {%3, %4};\n}"
      ::"l"(at), "r"(now), "r"(id), "r"(ns), "r"(tid), "r"(on)
      : "memory");
}

// A cursor over one warp's row: a POD of registers. `to` is past the head, `n` the entries issued (counted past
// capacity), `mine` is 1 on the lane that writes (the warp's first). The convenience the host half's rows expect; a
// user may ignore it.
template <uint32_t CAP = 4096>
struct Cursor {
  static constexpr uint32_t kCap = CAP;
  static constexpr uint32_t kRowEntries = kHead + CAP;
  Entry* to;
  uint32_t n;
  uint32_t mine;

  // The row `lane_row` of `rows` (rows are kRowEntries apart); `after`: entries already written to it.
  __device__ __forceinline__ void bind(Entry* rows, uint32_t lane_row, uint32_t after = 0) {
    to = rows + static_cast<size_t>(lane_row) * kRowEntries + kHead;
    n = after;
    mine = lane_id() == 0;
  }

  // The address of the next entry, advancing; `on` says whether to store there: this is the writing lane and the
  // row has room. An entry past capacity is counted and not stored.
  __device__ __forceinline__ Entry* next(uint32_t& on) {
    on = mine & (n < CAP ? 1u : 0u);
    return to + n++;
  }

  __device__ __forceinline__ void stamp(uint32_t id) {
    uint32_t on;
    Entry* at = next(on);
    stamp_if(at, id, on);
  }

  // A mark takes two entries; both stored or neither (a row too full for two stores none, and counts both).
  __device__ __forceinline__ void mark(uint32_t id) {
    const uint32_t on = mine & (n + 1 < CAP ? 1u : 0u);
    Entry* at = to + n;
    n += 2;
    mark_if(at, id, on);
  }

  // Write the count and the capacity to the row's head. Call once, at the kernel's exit.
  __device__ __forceinline__ void close() {
    if (mine) put(to - kHead, n, CAP);
  }
};

// Copy `count` entries from a shared-memory ring to a global row, `from` -> `to`. Run by ONE WARP, all its lanes,
// where the caller chooses (a quiet phase edge, the window's end, the exit): each lane copies every 32nd entry.
__device__ __forceinline__ void dump(Entry* to, const Entry* from, uint32_t count) {
  for (uint32_t i = lane_id(); i < count; i += 32) to[i] = from[i];
  __syncwarp();
}

}  // namespace warpline
