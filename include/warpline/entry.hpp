// warpline: the entry layout both halves share. Plain C++17: no CUDA is needed to include it.
//
// An ENTRY is eight bytes: a clock word and an event id. The id splits into a KIND (high 16 bits) and a SITE (low 16
// bits). The kinds below are RESERVED: the reader's host half gives them their meaning. Every kind from kUser up is the
// user's; the library never looks at it.
//
// A ROW is one warp's entries in global memory: kHead entries of head, then `cap` entries. The head's first entry
// holds, once the kernel closed the row, the count of entries issued (word) and the capacity (event). A count above
// the capacity means the difference was counted and not stored.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define WARPLINE_HD __host__ __device__
#else
#define WARPLINE_HD
#endif

namespace warpline {

struct Entry {
  uint32_t word;   // the cycle counter's low word; for a `kTime` entry, the nanosecond timer's low word
  uint32_t event;  // kind << 16 | site
};
static_assert(sizeof(Entry) == 8, "an entry is eight bytes");

enum Kind : uint32_t {
  kNone = 0,         // an untouched (zeroed) slot
  kPhaseBegin = 1,   // a warp starts a phase (the previous one, if any, ends here)
  kWaitBegin = 2,    // a warp starts waiting at the site's barrier
  kWaitEnd = 3,      // the wait ended
  kArrive = 4,       // a warp arrived at the site's barrier
  kIssue = 5,        // a warp issued an asynchronous operation on the site's resource
  kTime = 6,         // the nanosecond timer's reading of the entry written just before it (a mark's second entry)
  kLaunchBegin = 7,  // the launch's entry edge (a mark)
  kLaunchEnd = 8,    // the launch's exit edge (a mark)
  kUser = 16         // kinds from here up are the user's
};

constexpr uint32_t kHead = 16;  // a row's head, in entries (128 bytes)

WARPLINE_HD constexpr uint32_t id(uint32_t kind, uint32_t site) { return (kind << 16) | (site & 0xFFFFu); }
WARPLINE_HD constexpr uint32_t kind_of(uint32_t event) { return event >> 16; }
WARPLINE_HD constexpr uint32_t site_of(uint32_t event) { return event & 0xFFFFu; }

}  // namespace warpline
