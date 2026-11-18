// Writes records through include/warpline/host.hpp from HAND-MADE rows (no GPU): the 32-bit words wrap, a row is full,
// a mark's time entry is cut off. tests/test_host.py loads and checks what this writes.
//   host_gen <dir>        writes wrap.json, full.json, cut.json
//   host_gen --bad        exits 0 only if a time entry with no stamp before it is refused
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "warpline/host.hpp"

namespace wl = warpline;
using wl::Entry;

enum : uint32_t { kFull = 1, kFill = 10, kWork = 11, kEntry = 20, kExit = 21, kUserPing = 30 };
constexpr uint32_t kUserKind = wl::kUser + 1;

struct Row {
  std::vector<Entry> e;
  uint32_t issued;
  uint32_t cap;
  explicit Row(uint32_t cap_) : e(wl::kHead + cap_, Entry{0, 0}), issued(0), cap(cap_) {}
  // an entry the kernel issued: stored when there is room, counted either way
  Row& put(uint32_t kind, uint32_t site, uint32_t word) {
    if (issued < cap) e[wl::kHead + issued] = Entry{word, wl::id(kind, site)};
    ++issued;
    return *this;
  }
  // a mark: the stamp, then the time entry
  Row& mark(uint32_t kind, uint32_t site, uint32_t cycles, uint32_t ns) {
    put(kind, site, cycles);
    return put(wl::kTime, site, ns);
  }
  const Entry* data() {
    e[0] = Entry{issued, cap};  // what close() writes
    return e.data();
  }
};

static wl::host::Record make() {
  wl::host::Record r;
  r.name(kFull, "full");
  r.name(kFill, "fill");
  r.name(kWork, "work");
  r.name(kEntry, "entry");
  r.name(wl::id(kUserKind, kUserPing), "ping");
  r.barrier("full", 1);
  return r;
}

int main(int argc, char** argv) {
  if (argc > 1 && !std::strcmp(argv[1], "--bad")) {
    Row r(8);
    r.put(wl::kTime, kEntry, 5);  // a time entry with no stamp before it
    wl::host::Record rec = make();
    rec.add_row("cta0", "warp0", "producer", r.data(), r.cap);
    try {
      rec.json();
    } catch (const std::runtime_error&) {
      return 0;
    }
    return 1;
  }
  if (argc < 2) return 2;
  const std::string dir = argv[1];

  // WRAP. One scope, two lanes. Both clocks' 32-bit words cross zero; the consumer's first word is BEFORE the
  // producer's (a signed difference of -50 cycles, -30 ns across the wrap); the end edge is a second wrap away.
  {
    const uint32_t C = 0xFFFFFF00u, T = 0xFFFFFF00u;  // 256 cycles, 256 ns below the wrap
    Row p(64), c(64);
    p.mark(wl::kLaunchBegin, kEntry, C, T);
    p.mark(wl::kPhaseBegin, kFill, C + 100, T + 100);
    p.put(wl::kArrive, kFull, C + 350);   // 0x...5e: past the wrap
    p.put(kUserKind, kUserPing, C + 360);
    p.mark(wl::kLaunchEnd, kEntry + 1, C + 1000, T + 700);   // 2^32 - 256 + 1000: wrapped
    c.mark(wl::kLaunchBegin, kEntry, C - 50, T - 30);
    c.mark(wl::kPhaseBegin, kWork, C + 120, T + 80);
    c.put(wl::kWaitBegin, kFull, C + 130);
    c.put(wl::kWaitEnd, kFull, C + 410);
    c.mark(wl::kLaunchEnd, kEntry + 1, C + 990, T + 690);
    wl::host::Record rec = make();
    rec.name(kEntry + 1, "exit");
    rec.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    rec.add_row("cta0", "warp1", "consumer", c.data(), c.cap);
    rec.write(dir + "/wrap.json");
  }
  // FULL. A row of capacity 12 into which 14 entries were issued: the first 12 fill it exactly (both launch edges
  // among them), 2 are dropped, and counted.
  {
    Row p(12);
    p.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    p.put(wl::kArrive, kFull, 1090);
    p.put(kUserKind, kUserPing, 1095);
    for (int i = 0; i < 3; ++i) {
      p.put(wl::kWaitBegin, kFull, 1100 + 100 * i);
      p.put(wl::kWaitEnd, kFull, 1150 + 100 * i);
    }
    p.mark(wl::kLaunchEnd, kEntry + 1, 2000, 6000);   // entries 10, 11: the last that fit
    p.put(wl::kArrive, kFull, 2100);                   // 12.. dropped
    p.put(wl::kArrive, kFull, 2200);
    wl::host::Record rec = make();
    rec.name(kEntry + 1, "exit");
    rec.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    rec.write(dir + "/full.json");
  }
  // CUT. The row's capacity ends between the end mark's stamp and its time entry: the stamp is stored, its tie is not.
  {
    Row p(3);
    p.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    p.mark(wl::kLaunchEnd, kEntry + 1, 1200, 5200);   // entry 2 fits, entry 3 is dropped
    wl::host::Record rec = make();
    rec.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    rec.write(dir + "/cut.json");
  }
  return 0;
}
