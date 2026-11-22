// Writes records through include/warpline/host.hpp from HAND-MADE rows (no GPU): the 32-bit words wrap, a row is full,
// a mark's time entry is cut off. tests/test_host.py loads and checks what this writes.
//   host_gen <dir>        writes wrap.json, full.json, cut.json, head_a.json and head_b.json (one row, two ways of
//                         giving it), ops.json and ops_undeclared.json, rdv.json, rdv_skew.json, rdv_noskew.json and
//                         rdv_undeclared.json, noexit.json, variant.json
//   host_gen --bad        exits 0 only if a time entry with no stamp before it is refused
//   host_gen --bad-rdv    exits 0 only if an arrive entry at a rendezvous is refused
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "warpline/host.hpp"

namespace wl = warpline;
using wl::Entry;

enum : uint32_t { kFull = 1, kMeet = 2, kLanded = 3, kFill = 10, kWork = 11, kEntry = 20, kExit = 21, kUserPing = 30,
                  kCopy = 40 };
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
  r.name(kMeet, "meet");
  r.name(kLanded, "landed");
  r.name(kCopy, "copy");
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
  if (argc > 1 && !std::strcmp(argv[1], "--bad-rdv")) {
    Row r(8);
    r.put(wl::kArrive, kMeet, 5);  // an arrive entry at a barrier declared a rendezvous
    wl::host::Record rec = make();
    rec.rendezvous("meet", 1);
    rec.add_row("cta0", "warp0", "w", r.data(), r.cap);
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

  // A ROW WITH A HEAD OF ITS OWN: 32 words (the issued count at word 0, a base at word 1), the entries from word 32.
  // It is given as it lies, through the overload, and again in warpline's layout through the first: one record. The row
  // is full (capacity 4, 5 issued), so the dropped count is carried too.
  {
    const uint32_t kOwnHead = 16;  // entries of a head of its own (32 words)
    Row p(4);
    p.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    p.put(wl::kArrive, kFull, 1090);
    p.put(wl::kArrive, kFull, 1100);
    p.put(wl::kArrive, kFull, 1110);   // dropped
    std::vector<Entry> own(kOwnHead + p.cap, Entry{0xDEAD, 0xBEEF});   // the head is the kernel's
    own[0] = Entry{p.issued, 7};
    for (uint32_t i = 0; i < p.cap; ++i) own[kOwnHead + i] = p.e[wl::kHead + i];
    wl::host::Record a = make(), b = make();
    a.name(kEntry + 1, "exit");
    b.name(kEntry + 1, "exit");
    a.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    b.add_row("cta0", "warp0", "producer", own.data() + kOwnHead, p.issued, p.cap);
    a.write(dir + "/head_a.json");
    b.write(dir + "/head_b.json");
  }
  // OPERATIONS. warp0 issues a copy twice (site kCopy) and waits on `landed` after each; warp1 only waits on it. The
  // copy's landing is the barrier's one arrival, not stamped. Declared, `check` accepts it; not declared, each issue is
  // an operation never learned done.
  for (const bool declared : {true, false}) {
    Row p(16), c(16);
    p.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    c.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    for (int i = 0; i < 2; ++i) {
      p.put(wl::kIssue, kCopy, 1100 + 400 * i);
      p.put(wl::kWaitBegin, kLanded, 1110 + 400 * i);
      p.put(wl::kWaitEnd, kLanded, 1300 + 400 * i);
      c.put(wl::kWaitBegin, kLanded, 1120 + 400 * i);
      c.put(wl::kWaitEnd, kLanded, 1300 + 400 * i);
    }
    p.mark(wl::kLaunchEnd, kExit, 2000, 6000);
    c.mark(wl::kLaunchEnd, kExit, 2000, 6000);
    wl::host::Record rec = make();
    rec.name(kExit, "exit");
    if (declared) {
      rec.operation("copy", kCopy, "landed");
    } else {
      rec.barrier("landed", 1, 0, "copy");
    }
    rec.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    rec.add_row("cta0", "warp1", "consumer", c.data(), c.cap);
    rec.write(dir + (declared ? "/ops.json" : "/ops_undeclared.json"));
  }
  // A RENDEZVOUS. Two warps meet at `meet` (count 2), each waiting from its own arrival. warp1 reaches it last. `rdv`:
  // both waits end after the last begin. `rdv_skew` / `rdv_noskew`: warp0's wait ends 5 cycles BEFORE warp1's begin is
  // stamped (a clock read not ordered with the barrier), with a declared skew of 8 and of 0. `rdv_undeclared`: the
  // same entries as `rdv` at a barrier of count 2 that is not declared a rendezvous (no arrival is stamped).
  for (const char* which : {"rdv", "rdv_skew", "rdv_noskew", "rdv_undeclared"}) {
    const std::string w = which;
    const bool skewed = w == "rdv_skew" || w == "rdv_noskew";
    Row a(8), b(8);
    a.put(wl::kWaitBegin, kMeet, 100);
    a.put(wl::kWaitEnd, kMeet, skewed ? 200 : 210);
    b.put(wl::kWaitBegin, kMeet, skewed ? 205 : 110);
    b.put(wl::kWaitEnd, kMeet, 210);
    wl::host::Record rec = make();
    if (w == "rdv_undeclared") rec.barrier("meet", 2);
    else rec.rendezvous("meet", 2, w == "rdv_skew" ? 8 : 0);
    rec.add_row("cta0", "warp0", "first", a.data(), a.cap);
    rec.add_row("cta0", "warp1", "second", b.data(), b.cap);
    rec.write(dir + "/" + which + ".json");
  }
  // NO LAUNCH EXIT. warp0 stamps both of the launch's edges; warp1 stamps a phase and then nothing: its last phase is
  // open at its last stamp, and it has no edge at all.
  {
    Row a(8), b(8);
    a.mark(wl::kLaunchBegin, kEntry, 1000, 5000);
    a.mark(wl::kLaunchEnd, kExit, 1500, 5500);
    b.put(wl::kPhaseBegin, kWork, 1010);
    b.put(wl::kPhaseBegin, kFill, 1100);
    b.put(kUserKind, kUserPing, 1200);
    wl::host::Record rec = make();
    rec.name(kExit, "exit");
    rec.add_row("cta0", "warp0", "producer", a.data(), a.cap);
    rec.add_row("cta0", "warp1", "consumer", b.data(), b.cap);
    rec.write(dir + "/noexit.json");
  }
  // THE VARIANT'S NAME.
  {
    Row p(4);
    p.put(wl::kPhaseBegin, kFill, 10);
    wl::host::Record rec = make();
    rec.variant("fine");
    rec.add_row("cta0", "warp0", "producer", p.data(), p.cap);
    rec.write(dir + "/variant.json");
  }
  return 0;
}
