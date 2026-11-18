// warpline: the native record, written on the host. Plain C++17: no CUDA is needed to compile it.
//
// Given the rows copied back from the device (see entry.hpp), the lanes' names and roles, the barriers' counts and a
// legend {site -> name}, `Record::json()` writes a `warpline/0` document that `python -m warpline check` accepts as it
// stands: stamps named by the convention, the launch's two edges with their ties, a phase's begin and end, and the
// 32-bit clock words UNWRAPPED. Nothing here is specific to a kernel.
//
//   warpline::host::Record rec;
//   rec.name(kSiteFill, "fill");                  // legend: a site's name (a phase, a barrier, a resource)
//   rec.barrier("full", /*count=*/1);             // an arrival count a phase; barriers are declared, not inferred
//   rec.add_row("cta0", "warp0", "producer", host_rows + 0 * row_entries, cap);   // a row, head and entries
//   rec.write("record.json");
//
// HOW AN ENTRY BECOMES A STAMP, by kind (the name comes from the legend, by the site):
//   kPhaseBegin   phase.<name>.begin, after a phase.<previous>.end at the same time
//   kWaitBegin    barrier.<name>.wait.begin          kWaitEnd  barrier.<name>.wait.end
//   kArrive       barrier.<name>.arrive              kIssue    op.<name>.issue, payload: this lane's count of issues here
//   kLaunchBegin  launch.begin                        kLaunchEnd  launch.end (closes the lane's open phase first)
//   kTime         no stamp: the TIE of the entry just before it (same site), cycles and nanoseconds on one lane
//   a user kind   a mark named by the legend entry for the FULL id (warpline::id(kind, site)), else for the site
// A legend key is looked up as the full id first, then as the site: `name(site, ...)` is enough for the reserved kinds.
//
// UNWRAPPING. A word is 32 bits, the cycle counter wraps in about a second and the timer in about 4.3 s. A lane's words
// are unwrapped in the order written: each is the previous plus the 32-bit difference, so TWO CONSECUTIVE ENTRIES OF A
// LANE MUST BE LESS THAN 2^32 APART (stamp at least once per wrap; the launch's edges alone do not carry a long run).
// The first word of each lane is placed against the scope's first word by a signed 32-bit difference (lanes of a CTA
// start within 2^31 cycles of each other). Cycles are shifted so a scope's earliest stamp is 0 (scopes' clocks are
// unrelated); nanoseconds are shifted so the record's earliest tie is 0, one axis for every scope (the timer is global).
#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "entry.hpp"

namespace warpline {
namespace host {

class Record {
 public:
  // The legend: `key` is a site (or, for a user kind, a full id) and `name` its name in the record.
  void name(uint32_t key, const std::string& name) { legend_[key] = name; }

  // A barrier's arrival count a phase; `stamped`: how many of those arrivals are stamped (default: all); `issue`: for an
  // unstamped arrival, the resource whose `op.<resource>.issue` is named as its cause. Declared once, by the user.
  void barrier(const std::string& name, uint32_t count, int64_t stamped = -1, const std::string& issue = "") {
    barriers_[name] = Barrier{count, stamped < 0 ? static_cast<int64_t>(count) : stamped, issue};
  }

  // The same kernel with its stamps compiled out, measured in the same session: its time and the recorded kernel's, in
  // one unit (the reader flags a recorded/plain above 1 + 2 x tolerance).
  void twin(double plain, double recorded) { twin_ = {true, plain, recorded}; }

  // The clock's frequency, if known (written as `clock.hz`).
  void clock_hz(double hz) { hz_ = hz; }

  // A free-form note for `source` (a string pair, written as is).
  void source(const std::string& key, const std::string& value) { source_.push_back({key, value}); }

  // One warp's row as copied from the device: `row` points at its head (kHead entries), then `cap` entries. `scope` is
  // the CTA ("cta0"): the lanes of a scope share a clock; the lane's id is `scope/lane`.
  void add_row(const std::string& scope, const std::string& lane, const std::string& role, const Entry* row,
               uint32_t cap, const std::string& group = "") {
    Row r;
    r.scope = scope;
    r.lane = lane;
    r.role = role;
    r.group = group.empty() ? role : group;
    r.issued_entries = row[0].word;
    uint32_t stored = r.issued_entries < cap ? r.issued_entries : cap;
    r.entries.assign(row + kHead, row + kHead + stored);
    rows_.push_back(std::move(r));
  }

  // The record, as JSON text. Throws std::runtime_error on a row that cannot be a record (a time entry with no stamp
  // before it, a launch edge twice).
  std::string json() const {
    Built b = build();
    std::ostringstream o;
    o << "{\"format\":\"warpline/0\",\"clock\":{\"unit\":\"cycles\",\"hz\":";
    if (hz_ > 0) o << hz_; else o << "null";
    if (!b.ties.empty()) o << ",\"time_unit\":\"ns\",\"time_origin\":\"the earliest tie of this record\"";
    o << "},\"names\":{";
    for (size_t i = 0; i < b.names.size(); ++i) o << (i ? "," : "") << "\"" << i << "\":" << quote(b.names[i]);
    o << "},\"lanes\":[";
    for (size_t i = 0; i < rows_.size(); ++i) {
      const Row& r = rows_[i];
      o << (i ? "," : "") << "{\"id\":" << quote(r.scope + "/" + r.lane) << ",\"role\":" << quote(r.role)
        << ",\"group\":" << quote(r.group) << ",\"issued\":" << b.issued[i] << "}";
    }
    o << "],\"stamps\":[";
    for (size_t i = 0; i < b.stamps.size(); ++i) {
      const Stamp& s = b.stamps[i];
      o << (i ? "," : "") << "[" << quote(s.lane) << "," << s.name << "," << s.time;
      if (s.has_payload) o << "," << s.payload;
      o << "]";
    }
    o << "],\"dropped\":{";
    bool first = true;
    for (size_t i = 0; i < rows_.size(); ++i) {
      if (!b.dropped[i]) continue;
      o << (first ? "" : ",") << quote(rows_[i].scope + "/" + rows_[i].lane) << ":" << b.dropped[i];
      first = false;
    }
    o << "},\"facts\":{\"barriers\":{";
    first = true;
    for (const auto& kv : barriers_) {
      o << (first ? "" : ",") << quote(kv.first) << ":{\"count\":" << kv.second.count;
      if (kv.second.stamped != static_cast<int64_t>(kv.second.count)) o << ",\"stamped\":" << kv.second.stamped;
      if (!kv.second.issue.empty()) o << ",\"issue\":" << quote(kv.second.issue);
      o << "}";
      first = false;
    }
    o << "},\"operations\":{}}";
    if (!b.ties.empty()) {
      o << ",\"ties\":[";
      for (size_t i = 0; i < b.ties.size(); ++i)
        o << (i ? "," : "") << "[" << quote(b.ties[i].lane) << "," << b.ties[i].cycles << "," << b.ties[i].ns << "]";
      o << "]";
    }
    if (twin_.set) o << ",\"twin\":{\"plain\":" << twin_.plain << ",\"recorded\":" << twin_.recorded << "}";
    if (!source_.empty()) {
      o << ",\"source\":{";
      for (size_t i = 0; i < source_.size(); ++i)
        o << (i ? "," : "") << quote(source_[i].first) << ":" << quote(source_[i].second);
      o << "}";
    }
    o << "}";
    return o.str();
  }

  void write(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("warpline: cannot write " + path);
    f << json();
  }

 private:
  struct Barrier { uint32_t count; int64_t stamped; std::string issue; };
  struct Row {
    std::string scope, lane, role, group;
    uint32_t issued_entries = 0;
    std::vector<Entry> entries;
  };
  struct Stamp { std::string lane; uint32_t name; int64_t time; bool has_payload; int64_t payload; };
  struct Tie { std::string lane; int64_t cycles; int64_t ns; };
  struct Built {
    std::vector<std::string> names;
    std::vector<Stamp> stamps;
    std::vector<Tie> ties;
    std::vector<uint64_t> issued, dropped;
  };
  struct Twin { bool set = false; double plain = 0, recorded = 0; };

  static std::string quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') { o += '\\'; o += c; }
      else if (static_cast<unsigned char>(c) < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); o += buf; }
      else o += c;
    }
    return o + "\"";
  }

  std::string lookup(uint32_t event) const {
    auto full = legend_.find(event);
    if (full != legend_.end()) return full->second;
    auto site = legend_.find(site_of(event));
    if (site != legend_.end()) return site->second;
    char buf[32];
    snprintf(buf, sizeof buf, "site_0x%x", site_of(event));
    return buf;
  }

  Built build() const {
    Built b;
    std::map<std::string, uint32_t> ids;
    auto nid = [&](const std::string& n) {
      auto it = ids.find(n);
      if (it != ids.end()) return it->second;
      uint32_t v = static_cast<uint32_t>(b.names.size());
      b.names.push_back(n);
      ids[n] = v;
      return v;
    };
    // the references the unwrapping is placed against: a scope's first cycle word, the record's first time word
    std::map<std::string, uint32_t> cycle_ref;
    bool have_ns = false;
    uint32_t ns_ref = 0;
    for (const Row& r : rows_)
      for (const Entry& e : r.entries) {
        if (kind_of(e.event) == kTime) {
          if (!have_ns) { have_ns = true; ns_ref = e.word; }
        } else if (!cycle_ref.count(r.scope)) {
          cycle_ref[r.scope] = e.word;
        }
      }
    std::map<std::string, int64_t> scope_min;  // earliest unwrapped cycle of a scope
    std::map<std::string, bool> scope_seen;
    int64_t ns_min = 0;
    bool ns_seen = false;
    struct Pending { size_t stamp_begin, stamp_end, tie_begin, tie_end; std::string scope; };
    std::vector<Pending> spans;
    for (const Row& r : rows_) {
      const std::string lane = r.scope + "/" + r.lane;
      Pending span{b.stamps.size(), 0, b.ties.size(), 0, r.scope};
      bool have_c = false, have_t = false;
      uint32_t prev_c = 0, prev_t = 0;
      int64_t cur_c = 0, cur_t = 0;
      std::string phase;
      bool has_phase = false, began = false, ended = false;
      bool last_valid = false;      // the previous entry was a stamp whose tie may follow
      uint32_t last_site = 0;
      int64_t last_cycles = 0;
      std::map<uint32_t, int64_t> issues;
      uint64_t time_stored = 0;
      for (const Entry& e : r.entries) {
        const uint32_t kind = kind_of(e.event), site = site_of(e.event);
        if (kind == kNone) continue;
        if (kind == kTime) {
          if (!last_valid || last_site != site)
            throw std::runtime_error("warpline: " + lane + ": a time entry must follow a stamp of the same site");
          int64_t ns;
          if (!have_t) { have_t = true; ns = static_cast<int32_t>(e.word - ns_ref); }
          else ns = cur_t + static_cast<int64_t>(static_cast<uint32_t>(e.word - prev_t));
          cur_t = ns;
          prev_t = e.word;
          b.ties.push_back({lane, last_cycles, ns});
          if (!ns_seen || ns < ns_min) { ns_min = ns; ns_seen = true; }
          last_valid = false;
          ++time_stored;
          continue;
        }
        int64_t c;
        if (!have_c) { have_c = true; c = static_cast<int32_t>(e.word - cycle_ref.at(r.scope)); }
        else c = cur_c + static_cast<int64_t>(static_cast<uint32_t>(e.word - prev_c));
        cur_c = c;
        prev_c = e.word;
        if (!scope_seen[r.scope] || c < scope_min[r.scope]) { scope_min[r.scope] = c; scope_seen[r.scope] = true; }
        last_valid = true;
        last_site = site;
        last_cycles = c;
        auto push = [&](const std::string& n, bool has_payload = false, int64_t payload = 0) {
          b.stamps.push_back({lane, nid(n), c, has_payload, payload});
        };
        const std::string nm = lookup(e.event);
        switch (kind) {
          case kPhaseBegin:
            if (has_phase) push("phase." + phase + ".end");
            phase = nm;
            has_phase = true;
            push("phase." + phase + ".begin");
            break;
          case kWaitBegin: push("barrier." + nm + ".wait.begin"); break;
          case kWaitEnd: push("barrier." + nm + ".wait.end"); break;
          case kArrive: push("barrier." + nm + ".arrive"); break;
          case kIssue: push("op." + nm + ".issue", true, issues[site]++); break;
          case kLaunchBegin:
            if (began) throw std::runtime_error("warpline: " + lane + ": two launch begins");
            began = true;
            push("launch.begin");
            break;
          case kLaunchEnd:
            if (ended) throw std::runtime_error("warpline: " + lane + ": two launch ends");
            ended = true;
            if (has_phase) { push("phase." + phase + ".end"); has_phase = false; }
            push("launch.end");
            break;
          default: push(nm); break;
        }
      }
      span.stamp_end = b.stamps.size();
      span.tie_end = b.ties.size();
      spans.push_back(span);
      const uint64_t stored = r.entries.size();
      const uint64_t drop = r.issued_entries > stored ? r.issued_entries - stored : 0;
      b.dropped.push_back(drop);
      // `issued` = the stamps written (a phase's synthesized end among them) + those dropped, as `check` accounts them
      b.issued.push_back((span.stamp_end - span.stamp_begin) + drop);
      (void)time_stored;
    }
    for (const Pending& p : spans) {
      const int64_t shift = scope_seen[p.scope] ? scope_min[p.scope] : 0;
      for (size_t i = p.stamp_begin; i < p.stamp_end; ++i) b.stamps[i].time -= shift;
      for (size_t i = p.tie_begin; i < p.tie_end; ++i) { b.ties[i].cycles -= shift; b.ties[i].ns -= ns_min; }
    }
    return b;
  }

  std::map<uint32_t, std::string> legend_;
  std::map<std::string, Barrier> barriers_;
  std::vector<Row> rows_;
  std::vector<std::pair<std::string, std::string>> source_;
  Twin twin_;
  double hz_ = 0;
};

}  // namespace host
}  // namespace warpline
