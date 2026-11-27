// warpline: a run of the user's own task, instrumented. Host C++17 with the CUDA runtime: include it where the task
// launches its kernels (it needs cuda_runtime.h and cuda_profiler_api.h, and links the runtime, as any CUDA host code).
//
// Two pieces the user composes around their own launch, beside host.hpp's Record. Neither launches anything and neither
// switches a stamp: the kernel, its stamps and their switch are the user's (stamp.cuh, docs/guide.md).
//
//   Rows<CAP>   THE STAMP BUFFER: `n` rows of kHead + CAP entries on the device, cleared. `device()` is what the user's
//               kernel takes (`Cursor<CAP>::bind(rows, i)` binds row i); `collect(rec, lane_of)` copies the rows back
//               and adds each to a host::Record, row i named by lane_of(i).
//   Region      THE MARKED REGION: what a profiler captures. Its start synchronizes the device and calls cudaProfilerStart;
//               its end synchronizes and calls cudaProfilerStop. `warpline run --profile` runs the task under Nsight
//               Compute with `--profile-from-start off`, so the kernels launched inside a region are profiled and no other.
//               Outside a profiler both calls do nothing.
//
// What a user writes (examples/two_roles.cu is a whole program):
//
//   warpline::run::Rows<kCap> rows(n_warps);                       // before the region: allocating is not the kernel's
//   {
//     warpline::run::Region region;                                // the profiler captures what launches in here
//     my_kernel<true><<<grid, block>>>(rows.device(), args);       // the stamped variant, the user's own
//   }                                                              // the region ends: synchronized, the capture stops
//   warpline::host::Record rec;                                    // the legend and the barriers, as host.hpp says
//   rec.name(kFill, "fill");
//   rec.barrier("full", 1);
//   rows.collect(rec, [](uint32_t i) { return warpline::host::Lane{"cta" + std::to_string(i / 2),
//                                                                  "warp" + std::to_string(i % 2), "producer"}; });
//   rec.write_to_run("tile");                                      // under `warpline run --stamps`: DIR/records/tile.record.json
//
// and runs it as `warpline run --stamps --profile SourceCounters --place 'ns::my_kernel()' -o DIR -- ./task args`.
//
// A CUDA call that fails throws std::runtime_error naming the call. A Region left by an exception still stops the capture
// (its destructor), and says on stderr what failed there, since a destructor cannot throw.
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "entry.hpp"
#include "host.hpp"

namespace warpline {
namespace run {

inline void check(cudaError_t e, const char* call) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("warpline: ") + call + ": " + cudaGetErrorString(e));
}

// The stamp buffer of `n` rows, laid out as Cursor<CAP> binds them.
template <uint32_t CAP>
class Rows {
 public:
  static constexpr uint32_t kRowEntries = kHead + CAP;

  explicit Rows(uint32_t n) : n_(n) {
    check(cudaMalloc(&rows_, bytes()), "cudaMalloc");
    clear();
  }
  ~Rows() { cudaFree(rows_); }
  Rows(const Rows&) = delete;
  Rows& operator=(const Rows&) = delete;

  // The rows on the device, for the user's kernel.
  Entry* device() const { return rows_; }
  uint32_t size() const { return n_; }

  // Zero every row (an untouched entry is kNone, a row's head its count): before a launch whose record is taken, when the
  // buffer is used again.
  void clear() {
    check(cudaMemset(rows_, 0, bytes()), "cudaMemset");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  // The rows copied back, each added to `rec` as `lane_of(i)` names row i (Record::add_rows).
  template <typename LaneOf>
  void collect(host::Record& rec, LaneOf lane_of) const {
    std::vector<Entry> h(static_cast<size_t>(n_) * kRowEntries);
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaMemcpy(h.data(), rows_, bytes(), cudaMemcpyDeviceToHost), "cudaMemcpy");
    rec.add_rows(h.data(), n_, CAP, lane_of);
  }

 private:
  size_t bytes() const { return static_cast<size_t>(n_) * kRowEntries * sizeof(Entry); }
  uint32_t n_;
  Entry* rows_ = nullptr;
};

// The marked region, from its construction to `end()` (or its destructor).
class Region {
 public:
  Region() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaProfilerStart(), "cudaProfilerStart");
  }
  ~Region() {
    if (open_) {
      const cudaError_t a = cudaDeviceSynchronize(), b = cudaProfilerStop();
      if (a != cudaSuccess || b != cudaSuccess)
        std::fprintf(stderr, "warpline: a region's end failed: %s\n", cudaGetErrorString(a != cudaSuccess ? a : b));
    }
  }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

  // The region's end: the device synchronized (its kernels are inside), then the capture stopped.
  void end() {
    if (!open_) return;
    open_ = false;
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaProfilerStop(), "cudaProfilerStop");
  }

 private:
  bool open_ = true;
};

}  // namespace run
}  // namespace warpline
