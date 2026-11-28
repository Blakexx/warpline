// examples/floor/floor_examples.cu -- THE FLOOR'S EXAMPLES (docs/floor.md#the-examples): four kernels, each with its work
// declared beside it (`<name>.work.json`), stamped so a run manifest measures its launch and its phases, and run on a card by
// `//:record_floor_fixtures`. Their recorded runs are what holds the floor: never above a measured time, and naming the stream
// the counters show (`tests/test_floor_fixtures.py`).
//
//   floor_examples.bin stream|shared|mma|serial [record.json]     # the record into the run, else to the path given
//
// A CTA an SM, 512 threads; warp 0 of each CTA stamps (the launch's two edges and each window's phases, as marks), so a
// stamp costs a window nothing measurable. Each kernel runs kWindows windows:
//   stream   role reader, phase read: a CTA its own 8 MiB of a table from HBM, a window's share of it, summed (HBM binds)
//   shared   role reader, phase read: 16-byte loads of 48 KB of shared memory, conflict-free (shared memory binds)
//   mma      role math,   phase mma:  4 independent m16n8k16 MMAs a warp a trip (the tensor pipe binds)
//   serial   role worker, phases load then math, between barriers: a window's HBM read, then an FFMA loop that moves no
//            byte: a phase that idles the binding stream, on purpose (declared `serial`; undeclared, the idle check refuses it)
// The phases a declaration puts in sequence are the launch's: a CTA's warps pass them together (a barrier between), and every
// CTA alike; phases that one warp runs while another runs a different one are an overlap set.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "warpline/host.hpp"
#include "warpline/run.hpp"
#include "warpline/stamp.cuh"

namespace wl = warpline;

namespace floor_examples {

constexpr int kThreads = 512, kWindows = 16;
//: a CTA's slice of the table: 8 MiB, 2^19 units of 16 bytes; a window reads 2^15 of them
constexpr unsigned long long kUnitsPerCta = 1ull << 19;
constexpr int kTrip = 8;
//: the shared example: 48 KB, 6 loads a thread a trip, 1024 trips a window
constexpr int kSharedTrip = 6, kSharedTrips = 1024;
//: the MMA example: 4 MMAs a warp a trip, 4096 trips a window
constexpr int kMma = 4, kMmaTrips = 4096;
//: the serial example's math: 16 FFMA chains, 128 FFMA a trip, 48 trips a window (about half its load's time: HBM binds)
constexpr int kMathTrips = 48;
constexpr uint32_t kCap = 128;

enum Site : uint32_t { kEntry = 1, kExit = 2, kRead = 10, kMmaSite = 11, kLoad = 12, kMath = 13 };

//: the stamps: warp 0 of each CTA, row `blockIdx.x`, when `Stamp`
template <bool Stamp>
struct Line {
  wl::Cursor<kCap> c;
  __device__ void bind(wl::Entry* rows) {
    if constexpr (Stamp)
      if (threadIdx.x < 32) c.bind(rows, blockIdx.x);
  }
  __device__ void mark(uint32_t kind, uint32_t site) {
    if constexpr (Stamp)
      if (threadIdx.x < 32) c.mark(wl::id(kind, site));
  }
  __device__ void close() {
    if constexpr (Stamp)
      if (threadIdx.x < 32) c.close();
  }
};

__device__ __forceinline__ uint32_t read_window(const uint4* base, unsigned long long first, uint32_t sum) {
  constexpr unsigned long long kPer = kUnitsPerCta / kWindows;
#pragma unroll 1
  for (unsigned long long i = first + threadIdx.x; i < first + kPer; i += kThreads * kTrip) {
    uint4 v[kTrip];
#pragma unroll
    for (int k = 0; k < kTrip; ++k) v[k] = base[i + k * kThreads];
#pragma unroll
    for (int k = 0; k < kTrip; ++k) sum += v[k].x + v[k].y + v[k].z + v[k].w;
  }
  return sum;
}

template <bool Stamp>
__global__ void __launch_bounds__(kThreads, 1) stream(wl::Entry* rows, const uint4* table, uint32_t* out) {
  Line<Stamp> t;
  t.bind(rows);
  t.mark(wl::kLaunchBegin, kEntry);
  const uint4* base = table + blockIdx.x * kUnitsPerCta;
  uint32_t sum = 0;
  for (int w = 0; w < kWindows; ++w) {
    t.mark(wl::kPhaseBegin, kRead);
    sum = read_window(base, w * (kUnitsPerCta / kWindows), sum);
  }
  out[blockIdx.x * kThreads + threadIdx.x] = sum;
  t.mark(wl::kLaunchEnd, kExit);
  t.close();
}

template <bool Stamp>
__global__ void __launch_bounds__(kThreads, 1) shared(wl::Entry* rows, uint32_t* out) {
  __shared__ uint4 tile[kSharedTrip * kThreads];
  Line<Stamp> t;
  t.bind(rows);
  t.mark(wl::kLaunchBegin, kEntry);
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k) tile[k * kThreads + threadIdx.x] = make_uint4(k, threadIdx.x, 1, 2);
  __syncthreads();
  uint32_t at[kSharedTrip];
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k) at[k] = static_cast<uint32_t>(__cvta_generic_to_shared(&tile[k * kThreads + threadIdx.x]));
  uint32_t sum = 0;
  for (int w = 0; w < kWindows; ++w) {
    t.mark(wl::kPhaseBegin, kRead);
#pragma unroll 1
    for (int it = 0; it < kSharedTrips; ++it) {
#pragma unroll
      for (int k = 0; k < kSharedTrip; ++k) {
        uint32_t x, y, z, v;
        asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(v) : "r"(at[k]));
        sum += x + y + z + v;
      }
    }
  }
  out[blockIdx.x * kThreads + threadIdx.x] = sum;
  t.mark(wl::kLaunchEnd, kExit);
  t.close();
}

template <bool Stamp>
__global__ void __launch_bounds__(kThreads, 1) mma(wl::Entry* rows, float* out) {
  const uint32_t a = 0x3C003C00u, b = 0x1C001C00u;
  float c[kMma][4] = {};
  Line<Stamp> t;
  t.bind(rows);
  t.mark(wl::kLaunchBegin, kEntry);
  for (int w = 0; w < kWindows; ++w) {
    t.mark(wl::kPhaseBegin, kMmaSite);
#pragma unroll 1
    for (int it = 0; it < kMmaTrips; ++it) {
#pragma unroll
      for (int m = 0; m < kMma; ++m)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(c[m][0]), "+f"(c[m][1]), "+f"(c[m][2]), "+f"(c[m][3])
            : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
    }
  }
  float s = 0;
#pragma unroll
  for (int m = 0; m < kMma; ++m) s += c[m][0] + c[m][1] + c[m][2] + c[m][3];
  out[blockIdx.x * kThreads + threadIdx.x] = s;
  t.mark(wl::kLaunchEnd, kExit);
  t.close();
}

template <bool Stamp>
__global__ void __launch_bounds__(kThreads, 1) serial(wl::Entry* rows, const uint4* table, uint32_t* out) {
  Line<Stamp> t;
  t.bind(rows);
  t.mark(wl::kLaunchBegin, kEntry);
  const uint4* base = table + blockIdx.x * kUnitsPerCta;
  uint32_t sum = 0;
  float x[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) x[j] = static_cast<float>(j);
  for (int w = 0; w < kWindows; ++w) {
    //: the phases are the CTA's, between barriers: no warp's math runs under another's load
    __syncthreads();
    t.mark(wl::kPhaseBegin, kLoad);
    sum = read_window(base, w * (kUnitsPerCta / kWindows), sum);
    __syncthreads();
    t.mark(wl::kPhaseBegin, kMath);
#pragma unroll 1
    for (int it = 0; it < kMathTrips; ++it)
#pragma unroll
      for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int j = 0; j < 16; ++j) asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3F800000;" : "+f"(x[j]));
  }
  float s = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) s += x[j];
  out[blockIdx.x * kThreads + threadIdx.x] = sum + static_cast<uint32_t>(s);
  t.mark(wl::kLaunchEnd, kExit);
  t.close();
}

}  // namespace floor_examples

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e = (x);                                                         \
    if (e != cudaSuccess) {                                                      \
      std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));               \
      return 2;                                                                  \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  using namespace floor_examples;
  if (argc < 2) {
    std::fprintf(stderr, "usage: floor_examples stream|shared|mma|serial [record.json]\n");
    return 2;
  }
  const std::string which = argv[1];
  int ctas = 0;
  CK(cudaDeviceGetAttribute(&ctas, cudaDevAttrMultiProcessorCount, 0));
  uint4* table = nullptr;
  uint32_t* out = nullptr;
  CK(cudaMalloc(&out, ctas * kThreads * 4));
  if (which == "stream" || which == "serial") {
    CK(cudaMalloc(&table, ctas * kUnitsPerCta * 16));
    CK(cudaMemset(table, 1, ctas * kUnitsPerCta * 16));
  }
  wl::run::Rows<kCap> rows(ctas);
  const auto launch = [&](bool stamped) {
    if (which == "stream") {
      if (stamped) stream<true><<<ctas, kThreads>>>(rows.device(), table, out);
      else stream<false><<<ctas, kThreads>>>(rows.device(), table, out);
    } else if (which == "shared") {
      if (stamped) shared<true><<<ctas, kThreads>>>(rows.device(), out);
      else shared<false><<<ctas, kThreads>>>(rows.device(), out);
    } else if (which == "mma") {
      if (stamped) mma<true><<<ctas, kThreads>>>(rows.device(), reinterpret_cast<float*>(out));
      else mma<false><<<ctas, kThreads>>>(rows.device(), reinterpret_cast<float*>(out));
    } else {
      if (stamped) serial<true><<<ctas, kThreads>>>(rows.device(), table, out);
      else serial<false><<<ctas, kThreads>>>(rows.device(), table, out);
    }
  };
  if (which != "stream" && which != "shared" && which != "mma" && which != "serial") {
    std::fprintf(stderr, "floor_examples: no example %s\n", which.c_str());
    return 2;
  }
  //: two plain launches to warm the clocks and the caches, then the stamped one inside the region
  for (int i = 0; i < 2; ++i) launch(false);
  CK(cudaDeviceSynchronize());
  rows.clear();
  {
    wl::run::Region region;
    launch(true);
  }
  CK(cudaGetLastError());
  wl::host::Record rec;
  const std::string role = which == "mma" ? "math" : which == "serial" ? "worker" : "reader";
  rec.name(kRead, "read");
  rec.name(kMmaSite, "mma");
  rec.name(kLoad, "load");
  rec.name(kMath, "math");
  rows.collect(rec, [&](uint32_t i) { return wl::host::Lane{"cta" + std::to_string(i), "warp0", role}; });
  rec.source("kernel", "examples/floor/floor_examples.cu " + which);
  //: the record goes to the run (`warpline run --stamps`), else to the path given, else nowhere (a profiled run's)
  if (!rec.write_to_run(which) && argc > 2) {
    rec.write(argv[2]);
    std::printf("wrote %s\n", argv[2]);
  }
  return 0;
}
