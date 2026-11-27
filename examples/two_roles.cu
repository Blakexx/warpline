// A small kernel with two warp roles in one CTA, recorded with warpline.
//
//   warp 0, the PRODUCER: each window, fill a tile in shared memory, tell the consumer ("full"), wait for it ("empty").
//   warp 1, the CONSUMER: each window, wait for "full", compute on the tile, tell the producer ("empty").
//
// The stamps are the user's, and so is their switch: `Coarse` and `Detail` below are this file's own compile-time
// conditions, the library knows neither. Coarse: the launch's two edges and each phase's start, as marks (both clocks).
// Detail: every wait and arrival, as plain stamps. With both false, the kernel holds no clock read.
//
// The host side is the run layer's pieces (include/warpline/run.hpp): the rows are a `Rows`, the launch whose record is
// written is inside a `Region` (what a profiler captures), and the record goes to the run when `warpline run --stamps` runs it.
//
//   nvcc -std=c++17 -arch=sm_80 -I../include two_roles.cu -o two_roles
//   ./two_roles record.json            # the recorded kernel; its record
//   python -m warpline verify record.json
//   python -m warpline run --stamps --profile SourceCounters -o run -- ./two_roles    # a run: the record, the exports,
//     with --place 'void two_roles<true, true>(warpline::Entry*, float*)'              # and the run manifest
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "warpline/host.hpp"
#include "warpline/run.hpp"
#include "warpline/stamp.cuh"

namespace wl = warpline;

// The sites: a barrier's, a phase's. Names go in the legend, on the host.
enum Site : uint32_t { kFull = 1, kEmpty = 2, kFill = 10, kCompute = 11, kEntry = 20, kExit = 21 };
constexpr uint32_t kCap = 1024;        // entries a warp's row can hold
constexpr int kWindows = 64;           // iterations of the window loop
constexpr int kCtas = 2;
constexpr int kFillIters = 600, kComputeIters = 1800;  // a window of about 65k cycles: long against the timer's 1 us step
using Cur = wl::Cursor<kCap>;

// The user's own switch, around the library's pieces. `if constexpr (false)` compiles to nothing.
template <bool Coarse, bool Detail>
struct Line {
  Cur c;
  __device__ void bind(wl::Entry* rows) {
    if constexpr (Coarse || Detail) c.bind(rows, blockIdx.x * 2 + threadIdx.x / 32);
  }
  __device__ void edge_mark(uint32_t kind, uint32_t site) {
    if constexpr (Coarse) c.mark(wl::id(kind, site));
  }
  __device__ void phase(uint32_t site) {
    if constexpr (Coarse) c.mark(wl::id(wl::kPhaseBegin, site));
  }
  __device__ void at(uint32_t kind, uint32_t site) {
    if constexpr (Detail) {
      if (kind == wl::kWaitEnd) wl::after_wait();  // a named-barrier wait: see after_wait()
      c.stamp(wl::id(kind, site));
    }
  }
  __device__ void close() {
    if constexpr (Coarse || Detail) c.close();
  }
};

template <bool Coarse, bool Detail>
__global__ void __launch_bounds__(64) two_roles(wl::Entry* rows, float* out) {
  __shared__ float tile[32];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  Line<Coarse, Detail> t;
  t.bind(rows);
  t.edge_mark(wl::kLaunchBegin, kEntry);
  float acc = 0.f;
  for (int w = 0; w < kWindows; ++w) {
    if (warp == 0) {  // producer
      t.phase(kFill);
      if (w > 0) {
        t.at(wl::kWaitBegin, kEmpty);
        asm volatile("bar.sync 2, 64;" ::: "memory");
        t.at(wl::kWaitEnd, kEmpty);
      }
      float v = lane + w;
#pragma unroll 1
      for (int i = 0; i < kFillIters; ++i) v = v * 1.0001f + 0.5f;
      tile[lane] = v;
      __threadfence_block();
      t.at(wl::kArrive, kFull);
      asm volatile("bar.arrive 1, 64;" ::: "memory");
    } else {  // consumer
      t.phase(kCompute);
      t.at(wl::kWaitBegin, kFull);
      asm volatile("bar.sync 1, 64;" ::: "memory");
      t.at(wl::kWaitEnd, kFull);
      float v = tile[lane];
#pragma unroll 1
      for (int i = 0; i < kComputeIters; ++i) v = v * 0.9999f + 0.25f;
      acc += v;
      if (w + 1 < kWindows) {
        t.at(wl::kArrive, kEmpty);
        asm volatile("bar.arrive 2, 64;" ::: "memory");
      }
    }
  }
  out[blockIdx.x * 64 + threadIdx.x] = acc;  // a result the work feeds, so nothing is dropped
  t.edge_mark(wl::kLaunchEnd, kExit);
  t.close();
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

template <bool Coarse, bool Detail>
float run(wl::run::Rows<kCap>& rows, float* out) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  rows.clear();
  cudaEventRecord(a);
  two_roles<Coarse, Detail><<<kCtas, 64>>>(rows.device(), out);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms;
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "record.json";
  wl::run::Rows<kCap> rows(kCtas * 2);  // a row a warp: row cta * 2 + warp, as Line::bind binds it
  float* out;
  CK(cudaMalloc(&out, kCtas * 64 * sizeof(float)));
  // a warm-up, then the plain kernel (no stamps) and the recorded one (coarse + detail), for the cost
  run<false, false>(rows, out);
  float plain = 0, coarse = 0, full = 0;
  for (int i = 0; i < 5; ++i) plain += run<false, false>(rows, out) / 5;
  for (int i = 0; i < 5; ++i) coarse += run<true, false>(rows, out) / 5;
  for (int i = 0; i < 5; ++i) full += run<true, true>(rows, out) / 5;
  CK(cudaGetLastError());
  std::printf("kernel ms: plain %.4f  coarse %.4f  coarse+detail %.4f\n", plain, coarse, full);
  rows.clear();
  {
    wl::run::Region region;  // the run whose record is written: the one launch a profiler captures
    two_roles<true, true><<<kCtas, 64>>>(rows.device(), out);
  }
  CK(cudaGetLastError());

  wl::host::Record rec;
  rec.name(kFull, "full");
  rec.name(kEmpty, "empty");
  rec.name(kFill, "fill");
  rec.name(kCompute, "compute");
  rec.barrier("full", 1);   // one stamped arrival completes a phase of each barrier
  rec.barrier("empty", 1);
  rows.collect(rec, [](uint32_t i) {
    return wl::host::Lane{"cta" + std::to_string(i / 2), "warp" + std::to_string(i % 2), i % 2 ? "consumer" : "producer"};
  });
  rec.source("kernel", "examples/two_roles.cu");
  if (rec.write_to_run("two_roles")) {
    std::printf("wrote the run's record two_roles\n");
  } else {
    rec.write(path);
    std::printf("wrote %s\n", path);
  }
  return 0;
}
