// probes/rates.cu -- THE RATES TABLE a floor is priced at (docs/floor.md): each stream's work a cycle an SM on the card
// this runs on, measured, each row with its own exact check, written as `warpline-rates/1`.
//
//   rates.bin rates.json          # every row; a row whose check fails is not written and the program exits 1
//
// METHOD. A CTA an SM, 512 threads (16 warps). Each launch records, a CTA, the nanosecond timer and the SM's cycle counter
// at its start and end (after a barrier): the launch's SPAN is the last end less the first start on the timer, its CLOCK
// the CTAs' mean cycles over nanoseconds, and a row's RATE the launch's work over (SMs x span x clock). A row is the best of
// kRepeats launches after a warm-up (the rate the stream reached: a floor priced at it is never above what the stream can
// do), its other runs written beside it, with its launch's clock; the table's clock is the highest a row ran at (an SM
// stream's floor in nanoseconds is taken at it). Every launch's result is checked exactly against the host (a sum of every word a
// thread read, every word written, every accumulator), and a row whose check fails is not written.
//
// THE WORK A TRIP is a constant here (kGlobalTrip, kSharedTrip, kMmaTrip, kMixTrip, kFloatTrip), and the build holds each
// against the compiled loop: `probes/BUILD.bazel`'s `rates_check` bounds the loop's class counts and instructions exactly
// (`rates.declarations.json`), so the rate is the work the SASS does.
//
// THE COUNTERS: each row names the profiler's metrics that count its work (`counter`), which a profiled run of a kernel reads
// to say which stream its counters show (`floor.shown`).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace probes {

constexpr int kThreads = 512, kRepeats = 5;
//: a global trip: 8 independent 16-byte loads (or stores) a thread, 512 apart: 64 KB a CTA
constexpr int kGlobalTrip = 8;
//: a shared trip: 6 16-byte loads (or stores) a thread at fixed conflict-free addresses: the 48 KB a CTA holds
constexpr int kSharedTrip = 6;
//: a tensor trip: 4 independent m16n8k16 MMAs a warp
constexpr int kMmaTrip = 4;
//: an issue trip: 64 FFMA and 64 integer adds over 16 chains, and the loop's own 3 instructions (counter, compare, branch)
constexpr int kMixTrip = 131;
//: a float trip: 128 FFMA over 16 chains
constexpr int kFloatTrip = 128;

struct Span {
  unsigned long long t0, t1, c0, c1;
};

__device__ __forceinline__ unsigned long long timer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void span_begin(Span* s) {
  __syncthreads();
  if (threadIdx.x == 0) {
    s[blockIdx.x].t0 = timer();
    s[blockIdx.x].c0 = clock64();
  }
  __syncthreads();
}

__device__ __forceinline__ void span_end(Span* s) {
  __syncthreads();
  if (threadIdx.x == 0) {
    s[blockIdx.x].c1 = clock64();
    s[blockIdx.x].t1 = timer();
  }
}

//: the table's 32-bit word at index `w`
__host__ __device__ constexpr uint32_t word_of(uint64_t w) { return static_cast<uint32_t>(w) * 2654435761u + 1u; }

__host__ __device__ inline uint32_t fold(uint64_t e) {
  return word_of(4 * e) + word_of(4 * e + 1) + word_of(4 * e + 2) + word_of(4 * e + 3);
}

__global__ void fill(uint4* table, uint64_t count) {
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    table[i] = make_uint4(word_of(4 * i), word_of(4 * i + 1), word_of(4 * i + 2), word_of(4 * i + 3));
}

//: HBM READ: a CTA its own slice of `per` 16-byte units once, a trip 8 units a thread in flight
__global__ void __launch_bounds__(kThreads, 1) hbm_read(const uint4* src, uint64_t per, uint32_t* sums, Span* s) {
  span_begin(s);
  const uint4* base = src + blockIdx.x * per;
  uint32_t sum = 0;
#pragma unroll 1
  for (uint64_t i = threadIdx.x; i < per; i += kThreads * kGlobalTrip) {
    uint4 v[kGlobalTrip];
#pragma unroll
    for (int k = 0; k < kGlobalTrip; ++k) v[k] = base[i + k * kThreads];
#pragma unroll
    for (int k = 0; k < kGlobalTrip; ++k) sum += v[k].x + v[k].y + v[k].z + v[k].w;
  }
  span_end(s);
  sums[blockIdx.x * kThreads + threadIdx.x] = sum;
}

//: HBM WRITE: a CTA its own slice, each unit the words of its index
__global__ void __launch_bounds__(kThreads, 1) hbm_write(uint4* dst, uint64_t per, Span* s) {
  span_begin(s);
  const uint64_t first = blockIdx.x * per;
#pragma unroll 1
  for (uint64_t i = threadIdx.x; i < per; i += kThreads * kGlobalTrip) {
#pragma unroll
    for (int k = 0; k < kGlobalTrip; ++k) {
      const uint64_t e = first + i + k * kThreads;
      dst[e] = make_uint4(word_of(4 * e), word_of(4 * e + 1), word_of(4 * e + 2), word_of(4 * e + 3));
    }
  }
  span_end(s);
}

//: HBM COPY: a CTA its own slice of `src` into the same slice of `dst`
__global__ void __launch_bounds__(kThreads, 1) hbm_copy(const uint4* src, uint4* dst, uint64_t per, Span* s) {
  span_begin(s);
  const uint64_t first = blockIdx.x * per;
#pragma unroll 1
  for (uint64_t i = threadIdx.x; i < per; i += kThreads * kGlobalTrip) {
    uint4 v[kGlobalTrip];
#pragma unroll
    for (int k = 0; k < kGlobalTrip; ++k) v[k] = src[first + i + k * kThreads];
#pragma unroll
    for (int k = 0; k < kGlobalTrip; ++k) dst[first + i + k * kThreads] = v[k];
  }
  span_end(s);
}

//: L2 PAST L1 (`ld.global.cg`): a CTA reads a footprint of `slice` units (a power of two), `reps` times. OWN: its own
//: slice of the table (base `blockIdx.x * slice`); SHARED: every CTA the table's first `slice` units, from its own offset
__global__ void __launch_bounds__(kThreads, 1) l2_read(const uint4* src, uint64_t slice, int own, int reps, uint32_t* sums,
                                                       Span* s) {
  span_begin(s);
  const uint4* base = src + (own ? blockIdx.x * slice : 0);
  const uint64_t mask = slice - 1, rot = own ? 0 : (blockIdx.x * 4096ull) & mask;
  uint32_t sum = 0;
#pragma unroll 1
  for (int r = 0; r < reps; ++r) {
#pragma unroll 1
    for (uint64_t i = threadIdx.x; i < slice; i += kThreads * kGlobalTrip) {
      uint4 v[kGlobalTrip];
#pragma unroll
      for (int k = 0; k < kGlobalTrip; ++k) v[k] = __ldcg(base + ((rot + i + k * kThreads) & mask));
#pragma unroll
      for (int k = 0; k < kGlobalTrip; ++k) sum += v[k].x + v[k].y + v[k].z + v[k].w;
    }
  }
  span_end(s);
  sums[blockIdx.x * kThreads + threadIdx.x] = sum;
}

__device__ __forceinline__ uint32_t shared_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

//: SHARED READ: 48 KB a CTA, each thread 6 16-byte loads a trip at fixed addresses (a warp's 512 bytes contiguous: 4
//: wavefronts, no conflict), `ld.volatile` so none is hoisted out of the loop (the assembler hoists a plain one)
__global__ void __launch_bounds__(kThreads, 1) shared_read(int iters, uint32_t* sums, Span* s) {
  __shared__ uint4 tile[kSharedTrip * kThreads];
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k) {
    const int i = k * kThreads + threadIdx.x;
    tile[i] = make_uint4(word_of(4 * i), word_of(4 * i + 1), word_of(4 * i + 2), word_of(4 * i + 3));
  }
  span_begin(s);
  uint32_t at[kSharedTrip];
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k) at[k] = shared_address(&tile[k * kThreads + threadIdx.x]);
  uint32_t sum = 0;
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < kSharedTrip; ++k) {
      uint32_t x, y, z, w;
      asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(at[k]));
      sum += x + y + z + w;
    }
  }
  span_end(s);
  sums[blockIdx.x * kThreads + threadIdx.x] = sum;
}

//: SHARED WRITE: the same addresses, each trip's words formed from the trip; the last trip's are copied out for the check
__global__ void __launch_bounds__(kThreads, 1) shared_write(int iters, uint4* out, Span* s) {
  __shared__ uint4 tile[kSharedTrip * kThreads];
  span_begin(s);
  uint32_t at[kSharedTrip];
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k) at[k] = shared_address(&tile[k * kThreads + threadIdx.x]);
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const uint32_t v = it * 7u + threadIdx.x;
#pragma unroll
    for (int k = 0; k < kSharedTrip; ++k)
      asm volatile("st.volatile.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(at[k]), "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + k)
                   : "memory");
  }
  span_end(s);
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kSharedTrip; ++k)
    out[blockIdx.x * kSharedTrip * kThreads + k * kThreads + threadIdx.x] = tile[k * kThreads + threadIdx.x];
}

//: TENSOR: 4 independent m16n8k16 f16 MMAs into f32 a trip a warp; A all 1.0, B all 2^-8, so each MMA adds 16 x 2^-8 = 1/16
//: to every accumulator, exactly
__global__ void __launch_bounds__(kThreads, 1) mma_loop(int iters, float* out, Span* s) {
  const uint32_t a = 0x3C003C00u, b = 0x1C001C00u;
  float c[kMmaTrip][4] = {};
  span_begin(s);
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int m = 0; m < kMmaTrip; ++m)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
          : "+f"(c[m][0]), "+f"(c[m][1]), "+f"(c[m][2]), "+f"(c[m][3])
          : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
  }
  span_end(s);
  float* o = out + (blockIdx.x * kThreads + threadIdx.x) * kMmaTrip * 4;
#pragma unroll
  for (int m = 0; m < kMmaTrip; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) o[m * 4 + j] = c[m][j];
}

//: the integer step of chain `j`: y_j ^= y_(j+1) & y_(j+2), one LOP3 whose result the next two steps read, so the assembler
//: can fold no two into one (it folds a chain of adds of a constant into a single add)
__host__ __device__ inline uint32_t mix_step(uint32_t a, uint32_t b, uint32_t c) { return a ^ (b & c); }

//: ISSUE: 8 float chains (x = x * 1 + 1) and 8 integer chains (`mix_step`), interleaved, 8 rounds a trip: two pipes, so the
//: warps' issue is what binds
__global__ void __launch_bounds__(kThreads, 1) issue_mix(int iters, float* outf, uint32_t* outi, Span* s) {
  float x[8];
  uint32_t y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = static_cast<float>(j);
    y[j] = word_of(threadIdx.x * 8 + j);
  }
  span_begin(s);
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3F800000;" : "+f"(x[j]));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x78;" : "+r"(y[j]) : "r"(y[(j + 1) % 8]), "r"(y[(j + 2) % 8]));
      }
  }
  span_end(s);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    outf[(blockIdx.x * kThreads + threadIdx.x) * 8 + j] = x[j];
    outi[(blockIdx.x * kThreads + threadIdx.x) * 8 + j] = y[j];
  }
}

//: FLOAT: 16 float chains, 8 rounds a trip: the float pipe binds
__global__ void __launch_bounds__(kThreads, 1) issue_float(int iters, float* outf, Span* s) {
  float x[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) x[j] = static_cast<float>(j);
  span_begin(s);
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
      for (int j = 0; j < 16; ++j) asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3F800000;" : "+f"(x[j]));
  }
  span_end(s);
#pragma unroll
  for (int j = 0; j < 16; ++j) outf[(blockIdx.x * kThreads + threadIdx.x) * 16 + j] = x[j];
}

}  // namespace probes

namespace {

using namespace probes;

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e = (x);                                                                      \
    if (e != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                            \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

struct Row {
  std::string stream, extra, probe, check, metrics;
  double per = 1;
  int sms = 0;
  double rate = 0, clock = 0;
  std::vector<double> runs;
  bool ok = false;
};

struct Device {
  int sms = 0;
  std::string name;
  Span* spans = nullptr;
};

//: a launch's rate: `work` over (CTAs x span x clock), and its clock, from the spans of `ctas` CTAs
std::pair<double, double> rate_of(const Device& d, int ctas, double work) {
  std::vector<Span> s(ctas);
  CK(cudaMemcpy(s.data(), d.spans, ctas * sizeof(Span), cudaMemcpyDeviceToHost));
  unsigned long long t0 = ~0ull, t1 = 0;
  double clock = 0;
  for (const Span& x : s) {
    t0 = std::min(t0, x.t0);
    t1 = std::max(t1, x.t1);
    clock += static_cast<double>(x.c1 - x.c0) / static_cast<double>(x.t1 - x.t0) / ctas;
  }
  return {work / (ctas * static_cast<double>(t1 - t0) * clock), clock};
}

//: kRepeats launches after a warm-up: the best rate, every run's, the clock of the best; `check()` after the last
template <class Launch, class Check>
Row measure(const Device& d, int ctas, double work, Row row, Launch launch, Check check) {
  launch();
  CK(cudaDeviceSynchronize());
  for (int i = 0; i < kRepeats; ++i) {
    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    auto [rate, clock] = rate_of(d, ctas, work);
    row.runs.push_back(rate);
    if (rate > row.rate) {
      row.rate = rate;
      row.clock = clock;
    }
  }
  row.sms = ctas;
  row.check = check();
  row.ok = row.check.rfind("exact", 0) == 0;
  if (!row.ok) std::fprintf(stderr, "%s: %s\n", row.stream.c_str(), row.check.c_str());
  return row;
}

std::string counted(long long wrong, long long checked, const char* what) {
  return wrong ? std::to_string(wrong) + " of " + std::to_string(checked) + " " + what + " differ"
               : "exact: " + std::to_string(checked) + " " + what;
}

std::string json_text(const std::string& s) {
  std::string out = "\"";
  for (char c : s) out += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
  return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "rates.json";
  Device d;
  CK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  d.name = prop.name;
  const int S = d.sms;
  CK(cudaMalloc(&d.spans, S * sizeof(Span)));
  //: the tables: 1 GiB each (2^26 units of 16 bytes)
  constexpr uint64_t kUnits = 1ull << 26;
  uint4 *src = nullptr, *dst = nullptr;
  uint32_t* sums = nullptr;
  float* outf = nullptr;
  uint32_t* outi = nullptr;
  CK(cudaMalloc(&src, kUnits * 16));
  CK(cudaMalloc(&dst, kUnits * 16));
  CK(cudaMalloc(&sums, S * kThreads * 4));
  CK(cudaMalloc(&outf, S * kThreads * 16 * 4));
  CK(cudaMalloc(&outi, S * kThreads * 8 * 4));
  probes::fill<<<S * 4, 256>>>(src, kUnits);
  CK(cudaDeviceSynchronize());
  std::vector<Row> rows;
  const uint64_t trip = kThreads * kGlobalTrip;

  //: each thread's sum of the words it read: `reps` x the words of the units `base + i`, i = tid mod 512, i < `slice`
  const auto sums_check = [&](uint64_t slice, int own, int reps, int ctas) {
    std::vector<uint32_t> got(ctas * kThreads);
    CK(cudaMemcpy(got.data(), sums, got.size() * 4, cudaMemcpyDeviceToHost));
    long long wrong = 0;
    std::vector<uint32_t> shared(kThreads, 0);
    if (!own)
      for (uint64_t i = 0; i < slice; ++i) shared[i % kThreads] += fold(i);
    for (int b = 0; b < ctas; ++b)
      for (int t = 0; t < kThreads; ++t) {
        uint32_t want = shared[t];
        if (own) {
          want = 0;
          for (uint64_t i = t; i < slice; i += kThreads) want += fold(b * slice + i);
        }
        wrong += got[b * kThreads + t] != want * static_cast<uint32_t>(reps);
      }
    return counted(wrong, static_cast<long long>(ctas) * kThreads, "threads' sums of the words read");
  };
  //: a table's units against their index's words (a write) or the source's, which are the same (a copy)
  const auto table_check = [&](uint64_t units) {
    std::vector<uint4> got(units);
    CK(cudaMemcpy(got.data(), dst, units * 16, cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (uint64_t e = 0; e < units; ++e)
      wrong += got[e].x != word_of(4 * e) || got[e].y != word_of(4 * e + 1) || got[e].z != word_of(4 * e + 2) ||
               got[e].w != word_of(4 * e + 3);
    return counted(wrong, static_cast<long long>(units), "units written");
  };

  //: HBM: a CTA its own slice of the 1 GiB table, a whole number of trips
  const uint64_t per = kUnits / S / trip * trip;
  rows.push_back(measure(d, S, per * S * 16.0,
                         {"hbm.read", "", "a CTA its own slice of a 1 GiB table once, 16-byte loads, 8 a thread in flight", "",
                          "\"dram__bytes_read.sum\""},
                         [&] { hbm_read<<<S, kThreads>>>(src, per, sums, d.spans); },
                         [&] { return sums_check(per, 1, 1, S); }));
  CK(cudaMemset(dst, 0, kUnits * 16));
  rows.push_back(measure(d, S, per * S * 16.0,
                         {"hbm.write", "", "a CTA its own slice of a 1 GiB table once, 16-byte stores", "",
                          "\"dram__bytes_write.sum\""},
                         [&] { hbm_write<<<S, kThreads>>>(dst, per, d.spans); }, [&] { return table_check(per * S); }));
  const uint64_t half = kUnits / 2 / S / trip * trip;
  CK(cudaMemset(dst, 0, kUnits * 16));
  rows.push_back(measure(d, S, half * S * 32.0,
                         {"hbm", "", "a copy: a CTA its own slice of 512 MB into 512 MB (the bytes read and written)", "",
                          "\"dram__bytes_read.sum\", \"dram__bytes_write.sum\""},
                         [&] { hbm_copy<<<S, kThreads>>>(src, dst, half, d.spans); }, [&] { return table_check(half * S); }));
  //: L2 OWN: a CTA its own slice, 8 MB a CTA in all: 128 KB, or the largest power of two whose slices, every SM's, fit
  //: the device's L2 (64 KB at least, a trip)
  int l2 = 0;
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  uint64_t own = 131072;
  while (own > 65536 && own * S > static_cast<uint64_t>(l2)) own /= 2;
  const int own_reps = static_cast<int>((8ull << 20) / own);
  rows.push_back(measure(d, S, static_cast<double>(own) * own_reps * S,
                         {"l2.own", "",
                          "past L1 (ld.global.cg): a CTA its own " + std::to_string(own / 1024) + " KB slice " +
                              std::to_string(own_reps) + " times (every SM's slice in an L2 of " + std::to_string(l2 >> 20) +
                              " MB)",
                          "", "\"lts__t_bytes.sum\""},
                         [&] { l2_read<<<S, kThreads>>>(src, own / 16, 1, own_reps, sums, d.spans); },
                         [&] { return sums_check(own / 16, 1, own_reps, S); }));
  //: L2 SHARED: every CTA one table of the footprint, from its own offset, 8 MB a CTA in all; a footprint the device's L2
  //: does not hold is no L2 row
  for (uint64_t footprint : {131072ull, 1048576ull, 4194304ull, 16777216ull}) {
    if (footprint > static_cast<uint64_t>(l2)) continue;
    const uint64_t slice = footprint / 16;
    const int reps = static_cast<int>(std::max<uint64_t>(1, (8ull << 20) / footprint));
    Row row{"l2.shared", ", \"footprint\": " + std::to_string(footprint),
            "past L1 (ld.global.cg): every CTA one " + std::to_string(footprint) + "-byte table " + std::to_string(reps) +
                " times from its own offset",
            "", "\"lts__t_bytes.sum\""};
    rows.push_back(measure(d, S, static_cast<double>(footprint) * reps * S, row,
                           [&] { l2_read<<<S, kThreads>>>(src, slice, 0, reps, sums, d.spans); },
                           [&] { return sums_check(slice, 0, reps, S); }));
  }
  //: SHARED: 48 KB a CTA, 6 16-byte accesses a thread a trip
  constexpr int kSharedIters = 1 << 14;
  rows.push_back(measure(d, S, 16.0 * kSharedTrip * kThreads * kSharedIters * S,
                         {"shared.read", "", "16-byte loads, a warp's 512 bytes contiguous (4 wavefronts, no conflict), 48 KB a CTA",
                          "", "\"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum\"", 128},
                         [&] { shared_read<<<S, kThreads>>>(kSharedIters, sums, d.spans); },
                         [&] {
                           std::vector<uint32_t> got(S * kThreads);
                           CK(cudaMemcpy(got.data(), sums, got.size() * 4, cudaMemcpyDeviceToHost));
                           long long wrong = 0;
                           for (int b = 0; b < S; ++b)
                             for (int t = 0; t < kThreads; ++t) {
                               uint32_t want = 0;
                               for (int k = 0; k < kSharedTrip; ++k) want += fold(k * kThreads + t);
                               wrong += got[b * kThreads + t] != want * static_cast<uint32_t>(kSharedIters);
                             }
                           return counted(wrong, static_cast<long long>(S) * kThreads, "threads' sums of the words read");
                         }));
  rows.push_back(measure(d, S, 16.0 * kSharedTrip * kThreads * kSharedIters * S,
                         {"shared.write", "", "16-byte stores at the same addresses", "",
                          "\"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum\"", 128},
                         [&] { shared_write<<<S, kThreads>>>(kSharedIters, dst, d.spans); },
                         [&] {
                           std::vector<uint4> got(S * kSharedTrip * kThreads);
                           CK(cudaMemcpy(got.data(), dst, got.size() * 16, cudaMemcpyDeviceToHost));
                           long long wrong = 0;
                           for (int b = 0; b < S; ++b)
                             for (int k = 0; k < kSharedTrip; ++k)
                               for (int t = 0; t < kThreads; ++t) {
                                 const uint32_t v = (kSharedIters - 1) * 7u + t;
                                 const uint4 g = got[(b * kSharedTrip + k) * kThreads + t];
                                 wrong += g.x != v || g.y != v + 1 || g.z != v + 2 || g.w != v + k;
                               }
                           return counted(wrong, static_cast<long long>(S) * kSharedTrip * kThreads, "units written");
                         }));
  //: TENSOR: m16n8k16, f16 in, f32 accumulated
  constexpr int kMmaIters = 1 << 18;
  rows.push_back(measure(d, S, static_cast<double>(kMmaTrip) * kMmaIters * (kThreads / 32) * S,
                         {"tensor", ", \"shape\": \"m16n8k16\", \"dtype\": \"f16.f32\"",
                          "mma.sync m16n8k16 f16 into f32: 4 independent accumulators a warp, 16 warps a CTA", "",
                          "\"sm__inst_executed_pipe_tensor_op_hmma.sum\""},
                         [&] { mma_loop<<<S, kThreads>>>(kMmaIters, outf, d.spans); },
                         [&] {
                           std::vector<float> got(S * kThreads * kMmaTrip * 4);
                           CK(cudaMemcpy(got.data(), outf, got.size() * 4, cudaMemcpyDeviceToHost));
                           long long wrong = 0;
                           for (float v : got) wrong += v != kMmaIters / 16.0f;
                           return counted(wrong, static_cast<long long>(got.size()), "accumulators");
                         }));
  //: ISSUE: the mix (every class the loop holds), and the float pipe alone
  constexpr int kIssueIters = 1 << 17;
  rows.push_back(measure(d, S, static_cast<double>(kMixTrip) * kIssueIters * (kThreads / 32) * S,
                         {"issue", "", "8 FFMA and 8 integer-add chains interleaved, 131 warp instructions a trip (the loop's 3 counted)",
                          "", "\"sm__inst_executed.sum\""},
                         [&] { issue_mix<<<S, kThreads>>>(kIssueIters, outf, outi, d.spans); },
                         [&] {
                           //: the float chains of the timed launch, and the integer chains of a short one (64 trips) against
                           //: the host's own steps
                           std::vector<float> f(S * kThreads * 8);
                           CK(cudaMemcpy(f.data(), outf, f.size() * 4, cudaMemcpyDeviceToHost));
                           constexpr int kShort = 64;
                           issue_mix<<<S, kThreads>>>(kShort, outf, outi, d.spans);
                           CK(cudaDeviceSynchronize());
                           std::vector<uint32_t> i(S * kThreads * 8);
                           CK(cudaMemcpy(i.data(), outi, i.size() * 4, cudaMemcpyDeviceToHost));
                           long long wrong = 0;
                           for (size_t n = 0; n < f.size(); ++n) wrong += f[n] != static_cast<float>(n % 8) + 8.0f * kIssueIters;
                           for (int t = 0; t < kThreads; ++t) {
                             uint32_t y[8];
                             for (int j = 0; j < 8; ++j) y[j] = word_of(t * 8 + j);
                             for (int step = 0; step < kShort * 8; ++step)
                               for (int j = 0; j < 8; ++j) y[j] = mix_step(y[j], y[(j + 1) % 8], y[(j + 2) % 8]);
                             for (int b = 0; b < S; ++b)
                               for (int j = 0; j < 8; ++j) wrong += i[(b * kThreads + t) * 8 + j] != y[j];
                           }
                           return counted(wrong, static_cast<long long>(2 * f.size()), "chains");
                         }));
  rows.push_back(measure(d, S, static_cast<double>(kFloatTrip) * kIssueIters * (kThreads / 32) * S,
                         {"issue.float", "", "16 FFMA chains, 128 FFMA a trip", "",
                          "\"smsp__sass_thread_inst_executed_op_fp32_pred_on.sum\"", 1.0 / 32},
                         [&] { issue_float<<<S, kThreads>>>(kIssueIters, outf, d.spans); },
                         [&] {
                           std::vector<float> f(S * kThreads * 16);
                           CK(cudaMemcpy(f.data(), outf, f.size() * 4, cudaMemcpyDeviceToHost));
                           long long wrong = 0;
                           for (size_t n = 0; n < f.size(); ++n) wrong += f[n] != static_cast<float>(n % 16) + 8.0f * kIssueIters;
                           return counted(wrong, static_cast<long long>(f.size()), "chains");
                         }));

  bool all = true;
  std::vector<double> clocks;
  for (const Row& r : rows) {
    all = all && r.ok;
    if (r.ok) clocks.push_back(r.clock);
  }
  std::sort(clocks.begin(), clocks.end());
  FILE* f = std::fopen(path, "w");
  if (!f || clocks.empty()) {
    std::fprintf(stderr, "rates: %s\n", f ? "no row passed its check" : "cannot write the table");
    return 2;
  }
  std::fprintf(f, "{\"format\": \"warpline-rates/1\", \"device\": %s, \"sms\": %d, \"clock_ghz\": %.4f,\n \"about\": %s,\n \"rows\": [",
               json_text(d.name).c_str(), S, clocks.back(),
               json_text("probes/rates.cu: a row's rate is the best of " + std::to_string(kRepeats) +
                         " launches, a CTA an SM, at its own clock; clock_ghz the highest clock a row ran at").c_str());
  bool first = true;
  for (const Row& r : rows) {
    if (!r.ok) continue;
    std::string runs;
    for (double x : r.runs) runs += (runs.empty() ? "" : ", ") + std::to_string(x);
    std::fprintf(f, "%s\n  {\"stream\": \"%s\"%s, \"rate\": %.6g, \"sms\": %d, \"clock_ghz\": %.4f, \"runs\": [%s],\n   \"probe\": %s, \"check\": %s,\n   \"counter\": {\"metrics\": [%s], \"per\": %.6g}}",
                 first ? "" : ",", r.stream.c_str(), r.extra.c_str(), r.rate, r.sms, r.clock, runs.c_str(),
                 json_text(r.probe).c_str(), json_text(r.check).c_str(), r.metrics.c_str(), r.per);
    first = false;
  }
  std::fprintf(f, "]}\n");
  std::fclose(f);
  std::printf("wrote %s: %zu rows\n", path, rows.size());
  return all ? 0 : 1;
}
