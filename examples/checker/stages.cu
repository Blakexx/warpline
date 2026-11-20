// A second example kernel file for the checker: one small kernel for each shape the remaining rules read, never run.
// Each `__global__` function holds ONE shape on purpose; tests/test_checker_stages.py names what the rules report for it.
//
//   loop_join       a divergent inner loop inside a loop of tensor steps       loop_convergence, step_branches
//   trip_load       a global load read in the trip that issued it            load_in_trip
//   trip_reload     a shared store loaded back in the same trip               store_reload
//   arrive_unfenced a shared store announced by a barrier arrival, no fence   store_unfenced
//   arrive_fenced   the same, with `fence.proxy.async` between                (quiet)
//   step_costs      a conversion and a global access in a loop of steps       step_conversion, step_global
//   clock_sum       a cycle counter read into a uniform register, summed     (quiet: the sum is in the vector datapath)
//                   by thread
//   step_guarded    a tensor step under a condition                           step_predicated (quiet: a branch, not a predicate)
//   stub_join       a divergent region after a suspended barrier wait         loop_convergence (quiet: the stub's return is no loop)
//
// Build, and save the three outputs the tests read (tests/checker_fixtures/stages/; docs/checker-model.md):
//
//   nvcc -arch=sm_90a -cubin stages.cu -o stages.cubin
//   cuobjdump -sass stages.cubin > kernel.sass ; nvdisasm -bbcfg -poff stages.cubin > kernel.cfg.dot
//   cuobjdump -res-usage stages.cubin > kernel.res ; cuobjdump -sass stages.cubin | grep Function | awk '{print $3}' | c++filt > kernel.names
#include <cstdint>

namespace stages {

struct Tile {
    uint64_t full;              // an mbarrier, in shared memory
    uint32_t a[256], b[256];    // the two operands' staging
};

__device__ inline uint64_t desc(const void* p) {
    return (uint64_t(static_cast<uint32_t>(__cvta_generic_to_shared(p))) >> 4) & 0x3fff | (1ull << 62);
}
__device__ inline void step(float (&d)[8], uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(da), "l"(db));
}
__device__ inline void fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ inline void commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ inline void wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ inline void arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))) : "memory");
}
__device__ inline void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// one tensor step from the tile, then its commit and wait: what a loop of steps holds in each of the kernels below
__device__ inline void one_step(float (&d)[8], Tile& t, int k) {
    fence();
    step(d, desc(&t.a[k * 64 % 256]), desc(&t.b[k * 64 % 256]));
    commit();
}
__device__ inline void drain(float (&d)[8], float* out) {
    for (int i = 0; i < 8; ++i) out[threadIdx.x * 8 + i] = d[i];
}

__global__ void __launch_bounds__(128) loop_join(float* out, const uint32_t* flags, int n) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = threadIdx.x; t.b[threadIdx.x] = threadIdx.x;
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        one_step(d, t, i);
        uint32_t f = flags[i * 128 + threadIdx.x];
        while (f > 3) f = f / 3 + (f & 1);                       // a data-dependent inner loop: lanes leave it at different trips
        d[0] += static_cast<float>(f);
        wait0();
    }
    drain(d, out);
}

__global__ void __launch_bounds__(128) trip_load(uint32_t* out, const uint32_t* words, int n) {
    uint32_t sum = 0;
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        uint32_t v = words[i * 128 + threadIdx.x];               // loaded and used in the same trip
        sum = sum * 3 + v;
    }
    out[threadIdx.x] = sum;
}

__global__ void __launch_bounds__(128) trip_reload(uint32_t* out, const uint32_t* words, int n) {
    __shared__ volatile uint32_t keep[128];
    uint32_t sum = words[threadIdx.x];
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        keep[threadIdx.x] = sum * 3 + i;                         // a value no other thread reads, kept in shared memory
        sum += keep[threadIdx.x];
    }
    out[threadIdx.x] = sum;
}

__global__ void __launch_bounds__(128) arrive_unfenced(float* out, int n) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = n + threadIdx.x;
    arrive(&t.full);                                             // announces the store, no fence between
    one_step(d, t, 0);
    wait0();
    drain(d, out);
}

__global__ void __launch_bounds__(128) arrive_fenced(float* out, int n) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = n + threadIdx.x;
    proxy_fence();
    arrive(&t.full);
    one_step(d, t, 0);
    wait0();
    drain(d, out);
}

__global__ void __launch_bounds__(128) step_costs(float* out, int* ints, const float* in, int n) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = threadIdx.x; t.b[threadIdx.x] = threadIdx.x;
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        one_step(d, t, i);
        ints[i * 128 + threadIdx.x] = __float2int_rn(d[1] * in[i]);      // a conversion and a global access in the loop
        wait0();
    }
    drain(d, out);
}

__global__ void __launch_bounds__(128) clock_sum(uint32_t* out, int n) {
    uint32_t total = 0;
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        uint32_t t0 = static_cast<uint32_t>(clock());
        __nanosleep(5);
        total += __reduce_min_sync(0xffffffffu, static_cast<uint32_t>(clock()) - t0);   // a lap per thread, then the warp's minimum
    }
    out[blockIdx.x] = total;
}

__global__ void __launch_bounds__(128) step_guarded(float* out, int n) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = threadIdx.x; t.b[threadIdx.x] = threadIdx.x;
    fence();
    if (n > 3) step(d, desc(&t.a[0]), desc(&t.b[0]));            // a step that may be absent
    commit();
    wait0();
    drain(d, out);
}

// a suspended `try_wait` (the toolkit moves its retry out of line, and the stub returns into the body with a backward
// branch) followed by a divergent region with its convergence barriers and a call to the divide's slow path
__device__ inline void spin(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 0x989680;\n"
        "@!p bra WAIT_%=;\n}" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity)
        : "memory");
}

__global__ void __launch_bounds__(128) stub_join(float* out, const float* in, int n) {
    __shared__ uint64_t bar;
    spin(&bar, n & 1);
    float x = in[threadIdx.x];
    if (x > 0.5f) { x = x / in[threadIdx.x + 128]; x = sqrtf(x); __nanosleep(20); }
    out[threadIdx.x] = x;
}

}  // namespace stages
