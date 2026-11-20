// An example kernel for the checker, never run: it holds, on purpose, the five shapes the checker's model reads.
//
//   (a) a counted loop with no tensor step                      the loop in `main` that sums `n` words
//   (b) a loop around tensor steps that also holds their wait   `loop_with_wait`
//   (c) a rolled loop around tensor steps WITHOUT their wait    `loop_without_wait` (the rule `step_rolled` fires)
//   (d) suspended mbarrier waits (`try_wait` with a time hint)  `wait_full`, `wait_empty`: the toolkit emits them as
//                                                               out-of-line stubs that jump back into the body
//   (e) a tensor wait directly after its steps, and one with    `bare_wait` (the rule `wait_after_issue` fires) and
//       work between                                            `wait_with_work`
//
// Build, and save the three outputs the tests read (tests/checker_fixtures/example/; docs/checker-model.md):
//
//   nvcc -arch=sm_90a -cubin steps.cu -o steps.cubin
//   cuobjdump -sass steps.cubin > kernel.sass ; nvdisasm -bbcfg -poff steps.cubin > kernel.cfg.dot
//   cuobjdump -res-usage steps.cubin > kernel.res
#include <cstdint>

namespace example {

struct Tile {
    uint64_t full, empty;       // two mbarriers, in shared memory
    uint32_t a[256], b[256];    // the two operands' staging
};

__device__ inline uint64_t desc(const void* p) {
    return (uint64_t(static_cast<uint32_t>(__cvta_generic_to_shared(p))) >> 4) & 0x3fff | (1ull << 62);
}

// one tensor step: a 64x16x16 bf16 matrix multiply-accumulate from two shared-memory descriptors into `d`
__device__ inline void step(float (&d)[8], uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(da), "l"(db));
}
__device__ inline void fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ inline void commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ inline void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

// a suspended wait on the barrier's phase: the instruction suspends the thread for up to the hint, the loop retries
__device__ inline void spin(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 0x989680;\n"
        "@!p bra WAIT_%=;\n}" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity)
        : "memory");
}
__device__ inline void wait_full(Tile& t, uint32_t phase) { spin(&t.full, phase); }
__device__ inline void wait_empty(Tile& t, uint32_t phase) { spin(&t.empty, phase); }

// (b) the steps and their wait are in the loop's body
__device__ inline void loop_with_wait(float (&d)[8], Tile& t, int n) {
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        fence();
        step(d, desc(&t.a[0]), desc(&t.b[0]));
        step(d, desc(&t.a[64]), desc(&t.b[64]));
        commit();
        wait<0>();
    }
}

// (c) the steps are in the loop, their wait after it
__device__ inline void loop_without_wait(float (&d)[8], Tile& t, int n) {
    fence();
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        step(d, desc(&t.a[i * 64 % 256]), desc(&t.b[i * 64 % 256]));
        commit();
    }
    wait<0>();
}

// (e) the wait directly after the steps, and the wait with loads and stores in between
__device__ inline void bare_wait(float (&d)[8], Tile& t) {
    fence();
    step(d, desc(&t.a[128]), desc(&t.b[128]));
    commit();
    wait<0>();
}
__device__ inline void wait_with_work(float (&d)[8], Tile& t, uint32_t* out) {
    fence();
    step(d, desc(&t.a[192]), desc(&t.b[192]));
    commit();
    out[threadIdx.x] = t.a[threadIdx.x] + t.b[threadIdx.x];
    t.a[threadIdx.x] = out[threadIdx.x + 128];
    wait<0>();
}

__global__ void __launch_bounds__(128) main(float* out, uint32_t* words, int n, int rounds) {
    __shared__ Tile t;
    float d[8] = {};
    uint32_t sum = 0;
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) sum += words[i * 128 + threadIdx.x];       // (a) a counted loop
    t.a[threadIdx.x] = sum;
    t.b[threadIdx.x] = sum + 1;
    for (int r = 0; r < rounds; ++r) {      // the long loop the waits sit in
        wait_full(t, r & 1);                                                // (d)
        loop_with_wait(d, t, n);
        loop_without_wait(d, t, n);
        bare_wait(d, t);
        wait_with_work(d, t, words);
        wait_empty(t, r & 1);                                               // (d)
    }
    for (int i = 0; i < 8; ++i) out[threadIdx.x * 8 + i] = d[i];
}

}  // namespace example
