// An example kernel for the checker, never run: a loop whose loop-carried accumulators (eight groups of eight registers)
// flow into two blocks of tensor steps behind a run-time branch that each thread decides for itself. The toolkit (13.0,
// sm_90a) serializes the steps (ptxas C7520: "program dependence on compiler-inserted WG.AR in divergent path"): every step
// is followed by its own wait for all groups, the rule `step_serialized` fires. `fenced.cu` is the same two blocks with an
// operand fence (an empty `asm volatile` naming every accumulator register) before the branch and after each block's wait,
// and with the `wgmma.fence` inside each block, not before the branch: it pipelines (one wait a block). On 13.0 the operand
// fence alone does not do it; the `wgmma.fence` in the block is what the toolkit needs.
//
//   nvcc -arch=sm_90a -cubin serialized.cu -o serialized.cubin
#include <cstdint>

namespace sample {

struct Tile {
    uint32_t a[1024], b[1024];    // the two operands' staging
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

__global__ void __launch_bounds__(128) serialized(float* out, const int* flags, int n) {
    __shared__ Tile t;
    float d[8][8] = {};
    t.a[threadIdx.x] = n;
    t.b[threadIdx.x] = n + 1;
    __syncthreads();
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        fence();
        if (flags[i * 128 + threadIdx.x]) {        // a run-time branch each thread takes for itself
            for (int j = 0; j < 8; ++j) step(d[j], desc(&t.a[j * 64]), desc(&t.b[j * 64]));
            commit();
            wait<0>();
        } else {
            for (int j = 0; j < 8; ++j) step(d[j], desc(&t.a[512 + j * 64]), desc(&t.b[512 + j * 64]));
            commit();
            wait<0>();
        }
    }
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) out[(threadIdx.x * 8 + j) * 8 + i] = d[j][i];
}

}  // namespace sample
