// An example kernel for the checker, never run: `wait_all_work.cu` with the work after the wait for every group in flight
// reading the accumulators the steps wrote (it stores them): the wait is needed, the rule `wait_all_with_work` stays quiet.
//
//   nvcc -arch=sm_90a -cubin wait_all_reads.cu -o wait_all_reads.cubin
#include <cstdint>

namespace sample {

struct Tile {
    float a[512], b[512];    // the two operands' staging
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

// four steps and their wait for every group in flight
__device__ inline void block(float (&d)[8], Tile& t, int base) {
    fence();
    for (int j = 0; j < 4; ++j) step(d, desc(&t.a[base + j * 64]), desc(&t.b[base + j * 64]));
    commit();
    wait<0>();
}

__global__ void __launch_bounds__(128) wait_all_reads(float* out) {
    __shared__ Tile t;
    float d[8] = {};
    t.a[threadIdx.x] = threadIdx.x;
    t.b[threadIdx.x] = threadIdx.x + 1;
    __syncthreads();
    block(d, t, 0);
    __syncthreads();
    t.a[threadIdx.x + 128] = t.b[(threadIdx.x + 1) % 128] + d[0];         // work that reads the accumulators
    t.b[threadIdx.x + 128] = t.a[(threadIdx.x + 3) % 128] + 2;
    __syncthreads();
    block(d, t, 256);
    for (int i = 0; i < 8; ++i) out[threadIdx.x * 8 + i] = d[i];
}

}  // namespace sample
