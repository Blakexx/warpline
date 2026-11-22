// An example kernel for the checker, never run: thirty-two accumulator registers that two blocks of tensor steps, behind a
// run-time branch each thread decides for itself, split differently: one block runs four steps of shape m64n16k16, each into
// its own group of eight registers, the other runs one step of shape m64n64k16 into all thirty-two. The toolkit (13.5,
// sm_90a) moves the accumulators at the switch between the two shapes: instructions that are not tensor steps write the
// registers the steps accumulate into (the rule `accumulator_moved` fires). The same kernel with one shape in both blocks
// is `serialized.cu` and `fenced.cu`, where no register moves.
//
//   nvcc -arch=sm_90a -cubin mixed_shapes.cu -o mixed_shapes.cubin
#include <cstdint>

namespace sample {

struct Tile {
    uint32_t a[1024], b[1024];    // the two operands' staging
};

__device__ inline uint64_t desc(const void* p) {
    return (uint64_t(static_cast<uint32_t>(__cvta_generic_to_shared(p))) >> 4) & 0x3fff | (1ull << 62);
}

// a 64x16x16 bf16 step into eight registers, which the caller names in the order the step's fragment has them: the
// first block's tile keeps each pair of neighbouring accumulators swapped
__device__ inline void step16(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;"
        : "+f"(d[1]), "+f"(d[0]), "+f"(d[3]), "+f"(d[2]), "+f"(d[5]), "+f"(d[4]), "+f"(d[7]), "+f"(d[6])
        : "l"(da), "l"(db));
}
// a 64x64x16 bf16 step into thirty-two registers
__device__ inline void step64(float* d, uint64_t da, uint64_t db) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db));
}
__device__ inline void fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ inline void commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ inline void wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

// an operand fence: the accumulators are inputs and outputs of an empty statement
__device__ inline void hold(float (&d)[32]) {
    asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]));
}

__global__ void __launch_bounds__(128) mixed_shapes(float* out, const int* flags, int n) {
    __shared__ Tile t;
    float d[32] = {};
    t.a[threadIdx.x] = n;
    t.b[threadIdx.x] = n + 1;
    __syncthreads();
    _Pragma("unroll 1") for (int i = 0; i < n; ++i) {
        hold(d);
        if (flags[i * 128 + threadIdx.x]) {        // a run-time branch each thread takes for itself
            fence();
            for (int j = 0; j < 4; ++j) step16(d + 8 * j, desc(&t.a[j * 64]), desc(&t.b[j * 64]));
            commit();
            wait<0>();
            hold(d);
        } else {
            fence();
            step64(d, desc(&t.a[512]), desc(&t.b[512]));
            commit();
            wait<0>();
            hold(d);
        }
    }
    for (int i = 0; i < 32; ++i) out[threadIdx.x * 32 + i] = d[i];
}

}  // namespace sample
