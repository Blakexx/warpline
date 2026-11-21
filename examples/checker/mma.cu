// A fourth example for the checker: synchronous tensor steps (`mma.sync`, SASS `HMMA`) on `sm_80`, never run. A synchronous
// step has no commit group and no wait: the model reads it as a step of kind `sync`. Three small kernels, one shape each:
//
//   mma_loop       a straight loop of steps (operands read from shared memory: no branch, no conversion, no global access)
//   mma_branch     the same loop with a data-dependent branch, a conversion and a global store in its body
//   mma_predicated a step the source guards with a lane-dependent condition
//
//   nvcc -arch=sm_80 -cubin mma.cu -o mma.cubin
//   (the outputs as in stages.cu, saved in tests/checker_fixtures/mma/)
__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void mma_loop(float* out, const unsigned* in, int trips) {
    __shared__ unsigned tile[1536];
    tile[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    unsigned a[4], b[2];
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int t = 0; t < trips; ++t) {
        const unsigned* p = tile + ((t * 6) & 7) * 32 + threadIdx.x;
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = p[k * 32];
        b[0] = p[4 * 32]; b[1] = p[5 * 32];
        mma(d, a, b);
        mma(d, a, b);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) out[threadIdx.x * 4 + k] = d[k];
}

__global__ void mma_branch(float* out, int* flags, const unsigned* in, int trips) {
    unsigned a[4], b[2];
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int t = 0; t < trips; ++t) {
        const unsigned* p = in + t * 192 + threadIdx.x * 6;
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = p[k];
        b[0] = p[4]; b[1] = p[5];
        mma(d, a, b);
        if (a[0] & 1) { __nanosleep(20); mma(d, a, b); }
        flags[t * 32 + threadIdx.x] = __float2int_rn(d[0]);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) out[threadIdx.x * 4 + k] = d[k];
}

__global__ void mma_predicated(float* out, const unsigned* in, int lane_cut) {
    unsigned a[4], b[2];
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const unsigned* p = in + threadIdx.x * 6;
#pragma unroll
    for (int k = 0; k < 4; ++k) a[k] = p[k];
    b[0] = p[4]; b[1] = p[5];
    mma(d, a, b);
    if ((int)(threadIdx.x & 31) < lane_cut) mma(d, a, b);
#pragma unroll
    for (int k = 0; k < 4; ++k) out[threadIdx.x * 4 + k] = d[k];
}
