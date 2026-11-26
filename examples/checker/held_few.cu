// An example kernel for the checker, never run: the same shape as `held_many.cu` with two words held across the loop
// instead of twenty-four. `loop_held` reports a figure of a handful of registers.
//
//   nvcc -arch=sm_90a --generate-line-info -cubin held_few.cu -o held_few.cubin
__global__ void held_few(const float* in, const float* sum_in, float* out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float held[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) held[i] = in[t + i * 4096];
    float total = 0.f;
#pragma unroll 1
    for (int k = 0; k < n; ++k) total += sum_in[t + k * 32];
#pragma unroll
    for (int i = 0; i < 2; ++i) out[t + i * 4096] = held[i] + total;
}
