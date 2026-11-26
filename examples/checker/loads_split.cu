// An example kernel for the checker, never run: the same two loads as `loads_together.cu`, with the first value stored before
// the second load is issued. The store waits for the first load to arrive, and `out` may be the memory `b` reads, so the second
// load stays behind it: the trip waits out two memory round trips, one after the other. `load_behind_load` reports the second load.
//
//   nvcc -arch=sm_90a -cubin loads_split.cu -o loads_split.cubin
__global__ void loads_split(const float* a, const float* b, float* out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
        const float x = a[t + k * 32];
        out[t + k * 32] = x * 2.f;
        const float y = b[t + k * 32];
        out[t + k * 32 + 4096] = y + 1.f;
    }
}
