// An example kernel for the checker, never run: each trip loads two words from two places and then uses them. Both loads are
// issued before either value is read, so the trip waits out one memory round trip. `load_behind_load` reports nothing.
//
//   nvcc -arch=sm_90a -cubin loads_together.cu -o loads_together.cubin
__global__ void loads_together(const float* a, const float* b, float* out, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
        const float x = a[t + k * 32];
        const float y = b[t + k * 32];
        out[t + k * 32] = x * 2.f;
        out[t + k * 32 + 4096] = y + 1.f;
    }
}
