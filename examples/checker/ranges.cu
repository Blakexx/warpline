// An example kernel for the checker, never run: registers read by their life ranges. Thirty-two words are loaded through a helper
// the compiler inlines (so the inline frames name it), held across an IEEE division whose slow path is a call into a subroutine,
// and summed after it: `flow.pressure` over the life-range marks counts the thirty-two before the call, where the listing's own
// count drops the registers it marks every call as writing.
//
//   nvcc -arch=sm_90a --generate-line-info -cubin ranges.cu -o ranges.cubin
//   nvdisasm --print-life-ranges ranges.cubin; nvdisasm --print-line-info-inline ranges.cubin
__device__ __forceinline__ float scaled(const float* x, int k) { return x[k * 256 + threadIdx.x] * (k + 1); }

__global__ void ranges(float* y, const float* d) {
    float keep[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) keep[k] = scaled(y, k);
    y[threadIdx.x] = 0.0f;
    float v = keep[0] / d[threadIdx.x];
#pragma unroll
    for (int k = 0; k < 32; ++k) v += keep[k] * (k + 1);
    y[threadIdx.x + 256] = v;
}
