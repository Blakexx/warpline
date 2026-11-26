// An example kernel for the checker, never run: no loop. `loop_held` has no finding.
//
//   nvcc -arch=sm_90a --generate-line-info -cubin held_none.cu -o held_none.cubin
__global__ void held_none(const float* in, float* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    out[t] = in[t] * 2.f;
}
