// examples/checker/expansions/slow_call.cu -- `expansion_slow_call` fires: a call into a body the toolkit outlined that no
// other expansion names, here a function kept out of line (docs/checker-rules.md#compiler-expansions).
__device__ __noinline__ float scaled(float x, float s) { return x * s + 1.0f; }

__global__ void slow_call(float* out, const float* a, float s) {
  out[threadIdx.x] = scaled(a[threadIdx.x], s);
}
