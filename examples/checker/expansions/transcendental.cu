// examples/checker/expansions/transcendental.cu -- `expansion_transcendental` fires: precise sine, exponential and logarithm
// (docs/checker-rules.md#compiler-expansions).
__global__ void transcendental(float* out, const float* a) {
  const float x = a[threadIdx.x];
  out[threadIdx.x] = sinf(x) + expf(x) + logf(x);
}
