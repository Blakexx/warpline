// examples/checker/expansions/float_division.cu -- `expansion_float_division` fires: an IEEE-exact float division
// (docs/checker-rules.md#compiler-expansions).
__global__ void float_division(float* out, const float* a, const float* b) {
  out[threadIdx.x] = a[threadIdx.x] / b[threadIdx.x];
}
