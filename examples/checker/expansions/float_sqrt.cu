// examples/checker/expansions/float_sqrt.cu -- `expansion_float_sqrt` fires: an IEEE-exact square root
// (docs/checker-rules.md#compiler-expansions).
__global__ void float_sqrt(float* out, const float* a) {
  out[threadIdx.x] = sqrtf(a[threadIdx.x]);
}
