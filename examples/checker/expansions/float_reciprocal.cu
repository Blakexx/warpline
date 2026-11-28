// examples/checker/expansions/float_reciprocal.cu -- `expansion_float_reciprocal` fires: an IEEE-exact reciprocal
// (docs/checker-rules.md#compiler-expansions).
__global__ void float_reciprocal(float* out, const float* a) {
  out[threadIdx.x] = __frcp_rn(a[threadIdx.x]);
}
