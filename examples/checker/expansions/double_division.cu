// examples/checker/expansions/double_division.cu -- `expansion_float_division` and `expansion_float_sqrt` on doubles: an
// IEEE-exact double division and square root (docs/checker-rules.md#compiler-expansions).
__global__ void double_division(double* out, const double* a, const double* b) {
  out[threadIdx.x] = a[threadIdx.x] / b[threadIdx.x] + sqrt(b[threadIdx.x]);
}
