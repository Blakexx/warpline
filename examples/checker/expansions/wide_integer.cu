// examples/checker/expansions/wide_integer.cu -- `expansion_wide_integer` fires: a 64-bit division by a runtime value and a
// 64-bit by 64-bit multiply (docs/checker-rules.md#compiler-expansions).
__global__ void wide_integer(unsigned long long* out, const unsigned long long* in, unsigned long long d, unsigned long long m) {
  const unsigned long long x = in[threadIdx.x];
  out[threadIdx.x] = x / d + x * m;
}
