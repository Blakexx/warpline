// examples/checker/expansions/int_division.cu -- `expansion_integer_division` fires: a 32-bit division and a modulo by a
// runtime value, each expanded into the reciprocal-estimate sequence (docs/checker-rules.md#compiler-expansions).
__global__ void int_division(unsigned* out, const unsigned* in, unsigned d) {
  const unsigned x = in[threadIdx.x];
  out[threadIdx.x] = x / d + x % (d + 1);
}
