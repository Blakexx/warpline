// examples/checker/expansions/wide_clean.cu -- the clean form: 32-bit indices (the address a 32-by-32 wide multiply-add) and
// a 32-bit division by a power of two; no expansion rule fires (docs/checker-rules.md#compiler-expansions).
__global__ void wide_clean(float* out, const float* in, unsigned stride) {
  const unsigned i = threadIdx.x * stride;
  out[i / 4] = in[i];
}
