// examples/checker/expansions/int_clean.cu -- the clean forms of a division: a power-of-two divisor (a shift and a mask) and a
// precomputed inverse (multiply-high and a shift); no expansion rule fires (docs/checker-rules.md#compiler-expansions).
__global__ void int_clean(unsigned* out, const unsigned* in, unsigned inverse, unsigned shift) {
  const unsigned x = in[threadIdx.x];
  out[threadIdx.x] = (x / 64) + (x % 32) + (__umulhi(x, inverse) >> shift);
}
