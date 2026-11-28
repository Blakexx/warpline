// examples/checker/expansions/float_clean.cu -- the clean forms: the fast intrinsics where the accuracy allows (a division, a
// square root, a reciprocal, sine, exponential, logarithm); no expansion rule fires (docs/checker-rules.md#compiler-expansions).
__global__ void float_clean(float* out, const float* a, const float* b) {
  const float x = a[threadIdx.x], y = b[threadIdx.x];
  float s, r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(s) : "f"(x));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(y));
  out[threadIdx.x] = __fdividef(x, y) + s + r + __sinf(x) + __expf(x) + __logf(x);
}
