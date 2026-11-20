// A third example for the checker: warp collectives under role-dependent control flow, never run. The toolkit outlines
// each one the warp may not reach converged into a call to a convergence subroutine, and GUARDS the call (past a
// `BRA.CONV`, or in the fallback a `BRA.DIV` enters), so a converged warp runs the inline collective: the rule
// `convergence_call` reads the call as cold code. Compiled for `sm_80`, where the toolkit emits these calls:
//
//   nvcc -arch=sm_80 -cubin collectives.cu -o collectives.cubin
//   (the three outputs as in stages.cu, saved in tests/checker_fixtures/collectives/)
__global__ void roles(unsigned* out, const unsigned* in, int role) {
    unsigned v = in[threadIdx.x];
    int w = threadIdx.x >> 5;
    if (w == role) { v += __match_any_sync(0xffffffffu, v); }
    else if (w == role + 1) { v += __ballot_sync(0xffffffffu, v & 1); v ^= __any_sync(0xffffffffu, v & 2); }
    else if (w == role + 2) { v += __reduce_max_sync(0xffffffffu, v); }
    else { v += __shfl_sync(0xffffffffu, v, w); }
    out[threadIdx.x] = v;
}
