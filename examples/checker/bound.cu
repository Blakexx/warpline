// An example kernel for the checker, never run: a launch bound the assembler cannot honour. `__launch_bounds__(1024, 4)` asks
// for four blocks of 1024 threads on one SM, past the 2048 threads an sm_90a SM holds, so ptxas discards the bound and warns:
// the base ledger's row `launch_bound_discarded` counts the notice.
//
//   nvcc -arch=sm_90a -cubin -Xptxas -v bound.cu -o bound.cubin
__global__ void __launch_bounds__(1024, 4) bound(float* out) { out[threadIdx.x] = 1.0f; }
