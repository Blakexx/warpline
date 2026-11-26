// An example kernel for the checker, never run: values formed before a loop and used after it. Twenty-four words are
// loaded for each thread, a loop of a run-time trip count sums another array, and the twenty-four words are stored after the
// loop with the sum added, so every one of them is held in a register across the loop: the rule `loop_held` reports the
// loop's top with a figure of about thirty registers, and `warpline held` names the source lines that made them (the loads).
// The loop's own work is one running sum.
//
// This copy of held_many.cu has lines inserted above its loops (the checker's manifest must not move: a loop is keyed by its
// file and position, never its line).
//
//   nvcc -arch=sm_90a --generate-line-info -cubin held_many.cu -o held_many.cubin
__global__ void held_many(const float* in, const float* sum_in, float* out, int n) {



    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    float held[24];
#pragma unroll
    for (int i = 0; i < 24; ++i) held[i] = in[t + i * 4096];
    float total = 0.f;
#pragma unroll 1
    for (int k = 0; k < n; ++k) total += sum_in[t + k * 32];
#pragma unroll
    for (int i = 0; i < 24; ++i) out[t + i * 4096] = held[i] + total;
}
