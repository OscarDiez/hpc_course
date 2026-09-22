#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); \
        std::exit(2); \
    } \
} while (0)

__global__ void record_mapping(int *block_id, int *thread_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    block_id[i] = blockIdx.x;
    thread_id[i] = threadIdx.x;
}

int main() {
    const int n = 20;
    const int threads = 8;
    const int blocks = (n + threads - 1) / threads;
    const int launched = blocks * threads;

    int *d_block=nullptr, *d_thread=nullptr;
    CUDA_CHECK(cudaMalloc(&d_block, launched * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_thread, launched * sizeof(int)));

    record_mapping<<<blocks, threads>>>(d_block, d_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> block(launched), thread(launched);
    CUDA_CHECK(cudaMemcpy(block.data(), d_block,
                          launched*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(thread.data(), d_thread,
                          launched*sizeof(int), cudaMemcpyDeviceToHost));

    std::printf("N=%d threads_per_block=%d blocks=%d launched_threads=%d\n",
                n, threads, blocks, launched);
    std::printf("%-8s %-8s %-10s %-8s\n",
                "global_i", "block", "thread", "active");
    std::printf("--------------------------------------\n");
    for (int i = 0; i < launched; ++i)
        std::printf("%-8d %-8d %-10d %-8s\n",
                    i, block[i], thread[i], i < n ? "yes" : "no");

    cudaFree(d_block);
    cudaFree(d_thread);
    return 0;
}
