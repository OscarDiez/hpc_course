#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int n = 1024;
    constexpr int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    float *a = static_cast<float *>(std::malloc(bytes));
    float *b = static_cast<float *>(std::malloc(bytes));
    float *c = static_cast<float *>(std::malloc(bytes));
    if (!a || !b || !c) {
        std::fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const float expected = a[i] + b[i];
        if (c[i] != expected) {
            ok = false;
            std::fprintf(stderr, "Mismatch at %d: got %.1f, expected %.1f\n",
                         i, c[i], expected);
            break;
        }
    }

    std::printf("Vector addition on the GPU\n");
    std::printf("N                 = %d elements\n", n);
    std::printf("Threads per block = %d\n", threads_per_block);
    std::printf("Blocks            = %d\n", blocks);
    std::printf("Launched threads  = %d\n", blocks * threads_per_block);
    std::printf("Result check      = %s\n\n", ok ? "PASS" : "FAIL");

    std::printf("First five results:\n");
    for (int i = 0; i < 5; ++i) {
        std::printf("c[%d] = %.0f + %.0f = %.0f\n", i, a[i], b[i], c[i]);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(a);
    std::free(b);
    std::free(c);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
