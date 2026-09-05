#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
    if (i < n) c[i] = a[i] + b[i];
}

static double cpu_add(const std::vector<float> &a,
                      const std::vector<float> &b,
                      std::vector<float> &c,
                      int repeats) {
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            c[i] = a[i] + b[i];
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void run_case(int n, int repeats) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), cpu_c(n), gpu_c(n);

    // CPU timing
    const double cpu_ms = cpu_add(a, b, cpu_c, repeats);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Warm-up: avoid charging first-use CUDA initialization to the measured kernel.
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel-only timing: data is already resident on the GPU.
    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));
    CUDA_CHECK(cudaEventRecord(k_start));
    for (int r = 0; r < repeats; ++r) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaEventRecord(k_stop));
    CUDA_CHECK(cudaEventSynchronize(k_stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_stop));

    // End-to-end GPU timing: copy inputs once, run all kernels, copy result once.
    const auto total_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    for (int r = 0; r < repeats; ++r) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(gpu_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    const auto total_end = std::chrono::steady_clock::now();
    const double gpu_total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();

    bool ok = true;
    for (int i = 0; i < std::min(n, 1000); ++i) {
        if (std::fabs(gpu_c[i] - 3.0f) > 1e-5f) {
            ok = false;
            break;
        }
    }

    std::printf("%-12d %-8d %12.3f %16.3f %16.3f %8s\n",
                n, repeats, cpu_ms, kernel_ms, gpu_total_ms,
                ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    std::printf("CPU vs GPU vector-add benchmark\n");
    std::printf("GPU total = H->D copies once + all kernels + D->H copy once\n\n");
    std::printf("%-12s %-8s %12s %16s %16s %8s\n",
                "N", "repeats", "CPU ms", "GPU kernel ms", "GPU total ms", "check");
    std::printf("--------------------------------------------------------------------------------\n");

    // Small, medium, and large cases. Exact crossover depends on the hardware.
    run_case(1'000, 1);
    run_case(1'000'000, 1);
    run_case(10'000'000, 1);

    std::printf("\nReuse experiment: same large vectors, many GPU operations while data stays resident\n");
    std::printf("%-12s %-8s %12s %16s %16s %8s\n",
                "N", "repeats", "CPU ms", "GPU kernel ms", "GPU total ms", "check");
    std::printf("--------------------------------------------------------------------------------\n");
    run_case(10'000'000, 100);

    std::printf("\nQuestions:\n");
    std::printf("1. When does kernel-only GPU time look much better than end-to-end GPU time?\n");
    std::printf("2. Which GPU number is fair to compare with the CPU when transfers are required?\n");
    std::printf("3. What changes when the same data is reused for many GPU operations?\n");

    return EXIT_SUCCESS;
}
