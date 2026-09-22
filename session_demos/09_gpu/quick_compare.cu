#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
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

__global__ void add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static void run_case(const char *label, int n, int reps) {
    const size_t bytes = (size_t)n * sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), cpu(n), gpu(n);

    auto c0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < n; ++i)
            cpu[i] = a[i] + b[i];
    auto c1 = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(c1-c0).count();

    float *da=nullptr, *db=nullptr, *dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ks, ke;
    CUDA_CHECK(cudaEventCreate(&ks));
    CUDA_CHECK(cudaEventCreate(&ke));
    CUDA_CHECK(cudaEventRecord(ks));
    for (int r = 0; r < reps; ++r)
        add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaEventRecord(ke));
    CUDA_CHECK(cudaEventSynchronize(ke));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ks, ke));

    auto g0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    for (int r = 0; r < reps; ++r)
        add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dc, bytes, cudaMemcpyDeviceToHost));
    auto g1 = std::chrono::steady_clock::now();

    const double gpu_total_ms =
        std::chrono::duration<double, std::milli>(g1-g0).count();

    bool ok = true;
    for (int i = 0; i < std::min(n, 1000); ++i)
        if (std::fabs(gpu[i] - 3.0f) > 1e-5f) ok = false;

    std::printf("%-16s %10d %6d %12.4f %14.4f %14.4f %8s\n",
                label, n, reps, cpu_ms, kernel_ms, gpu_total_ms,
                ok ? "PASS" : "FAIL");

    cudaEventDestroy(ks);
    cudaEventDestroy(ke);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}

int main() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU=%s\n", p.name);
    std::printf("%-16s %10s %6s %12s %14s %14s %8s\n",
                "case", "N", "reps", "CPU_ms",
                "GPU_kernel_ms", "GPU_total_ms", "check");
    std::printf("--------------------------------------------------------------------------------------\n");
    run_case("tiny", 1024, 1);
    run_case("large_reuse", 5000000, 20);
    return 0;
}
