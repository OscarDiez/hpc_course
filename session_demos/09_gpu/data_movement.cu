#include <cuda_runtime.h>
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

int main() {
    const int n = 2000000;
    const int reps = 20;
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
    float *da=nullptr, *db=nullptr, *dc=nullptr;

    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Warm-up
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add<<<blocks,threads>>>(da,db,dc,n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Strategy A: move data for every operation.
    auto a0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        add<<<blocks,threads>>>(da,db,dc,n);
        CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    }
    auto a1 = std::chrono::steady_clock::now();
    double copy_every_ms =
        std::chrono::duration<double,std::milli>(a1-a0).count();

    // Strategy B: move data once, do many operations, move result once.
    auto b0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    for (int r = 0; r < reps; ++r)
        add<<<blocks,threads>>>(da,db,dc,n);
    CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    auto b1 = std::chrono::steady_clock::now();
    double resident_ms =
        std::chrono::duration<double,std::milli>(b1-b0).count();

    bool ok = true;
    for (int i=0; i<1000; ++i)
        if (std::fabs(c[i]-3.0f) > 1e-5f) ok=false;

    std::printf("N=%d reps=%d\n", n, reps);
    std::printf("copy_every_iteration_ms=%.3f\n", copy_every_ms);
    std::printf("keep_data_resident_ms=%.3f\n", resident_ms);
    std::printf("residency_speedup=%.2fx\n", copy_every_ms/resident_ms);
    std::printf("check=%s\n", ok ? "PASS" : "FAIL");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return ok ? 0 : 1;
}
