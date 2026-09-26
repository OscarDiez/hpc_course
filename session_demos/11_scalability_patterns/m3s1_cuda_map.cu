#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void map_kernel(const float *x, float *y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 1.0001f + 0.1234f;
}

static void run_case(long n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    for (long i=0;i<n;++i) h_x[i] = (float)(i % 1000) * 0.001f;

    auto full0 = std::chrono::steady_clock::now();
    float *d_x=nullptr,*d_y=nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int block=256;
    int grid=(int)((n + block - 1)/block);
    cudaEventRecord(start);
    map_kernel<<<grid,block>>>(d_x,d_y,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms=0.0f;
    cudaEventElapsedTime(&kernel_ms,start,stop);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    auto full1 = std::chrono::steady_clock::now();
    double end_to_end_ms = std::chrono::duration<double,std::milli>(full1-full0).count();

    double checksum=0.0;
    long stride = n/16 + 1;
    for(long i=0;i<n;i+=stride) checksum += h_y[i];
    printf("GPU_MAP n=%ld kernel_ms=%.6f end_to_end_ms=%.6f checksum=%.6e\n",
           n,kernel_ms,end_to_end_ms,checksum);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_x); free(h_y);
}

int main(void) {
    run_case(1024);
    run_case(20000000L);
    return 0;
}
