#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void init_arrays(float *a, float *b, float *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)((i % 101) + 1) * 0.001f;
        b[i] = (float)((i % 67) + 1) * 0.002f;
        c[i] = 0.0f;
    }
}

static double checksum(const float *c, size_t n) {
    double s = 0.0;
    size_t step = n / 32 + 1;
    for (size_t i = 0; i < n; i += step) s += c[i];
    return s;
}

static void run_case(size_t n, int reps) {
    const size_t bytes = n * sizeof(float);
    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);
    float *c = (float *)malloc(bytes);
    if (!a || !b || !c) {
        fprintf(stderr, "Allocation failed for N=%zu\n", n);
        exit(2);
    }

    init_arrays(a, b, c, n);

    double t0 = now_seconds();
    for (int r = 0; r < reps; ++r) {
        float alpha = 0.25f + 0.0001f * (float)(r % 100);
        for (size_t i = 0; i < n; ++i) {
            c[i] = 0.99991f * c[i] + alpha * a[i] + b[i];
        }
    }
    double cpu_total = now_seconds() - t0;
    double cpu_sum = checksum(c, n);

    memset(c, 0, bytes);
    t0 = now_seconds();
    for (int r = 0; r < reps; ++r) {
        float alpha = 0.25f + 0.0001f * (float)(r % 100);
        #pragma acc parallel loop copyin(a[0:n], b[0:n]) copy(c[0:n])
        for (size_t i = 0; i < n; ++i) {
            c[i] = 0.99991f * c[i] + alpha * a[i] + b[i];
        }
    }
    double gpu_copy_each_total = now_seconds() - t0;
    double gpu_copy_sum = checksum(c, n);

    memset(c, 0, bytes);
    double gpu_kernel_seconds = 0.0;
    t0 = now_seconds();
    #pragma acc data copyin(a[0:n], b[0:n]) copy(c[0:n])
    {
        double tk = now_seconds();
        for (int r = 0; r < reps; ++r) {
            float alpha = 0.25f + 0.0001f * (float)(r % 100);
            #pragma acc parallel loop present(a[0:n], b[0:n], c[0:n])
            for (size_t i = 0; i < n; ++i) {
                c[i] = 0.99991f * c[i] + alpha * a[i] + b[i];
            }
        }
        #pragma acc wait
        gpu_kernel_seconds = now_seconds() - tk;
    }
    double gpu_resident_total = now_seconds() - t0;
    double gpu_resident_sum = checksum(c, n);

    const double cpu_ms = 1000.0 * cpu_total / reps;
    const double copy_ms = 1000.0 * gpu_copy_each_total / reps;
    const double kernel_ms = 1000.0 * gpu_kernel_seconds / reps;
    const double resident_ms = 1000.0 * gpu_resident_total / reps;

    printf("\nN=%zu, repetitions=%d\n", n, reps);
    printf("  CPU average / operation                 : %10.6f ms\n", cpu_ms);
    printf("  GPU total / operation (copy every time) : %10.6f ms\n", copy_ms);
    printf("  GPU kernel / operation (data resident)  : %10.6f ms\n", kernel_ms);
    printf("  GPU total / operation (copy once)       : %10.6f ms\n", resident_ms);
    printf("  Winner, CPU vs GPU total(copy once)     : %s\n",
           (resident_ms < cpu_ms) ? "GPU" : "CPU");
    printf("  Checksums CPU / copy-each / resident    : %.6f / %.6f / %.6f\n",
           cpu_sum, gpu_copy_sum, gpu_resident_sum);

    free(a);
    free(b);
    free(c);
}

int main(void) {
#ifndef _OPENACC
    fprintf(stderr,
            "This benchmark must be compiled with OpenACC enabled.\n"
            "Example with NVIDIA HPC SDK: nvc -O3 -acc -Minfo=accel openacc_cpu_gpu_benchmark.c -o openacc_benchmark\n");
    return 2;
#endif

    printf("OpenACC CPU vs GPU benchmark\n");
    printf("Compare total application time, not only kernel time.\n");
    printf("The exact crossover depends on the GPU, CPU, compiler and current system load.\n");

    run_case(1024,       10000);
    run_case(1000000,       50);
    run_case(8000000,       10);

    printf("\nQuestions:\n");
    printf("  1. For which size does the CPU win?\n");
    printf("  2. How much do repeated CPU<->GPU copies cost?\n");
    printf("  3. What changes when data stays on the GPU?\n");
    return 0;
}
