/* The same stencil runs in serial, with OpenMP, or with OpenACC. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s N STEPS output.bin\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]), steps = atoi(argv[2]);
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000) return 1;
    size_t count = (size_t)n * n;
    double *a = calloc(count, sizeof(double));
    double *b = calloc(count, sizeof(double));
    if (!a || !b) return 1;
    /* Both arrays have the same fixed edges. The top includes its corners. */
    for (int x = 0; x < n; x++) a[x] = b[x] = 100.0;
    int threads = 1;
    const char *mode = "serial";
#ifdef SERIAL_O0
    mode = "serial_O0";
#endif
#ifdef NAIVE
    mode = "naive";
#endif
#ifdef _OPENMP
    mode = "openmp";
    #pragma omp parallel
    {
        #pragma omp single
        threads = omp_get_num_threads();
    }
#endif
#ifdef _OPENACC
    mode = "openacc";
    /* Refuse a silent CPU fallback in the GPU exercise. */
    if (acc_get_num_devices(acc_device_nvidia) < 1) return 2;
    acc_set_device_type(acc_device_nvidia);
    /* Transfer once; keep both time levels on the GPU for all steps. */
    #pragma acc data copy(a[0:count], b[0:count])
#endif
    {
        double start = seconds();
        for (int step = 0; step < steps; step++) {
#ifdef NAIVE
            double *old = a, *next = b;
#else
            /* Alternate the arrays instead of copying the whole grid. */
            double *old = (step % 2 == 0) ? a : b;
            double *next = (step % 2 == 0) ? b : a;
#endif
#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
#ifdef _OPENACC
            #pragma acc parallel loop collapse(2) present(old[0:count], next[0:count])
#endif
            for (int y = 1; y < n - 1; y++) {
                for (int x = 1; x < n - 1; x++) {
                    size_t k = (size_t)y * n + x;
                    next[k] = 0.25 * (old[k-n] + old[k+n] + old[k-1] + old[k+1]);
                }
            }
#ifdef NAIVE
            /* This extra full-grid copy is the first cost we will remove. */
            for (size_t k = 0; k < count; k++) a[k] = b[k];
#endif
        }
        /* This timer excludes allocation, file output and GPU data transfers. */
        printf("%s,%d,%d,1,%d,%.9f\n", mode, n, steps, threads, seconds()-start);
    }
#ifdef NAIVE
    double *final = a;
#else
    double *final = (steps % 2 == 0) ? a : b;
#endif
    FILE *file = fopen(argv[3], "wb");
    if (!file) return 1;
    size_t written = fwrite(final, sizeof(double), count, file);
    int closed = fclose(file);
    free(a); free(b);
    return written == count && closed == 0 ? 0 : 1;
}
