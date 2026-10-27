#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

/* Read only the old grid; each iteration writes a different new cell. */
void update(double *old, double *next, int n) {
    #pragma acc parallel loop present(old[0:n*n], next[0:n*n])
    for (int y = 1; y < n-1; y++) {
        for (int x = 1; x < n-1; x++) {
            int k = y*n+x;
            next[k] = 0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
        }
    }
}

void save_grid(const char *name, const double *t, int n);

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "Usage: %s N STEPS output.vtk\n", argv[0]); return 1; }
    int n = atoi(argv[1]), steps = atoi(argv[2]);
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000) return 1;
    double *a = calloc((size_t)n*n, sizeof(double));
    double *b = calloc((size_t)n*n, sizeof(double));
    if (!a || !b) return 1;
    /* Both grids start cold, with the same fixed hot top edge. */
    for (int x = 0; x < n; x++) a[x] = b[x] = 100.0;
    /* A queued GPU job must have a real accelerator, not a CPU fallback. */
    if (acc_get_num_devices(acc_device_nvidia) < 1) return 2;
    acc_set_device_type(acc_device_nvidia);
    /* Copy once in and once out, with both time levels resident on the GPU. */
    #pragma acc data copy(a[0:n*n], b[0:n*n])
    {
        double start = seconds();
        for (int step = 0; step < steps; step++) {
            /* Alternate buffers; no whole-grid copy is needed. */
            double *old = step % 2 == 0 ? a : b;
            double *next = step % 2 == 0 ? b : a;
            update(old, next, n);
        }
        printf("openacc loop_s=%.9f\n", seconds()-start);
    }
    double *final = steps % 2 == 0 ? a : b;
    printf("Cell below hot edge: %.9f C\n", final[n+1]);
    save_grid(argv[3], final, n);
    free(a); free(b);
    return 0;
}

/* VTK adds grid dimensions so ParaView can open the result directly. */
void save_grid(const char *name, const double *t, int n) {
    FILE *f = fopen(name, "w");
    if (!f) { perror(name); exit(1); }
    fprintf(f, "# vtk DataFile Version 3.0\nHeat plate, Celsius\nASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n", n, n);
    fprintf(f, "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %d\n", n*n);
    fprintf(f, "SCALARS Temperature double 1\nLOOKUP_TABLE default\n");
    for (int k = 0; k < n*n; k++) fprintf(f, "%.17g\n", t[k]);
    if (fclose(f) != 0) exit(1);
}
