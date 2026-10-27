#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

double seconds(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec + 1e-9 * timestamp.tv_nsec;
}

/* Update interior cells only. Boundary temperatures stay fixed.
   Read the old time level and write the new one into a separate array. */
void update(double *old, double *next, int n) {
    #pragma acc parallel loop present(old[0:n*n], next[0:n*n])
    for (int y = 1; y < n - 1; y++) {
        for (int x = 1; x < n - 1; x++) {
            /* Flatten (row, column) into one array index. */
            int index = y * n + x;
            next[index] = 0.25 * (
                old[index - n] + old[index + n] +
                old[index - 1] + old[index + 1]
            );
        }
    }
}

void save_grid(const char *name, const double *temperatures, int n);

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s N STEPS output.vtk\n", argv[0]);
        return 1;
    }
    /* Read the grid size and number of time steps from the command line. */
    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000) {
        return 1;
    }
    /* calloc starts every cell at zero. We need two complete time levels. */
    double *grid_a = calloc((size_t)n * n, sizeof(double));
    double *grid_b = calloc((size_t)n * n, sizeof(double));
    if (!grid_a || !grid_b) {
        return 1;
    }
    /* Both grids start cold, with the same fixed hot top edge. */
    for (int x = 0; x < n; x++) {
        grid_a[x] = 100.0;
        grid_b[x] = 100.0;
    }
    /* A queued GPU job must have a real accelerator, not a CPU fallback. */
    if (acc_get_num_devices(acc_device_nvidia) < 1) {
        return 2;
    }
    acc_set_device_type(acc_device_nvidia);
    /* Copy once in and once out, with both time levels resident on the GPU. */
    #pragma acc data copy(grid_a[0:n*n], grid_b[0:n*n])
    {
        double start = seconds();
        for (int step = 0; step < steps; step++) {
            /* On even steps read grid_a and write grid_b; on odd steps reverse them.
               Both arrays keep the same fixed boundary values. */
            double *old = step % 2 == 0 ? grid_a : grid_b;
            double *next = step % 2 == 0 ? grid_b : grid_a;
            update(old, next, n);
        }
        printf("openacc loop_s=%.9f\n", seconds() - start);
    }
    /* Choose the array containing the most recently computed time level. */
    double *final = steps % 2 == 0 ? grid_a : grid_b;
    printf("Cell below hot edge: %.9f C\n", final[n + 1]);
    save_grid(argv[3], final, n);
    free(grid_a);
    free(grid_b);
    return 0;
}

/* Write the grid dimensions and temperatures in a text file. */
void save_grid(const char *name, const double *temperatures, int n) {
    FILE *file = fopen(name, "w");
    if (!file) {
        perror(name);
        exit(1);
    }
    fprintf(file, "# vtk DataFile Version 3.0\nHeat plate, Celsius\nASCII\n");
    fprintf(file, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n", n, n);
    fprintf(file, "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %d\n", n * n);
    fprintf(file, "SCALARS Temperature double 1\nLOOKUP_TABLE default\n");
    for (int index = 0; index < n * n; index++) {
        fprintf(file, "%.17g\n", temperatures[index]);
    }
    if (fclose(file) != 0) {
        exit(1);
    }
}
