#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void save_grid(const char *name, const double *temperatures, int n);

int main(int argc, char **argv) {
    int rank;
    int ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 4) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Read the grid size and number of time steps from the command line. */
    int n = atoi(argv[1]);
    int steps = atoi(argv[2]);
    /* Split only the interior rows. Each rank receives the same number.
       This example requires N-2 to be divisible by the number of ranks. */
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000 ||
        ranks > n - 2 || (n - 2) % ranks != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int local_rows = (n - 2) / ranks;
    double *old = calloc((size_t)(local_rows + 2) * n, sizeof(double));
    double *next = calloc((size_t)(local_rows + 2) * n, sizeof(double));
    if (!old || !next) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Local row 0 and local row local_rows+1 are halos.
       Rank 0's top halo represents the physical hot edge. */
    if (rank == 0) {
        for (int x = 0; x < n; x++) {
            old[x] = 100.0;
            next[x] = 100.0;
        }
    }
    /* MPI_PROC_NULL means there is no neighbour beyond the plate edge. */
    int previous_rank = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int next_rank = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
    fprintf(stderr, "rank %d/%d: %d interior rows\n", rank, ranks, local_rows);
    /* Finish setup on every rank before timing the repeated calculation. */
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < steps; step++) {
        /* Send the first owned row upwards; receive the bottom halo. */
        MPI_Sendrecv(
            old + n, n, MPI_DOUBLE, previous_rank, 0,
            old + (local_rows + 1) * n, n, MPI_DOUBLE, next_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        /* Send the last owned row downwards; receive the top halo. */
        MPI_Sendrecv(
            old + local_rows * n, n, MPI_DOUBLE, next_rank, 1,
            old, n, MPI_DOUBLE, previous_rank, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );
        for (int y = 1; y <= local_rows; y++) {
            for (int x = 1; x < n - 1; x++) {
                /* Flatten (row, column) into one array index. */
                int index = y * n + x;
                next[index] = 0.25 * (
                    old[index - n] + old[index + n] +
                    old[index - 1] + old[index + 1]
                );
            }
        }
        /* Change buffer roles without copying the local grid. */
        double *temporary = old;
        old = next;
        next = temporary;
    }
    /* The step completes only when the slowest rank finishes. */
    double elapsed = MPI_Wtime() - start;
    double maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double *plate = NULL;
    if (rank == 0) {
        plate = calloc((size_t)n * n, sizeof(double));
        if (!plate) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int x = 0; x < n; x++) {
            plate[x] = 100.0;
        }
    }
    /* Gather only owned rows, excluding the halos.
       Rank 0 keeps the complete plate for saving the result. */
    MPI_Gather(
        old + n, local_rows * n, MPI_DOUBLE,
        rank == 0 ? plate + n : NULL, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    if (rank == 0) {
        printf("mpi loop_s=%.9f\n", maximum);
        save_grid(argv[3], plate, n);
    }
    free(old);
    free(next);
    free(plate);
    MPI_Finalize();
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
