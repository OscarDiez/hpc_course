#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

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

int main(int argc, char **argv) {
    int rank, ranks;
    /* Only the original thread calls MPI. */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 4) MPI_Abort(MPI_COMM_WORLD, 1);
    int n = atoi(argv[1]), steps = atoi(argv[2]);
    /* Equal blocks keep this example short: N-2 must divide by ranks. */
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000 || ranks > n-2 || (n-2)%ranks != 0)
        MPI_Abort(MPI_COMM_WORLD, 1);
    int rows = (n-2)/ranks;
    double *old = calloc((size_t)(rows+2)*n, sizeof(double));
    double *next = calloc((size_t)(rows+2)*n, sizeof(double));
    if (!old || !next) MPI_Abort(MPI_COMM_WORLD, 1);
    /* Each rank has two halo rows. Rank 0's top halo is the hot edge. */
    if (rank == 0) for (int x = 0; x < n; x++) old[x] = next[x] = 100.0;
    int up = rank == 0 ? MPI_PROC_NULL : rank-1;
    int down = rank == ranks-1 ? MPI_PROC_NULL : rank+1;
    fprintf(stderr, "rank %d/%d: %d interior rows\n", rank, ranks, rows);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < steps; step++) {
        /* Exchange owned edge rows for neighbours' halo values. */
        MPI_Sendrecv(old+n, n, MPI_DOUBLE, up, 0,
                     old+(rows+1)*n, n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(old+rows*n, n, MPI_DOUBLE, down, 1,
                     old, n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Threads share this rank's arrays. MPI calls stay outside this loop. */
        #pragma omp parallel for
        for (int y = 1; y <= rows; y++) {
            for (int x = 1; x < n-1; x++) {
                int k = y*n+x;
                next[k] = 0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
            }
        }
        double *temporary = old; old = next; next = temporary;
    }
    /* The step completes only when the slowest rank finishes. */
    double elapsed = MPI_Wtime()-start, maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double *plate = NULL;
    if (rank == 0) {
        plate = calloc((size_t)n*n, sizeof(double));
        if (!plate) MPI_Abort(MPI_COMM_WORLD, 1);
        for (int x = 0; x < n; x++) plate[x] = 100.0;
    }
    /* Gather interior rows once, after the time stepping. */
    MPI_Gather(old+n, rows*n, MPI_DOUBLE, rank == 0 ? plate+n : NULL,
               rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("hybrid loop_s=%.9f\n", maximum);
        save_grid(argv[3], plate, n);
    }
    free(old); free(next); free(plate);
    MPI_Finalize();
    return 0;
}
