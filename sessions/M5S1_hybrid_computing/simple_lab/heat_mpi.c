/* MPI owns rows; OpenMP can share the rows within each MPI process. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    int rank, ranks, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 4 || provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int n = atoi(argv[1]), steps = atoi(argv[2]);
    /* Equal row blocks keep this teaching example short. */
    if (n < 3 || n > 4096 || steps < 1 || steps > 100000 || (n-2) % ranks != 0 || ranks > n-2) {
        if (rank == 0) fprintf(stderr, "Need N>=3, STEPS>=1 and N-2 divisible by the rank count.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rows = (n-2) / ranks;
    size_t count = (size_t)(rows+2) * n;
    double *old = calloc(count, sizeof(double));
    double *next = calloc(count, sizeof(double));
    if (!old || !next) MPI_Abort(MPI_COMM_WORLD, 1);
    /* Row 0 and row rows+1 are halos. Rank 0's upper halo is the hot edge. */
    if (rank == 0) for (int x = 0; x < n; x++) old[x] = next[x] = 100.0;
    int up = rank == 0 ? MPI_PROC_NULL : rank-1;
    int down = rank == ranks-1 ? MPI_PROC_NULL : rank+1;
    int threads = 1;
    const char *mode = "mpi";
#ifdef _OPENMP
    mode = "hybrid";
    #pragma omp parallel
    {
        #pragma omp single
        threads = omp_get_num_threads();
    }
#endif
    fprintf(stderr, "rank %d/%d owns %d interior rows, threads=%d\n", rank, ranks, rows, threads);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < steps; step++) {
        /* Send our first row up; receive the lower neighbour into our bottom halo. */
        MPI_Sendrecv(old+n, n, MPI_DOUBLE, up, 0,
                     old+(size_t)(rows+1)*n, n, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send our last row down; receive the upper neighbour into our top halo. */
        MPI_Sendrecv(old+(size_t)rows*n, n, MPI_DOUBLE, down, 1,
                     old, n, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int y = 1; y <= rows; y++) {
            for (int x = 1; x < n-1; x++) {
                size_t k = (size_t)y*n+x;
                next[k] = 0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
            }
        }
        double *temporary = old; old = next; next = temporary;
    }
    /* Report the slowest rank, since all ranks are needed to finish the step. */
    double elapsed = MPI_Wtime()-start, maximum;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double *global = NULL;
    if (rank == 0) {
        global = calloc((size_t)n*n, sizeof(double));
        if (!global) MPI_Abort(MPI_COMM_WORLD, 1);
        for (int x = 0; x < n; x++) global[x] = 100.0;
    }
    /* Gather only owned rows; the root keeps the physical top/bottom edges. */
    MPI_Gather(old+n, rows*n, MPI_DOUBLE, rank == 0 ? global+n : NULL,
               rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        FILE *file = fopen(argv[3], "wb");
        if (!file) MPI_Abort(MPI_COMM_WORLD, 1);
        size_t written = fwrite(global, sizeof(double), (size_t)n*n, file);
        int closed = fclose(file);
        if (written != (size_t)n*n || closed != 0) MPI_Abort(MPI_COMM_WORLD, 1);
        printf("%s,%d,%d,%d,%d,%.9f\n", mode, n, steps, ranks, threads, maximum);
    }
    free(old); free(next); free(global);
    MPI_Finalize();
    return 0;
}
