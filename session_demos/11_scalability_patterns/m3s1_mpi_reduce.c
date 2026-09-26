#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static inline double work_value(long i, int reps) {
    double x = 1.0 + (double)(i % 1000) * 1.0e-6;
    for (int k = 0; k < reps; ++k) {
        x = x * 1.0000001192092896 + 0.0000001;
        x = x / (1.0 + x * 1.0e-7);
    }
    return x;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long total_n = 12000000L;
    if (argc > 1) total_n = atol(argv[1]);
    long base = total_n / size;
    long rem = total_n % size;
    long local_n = base + (rank < rem ? 1 : 0);
    long start_index = rank * base + (rank < rem ? rank : rem);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double local_sum = 0.0;
    for (long j = 0; j < local_n; ++j) {
        local_sum += work_value(start_index + j, 40);
    }
    double t1 = MPI_Wtime();

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    double comp = t1 - t0;
    double reduce = t2 - t1;
    double total = t2 - t0;
    double max_comp = 0.0, max_reduce = 0.0, max_total = 0.0;
    MPI_Reduce(&comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&reduce, &max_reduce, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI_REDUCE ranks=%d n=%ld compute_max=%.6f reduce_max=%.6f total_max=%.6f sum=%.6e\n",
               size, total_n, max_comp, max_reduce, max_total, global_sum);
    }
    MPI_Finalize();
    return 0;
}
