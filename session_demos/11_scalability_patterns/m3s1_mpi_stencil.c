#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long global_n = 1600000L;
    int steps = 30;
    if (argc > 1) global_n = atol(argv[1]);
    if (argc > 2) steps = atoi(argv[2]);

    if (global_n < size || global_n > 1600000 || steps < 1 || steps > 100 || global_n % size != 0) {
        if (rank == 0) fprintf(stderr, "global_n must be divisible by ranks for this teaching example\n");
        MPI_Finalize();
        return 2;
    }

    long local_n = global_n / size;
    double *a = (double*)calloc((size_t)(local_n + 2), sizeof(double));
    double *b = (double*)calloc((size_t)(local_n + 2), sizeof(double));
    if (!a || !b) {
        fprintf(stderr, "rank %d allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    for (long i = 1; i <= local_n; ++i) a[i] = ((rank * local_n + i - 1) % 100) * 0.01;
    int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double comm_time = 0.0, compute_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double total0 = MPI_Wtime();

    double *cur = a, *next = b;
    for (int s = 0; s < steps; ++s) {
        double c0 = MPI_Wtime();
        MPI_Sendrecv(&cur[1], 1, MPI_DOUBLE, left, 10,
                     &cur[local_n + 1], 1, MPI_DOUBLE, right, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[local_n], 1, MPI_DOUBLE, right, 11,
                     &cur[0], 1, MPI_DOUBLE, left, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        comm_time += MPI_Wtime() - c0;

        double k0 = MPI_Wtime();
        for (long i = 1; i <= local_n; ++i) {
            long global_i = rank*local_n + i - 1;
            if (global_i == 0 || global_i == global_n-1) { next[i]=cur[i]; continue; }
            next[i] = (cur[i-1] + cur[i] + cur[i+1]) / 3.0;
        }
        compute_time += MPI_Wtime() - k0;
        double *tmp = cur; cur = next; next = tmp;
    }

    double total = MPI_Wtime() - total0;
    double local_checksum = 0.0;
    for (long i = 1; i <= local_n; ++i) local_checksum += cur[i];
    double checksum = 0.0;
    MPI_Reduce(&local_checksum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm=0.0,max_compute=0.0,max_total=0.0;
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double *gathered=NULL, *ref=NULL, *tmp=NULL;
    if (rank == 0) {
        gathered=malloc((size_t)global_n*sizeof(double));
        ref=malloc((size_t)global_n*sizeof(double));
        tmp=malloc((size_t)global_n*sizeof(double));
        if (!gathered || !ref || !tmp) MPI_Abort(MPI_COMM_WORLD, 3);
    }
    MPI_Gather(cur+1, (int)local_n, MPI_DOUBLE, gathered, (int)local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (long i=0; i<global_n; ++i) ref[i]=(i%100)*0.01;
        for (int step=0; step<steps; ++step) {
            tmp[0]=ref[0]; tmp[global_n-1]=ref[global_n-1];
            for (long i=1; i<global_n-1; ++i) tmp[i]=(ref[i-1]+ref[i]+ref[i+1])/3.0;
            double *swap=ref; ref=tmp; tmp=swap;
        }
        double max_error=0.0;
        for (long i=0; i<global_n; ++i) {
            double err=gathered[i]-ref[i]; if (err<0) err=-err;
            if (err>max_error) max_error=err;
        }
        if (max_error > 1e-12) MPI_Abort(MPI_COMM_WORLD, 4);
        printf("STENCIL_VALIDATION max_error=%.12e result=PASS\n", max_error);
        printf("MPI_STENCIL ranks=%d global_n=%ld local_n=%ld steps=%d compute_max=%.6f halo_max=%.6f total_max=%.6f checksum=%.12e\n",
               size, global_n, local_n, steps, max_compute, max_comm, max_total, checksum);
    }

    free(gathered); free(ref); free(tmp);
    free(a); free(b);
    MPI_Finalize();
    return 0;
}
