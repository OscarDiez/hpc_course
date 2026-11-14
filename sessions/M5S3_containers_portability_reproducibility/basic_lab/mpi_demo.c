#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    /* Require one communicator containing both processes. */
    if (ranks != 2) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char server[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(server, &name_length);

    /* Rank 0 contributes 1 and rank 1 contributes 2. Both receive the sum. */
    int contribution = rank + 1;
    int total;
    MPI_Allreduce(&contribution, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total != 3) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("MPI rank %d/%d on %s: sum = %d\n", rank, ranks, server, total);
    if (rank == 0) {
        printf("MPI_SUM_PASS: 3\n");
    }

    MPI_Finalize();
    return 0;
}
