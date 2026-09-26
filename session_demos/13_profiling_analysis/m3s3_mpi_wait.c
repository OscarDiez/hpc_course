#include <mpi.h>
#include <stdio.h>
#include <stdint.h>

static volatile double sink = 0.0;

__attribute__((noinline))
static void work(uint64_t iters){
    double x = 1.000001;
    for(uint64_t i=0;i<iters;++i)
        x = x * 1.00000001 + 0.00000003;
    sink += x;
}

int main(int argc, char **argv){
    MPI_Init(&argc,&argv);

    int rank=0,size=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(size < 2){
        if(rank==0) fprintf(stderr,"run with at least 2 MPI ranks\n");
        MPI_Finalize();
        return 1;
    }

    uint64_t base = 25000000ULL;
    uint64_t iters = (rank == size-1) ? base*4ULL : base;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    work(iters);
    double t_compute = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double t_wait = MPI_Wtime() - t0;

    printf("MPI_TRACE rank=%d size=%d work_iters=%llu compute=%.6f barrier_wait=%.6f\n",
           rank,size,(unsigned long long)iters,t_compute,t_wait);

    MPI_Finalize();
    return 0;
}
