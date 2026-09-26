#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline))
static double do_work(uint64_t iters){
    double x = 1.000001;
    for(uint64_t i=0;i<iters;++i){
        x = x * 1.00000001 + 0.00000003;
    }
    return x;
}

int main(void){
    const int requested = 4;
    omp_set_dynamic(0);
    omp_set_num_threads(requested);

    double compute[requested];
    double wait[requested];
    uint64_t work_iters[requested];
    double results[requested];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        if(nthreads != requested){
            if(tid == 0) fprintf(stderr, "Expected %d threads, got %d\n", requested, nthreads);
        }

        uint64_t base = 25000000ULL;
        uint64_t iters = (tid == requested - 1) ? base * 4ULL : base;
        work_iters[tid] = iters;

        double t0 = omp_get_wtime();
        results[tid] = do_work(iters);
        compute[tid] = omp_get_wtime() - t0;

        t0 = omp_get_wtime();
        #pragma omp barrier
        wait[tid] = omp_get_wtime() - t0;
    }

    volatile double sink = 0.0;
    for(int i=0;i<requested;++i) sink += results[i];

    for(int i=0;i<requested;++i){
        printf("OMP_TRACE thread=%d threads=%d work_iters=%llu compute=%.6f barrier_wait=%.6f\n",
               i, requested, (unsigned long long)work_iters[i], compute[i], wait[i]);
    }
    printf("OMP_TRACE_SINK=%f\n", sink);
    return 0;
}
