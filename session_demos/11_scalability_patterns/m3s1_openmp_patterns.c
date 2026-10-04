#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>

static long setting(const char *name, long fallback, long lo, long hi) {
    const char *value = getenv(name);
    if (!value) return fallback;
    char *end; errno = 0;
    long n = strtol(value, &end, 10);
    if (errno || *end || end == value || n < lo || n > hi) {
        fprintf(stderr, "invalid %s (allowed %ld..%ld)\n", name, lo, hi);
        exit(2);
    }
    return n;
}
static int max_threads(void) { return (int)setting("M3_MAX_THREADS", 16, 1, 16); }
static double last_map_sum;
static double last_farm_sum;
static volatile double sink_value = 0.0;

static inline double work_value(long i, int reps) {
    double x = 1.0 + (double)(i % 1000) * 1.0e-6;
    for (int k = 0; k < reps; ++k) {
        x = x * 1.0000001192092896 + 0.0000001;
        x = x / (1.0 + x * 1.0e-7);
    }
    return x;
}

static double median3(double a, double b, double c) {
    if (a > b) { double t=a; a=b; b=t; }
    if (b > c) { double t=b; b=c; c=t; }
    if (a > b) { double t=a; a=b; b=t; }
    return b;
}

static double run_map(long n, int reps, int threads) {
    omp_set_num_threads(threads);
    double sum = 0.0;
    double t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < n; ++i) {
        sum += work_value(i, reps);
    }
    double t = omp_get_wtime() - t0;
    last_map_sum = sum;
    sink_value += sum * 1e-30;
    return t;
}

static void experiment_strong(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long n = setting("M3_MAP_N", 1000000, 100, 12000000);
    const int reps = (int)setting("M3_WORK_REPS", 40, 1, 200);
    printf("# fixed problem: n=%ld reps=%d\n", n, reps);
    for (int j=0; j<5; ++j) {
        int p = threads_list[j];
        if (p > max_threads()) continue;
        (void)run_map(20000, 10, p);
        double a=run_map(n,reps,p), b=run_map(n,reps,p), c=run_map(n,reps,p);
        printf("OMP_STRONG threads=%d t1=%.9f t2=%.9f t3=%.9f median=%.9f sum=%.12e\n", p,a,b,c,median3(a,b,c),last_map_sum);
    }
}

static void experiment_weak(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long per_thread = setting("M3_WEAK_N", 100000, 100, 450000);
    const int reps = (int)setting("M3_WORK_REPS", 40, 1, 200);
    printf("# weak scaling: work_per_thread=%ld reps=%d\n", per_thread, reps);
    for (int j=0; j<5; ++j) {
        int p = threads_list[j];
        if (p > max_threads()) continue;
        long n = per_thread * p;
        (void)run_map(20000, 10, p);
        double a=run_map(n,reps,p), b=run_map(n,reps,p), c=run_map(n,reps,p);
        printf("OMP_WEAK threads=%d n=%ld t1=%.9f t2=%.9f t3=%.9f median=%.6f\n", p,n,a,b,c,median3(a,b,c));
    }
}

static void experiment_small(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long n = setting("M3_SMALL_N", 20000, 10, 200000);
    const int reps = 30;
    for (int j=0; j<5; ++j) {
        int p=threads_list[j];
        if (p > max_threads()) continue;
        double best=DBL_MAX;
        for (int r=0;r<7;++r) {
            double t=run_map(n,reps,p);
            if (t<best) best=t;
        }
        printf("OMP_SMALL threads=%d n=%ld best=%.9f\n",p,n,best);
    }
}

static void experiment_reduce(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long n = 5000000L;
    const int reps = 30;
    for (int j=0;j<5;++j) {
        int p=threads_list[j];
        if (p > max_threads()) continue;
        omp_set_num_threads(p);
        double sum=0.0;
        double t0=omp_get_wtime();
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for(long i=0;i<n;++i) sum += work_value(i,reps);
        double t=omp_get_wtime()-t0;
        printf("OMP_REDUCE threads=%d seconds=%.6f sum=%.6e\n",p,t,sum);
        sink_value += sum*1e-30;
    }
}

static double busy_units(int units) {
    double s=0.0;
    long loops = 30000L * units;
    for(long i=0;i<loops;++i) s += work_value(i,8);
    return s;
}

static double taskfarm_run(int dynamic_schedule) {
    const int tasks=64;
    const int heavy=(int)setting("M3_TASK_HEAVY",12,1,30);
    double total_work=0.0;
    double t0=omp_get_wtime();
    if(dynamic_schedule) {
        #pragma omp parallel for schedule(dynamic,1) reduction(+:total_work)
        for(int i=0;i<tasks;++i) {
            int weight = (i < 12) ? heavy : ((i < 28) ? 5 : 1);
            total_work += busy_units(weight);
        }
    } else {
        #pragma omp parallel for schedule(static) reduction(+:total_work)
        for(int i=0;i<tasks;++i) {
            int weight = (i < 12) ? heavy : ((i < 28) ? 5 : 1);
            total_work += busy_units(weight);
        }
    }
    double elapsed=omp_get_wtime()-t0;
    last_farm_sum=total_work;
    sink_value += total_work*1e-30;
    return elapsed;
}

static void experiment_taskfarm(void) {
    int threads = max_threads() < 8 ? max_threads() : 8;
    omp_set_num_threads(threads);
    double ts=taskfarm_run(0);
    double expected=last_farm_sum;
    double td=taskfarm_run(1);
    if (fabs(last_farm_sum-expected)>1e-9*fabs(expected)) { fprintf(stderr,"task farm sum mismatch\n"); exit(3); }
    printf("TASK_FARM threads=%d static=%.9f dynamic=%.9f speedup_dynamic_vs_static=%.3f\n",threads,ts,td,ts/td);
}

static void parallel_scan(const double *in, double *out, long n, int threads) {
    double *block_sums=(double*)calloc((size_t)threads,sizeof(double));
    double *offsets=(double*)calloc((size_t)threads,sizeof(double));
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        int nt=omp_get_num_threads();
        long start=(n*tid)/nt;
        long end=(n*(tid+1))/nt;
        double running=0.0;
        for(long i=start;i<end;++i) {
            out[i]=running;
            running += in[i];
        }
        block_sums[tid]=running;
        #pragma omp barrier
        #pragma omp single
        {
            double off=0.0;
            for(int t=0;t<nt;++t) { offsets[t]=off; off += block_sums[t]; }
        }
        double off=offsets[tid];
        for(long i=start;i<end;++i) out[i]+=off;
    }
    free(block_sums); free(offsets);
}

static void experiment_scan(void) {
    const long n=4000000L;
    double *in=(double*)malloc((size_t)n*sizeof(double));
    double *out=(double*)malloc((size_t)n*sizeof(double));
    if(!in||!out){fprintf(stderr,"allocation failed\n");exit(2);}
    for(long i=0;i<n;++i) in[i]=1.0;
    int pvals[] = {1,4,8};
    for(int pidx=0;pidx<3;++pidx){
        int p=pvals[pidx];
        if (p > max_threads()) continue;
        double t0=omp_get_wtime();
        parallel_scan(in,out,n,p);
        double t=omp_get_wtime()-t0;
        for(long i=0;i<n;++i) if(out[i]!=(double)i) { fprintf(stderr,"scan failed\n");exit(3); }
        printf("SCAN threads=%d seconds=%.6f last=%.1f expected=%.1f\n",p,t,out[n-1],(double)(n-1));
    }
    free(in);free(out);
}

static void experiment_stencil(void) {
    const long n=setting("M3_STENCIL_N",400000,16,1600000);
    const int steps=(int)setting("M3_STEPS", 20, 1, 100);
    double *a=(double*)malloc((size_t)n*sizeof(double));
    double *b=(double*)malloc((size_t)n*sizeof(double));
    if(!a||!b){fprintf(stderr,"allocation failed\n");exit(2);}
    int pvals[] = {1,4,16};
    for(int pidx=0;pidx<3;++pidx){
        int p=pvals[pidx];
        if (p > max_threads()) continue;
        omp_set_num_threads(p);
        for(long i=0;i<n;++i) a[i]=(i%100)*0.01;
        double *cur=a,*next=b;
        double t0=omp_get_wtime();
        for(int s=0;s<steps;++s){
            #pragma omp parallel for schedule(static)
            for(long i=1;i<n-1;++i) next[i]=(cur[i-1]+cur[i]+cur[i+1])/3.0;
            next[0]=cur[0];next[n-1]=cur[n-1];
            double *tmp=cur;cur=next;next=tmp;
        }
        double t=omp_get_wtime()-t0;
        double checksum=0.0;
        for(long i=0;i<n;++i) checksum+=cur[i];
        printf("OMP_STENCIL threads=%d n=%ld steps=%d seconds=%.6f checksum=%.12e\n",p,n,steps,t,checksum);
    }
    free(a);free(b);
}

static void experiment_search(void) {
    const long n=12000000L;
    int *a=(int*)malloc((size_t)n*sizeof(int));
    if(!a){fprintf(stderr,"allocation failed\n");exit(2);}
    for(long i=0;i<n;++i) a[i]=(int)(i%1000003);
    long target_index=n-12345;
    int target=42424242;
    a[target_index]=target;
    int pvals[] = {1,4,8};
    for(int pidx=0;pidx<3;++pidx){
        int p=pvals[pidx];
        if (p > max_threads()) continue;
        omp_set_num_threads(p);
        long found=n;
        double t0=omp_get_wtime();
        #pragma omp parallel for reduction(min:found) schedule(static)
        for(long i=0;i<n;++i) if(a[i]==target && i<found) found=i;
        double t=omp_get_wtime()-t0;
        printf("SEARCH threads=%d seconds=%.6f found=%ld expected=%ld\n",p,t,found,target_index);
    }
    free(a);
}


static double run_amdahl_case(long serial_n, long parallel_n, int reps, int threads,
                              double *serial_time, double *parallel_time) {
    omp_set_num_threads(threads);

    double serial_sum = 0.0;
    double parallel_sum = 0.0;

    double total0 = omp_get_wtime();

    /* Deliberately serial phase: this work cannot use extra threads. */
    double s0 = omp_get_wtime();
    for (long i = 0; i < serial_n; ++i) {
        serial_sum += work_value(i, reps);
    }
    *serial_time = omp_get_wtime() - s0;

    /* Parallel phase: the same kind of work can use all requested threads. */
    double p0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:parallel_sum) schedule(static)
    for (long i = 0; i < parallel_n; ++i) {
        parallel_sum += work_value(serial_n + i, reps);
    }
    *parallel_time = omp_get_wtime() - p0;

    double total = omp_get_wtime() - total0;
    sink_value += (serial_sum + parallel_sum) * 1e-30;
    return total;
}

static void experiment_amdahl(void) {
    const int threads_list[] = {1, 2, 4, 8, 16};
    const long total_n = setting("M3_MAP_N", 1000000, 100, 12000000);
    const long serial_n = total_n * setting("M3_SERIAL_PERCENT", 5, 0, 50) / 100;          /* 5% of loop iterations */
    const long parallel_n = total_n - serial_n;  /* 95% of loop iterations */
    const int reps = (int)setting("M3_WORK_REPS", 40, 1, 200);

    printf("# controlled Amdahl experiment: serial iterations are configurable; time fraction is measured\n");
    printf("# total_n=%ld serial_n=%ld parallel_n=%ld reps=%d\n",
           total_n, serial_n, parallel_n, reps);

    for (int j = 0; j < 5; ++j) {
        int p = threads_list[j];
        if (p > max_threads()) continue;

        /* Short warm-up so first-use runtime effects do not dominate. */
        double warm_s = 0.0, warm_p = 0.0;
        (void)run_amdahl_case(5000, 95000, 10, p, &warm_s, &warm_p);

        double s1, p1, s2, p2, s3, p3;
        double t1 = run_amdahl_case(serial_n, parallel_n, reps, p, &s1, &p1);
        double t2 = run_amdahl_case(serial_n, parallel_n, reps, p, &s2, &p2);
        double t3 = run_amdahl_case(serial_n, parallel_n, reps, p, &s3, &p3);

        double total_med = median3(t1, t2, t3);
        double serial_med = median3(s1, s2, s3);
        double parallel_med = median3(p1, p2, p3);

        printf("AMDAHL_REAL threads=%d serial=%.9f parallel=%.9f total=%.9f\n",
               p, serial_med, parallel_med, total_med);
    }
}



/* Counter-based samples: index and seed identify each point, independent of threads.
   SplitMix64 supplies deterministic teaching samples, not cryptographic randomness. */
static uint64_t mix64(uint64_t z) {
    z += UINT64_C(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static void experiment_montecarlo(void) {
    long n = setting("M3_MC_N", 1000000, 100, 10000000);
    uint64_t seed = (uint64_t)setting("M3_SEED", 42, 0, 1000000);
    const int ps[] = {1,2,4,8,16};
    long baseline = -1;
    for (int j=0; j<5; ++j) {
        int p=ps[j]; if (p > max_threads()) continue;
        omp_set_num_threads(p);
        long hits=0;
        double t0=omp_get_wtime();
        #pragma omp parallel for reduction(+:hits) schedule(static)
        for (long i=0; i<n; ++i) {
            double x=(mix64(2*(uint64_t)i + 2*seed) >> 11)*0x1.0p-53;
            double y=(mix64(2*(uint64_t)i + 2*seed + 1) >> 11)*0x1.0p-53;
            hits += x*x + y*y <= 1.0;
        }
        double elapsed=omp_get_wtime()-t0;
        if (baseline < 0) baseline=hits;
        if (hits != baseline) { fprintf(stderr,"Monte Carlo correctness failed\n"); exit(3); }
        printf("MONTE_CARLO threads=%d n=%ld seed=%llu hits=%ld pi=%.9f seconds=%.9f\n",
               p,n,(unsigned long long)seed,hits,4.0*hits/n,elapsed);
    }
}

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr,"usage: %s strong|weak|small|reduce|taskfarm|scan|stencil|search|amdahl|montecarlo\n",argv[0]);
        return 1;
    }
    if(strcmp(argv[1],"strong")==0) experiment_strong();
    else if(strcmp(argv[1],"weak")==0) experiment_weak();
    else if(strcmp(argv[1],"small")==0) experiment_small();
    else if(strcmp(argv[1],"reduce")==0) experiment_reduce();
    else if(strcmp(argv[1],"taskfarm")==0) experiment_taskfarm();
    else if(strcmp(argv[1],"scan")==0) experiment_scan();
    else if(strcmp(argv[1],"stencil")==0) experiment_stencil();
    else if(strcmp(argv[1],"search")==0) experiment_search();
    else if(strcmp(argv[1],"montecarlo")==0) experiment_montecarlo();
    else if(strcmp(argv[1],"amdahl")==0) experiment_amdahl();
    else { fprintf(stderr,"unknown experiment: %s\n",argv[1]); return 1; }
    if(sink_value==1234567.0) fprintf(stderr,"ignore %.12f\n",sink_value);
    return 0;
}
