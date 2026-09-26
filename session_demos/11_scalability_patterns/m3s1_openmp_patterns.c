#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

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
    sink_value += sum * 1e-30;
    return t;
}

static void experiment_strong(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long n = 6000000L;
    const int reps = 70;
    printf("# fixed problem: n=%ld reps=%d\n", n, reps);
    for (int j=0; j<5; ++j) {
        int p = threads_list[j];
        (void)run_map(20000, 10, p);
        double a=run_map(n,reps,p), b=run_map(n,reps,p), c=run_map(n,reps,p);
        printf("OMP_STRONG threads=%d t1=%.6f t2=%.6f t3=%.6f median=%.6f\n", p,a,b,c,median3(a,b,c));
    }
}

static void experiment_weak(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long per_thread = 450000L;
    const int reps = 70;
    printf("# weak scaling: work_per_thread=%ld reps=%d\n", per_thread, reps);
    for (int j=0; j<5; ++j) {
        int p = threads_list[j];
        long n = per_thread * p;
        (void)run_map(20000, 10, p);
        double a=run_map(n,reps,p), b=run_map(n,reps,p), c=run_map(n,reps,p);
        printf("OMP_WEAK threads=%d n=%ld t1=%.6f t2=%.6f t3=%.6f median=%.6f\n", p,n,a,b,c,median3(a,b,c));
    }
}

static void experiment_small(void) {
    const int threads_list[] = {1,2,4,8,16};
    const long n = 20000L;
    const int reps = 30;
    for (int j=0; j<5; ++j) {
        int p=threads_list[j];
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

static void busy_units(int units) {
    double s=0.0;
    long loops = 30000L * units;
    for(long i=0;i<loops;++i) s += work_value(i,8);
    sink_value += s*1e-30;
}

static double taskfarm_run(int dynamic_schedule) {
    const int tasks=64;
    double t0=omp_get_wtime();
    if(dynamic_schedule) {
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<tasks;++i) {
            int weight = (i < 12) ? 12 : ((i < 28) ? 5 : 1);
            busy_units(weight);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<tasks;++i) {
            int weight = (i < 12) ? 12 : ((i < 28) ? 5 : 1);
            busy_units(weight);
        }
    }
    return omp_get_wtime()-t0;
}

static void experiment_taskfarm(void) {
    omp_set_num_threads(8);
    double ts=taskfarm_run(0);
    double td=taskfarm_run(1);
    printf("TASK_FARM threads=8 static=%.6f dynamic=%.6f speedup_dynamic_vs_static=%.3f\n",ts,td,ts/td);
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
        double t0=omp_get_wtime();
        parallel_scan(in,out,n,p);
        double t=omp_get_wtime()-t0;
        printf("SCAN threads=%d seconds=%.6f last=%.1f expected=%.1f\n",p,t,out[n-1],(double)(n-1));
    }
    free(in);free(out);
}

static void experiment_stencil(void) {
    const long n=1600000L;
    const int steps=30;
    double *a=(double*)malloc((size_t)n*sizeof(double));
    double *b=(double*)malloc((size_t)n*sizeof(double));
    if(!a||!b){fprintf(stderr,"allocation failed\n");exit(2);}
    int pvals[] = {1,4,16};
    for(int pidx=0;pidx<3;++pidx){
        int p=pvals[pidx];
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
        printf("OMP_STENCIL threads=%d n=%ld steps=%d seconds=%.6f checksum=%.6e\n",p,n,steps,t,cur[n/2]);
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

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr,"usage: %s strong|weak|small|reduce|taskfarm|scan|stencil|search\n",argv[0]);
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
    else { fprintf(stderr,"unknown experiment: %s\n",argv[1]); return 1; }
    if(sink_value==1234567.0) fprintf(stderr,"ignore %.12f\n",sink_value);
    return 0;
}
