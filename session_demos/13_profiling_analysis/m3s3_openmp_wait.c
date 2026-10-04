#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

__attribute__((noinline))
static double do_work(uint64_t iters){
    double x=1.000001;
    for(uint64_t i=0;i<iters;++i) x=x*1.00000001+0.00000003;
    return x;
}
int main(int argc,char **argv){
    const char *mode=argc>1?argv[1]:"skewed";
    int requested=argc>2?atoi(argv[2]):4, skew=argc>3?atoi(argv[3]):8;
    uint64_t base=argc>4?strtoull(argv[4],NULL,10):2000000;
    if((strcmp(mode,"skewed")&&strcmp(mode,"balanced")) || requested<1 || requested>64 ||
       skew<1 || skew>32 || base<100 || base>10000000){fprintf(stderr,"invalid OpenMP settings\n");return 2;}
    int tasks=requested-1+skew, actual=0;
    double compute[64]={0}, wait[64]={0}, start[64]={0}, end[64]={0}, results[64]={0};
    int counts[64]={0};
    omp_set_dynamic(0);omp_set_num_threads(requested);
    double origin=0, wall0=omp_get_wtime();
    #pragma omp parallel shared(origin,actual)
    {
        int tid=omp_get_thread_num();
        #pragma omp single
        {actual=omp_get_num_threads();origin=omp_get_wtime();}
        start[tid]=omp_get_wtime()-origin;
        for(int task=0;task<tasks;++task){
            int owner=!strcmp(mode,"balanced")?task%actual:(task<actual-1?task:actual-1);
            if(owner==tid){results[tid]+=do_work(base);counts[tid]++;}
        }
        end[tid]=omp_get_wtime()-origin;
        compute[tid]=end[tid]-start[tid];
        double t0=omp_get_wtime();
        #pragma omp barrier
        wait[tid]=omp_get_wtime()-t0;
    }
    double wall=omp_get_wtime()-wall0, sum=0;int done=0;
    for(int i=0;i<actual;++i){sum+=results[i];done+=counts[i];}
    double ref=tasks*do_work(base);
    if(actual!=requested || done!=tasks || fabs(sum-ref)>1e-10*fmax(1,fabs(ref))){
        fprintf(stderr,"VALIDATION_FAIL parallel_task_ownership actual=%d requested=%d\n",actual,requested);return 3;
    }
    for(int i=0;i<actual;++i)
        printf("OMP_TRACE mode=%s thread=%d threads=%d tasks=%d work_iters=%llu start=%.9f compute_end=%.9f compute=%.9f barrier_wait=%.9f\n",
        mode,i,actual,counts[i],(unsigned long long)(base*counts[i]),start[i],end[i],compute[i],wait[i]);
    printf("OMP_TOTAL mode=%s tasks=%d seconds=%.9f checksum=%.12e\n",mode,tasks,wall,sum);
    printf("VALIDATION name=parallel_task_ownership result=PASS\n");return 0;
}
