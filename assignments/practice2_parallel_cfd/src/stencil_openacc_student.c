#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec/1e9;
}

static void parse(int argc,char **argv,int *n,int *steps,int *demo){
    *n=2048; *steps=200; *demo=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--demo")){*demo=1; *n=21; *steps=4;}
        else if(!strcmp(argv[i],"--n") && i+1<argc) *n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps") && i+1<argc) *steps=atoi(argv[++i]);
    }
}

int main(int argc,char **argv){
    int n,steps,demo; parse(argc,argv,&n,&steps,&demo);
    size_t sz=(size_t)n*n;
    double *current=calloc(sz,sizeof(double));
    double *next=calloc(sz,sizeof(double));
    if(!current||!next) return 2;
    current[(n/2)*n+n/2]=100.0;

    double t0=now_s();

    /* TODO OPENACC DATA REGION
       Add an OpenACC data region around the repeated timestep loop so that
       current and next do not have to be transferred host<->device every step.
       Hint: both arrays are needed on the device and the final current values
       must be available on the host after the loop.
    */

    for(int s=0;s<steps;s++){
        memset(next,0,sz*sizeof(double));

        /* TODO OPENACC PARALLEL LOOP
           Add the directive that offloads/parallelizes the two nested loops.
           Use collapse(2). Do not change the numerical expression.
        */
        for(int i=1;i<n-1;i++){
            for(int j=1;j<n-1;j++){
                next[(size_t)i*n+j]=0.25*(
                    current[(size_t)(i-1)*n+j] +
                    current[(size_t)(i+1)*n+j] +
                    current[(size_t)i*n+j-1] +
                    current[(size_t)i*n+j+1]);
            }
        }
        double *tmp=current; current=next; next=tmp;
    }

    double t1=now_s();
    double sum=0.0; long long nonzero=0;
    for(size_t k=0;k<sz;k++){
        sum+=current[k];
        if(fabs(current[k])>1e-12) nonzero++;
    }

    printf("MODE=OPENACC\nGRID=%dx%d\nSTEPS=%d\n",n,n,steps);
    printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",
           current[(n/2)*n+n/2],sum,nonzero);
    printf("PROGRAM_SECONDS=%.6f\n",t1-t0);

    free(current); free(next); return 0;
}
