#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

static double now_s(void){ return omp_get_wtime(); }

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
    double *u=calloc(sz,sizeof(double)), *v=calloc(sz,sizeof(double));
    if(!u||!v) return 2;
    u[(n/2)*n+n/2]=100.0;
    double t0=now_s();
    for(int s=0;s<steps;s++){
        #pragma omp parallel for schedule(static)
        for(size_t k=0;k<sz;k++) v[k]=0.0;

        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=1;i<n-1;i++)
            for(int j=1;j<n-1;j++)
                v[(size_t)i*n+j]=0.25*(u[(size_t)(i-1)*n+j]+u[(size_t)(i+1)*n+j]+u[(size_t)i*n+j-1]+u[(size_t)i*n+j+1]);
        double *tmp=u; u=v; v=tmp;
    }
    double t1=now_s(), sum=0.0; long long nonzero=0;
    #pragma omp parallel for reduction(+:sum,nonzero)
    for(size_t k=0;k<sz;k++){ sum+=u[k]; if(fabs(u[k])>1e-12) nonzero++; }
    printf("MODE=OPENMP\nGRID=%dx%d\nSTEPS=%d\nTHREADS=%d\n",n,n,steps,omp_get_max_threads());
    printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",u[(n/2)*n+n/2],sum,nonzero);
    printf("PROGRAM_SECONDS=%.6f\n",t1-t0);
    free(u); free(v); return 0;
}
