#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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

    double t0=omp_get_wtime();
    for(int s=0;s<steps;s++){
        memset(next,0,sz*sizeof(double));

        /* TODO OPENMP
           Add ONE OpenMP directive immediately below this comment so that
           the two-dimensional (i,j) iteration space is distributed among
           threads. Use the concepts parallel for + collapse(2).
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
    double t1=omp_get_wtime();

    double sum=0.0; long long nonzero=0;
    for(size_t k=0;k<sz;k++){
        sum+=current[k];
        if(fabs(current[k])>1e-12) nonzero++;
    }

    printf("MODE=OPENMP\nGRID=%dx%d\nSTEPS=%d\nTHREADS=%d\n",
           n,n,steps,omp_get_max_threads());
    printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",
           current[(n/2)*n+n/2],sum,nonzero);
    printf("PROGRAM_SECONDS=%.6f\n",t1-t0);

    free(current); free(next); return 0;
}
