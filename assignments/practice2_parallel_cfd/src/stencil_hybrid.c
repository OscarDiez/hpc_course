#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void parse(int argc,char **argv,int *n,int *steps,int *demo){
    *n=2048; *steps=200; *demo=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--demo")){*demo=1; *n=21; *steps=4;}
        else if(!strcmp(argv[i],"--n") && i+1<argc) *n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps") && i+1<argc) *steps=atoi(argv[++i]);
    }
}

int main(int argc,char **argv){
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    char host[MPI_MAX_PROCESSOR_NAME]; int hlen=0; MPI_Get_processor_name(host,&hlen);
    int n,steps,demo; parse(argc,argv,&n,&steps,&demo);

    int base=n/size, extra=n%size;
    int myrows=base+(rank<extra?1:0);
    int start=rank*base+(rank<extra?rank:extra);
    int end=start+myrows-1;
    double *u=calloc((size_t)(myrows+2)*n,sizeof(double));
    double *v=calloc((size_t)(myrows+2)*n,sizeof(double));
    if(!u||!v) MPI_Abort(MPI_COMM_WORLD,2);
    int center=n/2;
    if(center>=start && center<=end) u[(size_t)(center-start+1)*n+center]=100.0;

    if(rank==0){
        printf("MODE=HYBRID\nGRID=%dx%d\nSTEPS=%d\nMPI_RANKS=%d\nOMP_THREADS_PER_RANK=%d\nTOTAL_CPU_THREADS=%d\n",
               n,n,steps,size,omp_get_max_threads(),size*omp_get_max_threads());
    }
    printf("rank %d: rows=%d-%d count=%d host=%s threads=%d\n",rank,start,end,myrows,host,omp_get_max_threads());
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0=MPI_Wtime();
    for(int s=0;s<steps;s++){
        int up=rank-1,down=rank+1;
        if(up<0) up=MPI_PROC_NULL; if(down>=size) down=MPI_PROC_NULL;
        MPI_Sendrecv(&u[n],n,MPI_DOUBLE,up,10,&u[(myrows+1)*n],n,MPI_DOUBLE,down,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[myrows*n],n,MPI_DOUBLE,down,20,&u[0],n,MPI_DOUBLE,up,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        #pragma omp parallel for schedule(static)
        for(size_t k=0;k<(size_t)(myrows+2)*n;k++) v[k]=0.0;

        #pragma omp parallel for collapse(2) schedule(static)
        for(int li=1;li<=myrows;li++) for(int j=1;j<n-1;j++){
            int gi=start+li-1;
            if(gi>0 && gi<n-1)
                v[(size_t)li*n+j]=0.25*(u[(size_t)(li-1)*n+j]+u[(size_t)(li+1)*n+j]+u[(size_t)li*n+j-1]+u[(size_t)li*n+j+1]);
        }
        double *tmp=u; u=v; v=tmp;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1=MPI_Wtime();

    double local_sum=0.0,center_value=0.0; long long local_nonzero=0;
    #pragma omp parallel for reduction(+:local_sum,local_nonzero) collapse(2)
    for(int li=1;li<=myrows;li++) for(int j=0;j<n;j++){
        double x=u[(size_t)li*n+j]; local_sum+=x; if(fabs(x)>1e-12) local_nonzero++;
    }
    if(center>=start && center<=end) center_value=u[(size_t)(center-start+1)*n+center];
    double global_sum=0.0,global_center=0.0; long long global_nonzero=0;
    MPI_Reduce(&local_sum,&global_sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&center_value,&global_center,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_nonzero,&global_nonzero,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    double local_time=t1-t0,max_time=0.0;
    MPI_Reduce(&local_time,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",global_center,global_sum,global_nonzero);
        printf("PROGRAM_SECONDS=%.6f\n",max_time);
    }
    free(u); free(v); MPI_Finalize(); return 0;
}
