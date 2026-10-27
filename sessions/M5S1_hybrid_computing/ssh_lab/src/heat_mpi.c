/* Interior row decomposition; same stencil, optional OpenMP, MPI outside threads. */
#include "heat_common.h"
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
int main(int argc,char **argv){
 int provided,rank,p;double total_start=clock_seconds();
 /* FUNNELED means only the original thread makes MPI calls. */
 MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&p);
 /* Catch a launcher that accidentally starts independent one-rank programs. */
 if(getenv("LAB_EXPECT_RANKS") && p!=atoi(getenv("LAB_EXPECT_RANKS"))){
  if(!rank)fprintf(stderr,"MPI launcher mismatch: expected %s ranks, got %d. Check site launcher.\n",getenv("LAB_EXPECT_RANKS"),p);
  MPI_Abort(MPI_COMM_WORLD,2);
 }
 if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,2);
 if(argc!=4){if(!rank)fprintf(stderr,"usage: heat_mpi N STEPS field.bin\n");MPI_Abort(MPI_COMM_WORLD,2);}
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000),interior=n-2;
 if(p>interior){if(!rank)fprintf(stderr,"Use at most N-2 ranks\n");MPI_Abort(MPI_COMM_WORLD,2);}
 /* Divide N-2 interior rows; early ranks receive any leftover rows. */
 int rows=interior/p+(rank<interior%p),first=1+rank*(interior/p)+(rank<interior%p?rank:interior%p);
 /* Each process stores its owned rows and one halo row on either side. */
 size_t cells=(size_t)(rows+2)*n;
 double *old=(double*)calloc(cells,sizeof(double)),*next=(double*)calloc(cells,sizeof(double));
 if(!old||!next)MPI_Abort(MPI_COMM_WORLD,3);
 /* Rank 0 keeps the physical hot edge in its upper halo. */
 if(rank==0)for(int x=0;x<n;x++){old[x]=100;next[x]=100;}
 /* MPI_PROC_NULL marks a physical edge where there is no neighbour. */
 int up=rank?rank-1:MPI_PROC_NULL,down=rank==p-1?MPI_PROC_NULL:rank+1,threads=1;
 const char *mode="mpi";
#ifdef _OPENMP
 mode="hybrid";
 #pragma omp parallel
 {
  #pragma omp single
  threads=omp_get_num_threads();
 }
#endif
 char host[MPI_MAX_PROCESSOR_NAME];int host_len;MPI_Get_processor_name(host,&host_len);
 /* Print ownership so we can compare the predicted layout with the actual run. */
 printf("MAP rank=%d ranks=%d host=%s first_global_row=%d rows=%d threads=%d\n",rank,p,host,first,rows,threads);
 /* Start the measured loop after every rank has finished initialisation. */
 MPI_Barrier(MPI_COMM_WORLD);double setup=clock_seconds()-total_start;
 double t0=MPI_Wtime(),halo=0,update=0;
 for(int s=0;s<steps;s++){
  double h=MPI_Wtime();
  /* Send our top owned row up; receive the lower neighbour into our bottom halo. */
  MPI_Sendrecv(old+n,n,MPI_DOUBLE,up,10,old+(size_t)(rows+1)*n,n,MPI_DOUBLE,down,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  /* Send our bottom owned row down; receive the upper neighbour into our top halo. */
  MPI_Sendrecv(old+(size_t)rows*n,n,MPI_DOUBLE,down,11,old,n,MPI_DOUBLE,up,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  halo+=MPI_Wtime()-h;double u=MPI_Wtime();
#ifdef _OPENMP
  /* OpenMP shares only this process's local rows. MPI stays outside this region. */
  #pragma omp parallel for schedule(static)
#endif
  for(int y=1;y<=rows;y++)for(int x=1;x<n-1;x++){
   size_t k=(size_t)y*n+x;next[k]=0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
  }
  /* Both halo exchanges and the local update have finished before roles swap. */
  double *tmp=old;old=next;next=tmp;update+=MPI_Wtime()-u;
 }
 /* Maximum rank duration excludes gather and output; phase maxima need not add up. */
 double local[3]={MPI_Wtime()-t0,halo,update},times[3];
 MPI_Reduce(local,times,3,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 double io_start=clock_seconds();double *global=NULL;int *counts=NULL,*offsets=NULL;
 if(rank==0){
  global=grid(n,100);counts=(int*)malloc(p*sizeof(int));offsets=(int*)malloc(p*sizeof(int));
  if(!counts||!offsets)MPI_Abort(MPI_COMM_WORLD,3);
  for(int r=0;r<p;r++){
   counts[r]=(interior/p+(r<interior%p))*n;
   offsets[r]=(1+r*(interior/p)+(r<interior%p?r:interior%p))*n;
  }
 }
 /* Counts and offsets allow ranks to own different numbers of rows.
    Only root allocates the full global field, and gathering happens once. */
 MPI_Gatherv(old+n,rows*n,MPI_DOUBLE,global,counts,offsets,MPI_DOUBLE,0,MPI_COMM_WORLD);
 if(rank==0){
  /* Validate this complete field against serial, not just its checksum. */
  save_grid(argv[3],global,n);
  result(mode,n,steps,p,threads,setup,times[0],times[1],times[2],clock_seconds()-io_start,clock_seconds()-total_start,global);
 }
 free(old);free(next);free(global);free(counts);free(offsets);MPI_Finalize();return 0;
}
