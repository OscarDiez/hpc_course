/* M5S5 guided exact top-1 search. Deterministic synthetic vectors, not text embeddings.
 * Serial: gcc -O3 -std=c11 search.c -lm -o search_serial
 * OpenMP: gcc -O3 -std=c11 -fopenmp search.c -lm -o search_omp
 * Hybrid: mpicc -O3 -std=c11 -fopenmp -DUSE_MPI search.c -lm -o search_mpi
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void stop(const char *why) {
 fprintf(stderr,"FAIL: %s\n",why);
#ifdef USE_MPI
 MPI_Abort(MPI_COMM_WORLD,2);
#endif
 exit(2);
}
/* Stateless 32-bit hash: each rank can generate its own rows by global ID. */
static uint32_t mix(uint32_t x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }
static void vector(float *v,int d,uint32_t id) {
 double norm=0;
 for(int j=0;j<d;j++){ v[j]=(float)((int)(mix(id*747796405U+(uint32_t)j*2891336453U)>>8)%2001-1000)/1000.0f; norm+=(double)v[j]*v[j]; }
 norm=sqrt(norm); if(norm==0) stop("zero vector");
 for(int j=0;j<d;j++)v[j]=(float)(v[j]/norm);
}
static int number(const char *s,int hi) { char *end; long v=strtol(s,&end,10); if(*end||v<1||v>hi)stop("invalid size");return (int)v; }
int main(int argc,char **argv) {
 int rank=0,ranks=1,threads=1;
#ifdef USE_MPI
 int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
 if(provided<MPI_THREAD_FUNNELED)stop("MPI thread support too low");
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
 if(argc!=4)stop("usage: search N D Q (N<=1000000,D<=512,Q<=128)");
 int n=number(argv[1],1000000),d=number(argv[2],512),nq=number(argv[3],128);
 if(ranks>n)stop("more ranks than vectors");
#ifdef _OPENMP
 #pragma omp parallel
 {
  #pragma omp single
  threads=omp_get_num_threads();
 }
#endif
 int begin=(int)((long long)n*rank/ranks),end=(int)((long long)n*(rank+1)/ranks),local=end-begin;
 float *x=malloc((size_t)local*d*sizeof(float)),*q=malloc((size_t)nq*d*sizeof(float));
 double *scores=malloc((size_t)local*sizeof(double)),*best=malloc((size_t)nq*sizeof(double));
 int *ids=malloc((size_t)nq*sizeof(int));
 double *gather_scores=malloc((size_t)ranks*sizeof(double));int *gather_ids=malloc((size_t)ranks*sizeof(int));
 if(!x||!q||!scores||!best||!ids||!gather_scores||!gather_ids)stop("allocation failed");
 for(int i=0;i<local;i++)vector(x+(size_t)i*d,d,(uint32_t)(begin+i+1));
 for(int k=0;k<nq;k++)vector(q+(size_t)k*d,d,(uint32_t)(10000001+k));
 /* One untimed dot-product pass warms memory/code. It does not initialize results. */
 volatile double warm=0; for(int j=0;j<d;j++)warm+=(double)x[j]*q[j];(void)warm;
#ifdef USE_MPI
 MPI_Barrier(MPI_COMM_WORLD);
#endif
 double start=now(),compute=0,selection=0,collective=0;
 for(int k=0;k<nq;k++) {
  double t=now();
  /* Each thread writes a different scores[i]; no shared best-index race. */
  #pragma omp parallel for schedule(static)
  for(int i=0;i<local;i++) {
   double dot=0;
   for(int j=0;j<d;j++)dot+=(double)x[(size_t)i*d+j]*q[(size_t)k*d+j];
   scores[i]=dot;
  }
  compute+=now()-t;t=now();
  int id=begin;double value=scores[0];
  for(int i=1;i<local;i++)if(scores[i]>value||(scores[i]==value&&begin+i<id)){value=scores[i];id=begin+i;}
  selection+=now()-t;t=now();
#ifdef USE_MPI
  /* All MPI calls are outside OpenMP regions: FUNNELED is sufficient. */
  MPI_Gather(&value,1,MPI_DOUBLE,gather_scores,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gather(&id,1,MPI_INT,gather_ids,1,MPI_INT,0,MPI_COMM_WORLD);
  if(rank==0)for(int r=0;r<ranks;r++)if(gather_scores[r]>value||(gather_scores[r]==value&&gather_ids[r]<id)){value=gather_scores[r];id=gather_ids[r];}
#endif
  if(ranks>1) collective+=now()-t;
  best[k]=value;ids[k]=id;
 }
 double elapsed=now()-start;
#ifdef USE_MPI
 double times[4]={elapsed,compute,selection,collective},maxima[4];
 MPI_Reduce(times,maxima,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 elapsed=maxima[0];compute=maxima[1];selection=maxima[2];collective=maxima[3];
#endif
 if(rank==0){
  printf("{\"n\":%d,\"d\":%d,\"queries\":%d,\"ranks\":%d,\"threads\":%d,\"seconds\":%.9g,\"compute_max_s\":%.9g,\"selection_max_s\":%.9g,\"collective_max_s\":%.9g,\"hits\":[",n,d,nq,ranks,threads,elapsed,compute,selection,collective);
  for(int k=0;k<nq;k++)printf("%s{\"id\":%d,\"score\":%.17g}",k?",":"",ids[k],best[k]);
  puts("]}");
 }
 free(x);free(q);free(scores);free(best);free(ids);free(gather_scores);free(gather_ids);
#ifdef USE_MPI
 MPI_Finalize();
#endif
 return 0;
}
