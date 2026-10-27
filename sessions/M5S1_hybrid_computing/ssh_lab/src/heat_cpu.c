/* Same mathematics. Swap pointers; optionally compile OpenMP or a library boundary. */
#include "heat_common.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef HOT_FROM_LIBRARY
extern double top_temperature(void);
#endif
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: heat_cpu N STEPS field.bin\n");return 2;}
 /* Keep the same grid and number of steps for every execution layout. */
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000),threads=1;
 const char *mode="serial";double hot=100;
#ifdef HOT_FROM_LIBRARY
 /* The shared library must return Celsius, the units expected by this solver. */
 hot=top_temperature();mode="library";
#endif
#ifdef _OPENMP
 mode="openmp";
 /* Measure the actual team size, rather than assuming the requested count. */
 #pragma omp parallel
 {
  #pragma omp single
  threads=omp_get_num_threads();
 }
#endif
 double start=clock_seconds();double *old=grid(n,hot),*next=grid(n,hot);
 double setup=clock_seconds()-start,t=clock_seconds();
 for(int s=0;s<steps;s++){
#ifdef _OPENMP
  /* Threads read shared old data and write disjoint rows in next.
     The implicit end-of-loop barrier completes all writes before the swap. */
  #pragma omp parallel for schedule(static)
#endif
  for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++){
   size_t k=(size_t)y*n+x;
   next[k]=0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
  }
  /* Swap names instead of copying every temperature. Boundaries exist in both arrays. */
  double *tmp=old;old=next;next=tmp;
 }
 /* I/O is timed separately so it cannot be mistaken for stencil computation. */
 double loop=clock_seconds()-t;t=clock_seconds();save_grid(argv[3],old,n);
 double io=clock_seconds()-t;
 result(mode,n,steps,1,threads,setup,loop,0,loop,io,clock_seconds()-start,old);
 free(old);free(next);return 0;
}
