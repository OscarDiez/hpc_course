/* First program: one CPU, explicit copy after every time step. */
#include "heat_common.h"
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: heat_naive N STEPS field.bin\n");return 2;}
 /* N includes boundary cells; STEPS is the number of time updates. */
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000);
 /* Two arrays keep all reads at the old time level while we write the next one. */
 double start=clock_seconds();double *old=grid(n,100),*next=grid(n,100);
 double setup=clock_seconds()-start, t=clock_seconds();
 /* Boundary rows and columns are excluded from the update loops. */
 for(int s=0;s<steps;s++){
  for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++){
   /* Flatten (row, column) to one index; neighbours are +/-n and +/-1. */
   size_t k=(size_t)y*n+x;
   next[k]=0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
  }
  /* This extra copy is correct but moves the full interior on every step. */
  for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++)old[(size_t)y*n+x]=next[(size_t)y*n+x];
 }
 /* Stop the loop timer before writing the field to storage. */
 double loop=clock_seconds()-t;t=clock_seconds();save_grid(argv[3],old,n);
 double io=clock_seconds()-t;
 result("naive",n,steps,1,1,setup,loop,0,loop,io,clock_seconds()-start,old);
 free(old);free(next);return 0;
}
