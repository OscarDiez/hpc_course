/* First program: one CPU, explicit copy after every time step. */
#include "heat_common.h"
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: heat_naive N STEPS field.bin\n");return 2;}
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000);
 double start=clock_seconds();double *old=grid(n,100),*next=grid(n,100);
 double setup=clock_seconds()-start, t=clock_seconds();
 for(int s=0;s<steps;s++){
  for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++){
   size_t k=(size_t)y*n+x;
   next[k]=0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);
  }
  for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++)old[(size_t)y*n+x]=next[(size_t)y*n+x];
 }
 double loop=clock_seconds()-t;t=clock_seconds();save_grid(argv[3],old,n);
 double io=clock_seconds()-t;
 result("naive",n,steps,1,1,setup,loop,0,loop,io,clock_seconds()-start,old);
 free(old);free(next);return 0;
}
