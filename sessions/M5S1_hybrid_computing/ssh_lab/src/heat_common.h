#ifndef HEAT_COMMON_H
#define HEAT_COMMON_H
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <string.h>
static inline int integer(const char *s,int lo,int hi){
 char *end; errno=0; long v=strtol(s,&end,10);
 if(errno||*end||end==s||v<lo||v>hi){fprintf(stderr,"Invalid integer: %s\n",s);exit(2);}return (int)v;
}
static inline double clock_seconds(void){
 struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t)){perror("clock_gettime");exit(2);}
 return t.tv_sec+1e-9*t.tv_nsec;
}
static inline double *grid(int n,double hot){
 double *a=(double*)calloc((size_t)n*n,sizeof(double));
 if(!a){perror("calloc");exit(2);}for(int x=0;x<n;x++)a[x]=hot;return a;
}
static inline void save_grid(const char *path,const double *a,int n){
 FILE *f=fopen(path,"wb");if(!f){perror(path);exit(2);}
 if(fwrite(a,sizeof(double),(size_t)n*n,f)!=(size_t)n*n){fprintf(stderr,"Short grid write\n");exit(2);}
 if(fclose(f)){perror("fclose");exit(2);}
}
static inline void result(const char *mode,int n,int steps,int ranks,int threads,
 double setup,double loop,double halo,double update,double io,double total,const double *a){
 double sum=0;for(size_t k=0;k<(size_t)n*n;k++)sum+=a[k];
 printf("RESULT mode=%s n=%d steps=%d ranks=%d threads=%d setup_s=%.9f loop_s=%.9f halo_s=%.9f update_s=%.9f io_s=%.9f total_s=%.9f checksum=%.17g\n",
 mode,n,steps,ranks,threads,setup,loop,halo,update,io,total,sum);
}
#endif
