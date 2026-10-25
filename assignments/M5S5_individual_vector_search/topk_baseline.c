/* M5S5 individual assignment: supplied SERIAL baseline, not the optimized answer. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void stop(const char *why) {
 fprintf(stderr,"FAIL: %s\n",why);
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
typedef struct { double score; int id; } Hit;
static int compare(const void *pa,const void *pb) {
 const Hit *a=pa,*b=pb;
 if(a->score>b->score)return -1;
 if(a->score<b->score)return 1;
 return (a->id>b->id)-(a->id<b->id);
}
int main(int argc,char **argv) {
 if(argc!=5 && argc!=6)stop("usage: topk N D Q K [ties]");
 int n=number(argv[1],1000000),d=number(argv[2],512),nq=number(argv[3],128),k=number(argv[4],100);
 if(k>n)stop("K must not exceed N");
 int ties=argc==6;
 if(ties && strcmp(argv[5],"ties"))stop("optional argument must be ties");
 if((size_t)n*d*sizeof(float)>256U*1024U*1024U)stop("keep collection at or below 256 MiB");
 float *x=malloc((size_t)n*d*sizeof(float)),*q=malloc((size_t)nq*d*sizeof(float));
 Hit *scores=malloc((size_t)n*sizeof(Hit)),*answer=malloc((size_t)nq*k*sizeof(Hit));
 if(!x||!q||!scores||!answer)stop("allocation failed");
 for(int i=0;i<n;i++)vector(x+(size_t)i*d,d,ties?1U:(uint32_t)(i+1));
 for(int j=0;j<nq;j++)vector(q+(size_t)j*d,d,(uint32_t)(10000001+j));
 double compute=0,selection=0,start=now();
 for(int query=0;query<nq;query++) {
  double t=now();
  for(int i=0;i<n;i++) {
   double dot=0;
   for(int j=0;j<d;j++)dot+=(double)x[(size_t)i*d+j]*q[(size_t)query*d+j];
   scores[i]=(Hit){dot,i};
  }
  compute+=now()-t;t=now();
  /* Deliberately simple O(N log N) full sort. Students must investigate its cost. */
  qsort(scores,(size_t)n,sizeof(Hit),compare);
  for(int j=0;j<k;j++)answer[(size_t)query*k+j]=scores[j];
  selection+=now()-t;
 }
 double elapsed=now()-start;
 printf("{\"n\":%d,\"d\":%d,\"queries\":%d,\"k\":%d,\"ties\":%s,\"ranks\":1,\"threads\":1,\"seconds\":%.9g,\"compute_s\":%.9g,\"selection_s\":%.9g,\"hits\":[",n,d,nq,k,ties?"true":"false",elapsed,compute,selection);
 for(int query=0;query<nq;query++) {
  printf("%s[",query?",":"");
  for(int j=0;j<k;j++){Hit h=answer[(size_t)query*k+j];printf("%s{\"id\":%d,\"score\":%.17g}",j?",":"",h.id,h.score);}
  printf("]");
 }
 puts("]}");free(x);free(q);free(scores);free(answer);return 0;
}
