/* OpenACC: fixed array identities, alternating read/write role; no pointer remapping. */
#include "heat_common.h"
#include <openacc.h>
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: heat_acc N STEPS field.bin\n");return 2;}
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000);
 /* Refuse CPU fallback: this exercise must actually use an NVIDIA accelerator. */
 if(acc_get_num_devices(acc_device_nvidia)<1){fprintf(stderr,"No NVIDIA accelerator\n");return 3;}
 acc_set_device_type(acc_device_nvidia);
 double start=clock_seconds();double *a=grid(n,100),*b=grid(n,100);size_t count=(size_t)n*n;
 double setup=0,loop=0,copy=0;
 /* Keep both arrays on the GPU while the entire time loop runs. */
 #pragma acc data copyin(a[0:count],b[0:count])
 {
  setup=clock_seconds()-start;double t=clock_seconds();
  for(int s=0;s<steps;s++){
   /* Alternating array roles avoids changing pointer-to-device mappings. */
   if((s%2)==0){
    /* collapse(2) distributes both grid loops; present reuses mapped device data. */
    #pragma acc parallel loop collapse(2) present(a[0:count],b[0:count])
    for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++){
     size_t k=(size_t)y*n+x;b[k]=0.25*(a[k-n]+a[k+n]+a[k-1]+a[k+1]);
    }
   }else{
    #pragma acc parallel loop collapse(2) present(a[0:count],b[0:count])
    for(int y=1;y<n-1;y++)for(int x=1;x<n-1;x++){
     size_t k=(size_t)y*n+x;a[k]=0.25*(b[k-n]+b[k+n]+b[k-1]+b[k+1]);
    }
   }
  }
  /* Finish device work before recording the host-clock duration. */
  #pragma acc wait
  loop=clock_seconds()-t;t=clock_seconds();
  /* Odd steps end in b, even steps in a. Transfer only the final array. */
  if(steps%2){
   #pragma acc update self(b[0:count])
  }else{
   #pragma acc update self(a[0:count])
  }
  copy=clock_seconds()-t;
 }
 /* The same binary output format lets serial validate the whole GPU field. */
 double *final=(steps%2)?b:a;double t=clock_seconds();save_grid(argv[3],final,n);
 result("openacc",n,steps,1,1,setup,loop,0,loop,copy+clock_seconds()-t,clock_seconds()-start,final);
 free(a);free(b);return 0;
}
