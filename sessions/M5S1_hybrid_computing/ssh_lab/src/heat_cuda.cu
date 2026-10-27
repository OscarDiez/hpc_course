/* One GPU heat solver. Both arrays stay on the device between time steps. */
#include "heat_common.h"
#include <cuda_runtime.h>
#define CUDA(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(3);}}while(0)
__global__ void stencil(const double *old,double *next,int n){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>0&&x<n-1&&y>0&&y<n-1){size_t k=(size_t)y*n+x;next[k]=0.25*(old[k-n]+old[k+n]+old[k-1]+old[k+1]);}
}
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: heat_cuda N STEPS field.bin\n");return 2;}
 int n=integer(argv[1],8,2048),steps=integer(argv[2],1,10000);
 double start=clock_seconds();double *a=grid(n,100),*b=grid(n,100),*da,*db;
 size_t bytes=(size_t)n*n*sizeof(double);CUDA(cudaMalloc((void**)&da,bytes));CUDA(cudaMalloc((void**)&db,bytes));
 CUDA(cudaMemcpy(da,a,bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(db,b,bytes,cudaMemcpyHostToDevice));
 cudaEvent_t e0,e1;CUDA(cudaEventCreate(&e0));CUDA(cudaEventCreate(&e1));
 double setup=clock_seconds()-start;dim3 block(16,16),blocks((n+15)/16,(n+15)/16);
 CUDA(cudaEventRecord(e0));
 for(int s=0;s<steps;s++){
  stencil<<<blocks,block>>>(da,db,n);CUDA(cudaGetLastError());double *tmp=da;da=db;db=tmp;
 }
 CUDA(cudaEventRecord(e1));CUDA(cudaEventSynchronize(e1));float ms;CUDA(cudaEventElapsedTime(&ms,e0,e1));
 double t=clock_seconds();CUDA(cudaMemcpy(a,da,bytes,cudaMemcpyDeviceToHost));save_grid(argv[3],a,n);
 double io=clock_seconds()-t;
 CUDA(cudaEventDestroy(e0));CUDA(cudaEventDestroy(e1));CUDA(cudaFree(da));CUDA(cudaFree(db));
 result("cuda",n,steps,1,1,setup,ms/1000.0,0,ms/1000.0,io,clock_seconds()-start,a);
 free(a);free(b);return 0;
}
