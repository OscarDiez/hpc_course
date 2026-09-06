#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

__global__ void stencil_step(const double *u,double *v,int n){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if(i<n && j<n){
        if(i==0 || j==0 || i==n-1 || j==n-1) v[(size_t)i*n+j]=0.0;
        else v[(size_t)i*n+j]=0.25*(u[(size_t)(i-1)*n+j]+u[(size_t)(i+1)*n+j]+u[(size_t)i*n+j-1]+u[(size_t)i*n+j+1]);
    }
}

int main(int argc,char **argv){
    int n=2048,steps=200,demo=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--demo")){demo=1;n=21;steps=4;}
        else if(!strcmp(argv[i],"--n") && i+1<argc) n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--steps") && i+1<argc) steps=atoi(argv[++i]);
    }
    size_t sz=(size_t)n*n, bytes=sz*sizeof(double);
    double *h=(double*)calloc(sz,sizeof(double));
    if(!h) return 2;
    h[(n/2)*n+n/2]=100.0;

    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,0));
    double *d_u,*d_v; CUDA_OK(cudaMalloc(&d_u,bytes)); CUDA_OK(cudaMalloc(&d_v,bytes));

    cudaEvent_t total0,total1,k0,k1; cudaEventCreate(&total0); cudaEventCreate(&total1); cudaEventCreate(&k0); cudaEventCreate(&k1);
    CUDA_OK(cudaEventRecord(total0));
    CUDA_OK(cudaMemcpy(d_u,h,bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_v,0,bytes));

    dim3 block(16,16); dim3 grid((n+block.x-1)/block.x,(n+block.y-1)/block.y);
    CUDA_OK(cudaEventRecord(k0));
    for(int s=0;s<steps;s++){
        stencil_step<<<grid,block>>>(d_u,d_v,n);
        CUDA_OK(cudaGetLastError());
        double *tmp=d_u; d_u=d_v; d_v=tmp;
    }
    CUDA_OK(cudaEventRecord(k1));
    CUDA_OK(cudaEventSynchronize(k1));
    CUDA_OK(cudaMemcpy(h,d_u,bytes,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(total1)); CUDA_OK(cudaEventSynchronize(total1));

    float kms=0,tms=0; cudaEventElapsedTime(&kms,k0,k1); cudaEventElapsedTime(&tms,total0,total1);
    double sum=0.0; long long nonzero=0;
    for(size_t k=0;k<sz;k++){ sum+=h[k]; if(fabs(h[k])>1e-12) nonzero++; }

    printf("MODE=CUDA\nGPU_MODEL=%s\nGRID=%dx%d\nSTEPS=%d\n",prop.name,n,n,steps);
    printf("THREADS_PER_BLOCK=%u\nBLOCKS_X=%u\nBLOCKS_Y=%u\n",block.x*block.y,grid.x,grid.y);
    printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",h[(n/2)*n+n/2],sum,nonzero);
    printf("GPU_KERNEL_SECONDS=%.6f\nGPU_TOTAL_SECONDS=%.6f\n",kms/1000.0,tms/1000.0);

    cudaFree(d_u); cudaFree(d_v); free(h); cudaEventDestroy(total0); cudaEventDestroy(total1); cudaEventDestroy(k0); cudaEventDestroy(k1);
    return 0;
}
