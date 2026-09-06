#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

__global__ void stencil_step(const double *current,double *next,int n){
    /* TODO CUDA
       Complete i and j using blockIdx, blockDim and threadIdx.
       Then complete the condition so that only interior grid points are updated.
       Each useful GPU thread should update one grid point.
    */
    int j = 0;  // TODO
    int i = 0;  // TODO

    if (0) {    // TODO: replace with the correct interior-boundary condition
        next[(size_t)i*n+j]=0.25*(
            current[(size_t)(i-1)*n+j] +
            current[(size_t)(i+1)*n+j] +
            current[(size_t)i*n+j-1] +
            current[(size_t)i*n+j+1]);
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
    double *d_current,*d_next;
    CUDA_OK(cudaMalloc(&d_current,bytes));
    CUDA_OK(cudaMalloc(&d_next,bytes));

    cudaEvent_t total0,total1,k0,k1;
    cudaEventCreate(&total0); cudaEventCreate(&total1);
    cudaEventCreate(&k0); cudaEventCreate(&k1);
    CUDA_OK(cudaEventRecord(total0));
    CUDA_OK(cudaMemcpy(d_current,h,bytes,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_next,0,bytes));

    dim3 block(16,16);
    dim3 grid((n+block.x-1)/block.x,(n+block.y-1)/block.y);

    CUDA_OK(cudaEventRecord(k0));
    for(int s=0;s<steps;s++){
        CUDA_OK(cudaMemset(d_next,0,bytes));
        stencil_step<<<grid,block>>>(d_current,d_next,n);
        CUDA_OK(cudaGetLastError());
        double *tmp=d_current; d_current=d_next; d_next=tmp;
    }
    CUDA_OK(cudaEventRecord(k1));
    CUDA_OK(cudaEventSynchronize(k1));
    CUDA_OK(cudaMemcpy(h,d_current,bytes,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(total1));
    CUDA_OK(cudaEventSynchronize(total1));

    float kms=0,tms=0;
    cudaEventElapsedTime(&kms,k0,k1);
    cudaEventElapsedTime(&tms,total0,total1);

    double sum=0.0; long long nonzero=0;
    for(size_t k=0;k<sz;k++){
        sum+=h[k];
        if(fabs(h[k])>1e-12) nonzero++;
    }

    printf("MODE=CUDA\nGPU_MODEL=%s\nGRID=%dx%d\nSTEPS=%d\n",prop.name,n,n,steps);
    printf("BLOCK_X=%u\nBLOCK_Y=%u\nTHREADS_PER_BLOCK=%u\nGRID_X=%u\nGRID_Y=%u\n",
           block.x,block.y,block.x*block.y,grid.x,grid.y);
    printf("CENTER_VALUE=%.6f\nCHECKSUM=%.6f\nNONZERO=%lld\n",
           h[(n/2)*n+n/2],sum,nonzero);
    printf("GPU_KERNEL_SECONDS=%.6f\nGPU_TOTAL_SECONDS=%.6f\n",kms/1000.0,tms/1000.0);

    cudaFree(d_current); cudaFree(d_next); free(h);
    cudaEventDestroy(total0); cudaEventDestroy(total1); cudaEventDestroy(k0); cudaEventDestroy(k1);
    return 0;
}
