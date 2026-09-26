
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static volatile double g_sink = 0.0;
static double now_sec(void){ return omp_get_wtime(); }

static void *xaligned(size_t align, size_t bytes){
    void *p=NULL;
    if(posix_memalign(&p,align,bytes)!=0 || !p){
        fprintf(stderr,"allocation failed: %.1f MiB\n",bytes/1048576.0);
        exit(2);
    }
    return p;
}

static uint64_t xorshift64(uint64_t *s){
    uint64_t x=*s;
    x^=x<<13; x^=x>>7; x^=x<<17; *s=x; return x;
}

static void bench_latency(void){
    const size_t n=4u*1024u*1024u;
    uint32_t *next=(uint32_t*)xaligned(64,n*sizeof(uint32_t));
    uint32_t *perm=(uint32_t*)xaligned(64,n*sizeof(uint32_t));
    for(size_t i=0;i<n;++i) perm[i]=(uint32_t)i;
    uint64_t state=0x123456789abcdefULL;
    for(size_t i=n-1;i>0;--i){
        size_t j=(size_t)(xorshift64(&state)%(i+1));
        uint32_t t=perm[i]; perm[i]=perm[j]; perm[j]=t;
    }
    for(size_t i=0;i+1<n;++i) next[perm[i]]=perm[i+1];
    next[perm[n-1]]=perm[0];
    volatile uint32_t idx=perm[0];
    const int laps=4;
    double t0=now_sec();
    for(int r=0;r<laps;++r)
        for(size_t i=0;i<n;++i) idx=next[idx];
    double t=now_sec()-t0;
    double accesses=(double)n*laps;
    printf("LATENCY n=%zu accesses=%.0f seconds=%.6f ns_per_access=%.2f final=%u\n",
           n,accesses,t,t*1e9/accesses,(unsigned)idx);
    free(next); free(perm);
}

static void init_three(double *a,double *b,double *c,size_t n,int threads){
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(size_t i=0;i<n;++i){
        a[i]=0.0; b[i]=1.0+(double)(i%17)*1e-3; c[i]=2.0+(double)(i%13)*1e-3;
    }
}

static void bench_stream(void){
    const size_t n=8u*1024u*1024u;
    const int reps=6;
    double *a=(double*)xaligned(64,n*sizeof(double));
    double *b=(double*)xaligned(64,n*sizeof(double));
    double *c=(double*)xaligned(64,n*sizeof(double));
    int max_threads=omp_get_max_threads();
    int cand[]={1,2,4,8,16};
    for(size_t ci=0;ci<sizeof(cand)/sizeof(cand[0]);++ci){
        int threads=cand[ci]; if(threads>max_threads) continue;
        init_three(a,b,c,n,threads);
        double scalar=3.0;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(size_t i=0;i<n;++i) a[i]=b[i]+scalar*c[i];
        double t0=now_sec();
        for(int r=0;r<reps;++r){
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t i=0;i<n;++i) a[i]=b[i]+scalar*c[i]+1e-12*r;
        }
        double t=now_sec()-t0;
        double bytes=3.0*(double)n*sizeof(double)*reps;
        double gb=bytes/t/1e9;
        double check=a[0]+a[n/2]+a[n-1];
        printf("STREAM threads=%d n=%zu reps=%d seconds=%.6f useful_GB_s=%.2f checksum=%.6f\n",
               threads,n,reps,t,gb,check);
    }
    free(a); free(b); free(c);
}

static void bench_stride(void){
    const size_t n=16u*1024u*1024u;
    double *a=(double*)xaligned(64,n*sizeof(double));
    for(size_t i=0;i<n;++i) a[i]=1.0+(double)(i&7)*1e-6;
    volatile double *v=a;
    int strides[]={1,2,4,8,16,32,64};
    const int reps=4;
    for(size_t si=0;si<sizeof(strides)/sizeof(strides[0]);++si){
        int stride=strides[si];
        volatile double sum=0.0;
        size_t accesses=0;
        double t0=now_sec();
        for(int r=0;r<reps;++r){
            for(size_t i=0;i<n;i+=(size_t)stride){ sum+=v[i]; ++accesses; }
        }
        double t=now_sec()-t0;
        printf("STRIDE stride=%d accesses=%zu seconds=%.6f ns_per_access=%.2f checksum=%.6f\n",
               stride,accesses,t,t*1e9/(double)accesses,(double)sum);
    }
    free(a);
}

static double sum_row(volatile double *a,int n){
    double s=0.0;
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) s+=a[(size_t)i*n+j];
    return s;
}
static double sum_col(volatile double *a,int n){
    double s=0.0;
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) s+=a[(size_t)i*n+j];
    return s;
}
static void bench_matrix(void){
    const int n=4096;
    size_t elems=(size_t)n*n;
    double *a=(double*)xaligned(64,elems*sizeof(double));
    for(size_t i=0;i<elems;++i) a[i]=1.0+(double)(i&31)*1e-6;
    double br=1e99,bc=1e99,sr=0,sc=0;
    for(int r=0;r<3;++r){
        double t0=now_sec(); sr=sum_row(a,n); double tr=now_sec()-t0;
        t0=now_sec(); sc=sum_col(a,n); double tc=now_sec()-t0;
        if(tr<br) br=tr; if(tc<bc) bc=tc;
    }
    g_sink+=sr+sc;
    printf("MATRIX order=row n=%d seconds=%.6f checksum=%.6e\n",n,br,sr);
    printf("MATRIX order=column n=%d seconds=%.6f checksum=%.6e\n",n,bc,sc);
    printf("MATRIX ratio_column_over_row=%.2f\n",bc/br);
    free(a);
}

static void bench_fusion(void){
    const size_t n=8u*1024u*1024u;
    const int reps=5;
    double *a=(double*)xaligned(64,n*sizeof(double));
    double *b=(double*)xaligned(64,n*sizeof(double));
    double *tmp=(double*)xaligned(64,n*sizeof(double));
    double *out=(double*)xaligned(64,n*sizeof(double));
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){ a[i]=1.0+(i&7)*1e-4; b[i]=2.0+(i&15)*1e-4; tmp[i]=out[i]=0.0; }
    const double scale=0.125;
    double t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) tmp[i]=a[i]+b[i];
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) out[i]=tmp[i]*scale+1e-12*r;
    }
    double two=now_sec()-t0;
    double c1=out[0]+out[n/2]+out[n-1];
    t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) out[i]=(a[i]+b[i])*scale+1e-12*r;
    }
    double fused=now_sec()-t0;
    double c2=out[0]+out[n/2]+out[n-1];
    printf("FUSION mode=two_pass n=%zu reps=%d seconds=%.6f checksum=%.6f\n",n,reps,two,c1);
    printf("FUSION mode=fused n=%zu reps=%d seconds=%.6f checksum=%.6f\n",n,reps,fused,c2);
    printf("FUSION speedup=%.2f\n",two/fused);
    free(a); free(b); free(tmp); free(out);
}

static void transpose_naive(double *dst,const double *src,int n){
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) dst[(size_t)j*n+i]=src[(size_t)i*n+j];
}
static void transpose_blocked(double *dst,const double *src,int n,int bs){
    for(int ii=0;ii<n;ii+=bs) for(int jj=0;jj<n;jj+=bs){
        int imax=ii+bs<n?ii+bs:n, jmax=jj+bs<n?jj+bs:n;
        for(int i=ii;i<imax;++i) for(int j=jj;j<jmax;++j)
            dst[(size_t)j*n+i]=src[(size_t)i*n+j];
    }
}
static void bench_tiling(void){
    const int n=4096;
    size_t elems=(size_t)n*n;
    double *src=(double*)xaligned(64,elems*sizeof(double));
    double *dst=(double*)xaligned(64,elems*sizeof(double));
    for(size_t i=0;i<elems;++i) src[i]=(double)(i&1023)*1e-3;
    double t0=now_sec(); transpose_naive(dst,src,n); double tn=now_sec()-t0;
    double c1=dst[0]+dst[elems/2]+dst[elems-1];
    memset(dst,0,elems*sizeof(double));
    t0=now_sec(); transpose_blocked(dst,src,n,32); double tb=now_sec()-t0;
    double c2=dst[0]+dst[elems/2]+dst[elems-1];
    printf("TILING workload=transpose mode=naive n=%d seconds=%.6f checksum=%.6f\n",n,tn,c1);
    printf("TILING workload=transpose mode=blocked block=32 n=%d seconds=%.6f checksum=%.6f\n",n,tb,c2);
    printf("TILING speedup=%.2f\n",tn/tb);
    free(src); free(dst);
}

typedef struct { volatile long long value; } CounterPacked;
typedef struct { volatile long long value; char pad[64-sizeof(long long)]; } CounterPadded;

static void bench_false_sharing(void){
    const long long iters=2000000LL;
    int max_threads=omp_get_max_threads();
    int cand[]={1,2,4,8,16};
    for(size_t ci=0;ci<sizeof(cand)/sizeof(cand[0]);++ci){
        int threads=cand[ci]; if(threads>max_threads) continue;
        CounterPacked *p=(CounterPacked*)xaligned(64,(size_t)threads*sizeof(CounterPacked));
        CounterPadded *q=(CounterPadded*)xaligned(64,(size_t)threads*sizeof(CounterPadded));
        for(int i=0;i<threads;++i){ p[i].value=0; q[i].value=0; }
        double t0=now_sec();
        #pragma omp parallel num_threads(threads)
        {
            int tid=omp_get_thread_num();
            for(long long k=0;k<iters;++k) p[tid].value++;
        }
        double tp=now_sec()-t0;
        long long sp=0; for(int i=0;i<threads;++i) sp+=p[i].value;
        t0=now_sec();
        #pragma omp parallel num_threads(threads)
        {
            int tid=omp_get_thread_num();
            for(long long k=0;k<iters;++k) q[tid].value++;
        }
        double tq=now_sec()-t0;
        long long sq=0; for(int i=0;i<threads;++i) sq+=q[i].value;
        printf("FALSE_SHARING threads=%d layout=packed seconds=%.6f total=%lld\n",threads,tp,sp);
        printf("FALSE_SHARING threads=%d layout=padded seconds=%.6f total=%lld\n",threads,tq,sq);
        printf("FALSE_SHARING threads=%d packed_over_padded=%.2f\n",threads,tp/tq);
        free(p); free(q);
    }
}

typedef struct {
    double x,y,z,vx,vy,vz,mass;
    uint32_t id,pad;
} Particle;

static void bench_aossoa(void){
    const size_t n=4u*1024u*1024u;
    const int reps=4;
    Particle *p=(Particle*)xaligned(64,n*sizeof(Particle));
    double *x=(double*)xaligned(64,n*sizeof(double));
    double *y=(double*)xaligned(64,n*sizeof(double));
    double *z=(double*)xaligned(64,n*sizeof(double));
    double *vx=(double*)xaligned(64,n*sizeof(double));
    double *vy=(double*)xaligned(64,n*sizeof(double));
    double *vz=(double*)xaligned(64,n*sizeof(double));
    double *mass=(double*)xaligned(64,n*sizeof(double));
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){
        double f=(double)(i&1023)*1e-6;
        p[i]=(Particle){1+f,2+f,3+f,0.1,0.2,0.3,1.0+(i&7),(uint32_t)i,0};
        x[i]=p[i].x; y[i]=p[i].y; z[i]=p[i].z;
        vx[i]=p[i].vx; vy[i]=p[i].vy; vz[i]=p[i].vz; mass[i]=p[i].mass;
    }
    double t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) p[i].x+=1e-6;
    }
    double ta=now_sec()-t0;
    double ca=p[0].x+p[n/2].x+p[n-1].x;
    t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) x[i]+=1e-6;
    }
    double ts=now_sec()-t0;
    double cs=x[0]+x[n/2]+x[n-1];
    printf("AOSSOA kernel=x_only layout=AoS n=%zu reps=%d seconds=%.6f checksum=%.6f struct_bytes=%zu\n",n,reps,ta,ca,sizeof(Particle));
    printf("AOSSOA kernel=x_only layout=SoA n=%zu reps=%d seconds=%.6f checksum=%.6f field_bytes=%zu\n",n,reps,ts,cs,sizeof(double));
    printf("AOSSOA kernel=x_only aos_over_soa=%.2f\n",ta/ts);

    const double dt=0.001;
    t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i){
            p[i].x+=p[i].vx*dt; p[i].y+=p[i].vy*dt; p[i].z+=p[i].vz*dt;
            p[i].vx+=p[i].mass*1e-9;
        }
    }
    double taf=now_sec()-t0;
    ca=p[0].x+p[n/2].y+p[n-1].z+p[n/3].vx;
    t0=now_sec();
    for(int r=0;r<reps;++r){
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i){
            x[i]+=vx[i]*dt; y[i]+=vy[i]*dt; z[i]+=vz[i]*dt;
            vx[i]+=mass[i]*1e-9;
        }
    }
    double tsf=now_sec()-t0;
    cs=x[0]+y[n/2]+z[n-1]+vx[n/3];
    printf("AOSSOA kernel=multi_field layout=AoS n=%zu reps=%d seconds=%.6f checksum=%.6f\n",n,reps,taf,ca);
    printf("AOSSOA kernel=multi_field layout=SoA n=%zu reps=%d seconds=%.6f checksum=%.6f\n",n,reps,tsf,cs);
    printf("AOSSOA kernel=multi_field aos_over_soa=%.2f\n",taf/tsf);
    free(p); free(x); free(y); free(z); free(vx); free(vy); free(vz); free(mass);
}

static double sum_parallel(const double *a,size_t n){
    double s=0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(size_t i=0;i<n;++i) s+=a[i];
    return s;
}
static void bench_first_touch(void){
    const size_t n=32u*1024u*1024u;
    const int reps=4;
    double *a=(double*)xaligned(64,n*sizeof(double));
    for(size_t i=0;i<n;++i) a[i]=1.0;
    double t0=now_sec(); double s1=0.0;
    for(int r=0;r<reps;++r) s1+=sum_parallel(a,n);
    double serial_touch=now_sec()-t0;
    free(a);

    a=(double*)xaligned(64,n*sizeof(double));
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i) a[i]=1.0;
    t0=now_sec(); double s2=0.0;
    for(int r=0;r<reps;++r) s2+=sum_parallel(a,n);
    double parallel_touch=now_sec()-t0;
    printf("FIRST_TOUCH init=serial n=%zu reps=%d seconds=%.6f checksum=%.6e\n",n,reps,serial_touch,s1);
    printf("FIRST_TOUCH init=parallel n=%zu reps=%d seconds=%.6f checksum=%.6e\n",n,reps,parallel_touch,s2);
    printf("FIRST_TOUCH serial_over_parallel=%.2f\n",serial_touch/parallel_touch);
    free(a);
}

static void bench_prefetch(void){
    const size_t n=16u*1024u*1024u;
    const int reps=3;
    double *a=(double*)xaligned(64,n*sizeof(double));
    for(size_t i=0;i<n;++i) a[i]=1.0+(i&15)*1e-6;
    volatile double sum=0.0;
    double t0=now_sec();
    for(int r=0;r<reps;++r) for(size_t i=0;i<n;++i) sum+=a[i];
    double tn=now_sec()-t0; double c1=sum;
    sum=0.0; const size_t d=64;
    t0=now_sec();
    for(int r=0;r<reps;++r) for(size_t i=0;i<n;++i){
        if(i+d<n) __builtin_prefetch(&a[i+d],0,1);
        sum+=a[i];
    }
    double tp=now_sec()-t0; double c2=sum;
    printf("PREFETCH mode=none n=%zu reps=%d seconds=%.6f checksum=%.6f\n",n,reps,tn,c1);
    printf("PREFETCH mode=software distance=%zu n=%zu reps=%d seconds=%.6f checksum=%.6f\n",d,n,reps,tp,c2);
    printf("PREFETCH speedup_none_over_software=%.2f\n",tn/tp);
    free(a);
}

static void bench_sparse(void){
    const int n=1000000, per=5, nnz=n*per;
    double *val=(double*)xaligned(64,(size_t)nnz*sizeof(double));
    int *col=(int*)xaligned(64,(size_t)nnz*sizeof(int));
    int *row=(int*)xaligned(64,(size_t)(n+1)*sizeof(int));
    double *x=(double*)xaligned(64,(size_t)n*sizeof(double));
    double *y=(double*)xaligned(64,(size_t)n*sizeof(double));
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        x[i]=1.0+(i&31)*1e-6;
        row[i]=i*per;
        for(int k=0;k<per;++k){
            int p=i*per+k;
            col[p]=(int)(((uint64_t)i*1315423911ULL+(uint64_t)(k+1)*2654435761ULL)%(uint64_t)n);
            val[p]=1.0/(double)(k+1);
        }
    }
    row[n]=nnz;
    double t0=now_sec();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        double s=0.0;
        for(int j=row[i];j<row[i+1];++j) s+=val[j]*x[col[j]];
        y[i]=s;
    }
    double t=now_sec()-t0;
    double csr=(double)nnz*(sizeof(double)+sizeof(int))+(double)(n+1)*sizeof(int);
    double dense=(double)n*(double)n*sizeof(double);
    double check=y[0]+y[n/2]+y[n-1];
    printf("SPARSE n=%d nnz=%d nnz_per_row=%d csr_MiB=%.2f dense_TiB=%.2f seconds=%.6f checksum=%.6f\n",
           n,nnz,per,csr/1048576.0,dense/1099511627776.0,t,check);
    free(val); free(col); free(row); free(x); free(y);
}

static void usage(const char *p){
    fprintf(stderr,"usage: %s latency|stream|stride|matrix|false_sharing|aossoa|tiling|fusion|first_touch|prefetch|sparse|all\n",p);
}
int main(int argc,char **argv){
    if(argc<2){ usage(argv[0]); return 1; }
    const char *m=argv[1];
    if(!strcmp(m,"latency")) bench_latency();
    else if(!strcmp(m,"stream")) bench_stream();
    else if(!strcmp(m,"stride")) bench_stride();
    else if(!strcmp(m,"matrix")) bench_matrix();
    else if(!strcmp(m,"false_sharing")) bench_false_sharing();
    else if(!strcmp(m,"aossoa")) bench_aossoa();
    else if(!strcmp(m,"tiling")) bench_tiling();
    else if(!strcmp(m,"fusion")) bench_fusion();
    else if(!strcmp(m,"first_touch")) bench_first_touch();
    else if(!strcmp(m,"prefetch")) bench_prefetch();
    else if(!strcmp(m,"sparse")) bench_sparse();
    else if(!strcmp(m,"all")){
        bench_latency(); bench_stream(); bench_stride(); bench_matrix();
        bench_false_sharing(); bench_aossoa(); bench_tiling(); bench_fusion();
        bench_first_touch(); bench_prefetch(); bench_sparse();
    } else { usage(argv[0]); return 1; }
    fprintf(stderr,"sink=%f\n",g_sink);
    return 0;
}
