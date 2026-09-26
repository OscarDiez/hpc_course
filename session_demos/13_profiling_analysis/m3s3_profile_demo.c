#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

static volatile double sink_value = 0.0;

static double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *xmalloc_aligned(size_t align, size_t bytes){
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0 || !p) {
        fprintf(stderr, "allocation failed for %.1f MiB\n", bytes / 1048576.0);
        exit(2);
    }
    return p;
}

__attribute__((noinline))
static void initialise(double *a, size_t n){
    for(size_t i=0;i<n;++i)
        a[i] = 1.0 + (double)(i & 1023u) * 1e-6;
}

__attribute__((noinline))
static double compute_bad(const double *a, int n, int reps){
    volatile const double *v = a;
    double sum = 0.0;
    for(int r=0;r<reps;++r){
        for(int j=0;j<n;++j){
            for(int i=0;i<n;++i){
                sum += v[(size_t)i*(size_t)n + (size_t)j];
            }
        }
    }
    return sum;
}

__attribute__((noinline))
static double compute_good(const double *a, int n, int reps){
    volatile const double *v = a;
    double sum = 0.0;
    for(int r=0;r<reps;++r){
        for(int i=0;i<n;++i){
            for(int j=0;j<n;++j){
                sum += v[(size_t)i*(size_t)n + (size_t)j];
            }
        }
    }
    return sum;
}

__attribute__((noinline))
static double reduction_work(const double *a, size_t n){
    volatile const double *v = a;
    double sum = 0.0;
    for(size_t i=0;i<n;++i)
        sum += v[i];
    return sum;
}

__attribute__((noinline))
static double other_work(uint64_t iters){
    double x = 1.000001;
    for(uint64_t i=0;i<iters;++i)
        x = x * 1.00000001 + 0.00000003;
    return x;
}

int main(int argc, char **argv){
    const char *mode = (argc > 1) ? argv[1] : "bad";
    const int n = (argc > 2) ? atoi(argv[2]) : 8192;
    const int reps = (argc > 3) ? atoi(argv[3]) : 3;

    if(strcmp(mode,"bad") != 0 && strcmp(mode,"good") != 0){
        fprintf(stderr,"usage: %s bad|good [N] [reps]\n", argv[0]);
        return 1;
    }

    size_t elems = (size_t)n * (size_t)n;
    size_t bytes = elems * sizeof(double);
    double *a = (double*)xmalloc_aligned(64, bytes);

    double total0 = now_sec();

    double t0 = now_sec();
    initialise(a, elems);
    double t_init = now_sec() - t0;

    t0 = now_sec();
    double c = (strcmp(mode,"bad")==0)
        ? compute_bad(a,n,reps)
        : compute_good(a,n,reps);
    double t_compute = now_sec() - t0;

    t0 = now_sec();
    double r = reduction_work(a, elems);
    double t_reduce = now_sec() - t0;

    t0 = now_sec();
    double o = other_work(30000000ULL);
    double t_other = now_sec() - t0;

    double total = now_sec() - total0;
    sink_value += c + r + o;

    printf("PROFILE_META mode=%s n=%d reps=%d matrix_MiB=%.1f\n",
           mode,n,reps,bytes/1048576.0);
    printf("PROFILE_PHASE mode=%s phase=initialise seconds=%.6f\n",mode,t_init);
    printf("PROFILE_PHASE mode=%s phase=compute seconds=%.6f\n",mode,t_compute);
    printf("PROFILE_PHASE mode=%s phase=reduction seconds=%.6f\n",mode,t_reduce);
    printf("PROFILE_PHASE mode=%s phase=other seconds=%.6f\n",mode,t_other);
    printf("PROFILE_TOTAL mode=%s seconds=%.6f checksum=%.12e\n",mode,total,c+r);
    fprintf(stderr,"sink=%f\n",sink_value);

    free(a);
    return 0;
}
