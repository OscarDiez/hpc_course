#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec/1e9;
}

static void parse(int argc, char **argv, long *n, int *steps,
                  const char **mode, const char **checkpoint){
    *n = 12000000;
    *steps = 80;
    *mode = "baseline";
    *checkpoint = "p3_checkpoint.bin";

    for(int k=1; k<argc; ++k){
        if(!strcmp(argv[k], "--demo")){
            *n = 10000;
            *steps = 4;
        } else if(!strcmp(argv[k], "--n") && k+1<argc){
            *n = atol(argv[++k]);
        } else if(!strcmp(argv[k], "--steps") && k+1<argc){
            *steps = atoi(argv[++k]);
        } else if(!strcmp(argv[k], "--mode") && k+1<argc){
            *mode = argv[++k];
        } else if(!strcmp(argv[k], "--checkpoint") && k+1<argc){
            *checkpoint = argv[++k];
        }
    }
}

__attribute__((noinline))
static void initialise(double *a, double *b, long n){
    for(long i=0; i<n; ++i){
        a[i] = 0.001 * (double)(i % 1000);
        b[i] = 0.002 * (double)((i*17) % 1000);
    }
}

__attribute__((noinline))
static void stage_mix(const double *a, const double *b, double *t1, long n){
    for(long i=0; i<n; ++i)
        t1[i] = 0.63*a[i] + 0.37*b[i];
}

__attribute__((noinline))
static void stage_score(const double *t1, double *t2, long n){
    for(long i=0; i<n; ++i){
        double x = t1[i];
        t2[i] = x*x + 0.015*x + 0.25;
    }
}

__attribute__((noinline))
static void stage_update(const double *t2, const double *b,
                         double *out, long n){
    for(long i=0; i<n; ++i)
        out[i] = 0.80*t2[i] + 0.20*b[i];
}

__attribute__((noinline))
static void pipeline_baseline(double *a, const double *b,
                              double *t1, double *t2, double *out,
                              long n, int steps){
    for(int s=0; s<steps; ++s){
        stage_mix(a, b, t1, n);
        stage_score(t1, t2, n);
        stage_update(t2, b, out, n);

        double *tmp = a;
        a = out;
        out = tmp;
    }
}

__attribute__((noinline))
static void pipeline_optimised(double *a, const double *b,
                               double *t1, double *t2, double *out,
                               long n, int steps){
    /*
      TODO P3 OPTIMISATION

      Do NOT edit this section until you have:
        1. collected the baseline,
        2. inspected the profile,
        3. written your bottleneck hypothesis.

      Replace the three full-array stages with ONE fused loop that computes
      exactly the same mathematical transformation for each element.

      Requirements:
        - keep the mathematics unchanged;
        - keep N and steps unchanged;
        - do not add OpenMP or MPI;
        - do not change compiler flags;
        - use scalar temporaries instead of storing t1 and t2 arrays
          during each timestep.

      The starter implementation below intentionally repeats the baseline.
      Your optimisation replaces only this function body.
    */

    for(int s=0; s<steps; ++s){
        stage_mix(a, b, t1, n);
        stage_score(t1, t2, n);
        stage_update(t2, b, out, n);

        double *tmp = a;
        a = out;
        out = tmp;
    }
}

static double checksum(const double *a, long n){
    double sum = 0.0;
    for(long i=0; i<n; ++i)
        sum += a[i];
    return sum;
}

static int write_checkpoint(const char *path, const double *a, long n){
    FILE *f = fopen(path, "wb");
    if(!f) return 0;

    long sample = n < 4096 ? n : 4096;
    size_t written = fwrite(a, sizeof(double), (size_t)sample, f);
    fclose(f);

    return written == (size_t)sample;
}

int main(int argc, char **argv){
    long n;
    int steps;
    const char *mode;
    const char *checkpoint;

    parse(argc, argv, &n, &steps, &mode, &checkpoint);

    size_t bytes = (size_t)n * sizeof(double);

    double *a   = malloc(bytes);
    double *b   = malloc(bytes);
    double *t1  = malloc(bytes);
    double *t2  = malloc(bytes);
    double *out = malloc(bytes);

    if(!a || !b || !t1 || !t2 || !out){
        fprintf(stderr, "Allocation failed\n");
        return 2;
    }

    initialise(a, b, n);
    memset(t1,  0, bytes);
    memset(t2,  0, bytes);
    memset(out, 0, bytes);

    double total0 = now_s();
    double compute0 = now_s();

    if(!strcmp(mode, "baseline")){
        pipeline_baseline(a, b, t1, t2, out, n, steps);
    } else if(!strcmp(mode, "optimised")){
        pipeline_optimised(a, b, t1, t2, out, n, steps);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 3;
    }

    double compute_s = now_s() - compute0;

    double sum = checksum(a, n);

    double io0 = now_s();
    int io_ok = write_checkpoint(checkpoint, a, n);
    double checkpoint_s = now_s() - io0;

    double total_s = now_s() - total0;

    printf("MODE=%s\n", mode);
    printf("N=%ld\n", n);
    printf("STEPS=%d\n", steps);
    printf("CHECKSUM=%.12e\n", sum);
    printf("COMPUTE_SECONDS=%.6f\n", compute_s);
    printf("CHECKPOINT_SECONDS=%.6f\n", checkpoint_s);
    printf("TOTAL_SECONDS=%.6f\n", total_s);
    printf("IO_OK=%d\n", io_ok);

    free(a);
    free(b);
    free(t1);
    free(t2);
    free(out);

    return io_ok ? 0 : 4;
}
