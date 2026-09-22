#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    const int n = 5000000;
    const int reps = 20;
    const size_t bytes = (size_t)n * sizeof(float);

    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);

    for (int i=0; i<n; ++i) {
        a[i]=1.0f;
        b[i]=2.0f;
        c[i]=0.0f;
    }

    double t0 = now_ms();

    #pragma acc data copyin(a[0:n],b[0:n]) copyout(c[0:n])
    {
        for (int r=0; r<reps; ++r) {
            #pragma acc parallel loop present(a[0:n],b[0:n],c[0:n])
            for (int i=0; i<n; ++i)
                c[i] = a[i] + b[i] + 0.0001f*r;
        }
    }

    double t1 = now_ms();
    const float expected = 3.0f + 0.0001f*(reps-1);
    const int ok = fabsf(c[0]-expected) < 1e-4f;

    printf("N=%d reps=%d\n", n, reps);
    printf("OpenACC_elapsed_ms=%.3f\n", t1-t0);
    printf("c[0]=%.4f expected=%.4f\n", c[0], expected);
    printf("OpenACC_check=%s\n", ok ? "PASS" : "FAIL");

    free(a); free(b); free(c);
    return ok ? 0 : 1;
}
