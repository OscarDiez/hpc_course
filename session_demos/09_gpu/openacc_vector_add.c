#include <stdio.h>
#include <stdlib.h>

int main(void) {
#ifndef _OPENACC
    fprintf(stderr, "Compile with OpenACC enabled (for example: nvc -O2 -acc openacc_vector_add.c -o openacc_vector_add)\n");
    return 2;
#endif

    const int n = 1024;
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    float *c = (float *)malloc((size_t)n * sizeof(float));
    if (!a || !b || !c) return 2;

    for (int i = 0; i < n; ++i) {
        a[i] = (float)i;
        b[i] = 2.0f * (float)i;
    }

    #pragma acc parallel loop copyin(a[0:n], b[0:n]) copyout(c[0:n])
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) { ok = 0; break; }
    }

    printf("OpenACC vector addition\n");
    printf("N = %d elements\n", n);
    printf("One loop iteration -> one vector element\n");
    printf("Result check: %s\n", ok ? "PASS" : "FAIL");
    for (int i = 0; i < 5; ++i)
        printf("  %.0f + %.0f = %.0f\n", a[i], b[i], c[i]);

    free(a); free(b); free(c);
    return ok ? 0 : 1;
}
