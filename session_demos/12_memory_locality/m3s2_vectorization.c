
#include <stddef.h>

__attribute__((noinline))
void vectorizable_saxpy(size_t n, double * restrict y, const double * restrict x, double a){
    for(size_t i=0;i<n;++i) y[i] = a*x[i] + y[i];
}

__attribute__((noinline))
void loop_carried_dependency(size_t n, double *x, const double *a){
    for(size_t i=1;i<n;++i) x[i] = x[i-1] + a[i];
}
