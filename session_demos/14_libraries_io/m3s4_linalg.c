/* Teaching references: row-major dense products and Gaussian elimination.
 * Same double precision and compiler flags for both manual products. */
#include <math.h>
#include <string.h>
void matmul_ijk(int n, const double *a, const double *b, double *c) {
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
        double sum=0;
        for(int k=0;k<n;k++) sum+=a[i*n+k]*b[k*n+j];
        c[i*n+j]=sum;
    }
}
void matmul_ikj(int n, const double *a, const double *b, double *c) {
    memset(c,0,(size_t)n*n*sizeof(double));
    for(int i=0;i<n;i++) for(int k=0;k<n;k++) {
        double v=a[i*n+k];
        for(int j=0;j<n;j++) c[i*n+j]+=v*b[k*n+j];
    }
}
/* Overwrites A and b, just like DGESV. A is row-major here.
 * Set pivot=0 to demonstrate why a nonzero leading pivot matters. */
int gaussian(int n, double *a, double *b, int pivot) {
    for(int k=0;k<n;k++) {
        int p=k;
        if(pivot) for(int i=k+1;i<n;i++) if(fabs(a[i*n+k])>fabs(a[p*n+k])) p=i;
        if(fabs(a[p*n+k])<1e-14) return k+1;
        if(p!=k) {
            for(int j=0;j<n;j++) {double t=a[k*n+j];a[k*n+j]=a[p*n+j];a[p*n+j]=t;}
            double t=b[k];b[k]=b[p];b[p]=t;
        }
        for(int i=k+1;i<n;i++) {
            double f=a[i*n+k]/a[k*n+k];a[i*n+k]=0;
            for(int j=k+1;j<n;j++) a[i*n+j]-=f*a[k*n+j];
            b[i]-=f*b[k];
        }
    }
    for(int i=n-1;i>=0;i--) {
        double v=b[i];
        for(int j=i+1;j<n;j++) v-=a[i*n+j]*b[j];
        b[i]=v/a[i*n+i];
    }
    return 0;
}
