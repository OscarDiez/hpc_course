/* Teaching reference: direct O(N^2) forward, unnormalized DFT of real data. */
#include <math.h>
void direct_dft(int n, const double *in, double *out) {
    const double tau = 2.0 * acos(-1.0);
    for (int k = 0; k < n; ++k) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < n; ++j) {
            double angle = -tau * (double)k * j / n;
            re += in[j] * cos(angle); im += in[j] * sin(angle);
        }
        out[2*k] = re; out[2*k+1] = im;
    }
}
