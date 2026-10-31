/* M6S1: small CPU concept simulations, never accelerator benchmarks. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void require(int ok, const char *name) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",name); exit(3); }
    printf("PASS: %s\n",name);
}
static double number(const char *s, double low, double high) {
    char *end; double x=strtod(s,&end);
    if (end==s || *end || !isfinite(x) || x<low || x>high) {
        fprintf(stderr,"FAIL: parameter must be in [%g,%g]\n",low,high); exit(2);
    }
    return x;
}
static int integer(const char *s,int low,int high) {
    double x=number(s,low,high);
    if (x!=floor(x)) { fprintf(stderr,"FAIL: expected integer\n"); exit(2); }
    return (int)x;
}

/* Compare two equivalent linear sensor aggregators. Sparse events are provided
   to the second algorithm. Constructing events from dense frames has a cost
   outside this operation-count model. A CPU can also exploit event streams. */
static void events(int period) {
    const int frames=1000, channels=16;
    double x[16]={0}, y=0, dense_total=0, event_total=0;
    int count=0, all_equal=1;
    for (int t=0;t<frames;t++) {
        if (t%period==0) {
            int c=count%channels;
            double delta=(double)((count%3)+1);
            x[c]+=delta;
            y+=(c+1)*delta; /* Update only the changed channel. */
            count++;
        }
        double reference=0;
        for (int c=0;c<channels;c++) reference+=(c+1)*x[c];
        if (fabs(y-reference)>=1e-12) all_equal=0;
        dense_total+=reference; event_total+=y;
    }
    require(all_equal,"all_event_steps_equal");
    require(dense_total==event_total,"event_checksums_equal");
    printf("RESULT events period=%d frames=%d channels=%d events=%d dense_updates=%d event_updates=%d checksum=%.0f\n",
           period,frames,channels,count,frames*channels,count,event_total);
}

/* Discrete leaky integrate-and-fire teaching model. No training rule. */
static void lif(double threshold) {
    const double decay=0.8, input=0.3;
    double v=0; int spikes=0;
    printf("TRACE step before_reset spike after_reset\n");
    for (int t=0;t<12;t++) {
        double before=decay*v+input;
        int spike=before>=threshold;
        v=spike ? 0 : before;
        spikes+=spike;
        require(v>=0 && (v<threshold || !spike),"lif_state_valid");
        printf("TRACE %d %.9f %d %.9f\n",t,before,spike,v);
    }
    printf("RESULT lif threshold=%.9f spikes=%d final_v=%.9f\n",threshold,spikes,v);
}

/* Same y=W*x mathematics may map to digital, resistive or photonic hardware.
   Quantizing weights is a simplified error model, not a device simulator. */
static void matvec(int bits) {
    double y[4]={0}, q[4]={0}, max_error=0;
    const double levels=(double)((1u<<bits)-1u);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i=0;i<4;i++) {
        for (int j=0;j<4;j++) {
            double w=(double)((i+1)*(j+1))/17.0;
            double x=(double)(j+1)/10.0;
            y[i]+=w*x;
            q[i]+=(round(w*levels)/levels)*x;
        }
    }
    printf("MATRIX row reference quantized abs_error\n");
    for (int i=0;i<4;i++) {
        double err=fabs(q[i]-y[i]);
        require(fabs(y[i]-(i+1)*3.0/17.0)<1e-12,"matvec_reference");
        /* Each quantized weight differs by at most half one level. */
        require(err<=0.5/levels+1e-12,"matvec_quantization_bound");
        if (err>max_error) max_error=err;
        printf("MATRIX %d %.12f %.12f %.12f\n",i,y[i],q[i],err);
    }
    printf("RESULT matvec bits=%d max_error=%.12f\n",bits,max_error);
}

/* Exact one-qubit state-vector simulation: H, phase(phi), H. */
static void quantum(double phi) {
    double complex a=1/sqrt(2.0), b=1/sqrt(2.0);
    b*=cexp(I*phi);
    double complex out0=(a+b)/sqrt(2.0), out1=(a-b)/sqrt(2.0);
    double p0=pow(cabs(out0),2), p1=pow(cabs(out1),2);
    require(fabs(p0+p1-1)<1e-12,"quantum_normalization");
    require(fabs(p1-pow(sin(phi/2),2))<1e-12,"quantum_analytic_probability");
    printf("RESULT quantum phi=%.12f p0=%.12f p1=%.12f\n",phi,p0,p1);
}

int main(int argc,char **argv) {
    if (argc!=3) {
        fprintf(stderr,"Usage: m6s1_lab events PERIOD | lif THRESHOLD | matvec BITS | quantum PHASE\n");
        return 2;
    }
    if (!strcmp(argv[1],"events")) events(integer(argv[2],1,1000));
    else if (!strcmp(argv[1],"lif")) lif(number(argv[2],0.1,2));
    else if (!strcmp(argv[1],"matvec")) matvec(integer(argv[2],2,12));
    else if (!strcmp(argv[1],"quantum")) quantum(number(argv[2],0,6.283185307179587));
    else { fprintf(stderr,"FAIL: unknown experiment\n"); return 2; }
    return 0;
}
