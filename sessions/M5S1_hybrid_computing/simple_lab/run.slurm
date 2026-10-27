#!/bin/bash -l
#SBATCH --job-name=m5s1-simple
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=cpu-%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
N=${N:-128}; STEPS=${STEPS:-150}; REPEATS=${REPEATS:-3}
module load OpenMPI/4.1.5-GCC-12.3.0
module list 2>modules.txt
gcc --version | head -1
# Each compiler command makes one change that we can explain.
gcc -O0 -DNAIVE heat.c -o naive
gcc -O0 -DSERIAL_O0 heat.c -o serial_O0
gcc -O2 heat.c -o serial
gcc -O2 -fopenmp heat.c -o openmp
mpicc -O2 heat_mpi.c -o mpi
mpicc -O2 -fopenmp heat_mpi.c -o hybrid
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
printf 'mode,n,steps,ranks,threads,seconds\n' > timings.csv
for repeat in $(seq 1 "$REPEATS"); do
    # One log row and a separate field file per measured run.
    srun -n1 -c1 ./naive "$N" "$STEPS" "naive-$repeat.bin" >>timings.csv
    srun -n1 -c1 ./serial_O0 "$N" "$STEPS" "serial_O0-$repeat.bin" >>timings.csv
    srun -n1 -c1 ./serial "$N" "$STEPS" "serial-$repeat.bin" >>timings.csv
    OMP_NUM_THREADS=2 srun -n1 -c2 ./openmp "$N" "$STEPS" "openmp-$repeat.bin" >>timings.csv
    OMP_NUM_THREADS=1 srun -n2 -c2 ./mpi "$N" "$STEPS" "mpi-$repeat.bin" >>timings.csv
    OMP_NUM_THREADS=2 srun -n2 -c2 ./hybrid "$N" "$STEPS" "hybrid-$repeat.bin" >>timings.csv
done
# A one-step run has a known answer, independent of the other implementations.
srun -n1 -c1 ./serial 8 1 first-step.bin > first-step.csv
# Profiling is separate from the timing comparison, because instrumentation has a cost.
if command -v gprof >/dev/null; then
    gcc -O0 -g -pg -DNAIVE heat.c -o heat_profile
    srun -n1 -c1 ./heat_profile 512 600 profile.bin > profile-time.csv
    gprof ./heat_profile gmon.out > profile.txt
fi
echo CPU_DONE
# Visualisation is post-processing. A render failure does not erase CPU results.
if module load ParaView/5.11.2-foss-2023a; then
    module list 2>paraview-modules.txt
    export LIBGL_ALWAYS_SOFTWARE=true GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless LP_NUM_THREADS=1
    unset DISPLAY WAYLAND_DISPLAY PV_DEBUG_REMOTE_RENDERING
    ulimit -c 0
    if pvbatch --force-offscreen-rendering view.py "$N" serial-1.bin; then
        echo PARAVIEW_DONE
    else
        echo 'PARAVIEW_FAILED: inspect this log for the rendering error.'
    fi
else
    echo 'PARAVIEW_UNAVAILABLE: the module could not be loaded.'
fi
