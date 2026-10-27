#!/bin/bash -l
#SBATCH --job-name=m5s1-basics
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=cpu-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
printf "cpu-%s.out\n" "$SLURM_JOB_ID" > latest_cpu_log.txt
N=${N:-128}; STEPS=${STEPS:-150}
module load OpenMPI/4.1.5-GCC-12.3.0
module list
# Four files, four executables. No compile-time switches choose the algorithm.
gcc -O2 heat_serial.c -o serial
gcc -O2 -fopenmp heat_openmp.c -o openmp
mpicc -O2 heat_mpi.c -o mpi
mpicc -O2 -fopenmp heat_hybrid.c -o hybrid
export OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE
export OMP_PLACES=cores OMP_PROC_BIND=close
# Save each submission in a different results folder.
out="$PWD/results/cpu-$SLURM_JOB_ID"
mkdir -p "$out"
for repeat in 1 2 3; do
    echo "Repeat $repeat" | tee -a "$out/timings.txt"
    srun -n1 -c1 ./serial "$N" "$STEPS" "$out/serial-$repeat.vtk" | tee -a "$out/timings.txt"
    srun -n1 -c2 ./openmp "$N" "$STEPS" "$out/openmp-$repeat.vtk" | tee -a "$out/timings.txt"
    srun -n2 -c2 ./mpi "$N" "$STEPS" "$out/mpi-$repeat.vtk" | tee -a "$out/timings.txt"
    srun -n2 -c2 ./hybrid "$N" "$STEPS" "$out/hybrid-$repeat.vtk" | tee -a "$out/timings.txt"
    for mode in serial openmp mpi hybrid; do
        cmp "$out/serial-1.vtk" "$out/$mode-$repeat.vtk"
        echo "PASS $mode repeat $repeat"
    done
done
# Check one known time step independently of the MPI/OpenMP comparisons.
srun -n1 -c1 ./serial 8 1 "$out/first-step.vtk"
awk 'NR>=19 && NR<=26 { expected=(NR==19 || NR==26)?0:25; if ($1!=expected) exit 1 }' "$out/first-step.vtk"
echo 'PASS first-step row: 0 25 25 25 25 25 25 0'
printf '%s\n' "$out" > latest_cpu.txt
echo CPU_PASS
# Profiling is outside the measured runs.
if command -v gprof >/dev/null; then
    gcc -O0 -g -pg heat_serial.c -o profile
    srun -n1 -c1 ./profile 512 600 "$out/profile.vtk"
    gprof ./profile gmon.out > "$out/profile.txt"
fi
