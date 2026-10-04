#!/bin/bash
#SBATCH --job-name=m3s1-scale
#SBATCH --partition=cpu
#SBATCH --time=00:06:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --output=m3s1_cpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU
unset SLURM_MEM_PER_NODE

module purge
module load foss/2023b

export OMP_DYNAMIC=FALSE
export M3_MAX_THREADS=${M3_MAX_THREADS:-16}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

printf '============================================================\n'
printf 'M3.S1 REAL CPU / MPI SCALABILITY MINI-LAB\n'
printf '============================================================\n'
printf 'USER=%s\n' "$USER"
printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
printf 'HOST=%s\n' "$(hostname)"
printf 'CPUS_PER_TASK=%s\n' "$SLURM_CPUS_PER_TASK"
printf 'CONFIG map_n=%s weak_n=%s serial_percent=%s stencil_n=%s steps=%s max_threads=%s\n' "${M3_MAP_N:-1000000}" "${M3_WEAK_N:-100000}" "${M3_SERIAL_PERCENT:-5}" "${M3_STENCIL_N:-400000}" "${M3_STEPS:-20}" "$M3_MAX_THREADS"

printf 'GCC=%s\n' "$(command -v gcc)"
printf 'MPICC=%s\n' "$(command -v mpicc)"
gcc --version | head -n 1
printf 'BUILD_FLAGS=-O3 -std=c11 (OpenMP: -fopenmp)\n'

gcc -O3 -fopenmp -std=c11 m3s1_openmp_patterns.c -o m3s1_openmp_patterns
mpicc -O3 -std=c11 m3s1_mpi_reduce.c -o m3s1_mpi_reduce
mpicc -O3 -std=c11 m3s1_mpi_stencil.c -o m3s1_mpi_stencil

section () {
    printf '\n===== EXPERIMENT %s: %s =====\n' "$1" "$2"
}
endsection () {
    printf '===== END EXPERIMENT %s =====\n' "$1"
}

section 1 'OPENMP MAP / STRONG SCALING'
./m3s1_openmp_patterns strong
endsection 1

section 2 'OPENMP MAP / WEAK SCALING'
./m3s1_openmp_patterns weak
endsection 2

section 3 'SMALL WORKLOAD / TOO MANY THREADS'
./m3s1_openmp_patterns small
endsection 3

section 4 'OPENMP REDUCTION'
./m3s1_openmp_patterns reduce
endsection 4

section 5 'TASK FARM / STATIC VS DYNAMIC'
./m3s1_openmp_patterns taskfarm
endsection 5

section 6 'PARALLEL SCAN'
./m3s1_openmp_patterns scan
endsection 6

section 7 'OPENMP STENCIL'
./m3s1_openmp_patterns stencil
endsection 7

section 8 'DIVIDE AND CONQUER / PARALLEL SEARCH'
./m3s1_openmp_patterns search
endsection 8

section 9 'MPI REDUCTION / MAX RANK TIME'
for p in 1 2 4 8; do
    (( p <= M3_MAX_THREADS )) || continue
    srun --exclusive -n "$p" -c 1 ./m3s1_mpi_reduce "${M3_MPI_N:-1000000}"
done
endsection 9

section 10 'MPI STENCIL / HALO EXCHANGE'
for p in 1 2 4 8; do
    (( p <= M3_MAX_THREADS )) || continue
    srun --exclusive -n "$p" -c 1 ./m3s1_mpi_stencil "${M3_STENCIL_N:-400000}" "${M3_STEPS:-20}"
done
endsection 10


section 11 'REAL AMDAHL / CONFIGURABLE SERIAL WORK'
./m3s1_openmp_patterns amdahl
endsection 11

section 12 'MONTE CARLO / REPRODUCIBLE INDEPENDENT SAMPLES'
./m3s1_openmp_patterns montecarlo
endsection 12

printf '\nM3S1_CPU_LAB_RESULT=PASS\n'
