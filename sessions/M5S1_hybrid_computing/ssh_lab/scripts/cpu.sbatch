#!/bin/bash -l
#SBATCH --job-name=m5s1-ssh
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=cpu-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
stage=${STAGE:-all};export LAB_DEVICE=cpu LAB_NEEDS_MPI=0
if [[ "$stage" == mpi || "$stage" == hybrid || "$stage" == all ]]; then LAB_NEEDS_MPI=1; fi
source scripts/site.sh
n=${N:-512};steps=${STEPS:-500};repeats=${REPEATS:-3}
out="results/cpu-$SLURM_JOB_ID";mkdir -p "$out"
{
 date -u;hostname;pwd;echo "JOB=$SLURM_JOB_ID STAGE=$stage N=$n STEPS=$steps REPEATS=$repeats"
 echo "MPI_LAUNCHER=${MPI_LAUNCHER:-srun} SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-site-default}"
 scontrol show job "$SLURM_JOB_ID";gcc --version
 if [[ "$LAB_NEEDS_MPI" == 1 ]]; then mpicc --showme; fi
 if type module >/dev/null 2>&1; then module list; fi
 env | sort | sed -n '/^OMP_/p;/^SLURM_/p'
 sha256sum src/* scripts/*
} >"$out/environment.txt" 2>&1
bash scripts/build.sh serial
if [[ "$LAB_NEEDS_MPI" == 1 ]]; then bash scripts/build.sh mpi; fi
if [[ "$stage" == libraries || "$stage" == all ]]; then bash scripts/build.sh libraries; fi
sha256sum bin/* >"$out/binary-hashes.txt"
if [[ "$stage" == libraries || "$stage" == all ]]; then
 sha256sum lib/*/*.so >"$out/library-hashes.txt"
fi
# All stages make an independent serial reference. Repeat 0 warms up; do not plot it.
bash scripts/measure.sh reference 1 1 bin/heat_serial "$n" "$steps" "$out" 0
modes=()
case "$stage" in
 serial) modes=(naive serial_O0 serial);;
 openmp) modes=(serial openmp);;
 mpi) modes=(serial mpi);;
 hybrid) modes=(serial mpi openmp hybrid);;
 libraries) modes=(library_v1 library_v2 library_fixed);;
 all) modes=(naive serial_O0 serial mpi openmp hybrid library_v1 library_v2 library_fixed);;
 *) echo 'STAGE must be serial/openmp/mpi/hybrid/libraries/all' >&2;exit 2;;
esac
for mode in "${modes[@]}"; do
 ranks=1;threads=1
 case "$mode" in
 naive) exe=bin/heat_naive;; serial_O0) exe=bin/heat_serial_O0;; serial) exe=bin/heat_serial;;
 mpi) exe=bin/heat_mpi;ranks=2;; openmp) exe=bin/heat_omp;threads=4;; hybrid) exe=bin/heat_hybrid;ranks=2;threads=2;;
 library_*) exe=bin/heat_library;version=${mode#library_};
 export LD_LIBRARY_PATH="$PWD/lib/$version${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
 ldd bin/heat_library >"$out/$mode-ldd.txt";;
 esac
 for ((r=0;r<=repeats;r++)); do
  bash scripts/measure.sh "$mode" "$ranks" "$threads" "$exe" "$n" "$steps" "$out" "$r"
  if [[ "$mode" == library_v2 ]]; then
   python3 scripts/validate.py "$out/reference-r0.bin" "$out/$mode-r$r.bin" "$n" --expect-different
  else
   python3 scripts/validate.py "$out/reference-r0.bin" "$out/$mode-r$r.bin" "$n"
  fi
 done
done
python3 scripts/summarize.py "$out"
echo "CPU_GUIDED_LAB_DONE $out"
