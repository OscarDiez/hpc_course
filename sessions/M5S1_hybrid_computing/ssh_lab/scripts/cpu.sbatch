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
# Match the compiler/runtime environment before building or running binaries.
source scripts/site.sh
n=${N:-512};steps=${STEPS:-500};repeats=${REPEATS:-3}
# One evidence directory per job keeps logs and measurements associated.
out="results/cpu-$SLURM_JOB_ID";mkdir -p "$out"
{
 date -u;hostname;pwd;echo "JOB=$SLURM_JOB_ID STAGE=$stage N=$n STEPS=$steps REPEATS=$repeats"
 echo "MPI_LAUNCHER=${MPI_LAUNCHER:-srun} SRUN_MPI_TYPE=${SRUN_MPI_TYPE:-site-default}"
 scontrol show job "$SLURM_JOB_ID";gcc --version
 if [[ "$LAB_NEEDS_MPI" == 1 ]]; then mpicc --showme 2>/dev/null || mpicc -show; fi
 if type module >/dev/null 2>&1; then module list; fi
 env | sort | sed -n '/^OMP_/p;/^SLURM_/p'
 sha256sum src/* scripts/*.sh scripts/*.py scripts/*.sbatch
} >"$out/environment.txt" 2>&1
bash scripts/build.sh serial
if [[ "$LAB_NEEDS_MPI" == 1 ]]; then bash scripts/build.sh mpi; fi
if [[ "$stage" == libraries || "$stage" == all ]]; then bash scripts/build.sh libraries; fi
ldd bin/heat_omp >"$out/openmp-libraries.txt"
if [[ "$LAB_NEEDS_MPI" == 1 ]];then ldd bin/heat_mpi >"$out/mpi-libraries.txt";fi
sha256sum bin/* >"$out/binary-hashes.txt"
if [[ "$stage" == libraries || "$stage" == all ]]; then
 sha256sum lib/*/*.so >"$out/library-hashes.txt"
fi
if [[ "$LAB_NEEDS_MPI" == 1 && ${MPI_LAUNCHER:-auto} == auto ]]; then
 echo 'STEP MPI launcher preflight'
 command -v timeout
 export LAB_EXPECT_RANKS=2
 bin/heat_serial 11 2 "$out/launcher-reference.bin" >"$out/launcher-reference.log"
 found=0
 for candidate in srun mpirun; do
  if [[ "$candidate" == srun ]]; then
   launch=(srun --ntasks=2 --cpus-per-task=1 --cpu-bind=cores)
   if [[ -n ${SRUN_MPI_TYPE:-} ]]; then launch+=("--mpi=$SRUN_MPI_TYPE"); fi
  else
   command -v mpirun >/dev/null || continue
   launch=(mpirun -np 2 --map-by slot:PE=1 --bind-to core)
  fi
  if timeout 25s "${launch[@]}" bin/heat_mpi 11 2 "$out/launcher-$candidate.bin" >"$out/launcher-$candidate.log" 2>&1 &&
    python3 scripts/validate.py "$out/launcher-reference.bin" "$out/launcher-$candidate.bin" 11 >>"$out/launcher-$candidate.log" 2>&1; then
   export MPI_LAUNCHER=$candidate;found=1;echo "CHECKPOINT PASS MPI launcher=$candidate";break
  fi
  echo "Launcher $candidate did not pass; log: $out/launcher-$candidate.log"
 done
 [[ "$found" == 1 ]] || { echo 'FAIL no validated MPI launcher';exit 2; }
 echo "VALIDATED_MPI_LAUNCHER=$MPI_LAUNCHER" >>"$out/environment.txt"
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
 all) modes=(naive serial_O0 serial openmp mpi hybrid library_v1 library_v2 library_fixed);;
 *) echo 'STAGE must be serial/openmp/mpi/hybrid/libraries/all' >&2;exit 2;;
esac
# Run configurations sequentially so they do not compete for the same cores.
for mode in "${modes[@]}"; do
 echo "STEP $mode: compile complete; execute and validate"
 ranks=1;threads=1
 case "$mode" in
 naive) exe=bin/heat_naive;; serial_O0) exe=bin/heat_serial_O0;; serial) exe=bin/heat_serial;;
 mpi) exe=bin/heat_mpi;ranks=2;; openmp) exe=bin/heat_omp;threads=2;; hybrid) exe=bin/heat_hybrid;ranks=2;threads=2;;
 library_*) exe=bin/heat_library;version=${mode#library_};
 export LD_LIBRARY_PATH="$PWD/lib/$version${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
 ldd bin/heat_library >"$out/$mode-ldd.txt";;
 esac
 # Repeat zero warms up; measured repetitions start at one.
 for ((r=0;r<=repeats;r++)); do
  bash scripts/measure.sh "$mode" "$ranks" "$threads" "$exe" "$n" "$steps" "$out" "$r"
  # Only the deliberate Kelvin error is expected to differ from serial.
  if [[ "$mode" == library_v2 ]]; then
   python3 scripts/validate.py "$out/reference-r0.bin" "$out/$mode-r$r.bin" "$n" --expect-different
  else
   python3 scripts/validate.py "$out/reference-r0.bin" "$out/$mode-r$r.bin" "$n"
  fi
 done
 echo "CHECKPOINT PASS $mode"
done
python3 scripts/summarize.py "$out"
echo "CPU_GUIDED_LAB_DONE $out"
