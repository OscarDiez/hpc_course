#!/bin/bash
# Called inside an allocation, once per program execution. Separate raw logs per trial.
set -euo pipefail
# Each call receives a mode, worker layout, executable, inputs and repeat number.
mode=$1; ranks=$2; threads=$3; exe=$4; n=$5; steps=$6; out=$7; repeat=$8
mkdir -p "$out"
# Thread count does not allocate CPUs; the job/step resource requests do that.
export OMP_NUM_THREADS=$threads
log="$out/$mode-r$repeat.log"; field="$out/$mode-r$repeat.bin"
# MPI modes need a launcher that creates a shared MPI communicator.
if [[ "$mode" == mpi || "$mode" == hybrid ]]; then
 export LAB_EXPECT_RANKS=$ranks
 case ${MPI_LAUNCHER:-srun} in
  srun) launch=(srun --ntasks="$ranks" --cpus-per-task="$threads" --cpu-bind=cores)
        if [[ -n ${SRUN_MPI_TYPE:-} ]]; then launch+=("--mpi=$SRUN_MPI_TYPE"); fi;;
  mpirun) launch=(mpirun -np "$ranks" --map-by "slot:PE=$threads" --bind-to core);;
  *) echo 'MPI_LAUNCHER must be srun or mpirun' >&2; exit 2;;
 esac
else
 launch=(srun --ntasks=1 --cpus-per-task="$threads" --cpu-bind=cores)
fi
# /usr/bin/time includes launcher/startup/compute/output; RESULT loop_s is solver-only.
if [[ -x /usr/bin/time ]]; then
 /usr/bin/time -f 'WALL elapsed_s=%e user_s=%U sys_s=%S maxrss_kb=%M' \
  "${launch[@]}" "$exe" "$n" "$steps" "$field" >"$log" 2>&1
else
 # Bash's timer still measures real execution; peak RSS is explicitly unavailable.
 TIMEFORMAT='WALL elapsed_s=%R user_s=%U sys_s=%S maxrss_kb=UNAVAILABLE'
 { time "${launch[@]}" "$exe" "$n" "$steps" "$field"; } >"$log" 2>&1
fi
cat "$log"
